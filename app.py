import os
import math
import time
import json
import sqlite3
import fnmatch
import logging
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ha-ml-predictor")

# --- Configuration via Environment Variables ---
HA_URL = os.getenv("HA_URL", "http://homeassistant.local:8123").rstrip("/")
HA_TOKEN = os.getenv("HA_TOKEN", "")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER", "")
MQTT_PASS = os.getenv("MQTT_PASS", "")
MQTT_PREFIX = os.getenv("MQTT_DISCOVERY_PREFIX", "homeassistant")

# Tier 1: Entities to predict (all person.* are always included automatically)
BINARY_SENSOR_PATTERNS = [p.strip() for p in os.getenv("BINARY_SENSOR_PATTERNS", "").split(",") if p.strip()]

# Bedroom presence patterns used for Sleep / Wake-Up inference
BEDROOM_SENSOR_PATTERNS = [
    p.strip()
    for p in os.getenv("BEDROOM_SENSOR_PATTERNS", "*bedroom*,*soba*,*spavaca*,*bed*").split(",")
    if p.strip()
]

# Tier 2: Human-triggered context entities for cross-sensor correlation (not predicted, rolling retention)
CONTEXT_ENTITY_PATTERNS = [
    p.strip()
    for p in os.getenv(
        "CONTEXT_ENTITY_PATTERNS",
        "binary_sensor.*_door,binary_sensor.*_window,binary_sensor.*_presence,binary_sensor.*_motion,lock.*,media_player.*",
    ).split(",")
    if p.strip()
]
EXCLUDE_PATTERNS = [
    p.strip()
    for p in os.getenv(
        "EXCLUDE_PATTERNS",
        "*_predicted,*_battery*,*_rssi*,*_linkquality*,*_update*,*_uptime*,*_power*,*_energy*,*_voltage*,*_current*,*_temperature*,*_humidity*,climate.*,light.*,cover.*",
    ).split(",")
    if p.strip()
]

CERTAINTY_THRESHOLD = float(os.getenv("CERTAINTY_THRESHOLD", "0.70"))
MIN_ACCEPTABLE_ACCURACY = float(os.getenv("MIN_ACCEPTABLE_ACCURACY", "0.70"))
PREDICT_HORIZON_MINS = int(os.getenv("PREDICT_HORIZON_MINS", "30"))
RESAMPLE_MINS = int(os.getenv("RESAMPLE_MINS", "5"))
VELOCITY_LOOKBACK_MINS = int(os.getenv("VELOCITY_LOOKBACK_MINS", "15"))
INITIAL_HISTORY_DAYS = int(os.getenv("INITIAL_HISTORY_DAYS", "10"))
CONTEXT_RETENTION_DAYS = int(os.getenv("CONTEXT_RETENTION_DAYS", "14"))
MIN_HISTORY_HOURS = int(os.getenv("MIN_HISTORY_HOURS", "24"))
DECAY_HALF_LIFE_DAYS = float(os.getenv("DECAY_HALF_LIFE_DAYS", "7.0"))
CORRELATION_MIN_STRENGTH = float(os.getenv("CORRELATION_MIN_STRENGTH", "0.35"))
CORRELATION_INTERVAL_HOURS = float(os.getenv("CORRELATION_INTERVAL_HOURS", "24.0"))
DB_PATH = os.getenv("DB_PATH", "/data/history.db")

HEADERS = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
HOME_COORDS = (None, None)
LOCAL_TZ = datetime.now().astimezone().tzinfo

ACTIVE_STATES = {"on", "home", "open", "unlocked", "playing", "heating", "cooling", "true"}
INACTIVE_STATES = {"off", "not_home", "closed", "locked", "idle", "paused", "standby", "false"}


# --- Geospatial Helper ---
def haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return 0.0
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_eta_mins(distance_km, approach_speed_kmh, is_home=False):
    if is_home or distance_km <= 0.1:
        return 0.0
    if approach_speed_kmh <= 2.0:
        return 1440.0
    return min(1440.0, (distance_km / approach_speed_kmh) * 60.0)


def state_to_binary(state_str, is_person=False):
    if state_str is None:
        return 0
    s = str(state_str).lower()
    if is_person:
        return 1 if s == "home" else 0
    return 1 if s in ACTIVE_STATES else 0


# --- Database Layer ---
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Tier 1: Permanent archive for predicted entities
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state_history (
            entity_id TEXT,
            timestamp TEXT,
            state TEXT,
            attributes TEXT,
            PRIMARY KEY (entity_id, timestamp)
        )
    """)
    # Tier 2: Rolling short-term archive for context entities
    conn.execute("""
        CREATE TABLE IF NOT EXISTS context_history (
            entity_id TEXT,
            timestamp TEXT,
            state TEXT,
            PRIMARY KEY (entity_id, timestamp)
        )
    """)
    # Self-scoring prediction audit log
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT,
            predicted_at TEXT,
            target_ts TEXT,
            predicted_state TEXT,
            certainty REAL,
            actual_state TEXT,
            is_correct INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_state_hist_eid ON state_history(entity_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ctx_hist_eid ON context_history(entity_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_eid_target ON prediction_audit(entity_id, target_ts)")
    conn.commit()
    return conn


def get_last_timestamp(conn, table, entity_id):
    cur = conn.execute(f"SELECT MAX(timestamp) FROM {table} WHERE entity_id = ?", (entity_id,))
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def prune_old_records(conn):
    ctx_cutoff = (datetime.now(timezone.utc) - timedelta(days=CONTEXT_RETENTION_DAYS)).isoformat()
    c1 = conn.execute("DELETE FROM context_history WHERE timestamp < ?", (ctx_cutoff,))

    audit_cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    c2 = conn.execute("DELETE FROM prediction_audit WHERE predicted_at < ?", (audit_cutoff,))
    conn.commit()
    if c1.rowcount > 0 or c2.rowcount > 0:
        log.info(f"Pruned {c1.rowcount} context rows and {c2.rowcount} old audit rows.")


def archive_states(conn, states_list, is_context=False):
    inserted = 0
    for entity_history in states_list:
        for entry in entity_history:
            entity_id = entry.get("entity_id")
            if not is_context and entity_id and entity_id.startswith("person."):
                ts = entry.get("last_updated") or entry.get("last_changed")
            else:
                ts = entry.get("last_changed") or entry.get("last_updated")

            state = entry.get("state")
            if not entity_id or not ts or state in ("unavailable", "unknown", None):
                continue

            try:
                if is_context:
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO context_history (entity_id, timestamp, state) VALUES (?, ?, ?)",
                        (entity_id, ts, state),
                    )
                else:
                    attrs = json.dumps(entry.get("attributes", {}))
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO state_history (entity_id, timestamp, state, attributes) VALUES (?, ?, ?, ?)",
                        (entity_id, ts, state, attrs),
                    )
                inserted += cur.rowcount
            except sqlite3.Error as e:
                log.error(f"DB Insert error: {e}")
    conn.commit()
    return inserted


# --- Self-Scoring Accuracy & Auto-Calibration ---
def log_and_score_predictions(conn, entity_id, predicted_state, certainty, live_state, now_dt):
    """
    1. Grades any past predictions whose target_ts has now elapsed.
    2. Logs the current prediction for future grading.
    3. Computes 7-day accuracy and auto-calibrates the certainty threshold if accuracy degrades.
    """
    now_iso = now_dt.isoformat()
    is_person = entity_id.startswith("person.")
    norm_live = ("home" if live_state == "home" else "not_home") if is_person else live_state

    # Grade matured predictions
    cur = conn.execute(
        "SELECT id, predicted_state FROM prediction_audit WHERE entity_id = ? AND actual_state IS NULL AND target_ts <= ?",
        (entity_id, now_iso),
    )
    ungraded = cur.fetchall()
    for row_id, pred_st in ungraded:
        if pred_st == "unknown":
            # 'unknown' abstentions are marked evaluated without counting as right or wrong
            conn.execute(
                "UPDATE prediction_audit SET actual_state = ?, is_correct = NULL WHERE id = ?",
                (norm_live, row_id),
            )
        else:
            correct = 1 if pred_st == norm_live else 0
            conn.execute(
                "UPDATE prediction_audit SET actual_state = ?, is_correct = ? WHERE id = ?",
                (norm_live, correct, row_id),
            )

    # Record current prediction for evaluation after PREDICT_HORIZON_MINS
    target_iso = (now_dt + timedelta(minutes=PREDICT_HORIZON_MINS)).isoformat()
    conn.execute(
        "INSERT INTO prediction_audit (entity_id, predicted_at, target_ts, predicted_state, certainty) VALUES (?, ?, ?, ?, ?)",
        (entity_id, now_iso, target_iso, predicted_state, float(certainty)),
    )
    conn.commit()

    # Calculate 7-day accuracy over non-unknown predictions
    cutoff_7d = (now_dt - timedelta(days=7)).isoformat()
    stats = conn.execute(
        "SELECT COUNT(*), SUM(is_correct) FROM prediction_audit WHERE entity_id = ? AND predicted_at >= ? AND is_correct IS NOT NULL",
        (entity_id, cutoff_7d),
    ).fetchone()

    total_graded = stats[0] or 0
    total_correct = stats[1] or 0
    accuracy_7d = round(total_correct / total_graded, 3) if total_graded > 0 else None

    # Auto-calibrate threshold: if accuracy over >= 20 predictions falls below MIN_ACCEPTABLE_ACCURACY,
    # tighten the required certainty so noisy sensors abstain ('unknown') more often.
    effective_threshold = CERTAINTY_THRESHOLD
    if total_graded >= 20 and accuracy_7d is not None and accuracy_7d < MIN_ACCEPTABLE_ACCURACY:
        deficit = MIN_ACCEPTABLE_ACCURACY - accuracy_7d
        effective_threshold = min(0.92, round(CERTAINTY_THRESHOLD + deficit * 0.75, 2))

    return accuracy_7d, total_graded, effective_threshold


# --- Home Assistant API ---
def fetch_home_coords():
    global HOME_COORDS
    try:
        resp = requests.get(f"{HA_URL}/api/states/zone.home", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            attrs = resp.json().get("attributes", {})
            HOME_COORDS = (attrs.get("latitude"), attrs.get("longitude"))
    except Exception as e:
        log.warning(f"Could not fetch zone.home coordinates: {e}")


def is_excluded(eid):
    return any(fnmatch.fnmatch(eid, pat) for pat in EXCLUDE_PATTERNS)


def get_entities():
    resp = requests.get(f"{HA_URL}/api/states", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    all_states = resp.json()

    targets = []
    target_ids = set()
    context_entities = []

    for item in all_states:
        eid = item["entity_id"]
        if eid.startswith("person."):
            targets.append(item)
            target_ids.add(eid)
        elif eid.startswith("binary_sensor.") and any(fnmatch.fnmatch(eid, pat) for pat in BINARY_SENSOR_PATTERNS):
            targets.append(item)
            target_ids.add(eid)

    for item in all_states:
        eid = item["entity_id"]
        if eid in target_ids or is_excluded(eid):
            continue
        if any(fnmatch.fnmatch(eid, pat) for pat in CONTEXT_ENTITY_PATTERNS):
            st = str(item.get("state", "")).lower()
            if st in ACTIVE_STATES or st in INACTIVE_STATES:
                context_entities.append(item)

    return targets, context_entities


def sync_entity_group(conn, entities, is_context=False):
    table = "context_history" if is_context else "state_history"
    default_days = min(INITIAL_HISTORY_DAYS, CONTEXT_RETENTION_DAYS) if is_context else INITIAL_HISTORY_DAYS

    for item in entities:
        eid = item["entity_id"]
        last_ts = get_last_timestamp(conn, table, eid)
        if last_ts:
            start_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        else:
            start_dt = datetime.now(timezone.utc) - timedelta(days=default_days)

        url = f"{HA_URL}/api/history/period/{start_dt.isoformat()}"
        params = {"filter_entity_id": eid}
        if is_context:
            params["minimal_response"] = "1"
            params["no_attributes"] = "1"

        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 200:
                count = archive_states(conn, resp.json(), is_context=is_context)
                if count > 0 and not is_context:
                    log.info(f"Archived {count} new state entries for {eid}")
        except Exception as e:
            log.warning(f"Failed syncing history for {eid}: {e}")

        if not is_context and eid.startswith("person."):
            ts_now = item.get("last_updated") or item.get("last_changed") or datetime.now(timezone.utc).isoformat()
        else:
            ts_now = item.get("last_changed") or datetime.now(timezone.utc).isoformat()

        state_now = item.get("state")
        if state_now not in ("unavailable", "unknown", None):
            if is_context:
                conn.execute(
                    "INSERT OR IGNORE INTO context_history (entity_id, timestamp, state) VALUES (?, ?, ?)",
                    (eid, ts_now, state_now),
                )
            else:
                conn.execute(
                    "INSERT OR IGNORE INTO state_history (entity_id, timestamp, state, attributes) VALUES (?, ?, ?, ?)",
                    (eid, ts_now, state_now, json.dumps(item.get("attributes", {}))),
                )
    conn.commit()


# --- Sleep, Wake-Up & Bedtime Inference ---
def infer_sleep_schedule(conn, targets):
    """
    Infers typical bedtime and wake-up time from bedroom presence sensors (weekday vs weekend)
    and projects the next expected wake-up and bedtime timestamps.
    """
    bedroom_eids = [
        t["entity_id"]
        for t in targets
        if t["entity_id"].startswith("binary_sensor.")
        and any(fnmatch.fnmatch(t["entity_id"], pat) for pat in BEDROOM_SENSOR_PATTERNS)
    ]

    if not bedroom_eids:
        return None

    # Use the primary bedroom sensor with the most recorded history
    best_eid = None
    best_df = pd.DataFrame()
    for eid in bedroom_eids:
        df = pd.read_sql_query(
            "SELECT timestamp, state FROM state_history WHERE entity_id = ? ORDER BY timestamp ASC",
            conn,
            params=(eid,),
        )
        if len(df) > len(best_df):
            best_df = df
            best_eid = eid

    if best_df.empty or len(best_df) < 4:
        return None

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(LOCAL_TZ)

    best_df["timestamp"] = pd.to_datetime(best_df["timestamp"], format="ISO8601", utc=True)
    best_df = best_df.drop_duplicates(subset=["timestamp"], keep="last")
    best_df["is_on"] = best_df["state"].apply(lambda s: 1 if str(s).lower() == "on" else 0)

    # Resample to 10-minute blocks and bridge brief <20m midnight bathroom breaks
    s = best_df.set_index("timestamp")["is_on"]
    s.loc[now_utc] = s.iloc[-1]
    s_10m = s.resample("10min").ffill().fillna(0)

    # Smooth out brief 1-2 block (10-20m) gaps inside a long sleep stretch
    s_smooth = s_10m.rolling(window=3, min_periods=1, center=True).max().astype(int)

    # Group contiguous 'on' blocks
    groups = (s_smooth != s_smooth.shift()).cumsum()
    sessions = []
    for _, grp in s_smooth.groupby(groups):
        if grp.iloc[0] != 1:
            continue
        start_local = grp.index.min().tz_convert(LOCAL_TZ)
        end_local = grp.index.max().tz_convert(LOCAL_TZ)
        duration_hrs = (end_local - start_local).total_seconds() / 3600.0

        # Valid overnight sleep session: >= 3.5 hours, starting between 19:00 and 05:00
        if duration_hrs >= 3.5 and (start_local.hour >= 19 or start_local.hour <= 5):
            # Ignore an ongoing session still active right now when measuring wake-up time
            is_ongoing = (now_local - end_local).total_seconds() < 900
            bed_hour = start_local.hour + start_local.minute / 60.0
            if bed_hour < 12:
                bed_hour += 24.0  # Represent 01:30 AM as 25.5 for clean averaging
            wake_hour = end_local.hour + end_local.minute / 60.0
            is_weekend_wake = end_local.weekday() >= 5
            sessions.append({
                "bed_hour": bed_hour,
                "wake_hour": wake_hour if not is_ongoing else None,
                "duration_hrs": duration_hrs if not is_ongoing else None,
                "is_weekend": is_weekend_wake,
            })

    if not sessions:
        return None

    sess_df = pd.DataFrame(sessions)
    completed = sess_df.dropna(subset=["wake_hour"])
    if completed.empty:
        return None

    overall_wake = float(completed["wake_hour"].median())
    wk
