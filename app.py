"""
HA ML Predictor (patched).

Main changes vs. the original:
  * History sync: explicit end_time, URL-encoded start, significant_changes_only=0 for persons,
    batched requests, per-entity sync watermark (sync_state table).
  * minimal_response handling: entity_id carried over from the first entry of each list.
  * Real timezone (HA config / TZ / LOCAL_TZ) instead of a fixed UTC offset.
  * State-group fix (mins_in_current_state no longer resets on not_home -> zone changes).
  * Holiday cache / home coordinates survive transient failures; missing GPS -> NaN, not 0.
  * Timestamps normalised to a fixed-width UTC string; one-time migration of old rows.
  * Accuracy tracking: logs every prediction (incl. abstentions) + persistence baseline.
  * Training window bounded and pruned (keeps one anchor row per entity); models cached
    and retrained every RETRAIN_INTERVAL_MINS instead of every loop.
  * Smoothed + hysteresis-based next ON/OFF estimates.
  * Constant-state shortcut capped and aware of the live state.
  * MQTT attribute payloads sanitised (NaN -> null) and size-limited.
"""
import os
import math
import time
import json
import sqlite3
import fnmatch
import logging
from datetime import datetime, timedelta, timezone, date
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

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

# Tier 2: Optional context entities for cross-sensor correlation
CONTEXT_ENTITY_PATTERNS = [
    p.strip()
    for p in os.getenv("CONTEXT_ENTITY_PATTERNS", "light.*,media_player.*,lock.*,cover.*,binary_sensor.*_door").split(",")
    if p.strip()
]
EXCLUDE_PATTERNS = [
    p.strip()
    for p in os.getenv(
        "EXCLUDE_PATTERNS",
        "*_battery*,*_rssi*,*_linkquality*,*_update*,*_uptime*,*_power*,*_energy*,*_voltage*,*_current*,*_temperature*,*_humidity*",
    ).split(",")
    if p.strip()
]

# Holiday Calendar (preferred, e.g. "calendar.croatia") & optional Workday binary_sensor fallback
HOLIDAY_CALENDAR_ENTITY_ID = os.getenv("HOLIDAY_CALENDAR_ENTITY_ID", "").strip()
WORKDAY_ENTITY_ID = os.getenv("WORKDAY_ENTITY_ID", "binary_sensor.workday_sensor").strip()

CERTAINTY_THRESHOLD = float(os.getenv("CERTAINTY_THRESHOLD", "0.70"))
MIN_ACCEPTABLE_ACCURACY = float(os.getenv("MIN_ACCEPTABLE_ACCURACY", "0.65"))
PREDICT_HORIZON_MINS = int(os.getenv("PREDICT_HORIZON_MINS", "30"))
EXTRA_HORIZONS_MINS = [
    int(h.strip()) for h in os.getenv("EXTRA_HORIZONS_MINS", "15,30,60,180").split(",") if h.strip()
]
RESAMPLE_MINS = int(os.getenv("RESAMPLE_MINS", "5"))
VELOCITY_LOOKBACK_MINS = int(os.getenv("VELOCITY_LOOKBACK_MINS", "15"))
INITIAL_HISTORY_DAYS = int(os.getenv("INITIAL_HISTORY_DAYS", "10"))
CONTEXT_RETENTION_DAYS = int(os.getenv("CONTEXT_RETENTION_DAYS", "14"))
MIN_HISTORY_HOURS = int(os.getenv("MIN_HISTORY_HOURS", "24"))
DECAY_HALF_LIFE_DAYS = float(os.getenv("DECAY_HALF_LIFE_DAYS", "7.0"))
CORRELATION_MIN_STRENGTH = float(os.getenv("CORRELATION_MIN_STRENGTH", "0.28"))
CORRELATION_INTERVAL_HOURS = float(os.getenv("CORRELATION_INTERVAL_HOURS", "24.0"))
DB_PATH = os.getenv("DB_PATH", "/data/history.db")

# New knobs
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", str(max(14, int(math.ceil(DECAY_HALF_LIFE_DAYS * 4))))))
RETRAIN_INTERVAL_MINS = int(os.getenv("RETRAIN_INTERVAL_MINS", "60"))
HISTORY_BATCH_SIZE = int(os.getenv("HISTORY_BATCH_SIZE", "20"))
LOOP_SECS = int(os.getenv("LOOP_SECS", "300"))
HOLIDAY_REFRESH_HOURS = float(os.getenv("HOLIDAY_REFRESH_HOURS", "6"))
MAX_ATTR_BYTES = 14000  # HA drops state attributes above 16 KB

HEADERS = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
HOME_COORDS = (None, None)
LOCAL_TZ = datetime.now().astimezone().tzinfo  # fallback only; replaced by init_timezone()

ACTIVE_STATES = {"on", "home", "open", "unlocked", "playing", "heating", "cooling", "true"}
INACTIVE_STATES = {"off", "not_home", "closed", "locked", "idle", "paused", "standby", "false"}

SCHEDULE_COLS = [
    "hour_sin", "hour_cos", "hour_sin2", "hour_cos2", "hour_sin3", "hour_cos3",
    "dow_sin", "dow_cos", "is_weekend", "is_workday", "is_tomorrow_workday",
]

MODEL_CACHE = {}


# --- Time helpers ---
def parse_ts(s):
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fmt_ts(dt):
    """Fixed-width UTC string (32 chars) so SQL string comparison == chronological comparison."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def norm_ts(s):
    try:
        return fmt_ts(parse_ts(s))
    except (ValueError, TypeError):
        return None


def init_timezone():
    global LOCAL_TZ
    names = []
    override = os.getenv("LOCAL_TZ", "").strip()
    if override:
        names.append(override)
    try:
        r = requests.get(f"{HA_URL}/api/config", headers=HEADERS, timeout=10)
        if r.status_code == 200 and r.json().get("time_zone"):
            names.append(r.json()["time_zone"])
    except Exception as e:
        log.warning(f"Could not read HA config for timezone: {e}")
    if os.getenv("TZ"):
        names.append(os.getenv("TZ"))

    for name in names:
        try:
            LOCAL_TZ = ZoneInfo(name)
            log.info(f"Using timezone {name}")
            return
        except Exception as e:
            log.warning(f"Invalid timezone '{name}': {e}")
    log.warning("Falling back to a fixed UTC offset; DST changes will not be followed.")


# --- Geospatial Helper ---
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance; returns NaN when any coordinate is missing (never 0)."""
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except (TypeError, ValueError):
        return float("nan")
    if any(math.isnan(v) for v in (lat1, lon1, lat2, lon2)):
        return float("nan")
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_approach_speed(prev_dist, curr_dist, hours):
    if hours <= 0 or math.isnan(prev_dist) or math.isnan(curr_dist):
        return float("nan")
    s = (prev_dist - curr_dist) / hours
    if abs(s) < 1.5:
        return 0.0
    return max(-250.0, min(250.0, s))


def compute_eta_mins(distance_km, approach_speed_kmh, is_home=False):
    if is_home:
        return 0.0
    if distance_km is None or math.isnan(distance_km):
        return float("nan")
    if distance_km <= 0.1:
        return 0.0
    if approach_speed_kmh is None or math.isnan(approach_speed_kmh) or approach_speed_kmh <= 2.0:
        return 1440.0
    return min(1440.0, (distance_km / approach_speed_kmh) * 60.0)


def state_to_binary(state_str, is_person=False):
    if state_str is None:
        return 0
    s = str(state_str).lower()
    if is_person:
        return 1 if s == "home" else 0
    return 1 if s in ACTIVE_STATES else 0


def normalize_label(eid, raw_state):
    if eid.startswith("person."):
        return "home" if raw_state == "home" else "not_home"
    return "on" if str(raw_state).lower() in ACTIVE_STATES else "off"


def safe_json(s):
    try:
        v = json.loads(s) if isinstance(s, str) else (s or {})
        return v if isinstance(v, dict) else {}
    except (ValueError, TypeError):
        return {}


def safe_round(x, n):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, n)


def sanitize(obj):
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def to_attr_json(attrs, drop_order=()):
    attrs = sanitize(attrs)
    payload = json.dumps(attrs, default=str)
    for key in drop_order:
        if len(payload.encode("utf-8")) <= MAX_ATTR_BYTES:
            break
        attrs.pop(key, None)
        payload = json.dumps(attrs, default=str)
    return payload


def slim_attributes(eid, attrs):
    """Only persons need GPS attributes; everything else is stored as {} to keep the DB small."""
    if not eid.startswith("person."):
        return {}
    attrs = attrs or {}
    return {k: attrs[k] for k in ("latitude", "longitude", "gps_accuracy") if k in attrs}


# --- Database Layer ---
def _column_names(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def init_db():
    d = os.path.dirname(DB_PATH)
    if d:
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state_history (
            entity_id TEXT, timestamp TEXT, state TEXT, attributes TEXT,
            PRIMARY KEY (entity_id, timestamp)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS context_history (
            entity_id TEXT, timestamp TEXT, state TEXT,
            PRIMARY KEY (entity_id, timestamp)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT, predicted_at TEXT, target_ts TEXT,
            predicted_state TEXT, actual_state TEXT, is_correct INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_state (
            entity_id TEXT, table_name TEXT, synced_until TEXT,
            PRIMARY KEY (entity_id, table_name)
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")

    # Migrate prediction_log (confidence / persistence baseline / abstention tracking)
    cols = _column_names(conn, "prediction_log")
    for name, ddl in (
        ("confidence", "REAL"),
        ("baseline_state", "TEXT"),
        ("abstained", "INTEGER"),
        ("is_baseline_correct", "INTEGER"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE prediction_log ADD COLUMN {name} {ddl}")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_state_hist_eid ON state_history(entity_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ctx_hist_eid ON context_history(entity_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_log_eid ON prediction_log(entity_id, target_ts)")
    conn.commit()
    normalize_existing_timestamps(conn)
    return conn


def normalize_existing_timestamps(conn):
    """One-time migration: rewrite old timestamps into the fixed-width UTC format."""
    if conn.execute("SELECT 1 FROM meta WHERE key = 'ts_normalized'").fetchone():
        return
    fixed = 0
    for table in ("state_history", "context_history"):
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE length(timestamp) != 32 OR timestamp NOT LIKE '%+00:00'"
        ).fetchall()
        for row in rows:
            conn.execute(f"DELETE FROM {table} WHERE entity_id = ? AND timestamp = ?", (row[0], row[1]))
            new_ts = norm_ts(row[1])
            if new_ts is None:
                continue
            new_row = (row[0], new_ts) + tuple(row[2:])
            conn.execute(
                f"INSERT OR IGNORE INTO {table} VALUES ({','.join('?' * len(new_row))})", new_row
            )
            fixed += 1
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('ts_normalized', '1')")
    conn.commit()
    if fixed:
        log.info(f"Normalized {fixed} legacy timestamps.")


def prune_history_table(conn, table, cutoff_iso):
    """Delete rows older than cutoff but keep the newest pre-cutoff row per entity as a state anchor."""
    rows = conn.execute(
        f"SELECT entity_id, MAX(timestamp) FROM {table} WHERE timestamp < ? GROUP BY entity_id", (cutoff_iso,)
    ).fetchall()
    deleted = 0
    for eid, keep_ts in rows:
        cur = conn.execute(f"DELETE FROM {table} WHERE entity_id = ? AND timestamp < ?", (eid, keep_ts))
        deleted += cur.rowcount
    return deleted


def prune_old_records(conn):
    now = datetime.now(timezone.utc)
    d1 = prune_history_table(conn, "state_history", fmt_ts(now - timedelta(days=TRAIN_WINDOW_DAYS)))
    d2 = prune_history_table(conn, "context_history", fmt_ts(now - timedelta(days=CONTEXT_RETENTION_DAYS)))
    cur3 = conn.execute("DELETE FROM prediction_log WHERE predicted_at < ?", (fmt_ts(now - timedelta(days=30)),))
    conn.commit()
    if d1 or d2 or cur3.rowcount:
        log.info(f"Pruned {d1} state rows, {d2} context rows and {cur3.rowcount} old prediction logs.")


def archive_states(conn, states_list, is_context=False, start_dt=None):
    """
    Insert history rows. With minimal_response, HA only includes entity_id on the first entry of
    each entity's list, so it is carried over to the following entries.
    """
    table = "context_history" if is_context else "state_history"
    inserted = 0
    rows_cache = {}

    def has_rows(eid):
        if eid not in rows_cache:
            rows_cache[eid] = conn.execute(f"SELECT 1 FROM {table} WHERE entity_id = ? LIMIT 1", (eid,)).fetchone() is not None
        return rows_cache[eid]

    for entity_history in states_list:
        if not entity_history:
            continue
        first_eid = entity_history[0].get("entity_id")
        for idx, entry in enumerate(entity_history):
            entity_id = entry.get("entity_id") or first_eid
            if not entity_id:
                continue
            if not is_context and entity_id.startswith("person."):
                raw_ts = entry.get("last_updated") or entry.get("last_changed")
            else:
                raw_ts = entry.get("last_changed") or entry.get("last_updated")
            ts = norm_ts(raw_ts) if raw_ts else None
            state = entry.get("state")
            if not ts or state in ("unavailable", "unknown", None):
                continue

            # HA's first entry is the "state at start_time" echo; skip it when we already have data.
            if idx == 0 and start_dt is not None and has_rows(entity_id):
                if abs((parse_ts(ts) - start_dt).total_seconds()) < 1.0:
                    continue

            try:
                if is_context:
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO context_history (entity_id, timestamp, state) VALUES (?, ?, ?)",
                        (entity_id, ts, state),
                    )
                else:
                    attrs = json.dumps(slim_attributes(entity_id, entry.get("attributes")))
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO state_history (entity_id, timestamp, state, attributes) VALUES (?, ?, ?, ?)",
                        (entity_id, ts, state, attrs),
                    )
                inserted += cur.rowcount
            except sqlite3.Error as e:
                log.error(f"DB Insert error: {e}")
    conn.commit()
    return inserted


# --- Self-Evaluating Accuracy Tracking ---
def evaluate_past_predictions(conn):
    now_iso = fmt_ts(datetime.now(timezone.utc))
    pending = conn.execute(
        "SELECT id, entity_id, target_ts, predicted_state, baseline_state FROM prediction_log "
        "WHERE actual_state IS NULL AND target_ts <= ?",
        (now_iso,),
    ).fetchall()

    for row_id, eid, target_ts, pred_state, base_state in pending:
        res = conn.execute(
            "SELECT state FROM state_history WHERE entity_id = ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
            (eid, target_ts),
        ).fetchone()
        if not res:
            continue
        norm_actual = normalize_label(eid, res[0])
        is_correct = 1 if pred_state == norm_actual else 0
        is_base_correct = (1 if base_state == norm_actual else 0) if base_state else None
        conn.execute(
            "UPDATE prediction_log SET actual_state = ?, is_correct = ?, is_baseline_correct = ? WHERE id = ?",
            (norm_actual, is_correct, is_base_correct, row_id),
        )
    conn.commit()

    cutoff_7d = fmt_ts(datetime.now(timezone.utc) - timedelta(days=7))
    rows = conn.execute(
        """
        SELECT entity_id, COUNT(*), AVG(is_correct), AVG(is_baseline_correct), AVG(abstained)
        FROM prediction_log
        WHERE actual_state IS NOT NULL AND baseline_state IS NOT NULL AND predicted_at >= ?
        GROUP BY entity_id
        """,
        (cutoff_7d,),
    ).fetchall()

    stats = {}
    for eid, count, acc, base_acc, abstain in rows:
        stats[eid] = {
            "count_7d": int(count),
            "accuracy_7d": round(float(acc), 3),
            "baseline_accuracy_7d": round(float(base_acc), 3) if base_acc is not None else None,
            "abstain_rate_7d": round(float(abstain), 3) if abstain is not None else None,
        }
    return stats


def record_prediction(conn, eid, now_dt, horizon_mins, predicted_state, confidence, baseline_state, abstained):
    """Logs every raw model prediction (including abstentions) so accuracy does not depend on the threshold."""
    if predicted_state == "unknown":
        return
    recent_cutoff = fmt_ts(now_dt - timedelta(minutes=14))
    exists = conn.execute(
        "SELECT 1 FROM prediction_log WHERE entity_id = ? AND predicted_at >= ? LIMIT 1",
        (eid, recent_cutoff),
    ).fetchone()
    if exists:
        return

    target_dt = now_dt + timedelta(minutes=horizon_mins)
    conn.execute(
        "INSERT INTO prediction_log (entity_id, predicted_at, target_ts, predicted_state, confidence, baseline_state, abstained) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (eid, fmt_ts(now_dt), fmt_ts(target_dt), predicted_state, float(confidence), baseline_state, 1 if abstained else 0),
    )
    conn.commit()


# --- Home Assistant API & Holiday Calendar ---
def fetch_home_coords():
    global HOME_COORDS
    try:
        resp = requests.get(f"{HA_URL}/api/states/zone.home", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            attrs = resp.json().get("attributes", {})
            lat, lon = attrs.get("latitude"), attrs.get("longitude")
            if lat is not None and lon is not None:
                HOME_COORDS = (lat, lon)
                return True
    except Exception as e:
        log.warning(f"Could not fetch zone.home coordinates: {e}")
    return False


def fetch_holiday_calendar_dates(days_back=30, days_forward=14):
    """Returns {date: name}; {} when no calendar configured; None when the fetch failed."""
    if not HOLIDAY_CALENDAR_ENTITY_ID:
        return {}

    now_utc = datetime.now(timezone.utc)
    start_iso = (now_utc - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00Z")
    end_iso = (now_utc + timedelta(days=days_forward)).strftime("%Y-%m-%dT23:59:59Z")
    url = f"{HA_URL}/api/calendars/{HOLIDAY_CALENDAR_ENTITY_ID}"

    holiday_map = {}
    try:
        resp = requests.get(url, headers=HEADERS, params={"start": start_iso, "end": end_iso}, timeout=15)
        if resp.status_code != 200:
            log.warning(f"Calendar fetch for {HOLIDAY_CALENDAR_ENTITY_ID} returned HTTP {resp.status_code}")
            return None

        for ev in resp.json():
            summary = ev.get("summary", "Holiday")
            start_obj = ev.get("start", {})
            end_obj = ev.get("end", {})

            if "date" in start_obj:
                s_date = date.fromisoformat(start_obj["date"])
                e_date = date.fromisoformat(end_obj["date"]) if "date" in end_obj else (s_date + timedelta(days=1))
                curr = s_date
                while curr < e_date:
                    holiday_map[curr] = summary
                    curr += timedelta(days=1)
            elif "dateTime" in start_obj:
                s_dt = datetime.fromisoformat(start_obj["dateTime"].replace("Z", "+00:00")).astimezone(LOCAL_TZ)
                e_dt = (
                    datetime.fromisoformat(end_obj["dateTime"].replace("Z", "+00:00")).astimezone(LOCAL_TZ)
                    if "dateTime" in end_obj
                    else s_dt
                )
                curr = s_dt.date()
                last = e_dt.date()
                while curr <= last:
                    holiday_map[curr] = summary
                    curr += timedelta(days=1)
    except Exception as e:
        log.warning(f"Failed fetching holiday calendar {HOLIDAY_CALENDAR_ENTITY_ID}: {e}")
        return None

    return holiday_map


def is_excluded(eid):
    return any(fnmatch.fnmatch(eid, pat) for pat in EXCLUDE_PATTERNS)


def get_entities():
    resp = requests.get(f"{HA_URL}/api/states", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    all_states = resp.json()

    targets = []
    target_ids = set()
    context_entities = []
    workday_item = None

    for item in all_states:
        eid = item["entity_id"]
        if WORKDAY_ENTITY_ID and eid == WORKDAY_ENTITY_ID:
            workday_item = item
        if eid.startswith("person."):
            targets.append(item)
            target_ids.add(eid)
        elif eid.startswith("binary_sensor.") and any(fnmatch.fnmatch(eid, pat) for pat in BINARY_SENSOR_PATTERNS):
            targets.append(item)
            target_ids.add(eid)

    for item in all_states:
        eid = item["entity_id"]
        if eid in target_ids or eid == WORKDAY_ENTITY_ID or eid.startswith("calendar.") or is_excluded(eid):
            continue
        if any(fnmatch.fnmatch(eid, pat) for pat in CONTEXT_ENTITY_PATTERNS):
            st = str(item.get("state", "")).lower()
            if st in ACTIVE_STATES or st in INACTIVE_STATES:
                context_entities.append(item)

    return targets, context_entities, workday_item


def insert_live_state(conn, item, is_context):
    eid = item["entity_id"]
    state_now = item.get("state")
    if state_now in ("unavailable", "unknown", None):
        return
    if not is_context and eid.startswith("person."):
        raw_ts = item.get("last_updated") or item.get("last_changed")
    else:
        raw_ts = item.get("last_changed")
    ts_now = (norm_ts(raw_ts) if raw_ts else None) or fmt_ts(datetime.now(timezone.utc))
    if is_context:
        conn.execute(
            "INSERT OR IGNORE INTO context_history (entity_id, timestamp, state) VALUES (?, ?, ?)",
            (eid, ts_now, state_now),
        )
    else:
        conn.execute(
            "INSERT OR IGNORE INTO state_history (entity_id, timestamp, state, attributes) VALUES (?, ?, ?, ?)",
            (eid, ts_now, state_now, json.dumps(slim_attributes(eid, item.get("attributes")))),
        )


def sync_entity_group(conn, entities, is_context=False, all_updates=False):
    """
    Batched history sync. Each entity has a sync watermark, so a chunk only re-reads what is new.
    all_updates=True disables HA's significant_changes_only (needed for person GPS attributes).
    """
    entities = [e for e in entities if e]
    if not entities:
        return

    table = "context_history" if is_context else "state_history"
    default_days = min(INITIAL_HISTORY_DAYS, CONTEXT_RETENTION_DAYS) if is_context else INITIAL_HISTORY_DAYS
    max_days = CONTEXT_RETENTION_DAYS if is_context else TRAIN_WINDOW_DAYS
    now_dt = datetime.now(timezone.utc)
    default_start = now_dt - timedelta(days=default_days)
    earliest = now_dt - timedelta(days=max_days)

    synced = {}
    for eid, ts in conn.execute("SELECT entity_id, synced_until FROM sync_state WHERE table_name = ?", (table,)):
        try:
            synced[eid] = parse_ts(ts)
        except (ValueError, TypeError):
            pass

    total_new = 0
    for i in range(0, len(entities), HISTORY_BATCH_SIZE):
        chunk = entities[i:i + HISTORY_BATCH_SIZE]
        ids = [e["entity_id"] for e in chunk]
        start_dt = max(min(synced.get(eid, default_start) for eid in ids), earliest)

        url = f"{HA_URL}/api/history/period/{quote(fmt_ts(start_dt), safe='')}"
        params = {"filter_entity_id": ",".join(ids), "end_time": fmt_ts(now_dt)}
        if is_context:
            params["minimal_response"] = "1"
            params["no_attributes"] = "1"
        elif all_updates:
            params["significant_changes_only"] = "0"

        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=90)
            if resp.status_code != 200:
                log.warning(f"History fetch ({table}, {len(ids)} entities) returned HTTP {resp.status_code}")
                continue
            total_new += archive_states(conn, resp.json(), is_context=is_context, start_dt=start_dt)
            for item in chunk:
                insert_live_state(conn, item, is_context)
            conn.executemany(
                "INSERT OR REPLACE INTO sync_state (entity_id, table_name, synced_until) VALUES (?, ?, ?)",
                [(eid, table, fmt_ts(now_dt)) for eid in ids],
            )
            conn.commit()
        except Exception as e:
            log.warning(f"Failed syncing history chunk ({table}): {e}")

    if total_new > 0 and not is_context:
        log.info(f"Archived {total_new} new state entries ({len(entities)} entities)")


def get_workday_series(conn, resample_rule, index_like, holiday_dates, live_workday_item):
    local_idx = index_like.tz_convert(LOCAL_TZ)
    base_weekday = np.asarray(local_idx.day_of_week < 5).astype(int)

    if HOLIDAY_CALENDAR_ENTITY_ID:
        if not holiday_dates:
            return pd.Series(base_weekday, index=index_like)
        local_dates = local_idx.date
        is_hol = np.fromiter((1 if d in holiday_dates else 0 for d in local_dates), dtype=int, count=len(local_dates))
        workdays = np.where((base_weekday == 1) & (is_hol == 0), 1, 0)
        return pd.Series(workdays, index=index_like, dtype=int)

    fallback = pd.Series(base_weekday, index=index_like)
    if not WORKDAY_ENTITY_ID:
        return fallback

    df_w = pd.read_sql_query(
        "SELECT timestamp, state FROM state_history WHERE entity_id = ? ORDER BY timestamp ASC",
        conn,
        params=(WORKDAY_ENTITY_ID,),
    )
    if df_w.empty:
        if live_workday_item and live_workday_item.get("state") in ("on", "off"):
            val = 1 if live_workday_item.get("state") == "on" else 0
            return pd.Series(val, index=index_like)
        return fallback

    df_w["timestamp"] = pd.to_datetime(df_w["timestamp"], format="ISO8601", utc=True)
    df_w = df_w.drop_duplicates(subset=["timestamp"], keep="last")
    s = df_w.set_index("timestamp")["state"].apply(lambda x: 1 if str(x).lower() == "on" else 0)
    s.loc[index_like.max()] = s.iloc[-1]
    s_res = s.resample(resample_rule).ffill()
    return s_res.reindex(index_like, method="ffill").fillna(fallback).astype(int)


def is_date_workday(target_date, holiday_dates, live_workday_item=None, is_today=False):
    if target_date.weekday() >= 5:
        return 0
    if HOLIDAY_CALENDAR_ENTITY_ID:
        return 0 if target_date in holiday_dates else 1
    if is_today and live_workday_item and live_workday_item.get("state") in ("on", "off"):
        return 1 if live_workday_item.get("state") == "on" else 0
    return 1


# --- Nightly Time-Deconfounded Cross-Sensor Correlation Engine ---
def run_correlation_analysis(conn, all_entity_map):
    log.info("Running time-deconfounded cross-sensor correlation analysis...")
    now_dt = datetime.now(timezone.utc)
    cutoff_str = fmt_ts(now_dt - timedelta(days=CONTEXT_RETENTION_DAYS))

    df_t1 = pd.read_sql_query(
        "SELECT entity_id, timestamp, state FROM state_history WHERE timestamp >= ? AND entity_id != ?",
        conn,
        params=(cutoff_str, WORKDAY_ENTITY_ID),
    )
    df_t2 = pd.read_sql_query(
        "SELECT entity_id, timestamp, state FROM context_history WHERE timestamp >= ?",
        conn,
        params=(cutoff_str,),
    )
    df_all = pd.concat([df_t1, df_t2], ignore_index=True)

    if df_all.empty:
        return {}, []

    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], format="ISO8601", utc=True)
    resample_rule = f"{RESAMPLE_MINS}min"

    series_dict = {}
    for eid, grp in df_all.groupby("entity_id"):
        grp = grp.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        is_p = eid.startswith("person.")
        s = grp.set_index("timestamp")["state"].apply(lambda x: state_to_binary(x, is_person=is_p))
        s.loc[now_dt] = s.iloc[-1]
        s_res = s.resample(resample_rule).ffill().dropna()
        if s_res.nunique() >= 2 and len(s_res) >= 48:
            series_dict[eid] = s_res

    if len(series_dict) < 2:
        return {}, []

    grid = pd.DataFrame(series_dict).ffill().bfill().dropna()
    if len(grid) < 48:
        return {}, []

    local_hours = grid.index.tz_convert(LOCAL_TZ).hour
    residuals = grid - grid.groupby(local_hours).transform("mean")

    target_eids = [
        eid for eid in grid.columns
        if eid.startswith("person.") or any(fnmatch.fnmatch(eid, pat) for pat in BINARY_SENSOR_PATTERNS)
    ]

    lags_mins = [0, 5, 10, 15]
    driver_map = {}
    all_findings = []

    def friendly(eid):
        return all_entity_map.get(eid, {}).get("attributes", {}).get("friendly_name", eid)

    for target in target_eids:
        target_res = residuals[target]
        candidates = []

        for driver in residuals.columns:
            if driver == target:
                continue

            best_lag = 0
            best_corr = 0.0

            for lag_m in lags_mins:
                shift_steps = lag_m // RESAMPLE_MINS
                shifted_driver = residuals[driver].shift(shift_steps) if shift_steps > 0 else residuals[driver]
                valid = pd.concat([target_res, shifted_driver], axis=1).dropna()
                if len(valid) < 48 or valid.iloc[:, 1].std() < 1e-6 or valid.iloc[:, 0].std() < 1e-6:
                    continue
                r = float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
                if not math.isnan(r) and abs(r) > abs(best_corr):
                    best_corr = r
                    best_lag = lag_m

            if abs(best_corr) >= CORRELATION_MIN_STRENGTH:
                d_name = friendly(driver)
                t_name = friendly(target)
                d_is_person = driver.startswith("person.")
                t_is_person = target.startswith("person.")

                if best_corr > 0:
                    d_action = "arrives home" if d_is_person else "activates"
                    t_action = "arrives home" if t_is_person else "activates"
                else:
                    d_action = "leaves home" if d_is_person else "turns off"
                    t_action = "arrives home" if t_is_person else "activates"

                timing = "at the same time" if best_lag == 0 else f"~{best_lag}m before {t_name} {t_action}"
                summary = f"{d_name} {d_action} {timing} (r={best_corr:+.2f})"
                candidates.append({
                    "driver_eid": driver,
                    "lag_mins": best_lag,
                    "corr": round(best_corr, 3),
                    "summary": summary,
                })
                all_findings.append((
                    abs(best_corr),
                    f"When {d_name} {d_action}, {t_name} {t_action} {timing if best_lag > 0 else 'simultaneously'} (r={best_corr:+.2f})",
                ))

        candidates.sort(key=lambda x: abs(x["corr"]), reverse=True)
        driver_map[target] = candidates[:3]

    all_findings.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    top_insights = []
    for _, text in all_findings:
        if text not in seen:
            seen.add(text)
            top_insights.append(text)
        if len(top_insights) >= 15:
            break

    log.info(f"De-confounded correlation analysis complete: found {len(top_insights)} strong rules across {len(grid.columns)} entities.")
    return driver_map, top_insights


def get_driver_history_series(conn, driver_eid, lag_mins, resample_rule, index_like):
    table = (
        "state_history"
        if (driver_eid.startswith("person.") or any(fnmatch.fnmatch(driver_eid, pat) for pat in BINARY_SENSOR_PATTERNS))
        else "context_history"
    )

    df_d = pd.read_sql_query(
        f"SELECT timestamp, state FROM {table} WHERE entity_id = ? ORDER BY timestamp ASC",
        conn,
        params=(driver_eid,),
    )
    if df_d.empty:
        return pd.Series(0, index=index_like)

    df_d["timestamp"] = pd.to_datetime(df_d["timestamp"], format="ISO8601", utc=True)
    df_d = df_d.drop_duplicates(subset=["timestamp"], keep="last")
    is_p = driver_eid.startswith("person.")
    s = df_d.set_index("timestamp")["state"].apply(lambda x: state_to_binary(x, is_person=is_p))
    s.loc[index_like.max()] = s.iloc[-1]
    s_res = s.resample(resample_rule).ffill()

    shift_steps = lag_mins // RESAMPLE_MINS
    if shift_steps > 0:
        s_res = s_res.shift(shift_steps)

    return s_res.reindex(index_like, method="ffill").fillna(0).astype(int)


# --- Feature Engineering & Models ---
def compute_fourier_time_df(timestamps_utc):
    """Multi-harmonic (24h, 12h, 8h) cyclical time features for sharp transition timing."""
    local_ts = timestamps_utc.tz_convert(LOCAL_TZ)
    hours = np.asarray(local_ts.hour + local_ts.minute / 60.0, dtype=float)
    dows = np.asarray(local_ts.day_of_week, dtype=int)
    angle = 2.0 * np.pi * hours / 24.0

    return pd.DataFrame({
        "hour_sin": np.sin(angle),
        "hour_cos": np.cos(angle),
        "hour_sin2": np.sin(2.0 * angle),
        "hour_cos2": np.cos(2.0 * angle),
        "hour_sin3": np.sin(3.0 * angle),
        "hour_cos3": np.cos(3.0 * angle),
        "dow_sin": np.sin(2.0 * np.pi * dows / 7.0),
        "dow_cos": np.cos(2.0 * np.pi * dows / 7.0),
        "is_weekend": (dows >= 5).astype(int),
    })


def build_features(df, is_person=False, driver_cols=None):
    df = df.copy()
    time_feats = compute_fourier_time_df(pd.DatetimeIndex(df["timestamp"]))
    for col in time_feats.columns:
        df[col] = time_feats[col].values

    if "is_workday" not in df.columns:
        df["is_workday"] = (df["is_weekend"] == 0).astype(int)
    if "is_tomorrow_workday" not in df.columns:
        df["is_tomorrow_workday"] = df["is_workday"]

    steps_30m = max(1, 30 // RESAMPLE_MINS)
    steps_1h = max(1, 60 // RESAMPLE_MINS)
    steps_2h = max(1, 120 // RESAMPLE_MINS)

    if "active_ratio_30m" not in df.columns:
        df["active_ratio_30m"] = df["current_state_bin"].rolling(steps_30m, min_periods=1).mean()
    if "active_ratio_2h" not in df.columns:
        df["active_ratio_2h"] = df["current_state_bin"].rolling(steps_2h, min_periods=1).mean()
    if "transitions_1h" not in df.columns:
        flips = (df["current_state_bin"].diff().abs() > 0).astype(int)
        df["transitions_1h"] = flips.rolling(steps_1h, min_periods=1).sum()

    schedule_cols = list(SCHEDULE_COLS)
    feature_cols = schedule_cols + [
        "current_state_bin",
        "mins_in_current_state",
        "active_ratio_30m",
        "active_ratio_2h",
        "transitions_1h",
    ]

    if is_person:
        def calc_dist(a):
            return haversine_km(a.get("latitude"), a.get("longitude"), HOME_COORDS[0], HOME_COORDS[1])

        df["distance_from_home_km"] = df["attr_dict"].apply(calc_dist).astype(float)
        df["gps_accuracy"] = df["attr_dict"].apply(lambda a: float(a.get("gps_accuracy", 0) or 0))

        if "home_approach_speed_kmh" not in df.columns:
            lookback_steps = max(1, VELOCITY_LOOKBACK_MINS // RESAMPLE_MINS)
            lookback_hours = (lookback_steps * RESAMPLE_MINS) / 60.0
            prev_dist = df["distance_from_home_km"].shift(lookback_steps).fillna(df["distance_from_home_km"])
            speed = (prev_dist - df["distance_from_home_km"]) / lookback_hours
            # NaN stays NaN (unknown), small values become 0 (stationary)
            speed = speed.mask(speed.abs() < 1.5, 0.0).clip(-250.0, 250.0)
            df["home_approach_speed_kmh"] = speed

        if "eta_mins" not in df.columns:
            df["eta_mins"] = [
                compute_eta_mins(d, s, bool(st == 1))
                for d, s, st in zip(
                    df["distance_from_home_km"],
                    df["home_approach_speed_kmh"],
                    df["current_state_bin"],
                )
            ]

        feature_cols.extend([
            "distance_from_home_km",
            "gps_accuracy",
            "home_approach_speed_kmh",
            "eta_mins",
        ])

    if driver_cols:
        feature_cols.extend(driver_cols)

    return df, feature_cols, schedule_cols


def fit_calibrated_hgb(X, y, sample_weight, max_depth=6):
    """Fits HistGradientBoostingClassifier with sigmoid probability calibration for horizon forecasts."""
    base_clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.06,
        max_depth=max_depth,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=42,
    )
    min_class_count = int(y.value_counts().min())
    if min_class_count >= 12 and len(y) >= 60:
        cal_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
        cal_clf.fit(X, y, sample_weight=sample_weight)
        return cal_clf

    base_clf.fit(X, y, sample_weight=sample_weight)
    return base_clf


def build_schedule_design_matrix(timestamps_utc, holiday_dates, live_workday_item):
    """Builds schedule feature matrix including is_workday and is_tomorrow_workday."""
    local_ts = timestamps_utc.tz_convert(LOCAL_TZ)
    today_date = datetime.now(timezone.utc).astimezone(LOCAL_TZ).date()

    X = compute_fourier_time_df(timestamps_utc)
    memo = {}

    def wd(d):
        if d not in memo:
            memo[d] = is_date_workday(d, holiday_dates, live_workday_item, is_today=(d == today_date))
        return memo[d]

    local_dates = local_ts.date
    X["is_workday"] = np.array([wd(d) for d in local_dates], dtype=int)
    X["is_tomorrow_workday"] = np.array([wd(d + timedelta(days=1)) for d in local_dates], dtype=int)
    return X[SCHEDULE_COLS]


def smooth_probs(prob, window):
    if window <= 1:
        return np.asarray(prob, dtype=float)
    return pd.Series(prob).rolling(window, center=True, min_periods=1).mean().to_numpy()


def compute_schedule_threshold(prob_pos):
    """
    Computes a dynamic threshold from the schedule probability curve so both
    low-duty-cycle sensors (e.g. active 2h/day) and high-duty-cycle entities
    (e.g. home 18h/day) produce clear ON/OFF transitions.
    """
    p_min = float(np.min(prob_pos))
    p_max = float(np.max(prob_pos))
    if (p_max - p_min) < 0.12:
        return 0.50

    p15 = float(np.percentile(prob_pos, 15))
    p85 = float(np.percentile(prob_pos, 85))
    midpoint = 0.5 * (p15 + p85)
    return float(np.clip(midpoint, 0.15, 0.85))


def schedule_prob_curve(schedule_model, X):
    probs = schedule_model.predict_proba(X)
    classes = list(schedule_model.classes_)
    return probs[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def build_24h_ascii_timeline(schedule_model, now_dt, threshold, holiday_dates, live_workday_item):
    """48-char (30 min/char) timeline, lightly smoothed so single noisy slots do not flicker."""
    slot_timestamps = pd.date_range(start=now_dt + timedelta(minutes=15), periods=48, freq="30min")
    X_slots = build_schedule_design_matrix(slot_timestamps, holiday_dates, live_workday_item)
    prob_pos = smooth_probs(schedule_prob_curve(schedule_model, X_slots), 2)

    chars = []
    for i, p in enumerate(prob_pos):
        if i > 0 and i % 12 == 0:
            chars.append("|")
        chars.append("█" if p >= threshold else "·")
    return "".join(chars)


def estimate_next_transitions(schedule_model, now_dt, current_state_bin, holiday_dates, live_workday_item):
    """
    7-day scan of the schedule curve. The curve is smoothed (~60 min) and a flip is only accepted
    after it persists (~15 min), so a noisy slot cannot produce a bogus 'next ON in 5 minutes'.
    """
    total_steps = (7 * 24 * 60) // RESAMPLE_MINS
    fut_timestamps = pd.date_range(
        start=now_dt + timedelta(minutes=RESAMPLE_MINS), periods=total_steps, freq=f"{RESAMPLE_MINS}min"
    )
    local_fut = fut_timestamps.tz_convert(LOCAL_TZ)

    X_fut = build_schedule_design_matrix(fut_timestamps, holiday_dates, live_workday_item)
    prob_pos = smooth_probs(schedule_prob_curve(schedule_model, X_fut), max(1, 60 // RESAMPLE_MINS))

    threshold = compute_schedule_threshold(prob_pos)
    preds = (prob_pos >= threshold).astype(int)

    confirm = max(1, 15 // RESAMPLE_MINS)
    next_on = None
    next_off = None
    sim_state = int(current_state_bin)
    run = 0
    for idx, pred in enumerate(preds):
        if pred != sim_state:
            run += 1
            if run >= confirm:
                t_idx = idx - confirm + 1
                if pred == 1 and next_on is None:
                    next_on = local_fut[t_idx].isoformat()
                elif pred == 0 and next_off is None:
                    next_off = local_fut[t_idx].isoformat()
                sim_state = int(pred)
                run = 0
                if next_on and next_off:
                    break
        else:
            run = 0

    timeline_24h = build_24h_ascii_timeline(schedule_model, now_dt, threshold, holiday_dates, live_workday_item)
    return next_on, next_off, timeline_24h


def fit_entity_models(base_df, feature_cols, schedule_cols, all_horizons, now_dt):
    """Trains calibrated HGB models for every horizon plus the pure schedule model."""
    horizon_models = {}
    for h_mins in all_horizons:
        h_steps = max(1, h_mins // RESAMPLE_MINS)
        h_target = base_df["current_state_bin"].shift(-h_steps)
        valid_mask = h_target.notna()
        h_train = base_df.loc[valid_mask]
        h_y = h_target.loc[valid_mask].astype(int)

        if len(h_y) == 0:
            horizon_models[h_mins] = ("prior", 0.5)
            continue
        if h_y.nunique() < 2 or len(h_train) < 20:
            # Laplace-smoothed prior; avoids reporting 100% certainty from a single class
            horizon_models[h_mins] = ("prior", (float(h_y.sum()) + 1.0) / (len(h_y) + 2.0))
            continue

        age_days = (now_dt - h_train["timestamp"]).dt.total_seconds() / 86400.0
        weights = np.exp(-np.log(2) * age_days / DECAY_HALF_LIFE_DAYS).to_numpy()
        clf = fit_calibrated_hgb(h_train[feature_cols], h_y, weights, max_depth=6)
        horizon_models[h_mins] = ("clf", clf)

    schedule_model = None
    sched_y = base_df["current_state_bin"].astype(int)
    if sched_y.nunique() >= 2 and len(base_df) >= 20:
        sched_age_days = (now_dt - base_df["timestamp"]).dt.total_seconds() / 86400.0
        sched_weights = np.exp(-np.log(2) * sched_age_days / DECAY_HALF_LIFE_DAYS).to_numpy()
        schedule_model = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=1
        )
        # Unshifted target so predicted timestamps align with the actual state
        schedule_model.fit(base_df[schedule_cols], sched_y, sample_weight=sched_weights)

    return {"trained_at": now_dt, "horizon_models": horizon_models, "schedule_model": schedule_model}


def train_and_predict(conn, entity_item, drivers, all_entity_map, holiday_dates, workday_item, acc_stats):
    eid = entity_item["entity_id"]
    is_person = eid.startswith("person.")
    positive_state = "home" if is_person else "on"
    negative_state = "not_home" if is_person else "off"
    now_dt = datetime.now(timezone.utc)

    df = pd.read_sql_query(
        "SELECT timestamp, state, attributes FROM state_history WHERE entity_id = ? ORDER BY timestamp ASC",
        conn,
        params=(eid,),
    )

    if df.empty:
        return "unknown", 0.0, {"reason": "no_history_recorded"}

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)

    span_hours = (now_dt - df["timestamp"].min()).total_seconds() / 3600.0
    if span_hours < MIN_HISTORY_HOURS:
        return "unknown", 0.0, {
            "reason": "insufficient_time_span",
            "recorded_hours": round(span_hours, 1),
            "required_hours": MIN_HISTORY_HOURS,
        }

    now_anchor = pd.DataFrame([{
        "timestamp": now_dt,
        "state": entity_item.get("state"),
        "attributes": json.dumps(slim_attributes(eid, entity_item.get("attributes"))),
    }])
    df = pd.concat([df, now_anchor], ignore_index=True)

    # Group on the binary state so e.g. not_home -> work does not reset the "time in state" counter
    raw_bin = (df["state"] == positive_state).astype(int)
    df["state_change_group"] = (raw_bin != raw_bin.shift()).cumsum()
    df["group_start_ts"] = df.groupby("state_change_group")["timestamp"].transform("first")

    resample_rule = f"{RESAMPLE_MINS}min"
    df_indexed = df.set_index("timestamp").resample(resample_rule).ffill().dropna(subset=["state"])

    # Attach workday/holiday flag series for both today and tomorrow
    s_workday = get_workday_series(conn, resample_rule, df_indexed.index, holiday_dates, workday_item)
    df_indexed["is_workday"] = s_workday.values

    next_day_index = df_indexed.index + pd.Timedelta(days=1)
    s_tomorrow_workday = get_workday_series(conn, resample_rule, next_day_index, holiday_dates, workday_item)
    df_indexed["is_tomorrow_workday"] = s_tomorrow_workday.values

    # Attach correlated driver history columns
    driver_cols = []
    live_driver_vals = {}
    for idx_d, drv in enumerate(drivers):
        col_name = f"driver_{idx_d}"
        d_eid = drv["driver_eid"]
        d_lag = drv["lag_mins"]
        s_drv = get_driver_history_series(conn, d_eid, d_lag, resample_rule, df_indexed.index)
        df_indexed[col_name] = s_drv.values
        driver_cols.append(col_name)

        if d_lag == 0 and d_eid in all_entity_map:
            live_st = all_entity_map[d_eid].get("state")
            live_driver_vals[col_name] = state_to_binary(live_st, is_person=d_eid.startswith("person."))
        else:
            live_driver_vals[col_name] = int(s_drv.iloc[-1]) if len(s_drv) > 0 else 0

    df = df_indexed.reset_index()
    df["attr_dict"] = df["attributes"].apply(safe_json)
    df["current_state_bin"] = (df["state"] == positive_state).astype(int)
    df["mins_in_current_state"] = (df["timestamp"] - df["group_start_ts"]).dt.total_seconds() / 60.0

    local_now = now_dt.astimezone(LOCAL_TZ)
    today_date = local_now.date()
    tomorrow_date = today_date + timedelta(days=1)

    is_workday_today = is_date_workday(today_date, holiday_dates, workday_item, is_today=True)
    is_workday_tomorrow = is_date_workday(tomorrow_date, holiday_dates, workday_item, is_today=False)
    holiday_today_name = holiday_dates.get(today_date)
    holiday_tomorrow_name = holiday_dates.get(tomorrow_date)

    curr_bin = 1 if entity_item.get("state") == positive_state else 0
    curr_label = positive_state if curr_bin == 1 else negative_state

    common_meta = {
        "timeline_start": local_now.strftime("%H:%M"),
        "timeline_legend": f"█={positive_state} ·={negative_state} |=6h (30m/char)",
        "is_workday": bool(is_workday_today),
        "is_tomorrow_workday": bool(is_workday_tomorrow),
        "holiday_today": holiday_today_name,
        "holiday_tomorrow": holiday_tomorrow_name,
        "correlated_triggers": [d["summary"] for d in drivers],
        "recorded_hours": round(span_hours, 1),
    }

    # --- Constant historical state ---
    if df["current_state_bin"].nunique() < 2:
        const_val = int(df["current_state_bin"].iloc[0])
        if const_val != curr_bin:
            # e.g. a sensor that was always off just turned on: nothing to learn from yet
            return "unknown", 0.0, {"reason": "state_never_seen_in_history", **common_meta}

        const_state = positive_state if const_val == 1 else negative_state
        glyph = "█" if const_val == 1 else "·"
        flat_line = "|".join([glyph * 12] * 4)
        # Certainty grows with the amount of evidence instead of being a flat 1.0
        certainty = round(min(0.95, 0.55 + 0.40 * min(1.0, span_hours / 168.0)), 3)
        final = const_state if certainty >= CERTAINTY_THRESHOLD else "unknown"
        return final, certainty, {
            "reason": "constant_historical_state",
            "certainty": certainty,
            "expected_next_on": None,
            "expected_next_off": None,
            "timeline_24h": flat_line,
            **common_meta,
        }

    base_df, feature_cols, schedule_cols = build_features(df, is_person=is_person, driver_cols=driver_cols)

    # Prepare live observation row
    last_changed_str = entity_item.get("last_changed")
    try:
        last_changed_dt = parse_ts(last_changed_str) if last_changed_str else now_dt
    except (ValueError, TypeError):
        last_changed_dt = now_dt
    mins_in_state = max(0.0, (now_dt - last_changed_dt).total_seconds() / 60.0)
    curr_attrs = entity_item.get("attributes", {}) or {}

    live_row = {
        "timestamp": now_dt,
        "attr_dict": curr_attrs,
        "current_state_bin": curr_bin,
        "mins_in_current_state": mins_in_state,
        "is_workday": is_workday_today,
        "is_tomorrow_workday": is_workday_tomorrow,
        "active_ratio_30m": float(base_df["active_ratio_30m"].iloc[-1]),
        "active_ratio_2h": float(base_df["active_ratio_2h"].iloc[-1]),
        "transitions_1h": float(base_df["transitions_1h"].iloc[-1]),
        **live_driver_vals,
    }

    if is_person:
        curr_dist = haversine_km(curr_attrs.get("latitude"), curr_attrs.get("longitude"), HOME_COORDS[0], HOME_COORDS[1])
        lookback_steps = max(1, VELOCITY_LOOKBACK_MINS // RESAMPLE_MINS)
        if len(df) > lookback_steps:
            past_attrs = df["attr_dict"].iloc[-(lookback_steps + 1)]
            past_dist = haversine_km(past_attrs.get("latitude"), past_attrs.get("longitude"), HOME_COORDS[0], HOME_COORDS[1])
            lookback_hours = (lookback_steps * RESAMPLE_MINS) / 60.0
            approach_speed = compute_approach_speed(past_dist, curr_dist, lookback_hours)
        else:
            approach_speed = float("nan")

        live_row["home_approach_speed_kmh"] = approach_speed
        live_row["eta_mins"] = compute_eta_mins(curr_dist, approach_speed, bool(curr_bin == 1))

    now_df = pd.DataFrame([live_row])
    now_df, _, _ = build_features(now_df, is_person=is_person, driver_cols=driver_cols)

    # --- Train (hourly) or reuse cached calibrated multi-horizon models ---
    all_horizons = sorted(set([PREDICT_HORIZON_MINS] + EXTRA_HORIZONS_MINS))
    sig = (tuple((d["driver_eid"], d["lag_mins"]) for d in drivers), tuple(all_horizons), tuple(feature_cols))
    cache = MODEL_CACHE.get(eid)
    stale = (
        cache is None
        or cache["sig"] != sig
        or (now_dt - cache["trained_at"]).total_seconds() >= RETRAIN_INTERVAL_MINS * 60
    )
    if stale:
        cache = fit_entity_models(base_df, feature_cols, schedule_cols, all_horizons, now_dt)
        cache["sig"] = sig
        MODEL_CACHE[eid] = cache
        log.info(f"Retrained models for {eid}")

    horizon_probs = {}
    X_now = now_df[feature_cols]
    for h_mins in all_horizons:
        kind, obj = cache["horizon_models"][h_mins]
        if kind == "prior":
            p_pos = obj
        else:
            p_arr = obj.predict_proba(X_now)[0]
            cls_list = list(obj.classes_)
            p_pos = p_arr[cls_list.index(1)] if 1 in cls_list else 0.0
        horizon_probs[h_mins] = round(float(p_pos), 3)

    prob_pos = horizon_probs.get(PREDICT_HORIZON_MINS, 0.0)
    prob_neg = 1.0 - prob_pos
    max_prob = max(prob_pos, prob_neg)
    predicted_state = positive_state if prob_pos >= prob_neg else negative_state

    if cache["schedule_model"] is not None:
        next_on, next_off, timeline_24h = estimate_next_transitions(
            cache["schedule_model"], now_dt, curr_bin, holiday_dates, workday_item
        )
    else:
        next_on, next_off, timeline_24h = None, None, None

    # --- Self-Evaluating Accuracy & Adaptive Threshold ---
    ent_acc = acc_stats.get(eid, {})
    acc_7d = ent_acc.get("accuracy_7d")
    base_acc_7d = ent_acc.get("baseline_accuracy_7d")
    eval_count_7d = ent_acc.get("count_7d", 0)

    effective_threshold = CERTAINTY_THRESHOLD
    if eval_count_7d >= 20 and acc_7d is not None and acc_7d < MIN_ACCEPTABLE_ACCURACY:
        penalty = min(0.18, (MIN_ACCEPTABLE_ACCURACY - acc_7d) * 0.75)
        effective_threshold = round(min(0.90, CERTAINTY_THRESHOLD + penalty), 3)

    meta = {
        "probability_active": prob_pos,
        "certainty": round(float(max_prob), 3),
        "threshold": CERTAINTY_THRESHOLD,
        "effective_threshold": effective_threshold,
        "horizon_mins": PREDICT_HORIZON_MINS,
        "expected_next_on": next_on,
        "expected_next_off": next_off,
        "timeline_24h": timeline_24h,
        "active_ratio_30m": round(float(live_row["active_ratio_30m"]), 2),
        "active_ratio_2h": round(float(live_row["active_ratio_2h"]), 2),
        "transitions_1h": int(live_row["transitions_1h"]),
        "accuracy_7d": acc_7d,
        "baseline_accuracy_7d": base_acc_7d,
        "skill_vs_baseline_7d": (
            round(acc_7d - base_acc_7d, 3) if (acc_7d is not None and base_acc_7d is not None) else None
        ),
        "abstain_rate_7d": ent_acc.get("abstain_rate_7d"),
        "evaluated_predictions_7d": eval_count_7d,
        "mins_in_current_state": round(mins_in_state, 1),
        "model_trained_at": cache["trained_at"].astimezone(LOCAL_TZ).isoformat(),
        "last_updated": local_now.isoformat(),
        **common_meta,
    }

    for h_mins, h_prob in horizon_probs.items():
        meta[f"prob_{h_mins}m"] = h_prob

    if is_person:
        meta["distance_from_home_km"] = safe_round(now_df["distance_from_home_km"].iloc[0], 2)
        meta["home_approach_speed_kmh"] = safe_round(now_df["home_approach_speed_kmh"].iloc[0], 1)
        meta["eta_mins"] = safe_round(now_df["eta_mins"].iloc[0], 0)

    final_state = "unknown" if max_prob < effective_threshold else predicted_state
    record_prediction(
        conn, eid, now_dt, PREDICT_HORIZON_MINS, predicted_state, max_prob, curr_label, final_state == "unknown"
    )

    return final_state, max_prob, meta


# --- MQTT Home Assistant Auto-Discovery ---
def setup_mqtt():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()
    return client


def publish_ha_discovery(client, entity_item, state, attributes):
    eid = entity_item["entity_id"]
    object_slug = eid.split(".", 1)[1]
    friendly_name = entity_item.get("attributes", {}).get("friendly_name", object_slug.replace("_", " ").title())

    default_entity_id = f"sensor.{object_slug}_predicted"
    unique_id = f"{object_slug}_predicted"

    base_topic = f"{MQTT_PREFIX}/sensor/{unique_id}"
    config_topic = f"{base_topic}/config"
    state_topic = f"{base_topic}/state"
    attr_topic = f"{base_topic}/attributes"

    discovery_payload = {
        "name": f"{friendly_name} Predicted",
        "default_entity_id": default_entity_id,
        "unique_id": unique_id,
        "state_topic": state_topic,
        "json_attributes_topic": attr_topic,
        "icon": "mdi:account-clock" if eid.startswith("person.") else "mdi:motion-sensor",
        "device": {
            "identifiers": ["ha_ml_predictor_service"],
            "name": "HA ML Predictor",
            "manufacturer": "Custom ML Service",
            "model": "Calibrated HGB State Forecaster",
        },
    }

    client.publish(config_topic, json.dumps(discovery_payload), retain=True)
    client.publish(state_topic, state, retain=True)
    client.publish(
        attr_topic,
        to_attr_json(attributes, drop_order=("timeline_legend", "correlated_triggers", "timeline_24h")),
        retain=True,
    )


def publish_insights_sensor(client, insights, context_entities, acc_stats, holiday_dates):
    unique_id = "ha_ml_behavioral_insights"
    base_topic = f"{MQTT_PREFIX}/sensor/{unique_id}"
    discovery_payload = {
        "name": "HA ML Behavioral Insights",
        "default_entity_id": "sensor.ha_ml_behavioral_insights",
        "unique_id": unique_id,
        "state_topic": f"{base_topic}/state",
        "json_attributes_topic": f"{base_topic}/attributes",
        "icon": "mdi:brain",
        "device": {
            "identifiers": ["ha_ml_predictor_service"],
            "name": "HA ML Predictor",
            "manufacturer": "Custom ML Service",
            "model": "Calibrated HGB State Forecaster",
        },
    }
    overall_acc = overall_base = None
    if acc_stats:
        overall_acc = round(float(np.mean([v["accuracy_7d"] for v in acc_stats.values()])), 3)
        bases = [v["baseline_accuracy_7d"] for v in acc_stats.values() if v.get("baseline_accuracy_7d") is not None]
        overall_base = round(float(np.mean(bases)), 3) if bases else None

    today_date = datetime.now(timezone.utc).astimezone(LOCAL_TZ).date()
    upcoming_holidays = {
        d.isoformat(): name for d, name in sorted(holiday_dates.items()) if d >= today_date
    }

    # The full context entity list is intentionally not published (HA's 16 KB attribute limit).
    attrs = {
        "insights": insights,
        "overall_accuracy_7d": overall_acc,
        "overall_baseline_accuracy_7d": overall_base,
        "entity_accuracy_7d": acc_stats,
        "context_entities_monitored": len(context_entities),
        "holiday_calendar": HOLIDAY_CALENDAR_ENTITY_ID or None,
        "upcoming_holidays": upcoming_holidays,
        "min_correlation_threshold": CORRELATION_MIN_STRENGTH,
        "last_analyzed": datetime.now(timezone.utc).astimezone(LOCAL_TZ).isoformat(),
    }
    client.publish(f"{base_topic}/config", json.dumps(discovery_payload), retain=True)
    client.publish(f"{base_topic}/state", str(len(insights)), retain=True)
    client.publish(
        f"{base_topic}/attributes",
        to_attr_json(attrs, drop_order=("entity_accuracy_7d", "upcoming_holidays", "insights")),
        retain=True,
    )


# --- Main Loop ---
def main():
    if not HA_TOKEN:
        raise ValueError("HA_TOKEN environment variable is required.")

    init_timezone()
    conn = init_db()
    mqtt_client = setup_mqtt()
    fetch_home_coords()
    log.info("HA ML Predictor started.")

    last_corr_run = None
    last_holiday_fetch = None
    driver_map = {}
    top_insights = []
    holiday_dates = {}

    while True:
        loop_started = time.time()
        try:
            if HOME_COORDS[0] is None:
                fetch_home_coords()

            targets, context_entities, workday_item = get_entities()
            all_entity_map = {e["entity_id"]: e for e in (targets + context_entities)}

            persons = [t for t in targets if t["entity_id"].startswith("person.")]
            others = [t for t in targets if not t["entity_id"].startswith("person.")]
            sync_entity_group(conn, persons, is_context=False, all_updates=True)
            sync_entity_group(conn, others + ([workday_item] if workday_item else []), is_context=False)
            sync_entity_group(conn, context_entities, is_context=True)

            acc_stats = evaluate_past_predictions(conn)

            now_utc = datetime.now(timezone.utc)

            # Holiday calendar: refresh periodically, keep the old data if a refresh fails
            if HOLIDAY_CALENDAR_ENTITY_ID and (
                last_holiday_fetch is None
                or (now_utc - last_holiday_fetch).total_seconds() >= HOLIDAY_REFRESH_HOURS * 3600
            ):
                new_holidays = fetch_holiday_calendar_dates(
                    days_back=max(30, TRAIN_WINDOW_DAYS + 2), days_forward=14
                )
                if new_holidays is not None:
                    holiday_dates = new_holidays
                    last_holiday_fetch = now_utc
                    log.info(f"Loaded {len(holiday_dates)} holiday dates from {HOLIDAY_CALENDAR_ENTITY_ID}.")

            if last_corr_run is None or (now_utc - last_corr_run).total_seconds() >= CORRELATION_INTERVAL_HOURS * 3600:
                try:
                    prune_old_records(conn)
                    driver_map, top_insights = run_correlation_analysis(conn, all_entity_map)
                except Exception as e:
                    log.error(f"Correlation analysis failed, keeping previous results: {e}", exc_info=True)
                last_corr_run = now_utc

            publish_insights_sensor(mqtt_client, top_insights, context_entities, acc_stats, holiday_dates)

            for item in targets:
                eid = item["entity_id"]
                try:
                    drivers = driver_map.get(eid, [])
                    pred_state, cert, meta = train_and_predict(
                        conn, item, drivers, all_entity_map, holiday_dates, workday_item, acc_stats
                    )
                    publish_ha_discovery(mqtt_client, item, pred_state, meta)
                    log.info(
                        f"{eid} -> {pred_state} "
                        f"(cert: {cert:.2f}, acc_7d: {meta.get('accuracy_7d')}, "
                        f"base_7d: {meta.get('baseline_accuracy_7d')}, 24h: {meta.get('timeline_24h')})"
                    )
                except Exception as e:
                    log.error(f"Prediction failed for {eid}: {e}", exc_info=True)

        except Exception as e:
            log.error(f"Error in main loop: {e}", exc_info=True)

        time.sleep(max(5.0, LOOP_SECS - (time.time() - loop_started)))


if __name__ == "__main__":
    main()
