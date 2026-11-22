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
    wkday_wake = float(completed[~completed["is_weekend"]]["wake_hour"].median()) if (~completed["is_weekend"]).any() else overall_wake
    wkend_wake = float(completed[completed["is_weekend"]]["wake_hour"].median()) if completed["is_weekend"].any() else overall_wake
    overall_bed = float(sess_df["bed_hour"].median())
    avg_dur = float(completed["duration_hrs"].mean())

    def hour_to_hhmm(h):
        h_mod = h % 24.0
        hh = int(h_mod)
        mm = int(round((h_mod - hh) * 60)) % 60
        return f"{hh:02d}:{mm:02d}"

    # Find next upcoming wake-up timestamp
    next_wake_dt = None
    for day_offset in (0, 1):
        cand_date = now_local.date() + timedelta(days=day_offset)
        target_h = wkend_wake if cand_date.weekday() >= 5 else wkday_wake
        hh = int(target_h % 24)
        mm = int(round((target_h - int(target_h)) * 60)) % 60
        cand_dt = datetime(cand_date.year, cand_date.month, cand_date.day, hh, mm, tzinfo=LOCAL_TZ)
        if cand_dt > now_local:
            next_wake_dt = cand_dt
            break

    # Find next upcoming bedtime timestamp
    next_bed_dt = None
    for day_offset in (-1, 0, 1):
        base_date = now_local.date() + timedelta(days=day_offset)
        actual_date = base_date + timedelta(days=1) if overall_bed >= 24.0 else base_date
        hh = int(overall_bed % 24)
        mm = int(round((overall_bed - int(overall_bed)) * 60)) % 60
        cand_dt = datetime(actual_date.year, actual_date.month, actual_date.day, hh, mm, tzinfo=LOCAL_TZ)
        if cand_dt > now_local:
            next_bed_dt = cand_dt
            break

    return {
        "source_entity": best_eid,
        "next_wake_up": next_wake_dt.isoformat() if next_wake_dt else None,
        "next_bedtime": next_bed_dt.isoformat() if next_bed_dt else None,
        "weekday_wake_avg": hour_to_hhmm(wkday_wake),
        "weekend_wake_avg": hour_to_hhmm(wkend_wake),
        "bedtime_avg": hour_to_hhmm(overall_bed),
        "avg_sleep_duration_hrs": round(avg_dur, 1),
        "sleep_sessions_analyzed": len(completed),
    }


# --- Nightly Cross-Sensor Lagged Correlation Engine ---
def run_correlation_analysis(conn, all_entity_map):
    log.info("Running cross-sensor lagged correlation analysis...")
    now_dt = datetime.now(timezone.utc)
    cutoff_str = (now_dt - timedelta(days=CONTEXT_RETENTION_DAYS)).isoformat()

    df_t1 = pd.read_sql_query(
        "SELECT entity_id, timestamp, state FROM state_history WHERE timestamp >= ?",
        conn,
        params=(cutoff_str,),
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

    target_eids = [
        eid
        for eid in grid.columns
        if eid.startswith("person.") or any(fnmatch.fnmatch(eid, pat) for pat in BINARY_SENSOR_PATTERNS)
    ]

    # Require >= 5m lead lag so simultaneous automation reactions (0m) never masquerade as causes
    lags_mins = [5, 10, 15]
    driver_map = {}
    all_findings = []

    def friendly(eid):
        return all_entity_map.get(eid, {}).get("attributes", {}).get("friendly_name", eid)

    for target in target_eids:
        target_series = grid[target]
        candidates = []

        for driver in grid.columns:
            if driver == target:
                continue

            best_lag = 5
            best_corr = 0.0

            for lag_m in lags_mins:
                shift_steps = lag_m // RESAMPLE_MINS
                shifted_driver = grid[driver].shift(shift_steps)
                valid = pd.concat([target_series, shifted_driver], axis=1).dropna()
                if len(valid) < 48 or valid.iloc[:, 1].nunique() < 2:
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

                summary = f"{d_name} {d_action} ~{best_lag}m before {t_name} {t_action} (r={best_corr:+.2f})"
                candidates.append({
                    "driver_eid": driver,
                    "lag_mins": best_lag,
                    "corr": round(best_corr, 3),
                    "summary": summary,
                })
                all_findings.append((abs(best_corr), f"When {d_name} {d_action}, {t_name} {t_action} ~{best_lag}m later (r={best_corr:+.2f})"))

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

    log.info(f"Correlation analysis complete: found {len(top_insights)} strong behavioral rules.")
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


# --- Feature Engineering & ML ---
def build_features(df, is_person=False, driver_cols=None):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["dow"] = df["timestamp"].dt.day_of_week
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    schedule_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
    feature_cols = schedule_cols + ["current_state_bin", "mins_in_current_state"]

    if is_person:
        def calc_dist(a):
            return haversine_km(a.get("latitude"), a.get("longitude"), HOME_COORDS[0], HOME_COORDS[1])

        df["distance_from_home_km"] = df["attr_dict"].apply(calc_dist)
        df["gps_accuracy"] = df["attr_dict"].apply(lambda a: float(a.get("gps_accuracy", 0) or 0))

        if "home_approach_speed_kmh" not in df.columns:
            lookback_steps = max(1, VELOCITY_LOOKBACK_MINS // RESAMPLE_MINS)
            lookback_hours = (lookback_steps * RESAMPLE_MINS) / 60.0
            prev_dist = df["distance_from_home_km"].shift(lookback_steps).fillna(df["distance_from_home_km"])
            speed = (prev_dist - df["distance_from_home_km"]) / lookback_hours
            speed = speed.where(speed.abs() >= 1.5, 0.0).clip(-250.0, 250.0)
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


def build_24h_ascii_timeline(schedule_model, now_dt, threshold):
    slot_offsets = np.arange(48) * 30 + 15
    slot_timestamps = pd.to_datetime([now_dt + timedelta(minutes=int(m)) for m in slot_offsets], utc=True)
    hours = slot_timestamps.hour + slot_timestamps.minute / 60.0
    dows = slot_timestamps.day_of_week

    X_slots = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hours / 24.0),
        "hour_cos": np.cos(2 * np.pi * hours / 24.0),
        "dow_sin": np.sin(2 * np.pi * dows / 7.0),
        "dow_cos": np.cos(2 * np.pi * dows / 7.0),
        "is_weekend": (dows >= 5).astype(int),
    })

    probs = schedule_model.predict_proba(X_slots)
    classes = list(schedule_model.classes_)
    prob_pos = probs[:, classes.index(1)] if 1 in classes else np.zeros(48)

    chars = []
    for i, p in enumerate(prob_pos):
        if i > 0 and i % 12 == 0:
            chars.append("|")
        chars.append("█" if p >= threshold else "·")

    return "".join(chars)


def estimate_next_transitions(schedule_model, now_dt, current_state_bin):
    next_on = None
    next_off = None

    total_steps = (7 * 24 * 60) // RESAMPLE_MINS
    steps = np.arange(1, total_steps + 1)
    fut_timestamps = pd.to_datetime([now_dt + timedelta(minutes=RESAMPLE_MINS * int(s)) for s in steps], utc=True)
    hours = fut_timestamps.hour + fut_timestamps.minute / 60.0
    dows = fut_timestamps.day_of_week

    X_fut = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hours / 24.0),
        "hour_cos": np.cos(2 * np.pi * hours / 24.0),
        "dow_sin": np.sin(2 * np.pi * dows / 7.0),
        "dow_cos": np.cos(2 * np.pi * dows / 7.0),
        "is_weekend": (dows >= 5).astype(int),
    })

    probs = schedule_model.predict_proba(X_fut)
    classes = list(schedule_model.classes_)
    prob_pos = probs[:, classes.index(1)] if 1 in classes else np.zeros(len(steps))

    threshold = 0.50 if prob_pos.max() >= 0.50 else max(0.25, float(prob_pos.max()) * 0.85)
    preds = (prob_pos >= threshold).astype(int)

    sim_state = current_state_bin
    for idx, pred in enumerate(preds):
        if sim_state == 0 and pred == 1 and next_on is None:
            next_on = fut_timestamps[idx].tz_convert(LOCAL_TZ).isoformat()
        elif sim_state == 1 and pred == 0 and next_off is None:
            next_off = fut_timestamps[idx].tz_convert(LOCAL_TZ).isoformat()

        sim_state = pred
        if next_on and next_off:
            break

    timeline_24h = build_24h_ascii_timeline(schedule_model, now_dt, threshold)
    return next_on, next_off, timeline_24h


def train_and_predict(conn, entity_item, drivers, all_entity_map):
    eid = entity_item["entity_id"]
    is_person = eid.startswith("person.")
    positive_state = "home" if is_person else "on"
    negative_state = "not_home" if is_person else "off"
    now_dt = datetime.now(timezone.utc)
    live_state = entity_item.get("state")

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
        "state": live_state,
        "attributes": json.dumps(entity_item.get("attributes", {})),
    }])
    df = pd.concat([df, now_anchor], ignore_index=True)

    df["state_change_group"] = (df["state"] != df["state"].shift()).cumsum()
    df["group_start_ts"] = df.groupby("state_change_group")["timestamp"].transform("first")

    resample_rule = f"{RESAMPLE_MINS}min"
    df_indexed = df.set_index("timestamp").resample(resample_rule).ffill().dropna(subset=["state"])

    driver_cols = []
    live_driver_vals = {}
    for idx_d, drv in enumerate(drivers):
        col_name = f"driver_{idx_d}"
        d_eid = drv["driver_eid"]
        d_lag = drv["lag_mins"]
        s_drv = get_driver_history_series(conn, d_eid, d_lag, resample_rule, df_indexed.index)
        df_indexed[col_name] = s_drv.values
        driver_cols.append(col_name)
        live_driver_vals[col_name] = int(s_drv.iloc[-1]) if len(s_drv) > 0 else 0

    df = df_indexed.reset_index()
    df["attr_dict"] = df["attributes"].apply(json.loads)
    df["current_state_bin"] = (df["state"] == positive_state).astype(int)
    df["mins_in_current_state"] = (df["timestamp"] - df["group_start_ts"]).dt.total_seconds() / 60.0

    shift_steps = max(1, PREDICT_HORIZON_MINS // RESAMPLE_MINS)
    df["future_target"] = df["current_state_bin"].shift(-shift_steps)
    train_df = df.dropna(subset=["future_target"]).copy()

    if train_df["future_target"].nunique() < 2:
        const_val = int(train_df["future_target"].iloc[0])
        const_state = positive_state if const_val == 1 else negative_state
        glyph = "█" if const_val == 1 else "·"
        flat_line = "|".join([glyph * 12] * 4)
        acc_7d, graded_cnt, eff_thresh = log_and_score_predictions(conn, eid, const_state, 1.0, live_state, now_dt)
        return const_state, 1.0, {
            "reason": "constant_historical_state",
            "expected_next_on": None,
            "expected_next_off": None,
            "timeline_24h": flat_line,
            "timeline_start": now_dt.astimezone(LOCAL_TZ).strftime("%H:%M"),
            "timeline_legend": f"█={positive_state} ·={negative_state} |=6h (30m/char)",
            "correlated_triggers": [d["summary"] for d in drivers],
            "accuracy_7d": acc_7d,
            "graded_predictions_7d": graded_cnt,
            "effective_threshold": eff_thresh,
            "recorded_hours": round(span_hours, 1),
        }

    train_df, feature_cols, schedule_cols = build_features(train_df, is_person=is_person, driver_cols=driver_cols)
    X = train_df[feature_cols]
    y = train_df["future_target"].astype(int)

    age_days = (now_dt - train_df["timestamp"]).dt.total_seconds() / 86400.0
    sample_weights = np.exp(-np.log(2) * age_days / DECAY_HALF_LIFE_DAYS)

    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y, sample_weight=sample_weights)

    schedule_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    schedule_model.fit(train_df[schedule_cols], y, sample_weight=sample_weights)

    last_changed_str = entity_item.get("last_changed")
    last_changed_dt = datetime.fromisoformat(last_changed_str.replace("Z", "+00:00")) if last_changed_str else now_dt
    mins_in_state = max(0.0, (now_dt - last_changed_dt).total_seconds() / 60.0)
    curr_bin = 1 if live_state == positive_state else 0
    curr_attrs = entity_item.get("attributes", {})

    live_row = {
        "timestamp": now_dt,
        "attr_dict": curr_attrs,
        "current_state_bin": curr_bin,
        "mins_in_current_state": mins_in_state,
        **live_driver_vals,
    }

    if is_person:
        curr_dist = haversine_km(curr_attrs.get("latitude"), curr_attrs.get("longitude"), HOME_COORDS[0], HOME_COORDS[1])
        lookback_steps = max(1, VELOCITY_LOOKBACK_MINS // RESAMPLE_MINS)
        if len(df) > lookback_steps:
            past_attrs = df["attr_dict"].iloc[-(lookback_steps + 1)]
            past_dist = haversine_km(past_attrs.get("latitude"), past_attrs.get("longitude"), HOME_COORDS[0], HOME_COORDS[1])
            lookback_hours = (lookback_steps * RESAMPLE_MINS) / 60.0
            raw_speed = (past_dist - curr_dist) / lookback_hours
            approach_speed = 0.0 if abs(raw_speed) < 1.5 else max(-250.0, min(250.0, raw_speed))
        else:
            approach_speed = 0.0

        live_row["home_approach_speed_kmh"] = approach_speed
        live_row["eta_mins"] = compute_eta_mins(curr_dist, approach_speed, bool(curr_bin == 1))

    now_df = pd.DataFrame([live_row])
    now_df, _, _ = build_features(now_df, is_person=is_person, driver_cols=driver_cols)

    probs = model.predict_proba(now_df[feature_cols])[0]
    classes = list(model.classes_)
    prob_pos = probs[classes.index(1)] if 1 in classes else 0.0
    prob_neg = probs[classes.index(0)] if 0 in classes else 0.0

    max_prob = max(prob_pos, prob_neg)
    raw_predicted_state = positive_state if prob_pos >= prob_neg else negative_state

    # Self-score past predictions and fetch auto-calibrated certainty threshold
    acc_7d, graded_cnt, effective_threshold = log_and_score_predictions(
        conn, eid, raw_predicted_state if max_prob >= CERTAINTY_THRESHOLD else "unknown", max_prob, live_state, now_dt
    )

    final_state = raw_predicted_state if max_prob >= effective_threshold else "unknown"
    next_on, next_off, timeline_24h = estimate_next_transitions(schedule_model, now_dt, curr_bin)

    meta = {
        "probability_active": round(float(prob_pos), 3),
        "certainty": round(float(max_prob), 3),
        "base_threshold": CERTAINTY_THRESHOLD,
        "effective_threshold": effective_threshold,
        "accuracy_7d": acc_7d,
        "graded_predictions_7d": graded_cnt,
        "horizon_mins": PREDICT_HORIZON_MINS,
        "expected_next_on": next_on,
        "expected_next_off": next_off,
        "timeline_24h": timeline_24h,
        "timeline_start": now_dt.astimezone(LOCAL_TZ).strftime("%H:%M"),
        "timeline_legend": f"█={positive_state} ·={negative_state} |=6h (30m/char)",
        "correlated_triggers": [d["summary"] for d in drivers],
        "mins_in_current_state": round(mins_in_state, 1),
        "recorded_hours": round(span_hours, 1),
        "last_updated": now_dt.astimezone(LOCAL_TZ).isoformat(),
    }

    if is_person:
        meta["distance_from_home_km"] = round(float(now_df["distance_from_home_km"].iloc[0]), 2)
        meta["home_approach_speed_kmh"] = round(float(now_df["home_approach_speed_kmh"].iloc[0]), 1)
        meta["eta_mins"] = round(float(now_df["eta_mins"].iloc[0]), 0)

    return final_state, max_prob, meta


# --- MQTT Home Assistant Auto-Discovery ---
def setup_mqtt():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
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
    discovery_payload = {
        "name": f"{friendly_name} Predicted",
        "default_entity_id": default_entity_id,
        "unique_id": unique_id,
        "state_topic": f"{base_topic}/state",
        "json_attributes_topic": f"{base_topic}/attributes",
        "icon": "mdi:account-clock" if eid.startswith("person.") else "mdi:motion-sensor",
        "device": {
            "identifiers": ["ha_ml_predictor_service"],
            "name": "HA ML Predictor",
            "manufacturer": "Custom ML Service",
            "model": "RandomForest State Forecaster",
        },
    }

    client.publish(f"{base_topic}/config", json.dumps(discovery_payload), retain=True)
    client.publish(f"{base_topic}/state", state, retain=True)
    client.publish(f"{base_topic}/attributes", json.dumps(attributes), retain=True)


def publish_insights_sensor(client, insights, context_count):
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
            "model": "RandomForest State Forecaster",
        },
    }
    attrs = {
        "insights": insights,
        "context_entities_monitored": context_count,
        "min_correlation_threshold": CORRELATION_MIN_STRENGTH,
        "last_analyzed": datetime.now(timezone.utc).astimezone(LOCAL_TZ).isoformat(),
    }
    client.publish(f"{base_topic}/config", json.dumps(discovery_payload), retain=True)
    client.publish(f"{base_topic}/state", str(len(insights)), retain=True)
    client.publish(f"{base_topic}/attributes", json.dumps(attrs), retain=True)


def publish_sleep_sensors(client, sleep_info):
    if not sleep_info:
        return

    device_block = {
        "identifiers": ["ha_ml_predictor_service"],
        "name": "HA ML Predictor",
        "manufacturer": "Custom ML Service",
        "model": "RandomForest State Forecaster",
    }

    # 1. sensor.expected_wake_up (native HA timestamp entity)
    if sleep_info.get("next_wake_up"):
        wake_id = "expected_wake_up"
        wake_topic = f"{MQTT_PREFIX}/sensor/{wake_id}"
        wake_cfg = {
            "name": "Expected Wake Up",
            "default_entity_id": "sensor.expected_wake_up",
            "unique_id": wake_id,
            "device_class": "timestamp",
            "state_topic": f"{wake_topic}/state",
            "json_attributes_topic": f"{wake_topic}/attributes",
            "icon": "mdi:weather-sunset-up",
            "device": device_block,
        }
        client.publish(f"{wake_topic}/config", json.dumps(wake_cfg), retain=True)
        client.publish(f"{wake_topic}/state", sleep_info["next_wake_up"], retain=True)
        client.publish(f"{wake_topic}/attributes", json.dumps(sleep_info), retain=True)

    # 2. sensor.expected_bedtime (native HA timestamp entity)
    if sleep_info.get("next_bedtime"):
        bed_id = "expected_bedtime"
        bed_topic = f"{MQTT_PREFIX}/sensor/{bed_id}"
        bed_cfg = {
            "name": "Expected Bedtime",
            "default_entity_id": "sensor.expected_bedtime",
            "unique_id": bed_id,
            "device_class": "timestamp",
            "state_topic": f"{bed_topic}/state",
            "json_attributes_topic": f"{bed_topic}/attributes",
            "icon": "mdi:bed-clock",
            "device": device_block,
        }
        client.publish(f"{bed_topic}/config", json.dumps(bed_cfg), retain=True)
        client.publish(f"{bed_topic}/state", sleep_info["next_bedtime"], retain=True)
        client.publish(f"{bed_topic}/attributes", json.dumps(sleep_info), retain=True)


# --- Main Loop ---
def main():
    if not HA_TOKEN:
        raise ValueError("HA_TOKEN environment variable is required.")

    conn = init_db()
    mqtt_client = setup_mqtt()
    fetch_home_coords()
    log.info("HA ML Predictor started.")

    last_nightly_run = None
    driver_map = {}
    top_insights = []

    while True:
        try:
            targets, context_entities = get_entities()
            all_entity_map = {e["entity_id"]: e for e in (targets + context_entities)}

            sync_entity_group(conn, targets, is_context=False)
            sync_entity_group(conn, context_entities, is_context=True)

            now_utc = datetime.now(timezone.utc)
            if last_nightly_run is None or (now_utc - last_nightly_run).total_seconds() >= CORRELATION_INTERVAL_HOURS * 3600:
                prune_old_records(conn)
                driver_map, top_insights = run_correlation_analysis(conn, all_entity_map)
                publish_insights_sensor(mqtt_client, top_insights, len(context_entities))
                last_nightly_run = now_utc

            # Update sleep / wake-up timestamp sensors
            sleep_info = infer_sleep_schedule(conn, targets)
            if sleep_info:
                publish_sleep_sensors(mqtt_client, sleep_info)

            for item in targets:
                eid = item["entity_id"]
                drivers = driver_map.get(eid, [])
                pred_state, cert, meta = train_and_predict(conn, item, drivers, all_entity_map)
                publish_ha_discovery(mqtt_client, item, pred_state, meta)
                log.info(
                    f"{eid} -> {pred_state} "
                    f"(cert: {cert:.2f}/{meta.get('effective_threshold')}, acc_7d: {meta.get('accuracy_7d')}, 24h: {meta.get('timeline_24h')})"
                )

        except Exception as e:
            log.error(f"Error in main loop: {e}", exc_info=True)

        time.sleep(300)


if __name__ == "__main__":
    main()
