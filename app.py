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

# Tier 2: Optional context entities for cross-sensor correlation (not predicted, rolling retention)
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

CERTAINTY_THRESHOLD = float(os.getenv("CERTAINTY_THRESHOLD", "0.70"))
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
    # Tier 2: Rolling short-term archive for context entities (pruned automatically)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS context_history (
            entity_id TEXT,
            timestamp TEXT,
            state TEXT,
            PRIMARY KEY (entity_id, timestamp)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_state_hist_eid ON state_history(entity_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ctx_hist_eid ON context_history(entity_id, timestamp)")
    conn.commit()
    return conn


def get_last_timestamp(conn, table, entity_id):
    cur = conn.execute(f"SELECT MAX(timestamp) FROM {table} WHERE entity_id = ?", (entity_id,))
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def prune_context_history(conn):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CONTEXT_RETENTION_DAYS)).isoformat()
    cur = conn.execute("DELETE FROM context_history WHERE timestamp < ?", (cutoff,))
    conn.commit()
    if cur.rowcount > 0:
        log.info(f"Pruned {cur.rowcount} expired context_history rows older than {CONTEXT_RETENTION_DAYS} days.")


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
            # Only include context entities with discrete/actionable states
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


# --- Nightly Cross-Sensor Lagged Correlation Engine ---
def run_correlation_analysis(conn, all_entity_map):
    """
    Finds significant simultaneous and lead/lag relationships across all Tier 1 and Tier 2 entities.
    Returns:
      - driver_map: {target_eid: [{"driver_eid": str, "lag_mins": int, "corr": float, "summary": str}, ...]}
      - top_insights: [str, ...] human-readable household behavioral rules
    """
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

    # Build aligned binary matrix (rows = time intervals, cols = entity_ids)
    series_dict = {}
    for eid, grp in df_all.groupby("entity_id"):
        grp = grp.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        is_p = eid.startswith("person.")
        s = grp.set_index("timestamp")["state"].apply(lambda x: state_to_binary(x, is_person=is_p))
        # Extend timeline to now and resample
        s.loc[now_dt] = s.iloc[-1]
        s_res = s.resample(resample_rule).ffill().dropna()
        if s_res.nunique() >= 2 and len(s_res) >= 48:
            series_dict[eid] = s_res

    if len(series_dict) < 2:
        log.info("Not enough active entities with state transitions for correlation analysis yet.")
        return {}, []

    grid = pd.DataFrame(series_dict).ffill().bfill().dropna()
    if len(grid) < 48:
        return {}, []

    target_eids = [eid for eid in grid.columns if eid.startswith("person.") or any(
        fnmatch.fnmatch(eid, pat) for pat in BINARY_SENSOR_PATTERNS
    )]

    # Test simultaneous (0m) and driver-leads-target lags (5m, 10m, 15m)
    lags_mins = [0, 5, 10, 15]
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

            best_lag = 0
            best_corr = 0.0

            for lag_m in lags_mins:
                shift_steps = lag_m // RESAMPLE_MINS
                shifted_driver = grid[driver].shift(shift_steps) if shift_steps > 0 else grid[driver]
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

                if best_lag == 0:
                    timing = "at the same time"
                else:
                    timing = f"~{best_lag}m before {t_name} {t_action}"

                summary = f"{d_name} {d_action} {timing} (r={best_corr:+.2f})"
                candidates.append({
                    "driver_eid": driver,
                    "lag_mins": best_lag,
                    "corr": round(best_corr, 3),
                    "summary": summary,
                })
                all_findings.append((abs(best_corr), f"When {d_name} {d_action}, {t_name} {t_action} {timing if best_lag > 0 else 'simultaneously'} (r={best_corr:+.2f})"))

        # Keep top 3 strongest drivers per target
        candidates.sort(key=lambda x: abs(x["corr"]), reverse=True)
        driver_map[target] = candidates[:3]

    # Deduplicate and sort global household insights
    all_findings.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    top_insights = []
    for _, text in all_findings:
        if text not in seen:
            seen.add(text)
            top_insights.append(text)
        if len(top_insights) >= 15:
            break

    log.info(f"Correlation analysis complete: found {len(top_insights)} strong behavioral rules across {len(grid.columns)} entities.")
    return driver_map, top_insights


def get_driver_history_series(conn, driver_eid, lag_mins, resample_rule, index_like):
    """Fetches a resampled, lag-shifted binary series for a correlated driver entity."""
    table = "state_history" if (
        driver_eid.startswith("person.") or any(fnmatch.fnmatch(driver_eid, pat) for pat in BINARY_SENSOR_PATTERNS)
    ) else "context_history"

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
    """Generates a 48-char (30-min/char) high-contrast ASCII timeline for the next 24 hours."""
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
    """Vectorized 7-day scan using pure schedule features + 24h ASCII timeline builder."""
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
        "attributes": json.dumps(entity_item.get("attributes", {})),
    }])
    df = pd.concat([df, now_anchor], ignore_index=True)

    df["state_change_group"] = (df["state"] != df["state"].shift()).cumsum()
    df["group_start_ts"] = df.groupby("state_change_group")["timestamp"].transform("first")

    resample_rule = f"{RESAMPLE_MINS}min"
    df_indexed = df.set_index("timestamp").resample(resample_rule).ffill().dropna(subset=["state"])

    # Attach correlated driver history columns onto the same resampled time index
    driver_cols = []
    live_driver_vals = {}
    for idx_d, drv in enumerate(drivers):
        col_name = f"driver_{idx_d}"
        d_eid = drv["driver_eid"]
        d_lag = drv["lag_mins"]
        s_drv = get_driver_history_series(conn, d_eid, d_lag, resample_rule, df_indexed.index)
        df_indexed[col_name] = s_drv.values
        driver_cols.append(col_name)

        # Live driver state (at t - lag_mins, or current state if lag == 0)
        if d_lag == 0 and d_eid in all_entity_map:
            live_st = all_entity_map[d_eid].get("state")
            live_driver_vals[col_name] = state_to_binary(live_st, is_person=d_eid.startswith("person."))
        else:
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
        return const_state, 1.0, {
            "reason": "constant_historical_state",
            "expected_next_on": None,
            "expected_next_off": None,
            "timeline_24h": flat_line,
            "timeline_start": now_dt.astimezone(LOCAL_TZ).strftime("%H:%M"),
            "timeline_legend": f"█={positive_state} ·={negative_state} |=6h (30m/char)",
            "correlated_triggers": [d["summary"] for d in drivers],
            "recorded_hours": round(span_hours, 1),
        }

    train_df, feature_cols, schedule_cols = build_features(train_df, is_person=is_person, driver_cols=driver_cols)
    X = train_df[feature_cols]
    y = train_df["future_target"].astype(int)

    age_days = (now_dt - train_df["timestamp"]).dt.total_seconds() / 86400.0
    sample_weights = np.exp(-np.log(2) * age_days / DECAY_HALF_LIFE_DAYS)

    # 1. Context-aware model (includes live distance, velocity, ETA, correlated drivers + time)
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y, sample_weight=sample_weights)

    # 2. Pure schedule model for next_on / next_off & 24h timeline forecasting
    schedule_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    schedule_model.fit(train_df[schedule_cols], y, sample_weight=sample_weights)

    last_changed_str = entity_item.get("last_changed")
    last_changed_dt = datetime.fromisoformat(last_changed_str.replace("Z", "+00:00")) if last_changed_str else now_dt
    mins_in_state = max(0.0, (now_dt - last_changed_dt).total_seconds() / 60.0)
    curr_bin = 1 if entity_item.get("state") == positive_state else 0
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
    predicted_state = positive_state if prob_pos >= prob_neg else negative_state

    next_on, next_off, timeline_24h = estimate_next_transitions(schedule_model, now_dt, curr_bin)

    meta = {
        "probability_active": round(float(prob_pos), 3),
        "certainty": round(float(max_prob), 3),
        "threshold": CERTAINTY_THRESHOLD,
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

    if max_prob < CERTAINTY_THRESHOLD:
        return "unknown", max_prob, meta

    return predicted_state, max_prob, meta


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
            "model": "RandomForest State Forecaster",
        },
    }

    client.publish(config_topic, json.dumps(discovery_payload), retain=True)
    client.publish(state_topic, state, retain=True)
    client.publish(attr_topic, json.dumps(attributes), retain=True)


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


# --- Main Loop ---
def main():
    if not HA_TOKEN:
        raise ValueError("HA_TOKEN environment variable is required.")

    conn = init_db()
    mqtt_client = setup_mqtt()
    fetch_home_coords()
    log.info("HA ML Predictor started.")

    last_corr_run = None
    driver_map = {}
    top_insights = []

    while True:
        try:
            targets, context_entities = get_entities()
            all_entity_map = {e["entity_id"]: e for e in (targets + context_entities)}

            sync_entity_group(conn, targets, is_context=False)
            sync_entity_group(conn, context_entities, is_context=True)

            # Run correlation discovery at startup and every CORRELATION_INTERVAL_HOURS
            now_utc = datetime.now(timezone.utc)
            if last_corr_run is None or (now_utc - last_corr_run).total_seconds() >= CORRELATION_INTERVAL_HOURS * 3600:
                prune_context_history(conn)
                driver_map, top_insights = run_correlation_analysis(conn, all_entity_map)
                publish_insights_sensor(mqtt_client, top_insights, len(context_entities))
                last_corr_run = now_utc

            for item in targets:
                eid = item["entity_id"]
                drivers = driver_map.get(eid, [])
                pred_state, cert, meta = train_and_predict(conn, item, drivers, all_entity_map)
                publish_ha_discovery(mqtt_client, item, pred_state, meta)
                log.info(
                    f"{eid} -> {pred_state} "
                    f"(cert: {cert:.2f}, drivers: {len(drivers)}, 24h: {meta.get('timeline_24h')})"
                )

        except Exception as e:
            log.error(f"Error in main loop: {e}", exc_info=True)

        time.sleep(300)


if __name__ == "__main__":
    main()
