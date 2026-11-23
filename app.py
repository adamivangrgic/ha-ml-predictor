"""
HA ML Predictor v3 (single file; same entrypoint as before).

Since the previous patch:
  Predictions
    * Household target "everyone home" (+ expected number of people home).  ("anyone home" is not useful.)
    * Arrival/departure time as a distribution (p10/p50/p90 minutes until the next change).
    * Extra features: solar elevation (computed, no dependency), time since last on/off,
      optional mode entities (vacation/guests...), optional per-person companion-app/travel-time sensors.
    * Zone prediction for persons (home / work / other ... probabilities).
    * Drift handling: shorter decay half-life while accuracy is poor.
    * Driver selection by permutation importance (more candidates, only useful ones are kept).
  Evaluation
    * Brier score / log loss per horizon, vs. persistence and vs. a weekday+time-of-day histogram baseline.
  Outputs
    * binary_sensor "likely home/active" per entity, anomaly binary_sensor, MQTT availability/LWT,
      diagnostics sensor, heartbeat file.
"""
import os
import math
import time
import json
import signal
import sqlite3
import fnmatch
import logging
from collections import deque
from datetime import datetime, timedelta, timezone, date
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ha-ml-predictor")


def _env_list(name, default=""):
    return [p.strip() for p in os.getenv(name, default).split(",") if p.strip()]


def _env_bool(name, default="1"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# --- Configuration via Environment Variables ---
HA_URL = os.getenv("HA_URL", "http://homeassistant.local:8123").rstrip("/")
HA_TOKEN = os.getenv("HA_TOKEN", "")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER", "")
MQTT_PASS = os.getenv("MQTT_PASS", "")
MQTT_PREFIX = os.getenv("MQTT_DISCOVERY_PREFIX", "homeassistant")
STATUS_TOPIC = os.getenv("MQTT_STATUS_TOPIC", "ha_ml_predictor/availability")

# Tier 1: Entities to predict (all person.* are always included automatically)
BINARY_SENSOR_PATTERNS = _env_list("BINARY_SENSOR_PATTERNS")
# Tier 2: Optional context entities for cross-sensor correlation
CONTEXT_ENTITY_PATTERNS = _env_list(
    "CONTEXT_ENTITY_PATTERNS", "light.*,media_player.*,lock.*,cover.*,binary_sensor.*_door"
)
EXCLUDE_PATTERNS = _env_list(
    "EXCLUDE_PATTERNS",
    "*_battery*,*_rssi*,*_linkquality*,*_update*,*_uptime*,*_power*,*_energy*,*_voltage*,*_current*,*_temperature*,*_humidity*",
)

HOLIDAY_CALENDAR_ENTITY_ID = os.getenv("HOLIDAY_CALENDAR_ENTITY_ID", "").strip()
WORKDAY_ENTITY_ID = os.getenv("WORKDAY_ENTITY_ID", "binary_sensor.workday_sensor").strip()

CERTAINTY_THRESHOLD = float(os.getenv("CERTAINTY_THRESHOLD", "0.70"))
MIN_ACCEPTABLE_ACCURACY = float(os.getenv("MIN_ACCEPTABLE_ACCURACY", "0.65"))
PREDICT_HORIZON_MINS = int(os.getenv("PREDICT_HORIZON_MINS", "30"))
EXTRA_HORIZONS_MINS = [int(h) for h in _env_list("EXTRA_HORIZONS_MINS", "15,30,60,180")]
RESAMPLE_MINS = int(os.getenv("RESAMPLE_MINS", "5"))
VELOCITY_LOOKBACK_MINS = int(os.getenv("VELOCITY_LOOKBACK_MINS", "15"))
MIN_HISTORY_HOURS = int(os.getenv("MIN_HISTORY_HOURS", "24"))
DECAY_HALF_LIFE_DAYS = float(os.getenv("DECAY_HALF_LIFE_DAYS", "7.0"))
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", str(max(14, int(math.ceil(DECAY_HALF_LIFE_DAYS * 4))))))
# Defaults now follow the training window (HA returns whatever its recorder still has)
INITIAL_HISTORY_DAYS = int(os.getenv("INITIAL_HISTORY_DAYS", str(TRAIN_WINDOW_DAYS)))
CONTEXT_RETENTION_DAYS = int(os.getenv("CONTEXT_RETENTION_DAYS", str(TRAIN_WINDOW_DAYS)))
CORRELATION_MIN_STRENGTH = float(os.getenv("CORRELATION_MIN_STRENGTH", "0.28"))
CORRELATION_INTERVAL_HOURS = float(os.getenv("CORRELATION_INTERVAL_HOURS", "24.0"))
DB_PATH = os.getenv("DB_PATH", "/data/history.db")

RETRAIN_INTERVAL_MINS = int(os.getenv("RETRAIN_INTERVAL_MINS", "60"))
HISTORY_BATCH_SIZE = int(os.getenv("HISTORY_BATCH_SIZE", "20"))
LOOP_SECS = int(os.getenv("LOOP_SECS", "300"))
HOLIDAY_REFRESH_HOURS = float(os.getenv("HOLIDAY_REFRESH_HOURS", "6"))
MAX_ATTR_BYTES = 14000  # HA drops state attributes above 16 KB

# New in v3
MAX_DRIVER_CANDIDATES = int(os.getenv("MAX_DRIVER_CANDIDATES", "5"))
DRIVER_SELECTION = _env_bool("DRIVER_SELECTION", "1")
HOUSEHOLD_ENABLED = _env_bool("HOUSEHOLD_ENABLED", "1")
HOUSEHOLD_PERSONS = _env_list("HOUSEHOLD_PERSONS")  # empty = all person.* entities
ZONE_PREDICTION = _env_bool("ZONE_PREDICTION", "1")
ARRIVAL_PREDICTION = _env_bool("ARRIVAL_PREDICTION", "1")
BASELINE_WEEKS = int(os.getenv("BASELINE_WEEKS", "4"))
PUBLISH_BINARY_HORIZONS = [int(h) for h in _env_list("PUBLISH_BINARY_HORIZONS", str(PREDICT_HORIZON_MINS))]
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.85"))
ANOMALY_MIN_MINS = float(os.getenv("ANOMALY_MIN_MINS", "30"))
MODE_ENTITY_IDS = _env_list("MODE_ENTITY_IDS")  # e.g. input_boolean.vacation,input_boolean.guests
HEARTBEAT_FILE = os.getenv("HEARTBEAT_FILE", "/tmp/ha_ml_predictor.heartbeat")
QUANTILES = (0.1, 0.5, 0.9)


def _load_person_features():
    """
    PERSON_FEATURE_ENTITIES (JSON), e.g.
      {"person.john": [
          {"entity_id": "sensor.john_phone_wifi_connection", "match": ["MyHomeWifi"]},
          {"entity_id": "sensor.john_phone_activity", "match": ["automotive"]},
          {"entity_id": "binary_sensor.john_phone_charging"},
          {"entity_id": "sensor.john_travel_time", "numeric": true}]}
    """
    raw = os.getenv("PERSON_FEATURE_ENTITIES", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as e:
        log.warning(f"PERSON_FEATURE_ENTITIES is not valid JSON: {e}")
        return {}
    out = {}
    for person, specs in data.items():
        lst = []
        for sp in specs:
            if isinstance(sp, str):
                sp = {"entity_id": sp}
            if sp.get("entity_id"):
                lst.append({
                    "entity_id": sp["entity_id"],
                    "match": [str(x).strip().lower() for x in sp.get("match", [])],
                    "numeric": bool(sp.get("numeric", False)),
                })
        out[person] = lst
    return out


PERSON_FEATURES = _load_person_features()
MODE_SPECS = [{"entity_id": e, "match": [], "numeric": False} for e in MODE_ENTITY_IDS]
EXPLICIT_FEATURE_IDS = set(MODE_ENTITY_IDS) | {sp["entity_id"] for v in PERSON_FEATURES.values() for sp in v}

HEADERS = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
HOME_COORDS = (None, None)
LOCAL_TZ = datetime.now().astimezone().tzinfo  # fallback only; replaced by init_timezone()

ACTIVE_STATES = {"on", "home", "open", "unlocked", "playing", "heating", "cooling", "true"}
INACTIVE_STATES = {"off", "not_home", "closed", "locked", "idle", "paused", "standby", "false"}

HOUSEHOLD_EID = "household.everyone_home"
SCHEDULE_COLS = [
    "hour_sin", "hour_cos", "hour_sin2", "hour_cos2", "hour_sin3", "hour_cos3",
    "dow_sin", "dow_cos", "is_weekend", "is_workday", "is_tomorrow_workday",
    "sun_elev", "sun_rising",
]

MODEL_CACHE = {}
RUNTIME = {"errors": deque(maxlen=100), "loop_secs": None, "loops": 0}

DEVICE_INFO = {
    "identifiers": ["ha_ml_predictor_service"],
    "name": "HA ML Predictor",
    "manufacturer": "Custom ML Service",
    "model": "Calibrated HGB State Forecaster",
}


def note_error(msg):
    RUNTIME["errors"].append((datetime.now(timezone.utc), str(msg)[:300]))


# --- Entity-kind helpers ---
def is_presence(eid):
    """person.* and the synthetic household entity use home/not_home semantics."""
    return eid.startswith("person.") or eid == HOUSEHOLD_EID


def has_gps(eid):
    return eid.startswith("person.")


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


# --- Geospatial / astronomy helpers ---
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


def solar_features(ts):
    """Solar elevation (deg) and 'before solar noon' flag via the NOAA approximation (no dependencies)."""
    n = len(ts)
    try:
        lat, lon = float(HOME_COORDS[0]), float(HOME_COORDS[1])
    except (TypeError, ValueError):
        return np.zeros(n), np.zeros(n, dtype=int)
    t = ts.tz_convert("UTC")
    doy = np.asarray(t.dayofyear, dtype=float)
    hour = np.asarray(t.hour + t.minute / 60.0 + t.second / 3600.0, dtype=float)
    g = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = (hour * 60.0 + eqtime + 4.0 * lon) % 1440.0
    ha = np.radians(tst / 4.0 - 180.0)
    latr = math.radians(lat)
    cosz = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(ha)
    elev = 90.0 - np.degrees(np.arccos(np.clip(cosz, -1.0, 1.0)))
    return elev, (ha < 0).astype(int)


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
    if is_presence(eid):
        return "home" if raw_state == "home" else "not_home"
    return "on" if str(raw_state).lower() in ACTIVE_STATES else "off"


def spec_value(state, spec):
    """Converts an arbitrary entity state into a numeric feature according to its spec."""
    if state is None:
        return float("nan")
    s = str(state).strip().lower()
    if s in ("", "unknown", "unavailable", "none"):
        return float("nan")
    if spec["numeric"]:
        try:
            return float(s)
        except ValueError:
            return float("nan")
    if spec["match"]:
        return 1.0 if s in spec["match"] else 0.0
    return float(state_to_binary(s))


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
        return {str(k): sanitize(v) for k, v in obj.items()}
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
    if not has_gps(eid):
        return {}
    attrs = attrs or {}
    return {k: attrs[k] for k in ("latitude", "longitude", "gps_accuracy") if k in attrs}


def decay_weights(timestamps, now_dt, half_life):
    age_days = (now_dt - timestamps).dt.total_seconds() / 86400.0
    return np.exp(-np.log(2) * age_days / half_life).to_numpy()


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
        CREATE TABLE IF NOT EXISTS horizon_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT, predicted_at TEXT, horizon_mins INTEGER, target_ts TEXT,
            prob REAL, persist_prob REAL, hist_prob REAL, actual INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_state (
            entity_id TEXT, table_name TEXT, synced_until TEXT,
            PRIMARY KEY (entity_id, table_name)
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")

    cols = _column_names(conn, "prediction_log")
    for name, ddl in (
        ("confidence", "REAL"), ("baseline_state", "TEXT"),
        ("abstained", "INTEGER"), ("is_baseline_correct", "INTEGER"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE prediction_log ADD COLUMN {name} {ddl}")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_state_hist_eid ON state_history(entity_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ctx_hist_eid ON context_history(entity_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_log_eid ON prediction_log(entity_id, target_ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hor_log_eid ON horizon_log(entity_id, target_ts)")
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
            conn.execute(f"INSERT OR IGNORE INTO {table} VALUES ({','.join('?' * len(new_row))})", new_row)
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
    log_cut = fmt_ts(now - timedelta(days=30))
    c3 = conn.execute("DELETE FROM prediction_log WHERE predicted_at < ?", (log_cut,)).rowcount
    c4 = conn.execute("DELETE FROM horizon_log WHERE predicted_at < ?", (log_cut,)).rowcount
    conn.commit()
    if d1 or d2 or c3 or c4:
        log.info(f"Pruned {d1} state rows, {d2} context rows, {c3 + c4} old prediction logs.")


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
            rows_cache[eid] = conn.execute(
                f"SELECT 1 FROM {table} WHERE entity_id = ? LIMIT 1", (eid,)
            ).fetchone() is not None
        return rows_cache[eid]

    for entity_history in states_list:
        if not entity_history:
            continue
        first_eid = entity_history[0].get("entity_id")
        for idx, entry in enumerate(entity_history):
            entity_id = entry.get("entity_id") or first_eid
            if not entity_id:
                continue
            if not is_context and has_gps(entity_id):
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


# --- Baselines ---
class HistBaseline:
    """Weekday + time-of-day occupancy histogram over the last BASELINE_WEEKS weeks."""

    def __init__(self, base_df, now_dt):
        ts = base_df["timestamp"]
        local = ts.dt.tz_convert(LOCAL_TZ)
        recent = (ts >= now_dt - timedelta(weeks=BASELINE_WEEKS)).to_numpy()
        self.dow = local.dt.dayofweek.to_numpy()[recent]
        self.mod = (local.dt.hour * 60 + local.dt.minute).to_numpy()[recent]
        self.y = base_df["current_state_bin"].to_numpy()[recent].astype(float)

    def prob(self, target_dt):
        if len(self.y) == 0:
            return None
        tl = target_dt.astimezone(LOCAL_TZ)
        t_mod = tl.hour * 60 + tl.minute
        d = np.abs(self.mod - t_mod)
        d = np.minimum(d, 1440 - d)
        m = (self.dow == tl.weekday()) & (d <= 10)
        if int(m.sum()) < 3:
            return None
        return float(self.y[m].mean())


# --- Self-Evaluating Accuracy / Proper Scoring ---
def compute_scores(conn):
    """Brier score + log loss per entity/horizon over 7 days, vs persistence and vs the histogram baseline."""
    cutoff = fmt_ts(datetime.now(timezone.utc) - timedelta(days=7))
    df = pd.read_sql_query(
        "SELECT entity_id, horizon_mins, prob, persist_prob, hist_prob, actual FROM horizon_log "
        "WHERE actual IS NOT NULL AND predicted_at >= ?",
        conn, params=(cutoff,),
    )
    scores = {}
    if df.empty:
        return scores
    for (eid, h), g in df.groupby(["entity_id", "horizon_mins"]):
        a = g["actual"].to_numpy(dtype=float)
        p = g["prob"].to_numpy(dtype=float)
        pp = g["persist_prob"].to_numpy(dtype=float)
        pc = np.clip(p, 0.02, 0.98)
        brier = float(np.mean((p - a) ** 2))
        brier_persist = float(np.mean((pp - a) ** 2))
        logloss = float(-np.mean(a * np.log(pc) + (1 - a) * np.log(1 - pc)))
        entry = {
            "n": int(len(g)),
            "brier": round(brier, 4),
            "brier_persist": round(brier_persist, 4),
            "logloss": round(logloss, 4),
            "bss_vs_persist": round(1 - brier / brier_persist, 3) if brier_persist > 1e-9 else None,
        }
        gh = g[g["hist_prob"].notna()]
        if len(gh) >= 20:
            ah = gh["actual"].to_numpy(dtype=float)
            bh = float(np.mean((gh["hist_prob"].to_numpy(dtype=float) - ah) ** 2))
            bm = float(np.mean((gh["prob"].to_numpy(dtype=float) - ah) ** 2))
            entry["brier_hist"] = round(bh, 4)
            entry["bss_vs_hist"] = round(1 - bm / bh, 3) if bh > 1e-9 else None
        scores.setdefault(eid, {})[f"{int(h)}m"] = entry
    return scores


def evaluate_past_predictions(conn):
    now_iso = fmt_ts(datetime.now(timezone.utc))

    def state_at(eid, ts):
        res = conn.execute(
            "SELECT state FROM state_history WHERE entity_id = ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
            (eid, ts),
        ).fetchone()
        return res[0] if res else None

    pending = conn.execute(
        "SELECT id, entity_id, target_ts, predicted_state, baseline_state FROM prediction_log "
        "WHERE actual_state IS NULL AND target_ts <= ?", (now_iso,),
    ).fetchall()
    for row_id, eid, target_ts, pred_state, base_state in pending:
        raw = state_at(eid, target_ts)
        if raw is None:
            continue
        norm_actual = normalize_label(eid, raw)
        conn.execute(
            "UPDATE prediction_log SET actual_state = ?, is_correct = ?, is_baseline_correct = ? WHERE id = ?",
            (norm_actual, 1 if pred_state == norm_actual else 0,
             (1 if base_state == norm_actual else 0) if base_state else None, row_id),
        )

    pend_h = conn.execute(
        "SELECT id, entity_id, target_ts FROM horizon_log WHERE actual IS NULL AND target_ts <= ?", (now_iso,)
    ).fetchall()
    for row_id, eid, target_ts in pend_h:
        raw = state_at(eid, target_ts)
        if raw is None:
            continue
        conn.execute(
            "UPDATE horizon_log SET actual = ? WHERE id = ?",
            (1 if normalize_label(eid, raw) in ("home", "on") else 0, row_id),
        )
    conn.commit()

    cutoff_7d = fmt_ts(datetime.now(timezone.utc) - timedelta(days=7))
    rows = conn.execute(
        """
        SELECT entity_id, COUNT(*), AVG(is_correct), AVG(is_baseline_correct), AVG(abstained)
        FROM prediction_log
        WHERE actual_state IS NOT NULL AND baseline_state IS NOT NULL AND predicted_at >= ?
        GROUP BY entity_id
        """, (cutoff_7d,),
    ).fetchall()

    stats = {}
    for eid, count, acc, base_acc, abstain in rows:
        stats[eid] = {
            "count_7d": int(count),
            "accuracy_7d": round(float(acc), 3),
            "baseline_accuracy_7d": round(float(base_acc), 3) if base_acc is not None else None,
            "abstain_rate_7d": round(float(abstain), 3) if abstain is not None else None,
        }
    for eid, sc in compute_scores(conn).items():
        stats.setdefault(eid, {})["scores"] = sc
    return stats


def record_prediction(conn, eid, now_dt, horizon_mins, predicted_state, confidence, baseline_state, abstained):
    """Logs every raw model prediction (including abstentions) so accuracy does not depend on the threshold."""
    if predicted_state == "unknown":
        return
    recent_cutoff = fmt_ts(now_dt - timedelta(minutes=14))
    if conn.execute(
        "SELECT 1 FROM prediction_log WHERE entity_id = ? AND predicted_at >= ? LIMIT 1", (eid, recent_cutoff)
    ).fetchone():
        return
    conn.execute(
        "INSERT INTO prediction_log (entity_id, predicted_at, target_ts, predicted_state, confidence, baseline_state, abstained) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (eid, fmt_ts(now_dt), fmt_ts(now_dt + timedelta(minutes=horizon_mins)), predicted_state,
         float(confidence), baseline_state, 1 if abstained else 0),
    )
    conn.commit()


def record_horizon_log(conn, eid, now_dt, rows):
    """rows: [(horizon_mins, prob, persist_prob, hist_prob)] -> one batch per ~15 minutes per entity."""
    recent_cutoff = fmt_ts(now_dt - timedelta(minutes=14))
    if conn.execute(
        "SELECT 1 FROM horizon_log WHERE entity_id = ? AND predicted_at >= ? LIMIT 1", (eid, recent_cutoff)
    ).fetchone():
        return
    conn.executemany(
        "INSERT INTO horizon_log (entity_id, predicted_at, horizon_mins, target_ts, prob, persist_prob, hist_prob) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(eid, fmt_ts(now_dt), h, fmt_ts(now_dt + timedelta(minutes=h)), p, pp, hp) for h, p, pp, hp in rows],
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
                    if "dateTime" in end_obj else s_dt
                )
                curr = s_dt.date()
                while curr <= e_dt.date():
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

    targets, target_ids, context_entities = [], set(), []
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
        if eid in target_ids or eid == WORKDAY_ENTITY_ID or eid.startswith("calendar."):
            continue
        if eid in EXPLICIT_FEATURE_IDS:  # explicit feature entities bypass exclude/state filters
            context_entities.append(item)
            continue
        if is_excluded(eid):
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
    if not is_context and has_gps(eid):
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
                note_error(f"history HTTP {resp.status_code}")
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
            note_error(f"history sync: {e}")

    if total_new > 0 and not is_context:
        log.info(f"Archived {total_new} new state entries ({len(entities)} entities)")


# --- Household (derived) target ---
def household_member_ids(person_items):
    ids = [p["entity_id"] for p in person_items]
    if HOUSEHOLD_PERSONS:
        ids = [i for i in ids if i in HOUSEHOLD_PERSONS]
    return ids


def sync_household_history(conn, member_ids):
    """
    Derives the 'everyone home' history from the members' histories and stores only its changes.
    Returns a synthetic entity item (state, last_changed) or None when there are < 2 members/no data.
    """
    if len(member_ids) < 2:
        return None

    series = []
    for eid in member_ids:
        df = pd.read_sql_query(
            """SELECT timestamp, state FROM (
                   SELECT timestamp, state, LAG(state) OVER (ORDER BY timestamp) AS prev
                   FROM state_history WHERE entity_id = ?)
               WHERE prev IS NULL OR (prev = 'home') != (state = 'home')
               ORDER BY timestamp""",
            conn, params=(eid,),
        )
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        s = (df["state"] == "home").astype(int)
        s.index = df["timestamp"]
        series.append(s[~s.index.duplicated(keep="last")])

    start = max(s.index.min() for s in series)
    idx = sorted(set().union(*[set(s.index) for s in series]))
    idx = pd.DatetimeIndex([t for t in idx if t >= start])
    if len(idx) == 0:
        return None
    grid = pd.DataFrame(index=idx)
    for eid, s in zip(member_ids, series):
        grid[eid] = s.reindex(s.index.union(idx)).ffill().reindex(idx)
    all_home = (grid.min(axis=1) == 1).astype(int)
    changes = all_home[all_home != all_home.shift()]

    conn.executemany(
        "INSERT OR IGNORE INTO state_history (entity_id, timestamp, state, attributes) VALUES (?, ?, ?, ?)",
        [(HOUSEHOLD_EID, fmt_ts(ts.to_pydatetime()), "home" if v == 1 else "not_home", "{}") for ts, v in changes.items()],
    )
    conn.commit()

    row = conn.execute(
        "SELECT timestamp, state FROM state_history WHERE entity_id = ? ORDER BY timestamp DESC LIMIT 1",
        (HOUSEHOLD_EID,),
    ).fetchone()
    if not row:
        return None
    return {
        "entity_id": HOUSEHOLD_EID,
        "state": row[1],
        "last_changed": row[0],
        "attributes": {"friendly_name": "Everyone Home"},
    }


# --- Workday / holiday series ---
def get_workday_series(conn, resample_rule, index_like, holiday_dates, live_workday_item):
    local_idx = index_like.tz_convert(LOCAL_TZ)
    base_weekday = np.asarray(local_idx.day_of_week < 5).astype(int)

    if HOLIDAY_CALENDAR_ENTITY_ID:
        if not holiday_dates:
            return pd.Series(base_weekday, index=index_like)
        local_dates = local_idx.date
        is_hol = np.fromiter((1 if d in holiday_dates else 0 for d in local_dates), dtype=int, count=len(local_dates))
        return pd.Series(np.where((base_weekday == 1) & (is_hol == 0), 1, 0), index=index_like, dtype=int)

    fallback = pd.Series(base_weekday, index=index_like)
    if not WORKDAY_ENTITY_ID:
        return fallback

    df_w = pd.read_sql_query(
        "SELECT timestamp, state FROM state_history WHERE entity_id = ? ORDER BY timestamp ASC",
        conn, params=(WORKDAY_ENTITY_ID,),
    )
    if df_w.empty:
        if live_workday_item and live_workday_item.get("state") in ("on", "off"):
            return pd.Series(1 if live_workday_item.get("state") == "on" else 0, index=index_like)
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
        conn, params=(cutoff_str, WORKDAY_ENTITY_ID),
    )
    df_t2 = pd.read_sql_query(
        "SELECT entity_id, timestamp, state FROM context_history WHERE timestamp >= ?",
        conn, params=(cutoff_str,),
    )
    df_all = pd.concat([df_t1, df_t2], ignore_index=True)
    # Explicit feature entities (SSIDs, numeric sensors, modes) are model features, not correlation drivers.
    df_all = df_all[~df_all["entity_id"].isin(EXPLICIT_FEATURE_IDS)]
    if df_all.empty:
        return {}, []

    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], format="ISO8601", utc=True)
    resample_rule = f"{RESAMPLE_MINS}min"

    series_dict = {}
    for eid, grp in df_all.groupby("entity_id"):
        grp = grp.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        is_p = is_presence(eid)
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
        if is_presence(eid) or any(fnmatch.fnmatch(eid, pat) for pat in BINARY_SENSOR_PATTERNS)
    ]

    lags_mins = [0, 5, 10, 15]
    driver_map, all_findings = {}, []

    def friendly(eid):
        return all_entity_map.get(eid, {}).get("attributes", {}).get("friendly_name", eid)

    for target in target_eids:
        target_res = residuals[target]
        candidates = []

        for driver in residuals.columns:
            if driver == target or driver == HOUSEHOLD_EID:  # household is a function of persons
                continue

            best_lag, best_corr = 0, 0.0
            for lag_m in lags_mins:
                shift_steps = lag_m // RESAMPLE_MINS
                shifted = residuals[driver].shift(shift_steps) if shift_steps > 0 else residuals[driver]
                valid = pd.concat([target_res, shifted], axis=1).dropna()
                if len(valid) < 48 or valid.iloc[:, 1].std() < 1e-6 or valid.iloc[:, 0].std() < 1e-6:
                    continue
                r = float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
                if not math.isnan(r) and abs(r) > abs(best_corr):
                    best_corr, best_lag = r, lag_m

            if abs(best_corr) >= CORRELATION_MIN_STRENGTH:
                d_name, t_name = friendly(driver), friendly(target)
                d_is_person, t_is_person = is_presence(driver), is_presence(target)

                if best_corr > 0:
                    d_action = "arrives home" if d_is_person else "activates"
                    t_action = "arrives home" if t_is_person else "activates"
                else:
                    d_action = "leaves home" if d_is_person else "turns off"
                    t_action = "arrives home" if t_is_person else "activates"

                timing = "at the same time" if best_lag == 0 else f"~{best_lag}m before {t_name} {t_action}"
                candidates.append({
                    "driver_eid": driver,
                    "lag_mins": best_lag,
                    "corr": round(best_corr, 3),
                    "summary": f"{d_name} {d_action} {timing} (r={best_corr:+.2f})",
                })
                all_findings.append((
                    abs(best_corr),
                    f"When {d_name} {d_action}, {t_name} {t_action} {timing if best_lag > 0 else 'simultaneously'} (r={best_corr:+.2f})",
                ))

        candidates.sort(key=lambda x: abs(x["corr"]), reverse=True)
        driver_map[target] = candidates[:MAX_DRIVER_CANDIDATES]

    all_findings.sort(key=lambda x: x[0], reverse=True)
    seen, top_insights = set(), []
    for _, text in all_findings:
        if text not in seen:
            seen.add(text)
            top_insights.append(text)
        if len(top_insights) >= 15:
            break

    log.info(f"Correlation analysis complete: {len(top_insights)} strong rules across {len(grid.columns)} entities.")
    return driver_map, top_insights


def get_driver_history_series(conn, driver_eid, lag_mins, resample_rule, index_like):
    table = (
        "state_history"
        if (is_presence(driver_eid) or any(fnmatch.fnmatch(driver_eid, pat) for pat in BINARY_SENSOR_PATTERNS))
        else "context_history"
    )
    df_d = pd.read_sql_query(
        f"SELECT timestamp, state FROM {table} WHERE entity_id = ? ORDER BY timestamp ASC",
        conn, params=(driver_eid,),
    )
    if df_d.empty:
        return pd.Series(0, index=index_like)

    df_d["timestamp"] = pd.to_datetime(df_d["timestamp"], format="ISO8601", utc=True)
    df_d = df_d.drop_duplicates(subset=["timestamp"], keep="last")
    is_p = is_presence(driver_eid)
    s = df_d.set_index("timestamp")["state"].apply(lambda x: state_to_binary(x, is_person=is_p))
    s.loc[index_like.max()] = s.iloc[-1]
    s_res = s.resample(resample_rule).ffill()
    shift_steps = lag_mins // RESAMPLE_MINS
    if shift_steps > 0:
        s_res = s_res.shift(shift_steps)
    return s_res.reindex(index_like, method="ffill").fillna(0).astype(int)


def get_spec_series(conn, spec, resample_rule, index_like):
    """Feature series for an explicit entity (mode / companion-app / travel-time); NaN where unknown."""
    df_d = pd.read_sql_query(
        "SELECT timestamp, state FROM context_history WHERE entity_id = ? ORDER BY timestamp ASC",
        conn, params=(spec["entity_id"],),
    )
    if df_d.empty:
        return pd.Series(np.nan, index=index_like)
    df_d["timestamp"] = pd.to_datetime(df_d["timestamp"], format="ISO8601", utc=True)
    df_d = df_d.drop_duplicates(subset=["timestamp"], keep="last")
    s = df_d.set_index("timestamp")["state"].map(lambda x: spec_value(x, spec)).astype(float)
    s.loc[index_like.max()] = s.iloc[-1]
    s_res = s.resample(resample_rule).ffill()
    return s_res.reindex(index_like, method="ffill")


# --- Feature Engineering & Models ---
def compute_fourier_time_df(timestamps_utc):
    """Multi-harmonic cyclical time features + solar elevation."""
    local_ts = timestamps_utc.tz_convert(LOCAL_TZ)
    hours = np.asarray(local_ts.hour + local_ts.minute / 60.0, dtype=float)
    dows = np.asarray(local_ts.day_of_week, dtype=int)
    angle = 2.0 * np.pi * hours / 24.0
    elev, rising = solar_features(timestamps_utc)

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
        "sun_elev": elev,
        "sun_rising": rising,
    })


def build_features(df, gps=False, driver_cols=None):
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

    if "mins_since_on" not in df.columns:
        b = df["current_state_bin"]
        ts = df["timestamp"]
        start_on = ts.where((b == 1) & (b.shift() != 1))
        start_off = ts.where((b == 0) & (b.shift() != 0))
        df["mins_since_on"] = (ts - start_on.ffill()).dt.total_seconds() / 60.0
        df["mins_since_off"] = (ts - start_off.ffill()).dt.total_seconds() / 60.0

    schedule_cols = list(SCHEDULE_COLS)
    feature_cols = schedule_cols + [
        "current_state_bin", "mins_in_current_state", "mins_since_on", "mins_since_off",
        "active_ratio_30m", "active_ratio_2h", "transitions_1h",
    ]

    if gps:
        def calc_dist(a):
            return haversine_km(a.get("latitude"), a.get("longitude"), HOME_COORDS[0], HOME_COORDS[1])

        df["distance_from_home_km"] = df["attr_dict"].apply(calc_dist).astype(float)
        df["gps_accuracy"] = df["attr_dict"].apply(lambda a: float(a.get("gps_accuracy", 0) or 0))

        if "home_approach_speed_kmh" not in df.columns:
            lookback_steps = max(1, VELOCITY_LOOKBACK_MINS // RESAMPLE_MINS)
            lookback_hours = (lookback_steps * RESAMPLE_MINS) / 60.0
            prev_dist = df["distance_from_home_km"].shift(lookback_steps).fillna(df["distance_from_home_km"])
            speed = (prev_dist - df["distance_from_home_km"]) / lookback_hours
            speed = speed.mask(speed.abs() < 1.5, 0.0).clip(-250.0, 250.0)  # NaN stays NaN
            df["home_approach_speed_kmh"] = speed

        if "eta_mins" not in df.columns:
            df["eta_mins"] = [
                compute_eta_mins(d, s, bool(st == 1))
                for d, s, st in zip(df["distance_from_home_km"], df["home_approach_speed_kmh"], df["current_state_bin"])
            ]
        feature_cols.extend(["distance_from_home_km", "gps_accuracy", "home_approach_speed_kmh", "eta_mins"])

    if driver_cols:
        feature_cols.extend(driver_cols)
    return df, feature_cols, schedule_cols


def fit_calibrated_hgb(X, y, sample_weight, max_depth=6):
    """HistGradientBoostingClassifier with sigmoid probability calibration for horizon forecasts."""
    base_clf = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.06, max_depth=max_depth,
        min_samples_leaf=15, l2_regularization=1.0, random_state=42,
    )
    if int(y.value_counts().min()) >= 12 and len(y) >= 60:
        cal_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
        cal_clf.fit(X, y, sample_weight=sample_weight)
        return cal_clf
    base_clf.fit(X, y, sample_weight=sample_weight)
    return base_clf


def build_schedule_design_matrix(timestamps_utc, holiday_dates, live_workday_item):
    """Schedule feature matrix including is_workday / is_tomorrow_workday and solar features."""
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
    """Dynamic threshold from the schedule probability curve (works for low and high duty-cycle entities)."""
    if (float(np.max(prob_pos)) - float(np.min(prob_pos))) < 0.12:
        return 0.50
    midpoint = 0.5 * (float(np.percentile(prob_pos, 15)) + float(np.percentile(prob_pos, 85)))
    return float(np.clip(midpoint, 0.15, 0.85))


def schedule_prob_curve(schedule_model, X):
    probs = schedule_model.predict_proba(X)
    classes = list(schedule_model.classes_)
    return probs[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def build_24h_ascii_timeline(schedule_model, now_dt, threshold, holiday_dates, live_workday_item):
    """48-char (30 min/char) timeline, lightly smoothed so single noisy slots do not flicker."""
    slots = pd.date_range(start=now_dt + timedelta(minutes=15), periods=48, freq="30min")
    prob_pos = smooth_probs(
        schedule_prob_curve(schedule_model, build_schedule_design_matrix(slots, holiday_dates, live_workday_item)), 2
    )
    chars = []
    for i, p in enumerate(prob_pos):
        if i > 0 and i % 12 == 0:
            chars.append("|")
        chars.append("█" if p >= threshold else "·")
    return "".join(chars)


def estimate_next_transitions(schedule_model, now_dt, current_state_bin, holiday_dates, live_workday_item):
    """7-day schedule scan: smoothed curve (~60 min) + a flip must persist (~15 min) to count."""
    total_steps = (7 * 24 * 60) // RESAMPLE_MINS
    fut = pd.date_range(start=now_dt + timedelta(minutes=RESAMPLE_MINS), periods=total_steps, freq=f"{RESAMPLE_MINS}min")
    local_fut = fut.tz_convert(LOCAL_TZ)
    X_fut = build_schedule_design_matrix(fut, holiday_dates, live_workday_item)
    prob_pos = smooth_probs(schedule_prob_curve(schedule_model, X_fut), max(1, 60 // RESAMPLE_MINS))

    threshold = compute_schedule_threshold(prob_pos)
    preds = (prob_pos >= threshold).astype(int)

    confirm = max(1, 15 // RESAMPLE_MINS)
    next_on = next_off = None
    sim_state, run = int(current_state_bin), 0
    for idx, pred in enumerate(preds):
        if pred != sim_state:
            run += 1
            if run >= confirm:
                t_idx = idx - confirm + 1
                if pred == 1 and next_on is None:
                    next_on = local_fut[t_idx].isoformat()
                elif pred == 0 and next_off is None:
                    next_off = local_fut[t_idx].isoformat()
                sim_state, run = int(pred), 0
                if next_on and next_off:
                    break
        else:
            run = 0

    timeline_24h = build_24h_ascii_timeline(schedule_model, now_dt, threshold, holiday_dates, live_workday_item)
    return next_on, next_off, timeline_24h


def compute_anomaly(schedule_model, now_dt, curr_bin, mins_in_state, holiday_dates, workday_item, presence, is_household):
    """
    Compares the live state with what the pure schedule model expects across a +-30 min window.
    A state is only anomalous if the schedule disagrees for the *whole* window (so normal
    transition-time jitter is ignored) and the state has lasted at least ANOMALY_MIN_MINS.
    """
    if schedule_model is None:
        return None
    window = pd.date_range(start=now_dt - timedelta(minutes=30), periods=13, freq="5min")
    p = schedule_prob_curve(schedule_model, build_schedule_design_matrix(window, holiday_dates, workday_item))
    p_now = float(p[6])
    if curr_bin == 0:
        score = float(np.min(p))
        kind = "expected_home_but_away" if presence else "expected_active_but_inactive"
        if is_household:
            kind = "expected_everyone_home_but_not"
    else:
        score = float(1.0 - np.max(p))
        kind = "unexpectedly_home" if presence else "unexpected_activity"
        if is_household:
            kind = "everyone_home_unexpectedly"
    return {
        "anomaly": bool(score >= ANOMALY_THRESHOLD and mins_in_state >= ANOMALY_MIN_MINS),
        "anomaly_score": round(score, 3),
        "anomaly_kind": kind,
        "schedule_expectation": round(p_now, 3),
    }


def minutes_to_next_change(bins, step_min):
    """For each row: minutes until the binary state next differs; NaN when no later change is observed."""
    b = np.asarray(bins, dtype=int)
    n = len(b)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    changes = np.where(b[1:] != b[:-1])[0] + 1
    pos = np.searchsorted(changes, np.arange(n), side="right")
    ok = pos < len(changes)
    out[ok] = (changes[pos[ok]] - np.arange(n)[ok]) * step_min
    return out


def select_drivers(base_df, feature_cols, driver_cols, now_dt, half_life):
    """Keeps only drivers whose permutation raises the validation Brier score (chronological 75/25 split)."""
    h_steps = max(1, PREDICT_HORIZON_MINS // RESAMPLE_MINS)
    y = base_df["current_state_bin"].shift(-h_steps)
    mask = y.notna()
    d, yy = base_df.loc[mask], y.loc[mask].astype(int)
    n = len(d)
    if n < 200 or yy.nunique() < 2:
        return list(driver_cols)
    split = int(n * 0.75)
    ytr, yva = yy.iloc[:split], yy.iloc[split:]
    if ytr.nunique() < 2 or yva.nunique() < 2:
        return list(driver_cols)

    w = decay_weights(d["timestamp"], now_dt, half_life)
    clf = HistGradientBoostingClassifier(
        max_iter=60, learning_rate=0.08, max_depth=5, min_samples_leaf=15, l2_regularization=1.0, random_state=42
    )
    clf.fit(d[feature_cols].iloc[:split], ytr, sample_weight=w[:split])
    pos_idx = list(clf.classes_).index(1)
    Xva, yv = d[feature_cols].iloc[split:].copy(), yva.to_numpy()

    def brier(X):
        return float(np.mean((clf.predict_proba(X)[:, pos_idx] - yv) ** 2))

    base = brier(Xva)
    rng = np.random.default_rng(0)
    kept = []
    for col in driver_cols:
        deltas = []
        for _ in range(3):
            Xp = Xva.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            deltas.append(brier(Xp) - base)
        if float(np.mean(deltas)) > 0.0005:
            kept.append(col)
    return kept


def fit_entity_models(base_df, feature_cols, driver_cols, schedule_cols, all_horizons, now_dt,
                      half_life=None, zone=False, full=True):
    """
    Trains: calibrated HGB per horizon, the pure schedule RF, arrival/departure quantile models and
    (optionally) a multiclass zone model. full=False skips selection/quantile/zone (used by the backtest).
    """
    half_life = half_life or DECAY_HALF_LIFE_DAYS
    kept = select_drivers(base_df, feature_cols, driver_cols, now_dt, half_life) if (
        full and DRIVER_SELECTION and driver_cols) else list(driver_cols)
    final_cols = [c for c in feature_cols if c not in driver_cols or c in kept]

    horizon_models = {}
    for h_mins in all_horizons:
        h_steps = max(1, h_mins // RESAMPLE_MINS)
        h_target = base_df["current_state_bin"].shift(-h_steps)
        valid_mask = h_target.notna()
        h_train = base_df.loc[valid_mask]
        h_y = h_target.loc[valid_mask].astype(int)

        if len(h_y) == 0:
            horizon_models[h_mins] = ("prior", 0.5)
        elif h_y.nunique() < 2 or len(h_train) < 20:
            horizon_models[h_mins] = ("prior", (float(h_y.sum()) + 1.0) / (len(h_y) + 2.0))  # Laplace
        else:
            horizon_models[h_mins] = (
                "clf", fit_calibrated_hgb(h_train[final_cols], h_y, decay_weights(h_train["timestamp"], now_dt, half_life))
            )

    weights_all = decay_weights(base_df["timestamp"], now_dt, half_life)
    schedule_model = None
    sched_y = base_df["current_state_bin"].astype(int)
    if sched_y.nunique() >= 2 and len(base_df) >= 20:
        schedule_model = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=1
        )
        schedule_model.fit(base_df[schedule_cols], sched_y, sample_weight=weights_all)

    quantile_models, zone_model = [], None
    if full and ARRIVAL_PREDICTION:
        y_t = minutes_to_next_change(base_df["current_state_bin"], RESAMPLE_MINS)
        ok = ~np.isnan(y_t)
        n_changes = int((np.diff(base_df["current_state_bin"].to_numpy()) != 0).sum())
        if int(ok.sum()) >= 60 and n_changes >= 4:
            Xq = base_df.loc[ok, final_cols]
            yq = np.clip(y_t[ok], 0.0, 1440.0)
            for q in QUANTILES:
                reg = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_iter=80, learning_rate=0.08, max_depth=5,
                    min_samples_leaf=20, l2_regularization=1.0, random_state=42,
                )
                reg.fit(Xq, yq, sample_weight=weights_all[ok])
                quantile_models.append(reg)

    if full and zone:
        h_steps = max(1, PREDICT_HORIZON_MINS // RESAMPLE_MINS)
        z = base_df["state"].astype(str).shift(-h_steps)
        zmask = z.notna()
        zs = z[zmask]
        if len(zs) >= 60:
            counts = zs.value_counts()
            keep = set(counts[counts >= max(15, 0.02 * len(zs))].index) | {"home", "not_home"}
            zs = zs.where(zs.isin(keep), "other")
            if zs.nunique() >= 3:
                zone_model = HistGradientBoostingClassifier(
                    max_iter=100, learning_rate=0.06, max_depth=5, min_samples_leaf=15,
                    l2_regularization=1.0, random_state=42,
                )
                zone_model.fit(base_df.loc[zmask, final_cols], zs, sample_weight=weights_all[zmask.to_numpy()])

    return {
        "trained_at": now_dt, "feature_cols": final_cols, "kept_drivers": kept, "half_life": half_life,
        "horizon_models": horizon_models, "schedule_model": schedule_model,
        "quantile_models": quantile_models, "zone_model": zone_model,
    }


def adaptive_half_life(ent_acc):
    """Drift handling: forget faster while accuracy is poor or worse than persistence."""
    factor = 1.0
    n, acc, base = ent_acc.get("count_7d", 0), ent_acc.get("accuracy_7d"), ent_acc.get("baseline_accuracy_7d")
    if n >= 20 and acc is not None:
        if acc < MIN_ACCEPTABLE_ACCURACY:
            factor *= 0.5
        if base is not None and acc < base - 0.03:
            factor *= 0.7
    return round(max(2.0, DECAY_HALF_LIFE_DAYS * factor), 2)


def entity_extra_specs(eid):
    return list(PERSON_FEATURES.get(eid, [])) + list(MODE_SPECS)


def prepare_entity_frame(conn, eid, state_now, attrs_now, now_dt, drivers, all_entity_map,
                         holiday_dates, workday_item, extra_specs):
    """
    Builds the resampled feature grid for one entity. Returns (frame_dict, None) or (None, early_meta).
    Shared by the live predictor and backtest.py.
    """
    presence = is_presence(eid)
    positive_state = "home" if presence else "on"

    df = pd.read_sql_query(
        "SELECT timestamp, state, attributes FROM state_history WHERE entity_id = ? ORDER BY timestamp ASC",
        conn, params=(eid,),
    )
    if df.empty:
        return None, {"reason": "no_history_recorded"}

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    span_hours = (now_dt - df["timestamp"].min()).total_seconds() / 3600.0
    if span_hours < MIN_HISTORY_HOURS:
        return None, {
            "reason": "insufficient_time_span",
            "recorded_hours": round(span_hours, 1),
            "required_hours": MIN_HISTORY_HOURS,
        }

    now_anchor = pd.DataFrame([{
        "timestamp": now_dt, "state": state_now,
        "attributes": json.dumps(slim_attributes(eid, attrs_now)),
    }])
    df = pd.concat([df, now_anchor], ignore_index=True)

    # Group on the binary state so e.g. not_home -> work does not reset the "time in state" counter
    raw_bin = (df["state"] == positive_state).astype(int)
    df["state_change_group"] = (raw_bin != raw_bin.shift()).cumsum()
    df["group_start_ts"] = df.groupby("state_change_group")["timestamp"].transform("first")

    resample_rule = f"{RESAMPLE_MINS}min"
    df_indexed = df.set_index("timestamp").resample(resample_rule).ffill().dropna(subset=["state"])

    df_indexed["is_workday"] = get_workday_series(
        conn, resample_rule, df_indexed.index, holiday_dates, workday_item).values
    next_day_index = df_indexed.index + pd.Timedelta(days=1)
    df_indexed["is_tomorrow_workday"] = get_workday_series(
        conn, resample_rule, next_day_index, holiday_dates, workday_item).values

    driver_cols, live_driver_vals = [], {}
    for idx_d, drv in enumerate(drivers):
        col, d_eid, d_lag = f"driver_{idx_d}", drv["driver_eid"], drv["lag_mins"]
        s_drv = get_driver_history_series(conn, d_eid, d_lag, resample_rule, df_indexed.index)
        df_indexed[col] = s_drv.values
        driver_cols.append(col)
        if d_lag == 0 and d_eid in all_entity_map:
            live_driver_vals[col] = state_to_binary(all_entity_map[d_eid].get("state"), is_person=is_presence(d_eid))
        else:
            live_driver_vals[col] = int(s_drv.iloc[-1]) if len(s_drv) > 0 else 0

    extra_cols, live_extra_vals = [], {}
    for i, spec in enumerate(extra_specs):
        col = f"x_{i}"
        s_x = get_spec_series(conn, spec, resample_rule, df_indexed.index)
        df_indexed[col] = s_x.values
        extra_cols.append(col)
        live_item = all_entity_map.get(spec["entity_id"])
        if live_item is not None:
            live_extra_vals[col] = spec_value(live_item.get("state"), spec)
        else:
            live_extra_vals[col] = float(s_x.iloc[-1]) if len(s_x) else float("nan")

    df = df_indexed.reset_index()
    df["attr_dict"] = df["attributes"].apply(safe_json)
    df["current_state_bin"] = (df["state"] == positive_state).astype(int)
    df["mins_in_current_state"] = (df["timestamp"] - df["group_start_ts"]).dt.total_seconds() / 60.0

    return {
        "df": df, "span_hours": span_hours,
        "driver_cols": driver_cols, "extra_cols": extra_cols,
        "live_driver_vals": live_driver_vals, "live_extra_vals": live_extra_vals,
    }, None


def train_and_predict(conn, entity_item, drivers, all_entity_map, holiday_dates, workday_item, acc_stats):
    eid = entity_item["entity_id"]
    presence = is_presence(eid)
    gps = has_gps(eid)
    is_household = eid == HOUSEHOLD_EID
    positive_state = "home" if presence else "on"
    negative_state = "not_home" if presence else "off"
    now_dt = datetime.now(timezone.utc)
    curr_attrs = entity_item.get("attributes", {}) or {}

    frame, early = prepare_entity_frame(
        conn, eid, entity_item.get("state"), curr_attrs, now_dt, drivers, all_entity_map,
        holiday_dates, workday_item, entity_extra_specs(eid),
    )
    if early:
        return "unknown", 0.0, early

    df = frame["df"]
    span_hours = frame["span_hours"]
    driver_cols, extra_cols = frame["driver_cols"], frame["extra_cols"]

    local_now = now_dt.astimezone(LOCAL_TZ)
    today_date = local_now.date()
    tomorrow_date = today_date + timedelta(days=1)
    is_workday_today = is_date_workday(today_date, holiday_dates, workday_item, is_today=True)
    is_workday_tomorrow = is_date_workday(tomorrow_date, holiday_dates, workday_item, is_today=False)

    curr_bin = 1 if entity_item.get("state") == positive_state else 0
    curr_label = positive_state if curr_bin == 1 else negative_state

    common_meta = {
        "timeline_start": local_now.strftime("%H:%M"),
        "timeline_legend": f"█={positive_state} ·={negative_state} |=6h (30m/char)",
        "is_workday": bool(is_workday_today),
        "is_tomorrow_workday": bool(is_workday_tomorrow),
        "holiday_today": holiday_dates.get(today_date),
        "holiday_tomorrow": holiday_dates.get(tomorrow_date),
        "correlated_triggers": [d["summary"] for d in drivers],
        "recorded_hours": round(span_hours, 1),
    }

    # --- Constant historical state ---
    if df["current_state_bin"].nunique() < 2:
        const_val = int(df["current_state_bin"].iloc[0])
        if const_val != curr_bin:
            return "unknown", 0.0, {"reason": "state_never_seen_in_history", **common_meta}
        const_state = positive_state if const_val == 1 else negative_state
        glyph = "█" if const_val == 1 else "·"
        certainty = round(min(0.95, 0.55 + 0.40 * min(1.0, span_hours / 168.0)), 3)
        final = const_state if certainty >= CERTAINTY_THRESHOLD else "unknown"
        return final, certainty, {
            "reason": "constant_historical_state", "certainty": certainty,
            "expected_next_on": None, "expected_next_off": None,
            "timeline_24h": "|".join([glyph * 12] * 4), **common_meta,
        }

    all_driver_like = driver_cols + extra_cols
    base_df, feature_cols, schedule_cols = build_features(df, gps=gps, driver_cols=all_driver_like)

    # --- Live observation row ---
    last_changed_str = entity_item.get("last_changed")
    try:
        last_changed_dt = parse_ts(last_changed_str) if last_changed_str else now_dt
    except (ValueError, TypeError):
        last_changed_dt = now_dt
    mins_in_state = max(0.0, (now_dt - last_changed_dt).total_seconds() / 60.0)

    last_row_ts = base_df["timestamp"].iloc[-1]
    gap_mins = max(0.0, (now_dt - last_row_ts).total_seconds() / 60.0)
    since_on_prev = float(base_df["mins_since_on"].iloc[-1]) + gap_mins
    since_off_prev = float(base_df["mins_since_off"].iloc[-1]) + gap_mins

    live_row = {
        "timestamp": now_dt, "attr_dict": curr_attrs,
        "current_state_bin": curr_bin, "mins_in_current_state": mins_in_state,
        "mins_since_on": mins_in_state if curr_bin == 1 else since_on_prev,
        "mins_since_off": mins_in_state if curr_bin == 0 else since_off_prev,
        "is_workday": is_workday_today, "is_tomorrow_workday": is_workday_tomorrow,
        "active_ratio_30m": float(base_df["active_ratio_30m"].iloc[-1]),
        "active_ratio_2h": float(base_df["active_ratio_2h"].iloc[-1]),
        "transitions_1h": float(base_df["transitions_1h"].iloc[-1]),
        **frame["live_driver_vals"], **frame["live_extra_vals"],
    }

    if gps:
        curr_dist = haversine_km(curr_attrs.get("latitude"), curr_attrs.get("longitude"), HOME_COORDS[0], HOME_COORDS[1])
        lookback_steps = max(1, VELOCITY_LOOKBACK_MINS // RESAMPLE_MINS)
        if len(df) > lookback_steps:
            past_attrs = df["attr_dict"].iloc[-(lookback_steps + 1)]
            past_dist = haversine_km(past_attrs.get("latitude"), past_attrs.get("longitude"), HOME_COORDS[0], HOME_COORDS[1])
            approach_speed = compute_approach_speed(past_dist, curr_dist, (lookback_steps * RESAMPLE_MINS) / 60.0)
        else:
            approach_speed = float("nan")
        live_row["home_approach_speed_kmh"] = approach_speed
        live_row["eta_mins"] = compute_eta_mins(curr_dist, approach_speed, bool(curr_bin == 1))

    now_df, _, _ = build_features(pd.DataFrame([live_row]), gps=gps, driver_cols=all_driver_like)

    # --- Train (hourly / on change) or reuse cached models ---
    ent_acc = acc_stats.get(eid, {})
    half_life = adaptive_half_life(ent_acc)
    all_horizons = sorted(set([PREDICT_HORIZON_MINS] + EXTRA_HORIZONS_MINS))
    sig = (tuple((d["driver_eid"], d["lag_mins"]) for d in drivers), tuple(all_horizons),
           tuple(feature_cols), half_life)
    cache = MODEL_CACHE.get(eid)
    if (cache is None or cache["sig"] != sig
            or (now_dt - cache["trained_at"]).total_seconds() >= RETRAIN_INTERVAL_MINS * 60):
        cache = fit_entity_models(
            base_df, feature_cols, driver_cols, schedule_cols, all_horizons, now_dt,
            half_life=half_life, zone=(gps and ZONE_PREDICTION),
        )
        cache["sig"] = sig
        MODEL_CACHE[eid] = cache
        log.info(f"Retrained models for {eid} (half-life {half_life}d, drivers kept: {len(cache['kept_drivers'])}/{len(driver_cols)})")

    X_now = now_df[cache["feature_cols"]]

    horizon_probs = {}
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
    max_prob = max(prob_pos, 1.0 - prob_pos)
    predicted_state = positive_state if prob_pos >= 0.5 else negative_state

    # --- Schedule outputs + anomaly ---
    next_on = next_off = timeline_24h = None
    anomaly = None
    if cache["schedule_model"] is not None:
        next_on, next_off, timeline_24h = estimate_next_transitions(
            cache["schedule_model"], now_dt, curr_bin, holiday_dates, workday_item)
        anomaly = compute_anomaly(
            cache["schedule_model"], now_dt, curr_bin, mins_in_state, holiday_dates, workday_item,
            presence, is_household)

    # --- Arrival / departure time distribution ---
    next_change = {}
    if cache["quantile_models"]:
        vals = sorted(float(np.clip(m_.predict(X_now)[0], 0.0, 1440.0)) for m_ in cache["quantile_models"])
        if is_household:
            kind = "someone_leaves" if curr_bin == 1 else "everyone_home"
        elif presence:
            kind = "departure" if curr_bin == 1 else "arrival"
        else:
            kind = "turns_off" if curr_bin == 1 else "turns_on"
        t_lo, t_mid, t_hi = (local_now + timedelta(minutes=v) for v in vals)
        next_change = {
            "next_change_kind": kind,
            "next_change_p10_mins": int(round(vals[0])),
            "next_change_p50_mins": int(round(vals[1])),
            "next_change_p90_mins": int(round(vals[2])),
            "next_change_at_p50": t_mid.isoformat(),
            "next_change_window": f"{t_lo.strftime('%H:%M')}-{t_hi.strftime('%H:%M')}",
        }

    # --- Zone probabilities (persons) ---
    zone_meta = {}
    if cache["zone_model"] is not None:
        zp = cache["zone_model"].predict_proba(X_now)[0]
        probs = {str(c): round(float(p), 3) for c, p in zip(cache["zone_model"].classes_, zp)}
        probs = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
        zone_meta = {"zone_probs": probs, "most_likely_zone": next(iter(probs))}

    # --- Accuracy-based adaptive threshold ---
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
        "skill_vs_baseline_7d": round(acc_7d - base_acc_7d, 3) if (acc_7d is not None and base_acc_7d is not None) else None,
        "abstain_rate_7d": ent_acc.get("abstain_rate_7d"),
        "evaluated_predictions_7d": eval_count_7d,
        "scores": ent_acc.get("scores"),
        "mins_in_current_state": round(mins_in_state, 1),
        "decay_half_life_days": half_life,
        "drivers_used": [drivers[int(c.split("_")[1])]["summary"] for c in cache["kept_drivers"]] if drivers else [],
        "model_trained_at": cache["trained_at"].astimezone(LOCAL_TZ).isoformat(),
        "last_updated": local_now.isoformat(),
        **common_meta, **next_change, **zone_meta, **(anomaly or {}),
    }
    for h_mins, h_prob in horizon_probs.items():
        meta[f"prob_{h_mins}m"] = h_prob
    if gps:
        meta["distance_from_home_km"] = safe_round(now_df["distance_from_home_km"].iloc[0], 2)
        meta["home_approach_speed_kmh"] = safe_round(now_df["home_approach_speed_kmh"].iloc[0], 1)
        meta["eta_mins"] = safe_round(now_df["eta_mins"].iloc[0], 0)

    final_state = "unknown" if max_prob < effective_threshold else predicted_state
    record_prediction(conn, eid, now_dt, PREDICT_HORIZON_MINS, predicted_state, max_prob,
                      curr_label, final_state == "unknown")

    hb = HistBaseline(base_df, now_dt)
    record_horizon_log(conn, eid, now_dt, [
        (h, horizon_probs[h], float(curr_bin), hb.prob(now_dt + timedelta(minutes=h))) for h in all_horizons
    ])

    return final_state, max_prob, meta


# --- MQTT Home Assistant Auto-Discovery ---
def setup_mqtt():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.will_set(STATUS_TOPIC, "offline", retain=True)  # entities go unavailable if we die
    client.on_connect = lambda c, userdata, flags, reason_code, properties=None: c.publish(
        STATUS_TOPIC, "online", retain=True)
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()
    return client


def publish_entity(client, component, unique_id, name, state, attributes=None, extra=None, drop_order=()):
    base = f"{MQTT_PREFIX}/{component}/{unique_id}"
    cfg = {
        "name": name,
        "default_entity_id": f"{component}.{unique_id}",
        "unique_id": unique_id,
        "state_topic": f"{base}/state",
        "availability_topic": STATUS_TOPIC,
        "device": DEVICE_INFO,
    }
    if attributes is not None:
        cfg["json_attributes_topic"] = f"{base}/attributes"
    if extra:
        cfg.update(extra)
    client.publish(f"{base}/config", json.dumps(cfg), retain=True)
    client.publish(f"{base}/state", str(state), retain=True)
    if attributes is not None:
        client.publish(f"{base}/attributes", to_attr_json(attributes, drop_order), retain=True)


def publish_ha_discovery(client, entity_item, state, attributes):
    """Publishes the predicted-state sensor, 'likely home/active' binary sensors and the anomaly sensor."""
    eid = entity_item["entity_id"]
    slug = eid.split(".", 1)[1]
    friendly = entity_item.get("attributes", {}).get("friendly_name", slug.replace("_", " ").title())
    is_household = eid == HOUSEHOLD_EID

    icon = "mdi:home-account" if is_household else ("mdi:account-clock" if has_gps(eid) else "mdi:motion-sensor")
    publish_entity(
        client, "sensor", f"{slug}_predicted", f"{friendly} Predicted", state, attributes,
        extra={"icon": icon},
        drop_order=("timeline_legend", "scores", "correlated_triggers", "drivers_used", "timeline_24h"),
    )

    eff = attributes.get("effective_threshold", CERTAINTY_THRESHOLD)
    for h in PUBLISH_BINARY_HORIZONS:
        p = attributes.get(f"prob_{h}m")
        if p is None:
            payload = "None"
        elif p >= eff:
            payload = "ON"
        elif (1.0 - p) >= eff:
            payload = "OFF"
        else:
            payload = "None"  # uncertain -> unknown
        if is_household:
            label, key = "Everyone Home", "likely_all_home"
        elif is_presence(eid):
            label, key = "Likely Home", "likely_home"
        else:
            label, key = "Likely Active", "likely_active"
        extra = {"payload_on": "ON", "payload_off": "OFF"}
        if is_presence(eid):
            extra["device_class"] = "presence"
        publish_entity(
            client, "binary_sensor", f"{slug}_{key}_{h}m", f"{friendly} {label} in {h}m", payload,
            {"probability": p, "threshold": eff, "horizon_mins": h}, extra=extra,
        )

    if "anomaly" in attributes:
        keys = ("anomaly_score", "anomaly_kind", "schedule_expectation", "mins_in_current_state")
        publish_entity(
            client, "binary_sensor", f"{slug}_ml_anomaly", f"{friendly} Unusual Behaviour",
            "ON" if attributes["anomaly"] else "OFF",
            {k: attributes.get(k) for k in keys},
            extra={"payload_on": "ON", "payload_off": "OFF", "device_class": "problem"},
        )


def publish_household_counts(client, results, member_ids):
    """Expected number of people home per horizon = sum of the members' probabilities."""
    horizons = sorted(set([PREDICT_HORIZON_MINS] + EXTRA_HORIZONS_MINS))
    attrs, included = {}, 0
    members = [r for e, r in results.items() if e in member_ids and r[2].get(f"prob_{PREDICT_HORIZON_MINS}m") is not None]
    included = len(members)
    if included == 0:
        return
    for h in horizons:
        vals = [r[2].get(f"prob_{h}m") for r in members if r[2].get(f"prob_{h}m") is not None]
        attrs[f"expected_people_home_{h}m"] = round(float(sum(vals)), 2)
    attrs["members_total"] = len(member_ids)
    attrs["members_with_predictions"] = included
    publish_entity(
        client, "sensor", "household_people_home_predicted", "Household People Home Predicted",
        attrs[f"expected_people_home_{PREDICT_HORIZON_MINS}m"], attrs,
        extra={"icon": "mdi:account-group", "state_class": "measurement"},
    )


def publish_insights_sensor(client, insights, context_entities, acc_stats, holiday_dates):
    accs = [v["accuracy_7d"] for v in acc_stats.values() if v.get("accuracy_7d") is not None]
    bases = [v["baseline_accuracy_7d"] for v in acc_stats.values() if v.get("baseline_accuracy_7d") is not None]
    main_key = f"{PREDICT_HORIZON_MINS}m"
    bss_p = [v["scores"][main_key]["bss_vs_persist"] for v in acc_stats.values()
             if v.get("scores", {}).get(main_key, {}).get("bss_vs_persist") is not None]
    bss_h = [v["scores"][main_key]["bss_vs_hist"] for v in acc_stats.values()
             if v.get("scores", {}).get(main_key, {}).get("bss_vs_hist") is not None]

    compact = {}
    for eid, v in acc_stats.items():
        sc = (v.get("scores") or {}).get(main_key, {})
        compact[eid] = {
            "n": v.get("count_7d"), "acc": v.get("accuracy_7d"), "persist_acc": v.get("baseline_accuracy_7d"),
            "brier": sc.get("brier"), "bss_persist": sc.get("bss_vs_persist"), "bss_hist": sc.get("bss_vs_hist"),
        }

    today_date = datetime.now(timezone.utc).astimezone(LOCAL_TZ).date()
    attrs = {
        "insights": insights,
        "overall_accuracy_7d": round(float(np.mean(accs)), 3) if accs else None,
        "overall_baseline_accuracy_7d": round(float(np.mean(bases)), 3) if bases else None,
        f"overall_bss_vs_persistence_{main_key}": round(float(np.mean(bss_p)), 3) if bss_p else None,
        f"overall_bss_vs_histogram_{main_key}": round(float(np.mean(bss_h)), 3) if bss_h else None,
        "entity_scores_7d": compact,
        "context_entities_monitored": len(context_entities),
        "holiday_calendar": HOLIDAY_CALENDAR_ENTITY_ID or None,
        "upcoming_holidays": {d.isoformat(): n for d, n in sorted(holiday_dates.items()) if d >= today_date},
        "min_correlation_threshold": CORRELATION_MIN_STRENGTH,
        "last_analyzed": datetime.now(timezone.utc).astimezone(LOCAL_TZ).isoformat(),
    }
    publish_entity(
        client, "sensor", "ha_ml_behavioral_insights", "HA ML Behavioral Insights", len(insights), attrs,
        extra={"icon": "mdi:brain"}, drop_order=("entity_scores_7d", "upcoming_holidays", "insights"),
    )


def publish_diagnostics(client, conn, info):
    """Diagnostic sensor (state = time of last successful loop) + heartbeat file for an external healthcheck."""
    now = datetime.now(timezone.utc)
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("state_history", "context_history", "prediction_log", "horizon_log")}
    oldest = conn.execute("SELECT MIN(timestamp) FROM state_history").fetchone()[0]
    span_days = round((now - parse_ts(oldest)).total_seconds() / 86400.0, 1) if oldest else 0.0
    errors_24h = sum(1 for ts, _ in RUNTIME["errors"] if (now - ts).total_seconds() < 86400)
    last_err = RUNTIME["errors"][-1][1] if RUNTIME["errors"] else None
    attrs = {
        "db_size_mb": round(os.path.getsize(DB_PATH) / 1e6, 2) if os.path.exists(DB_PATH) else None,
        "rows": counts,
        "history_span_days": span_days,
        "history_hint": (
            f"History ({span_days} d) is shorter than TRAIN_WINDOW_DAYS ({TRAIN_WINDOW_DAYS}); "
            "if it stops growing, raise recorder purge_keep_days in Home Assistant."
            if span_days < TRAIN_WINDOW_DAYS - 1 and span_days >= INITIAL_HISTORY_DAYS * 0.5 else None
        ),
        "loop_seconds": RUNTIME["loop_secs"],
        "loops_completed": RUNTIME["loops"],
        "models_cached": len(MODEL_CACHE),
        "last_training": max((c["trained_at"] for c in MODEL_CACHE.values()), default=None),
        "timezone": str(LOCAL_TZ),
        "home_coordinates_known": HOME_COORDS[0] is not None,
        "errors_24h": errors_24h,
        "last_error": last_err,
        **info,
    }
    if attrs["last_training"] is not None:
        attrs["last_training"] = attrs["last_training"].astimezone(LOCAL_TZ).isoformat()
    publish_entity(
        client, "sensor", "ha_ml_predictor_status", "HA ML Predictor Status",
        now.astimezone(LOCAL_TZ).isoformat(), attrs,
        extra={"device_class": "timestamp", "entity_category": "diagnostic", "icon": "mdi:heart-pulse"},
    )
    try:
        with open(HEARTBEAT_FILE, "w") as fh:
            fh.write(now.isoformat())
    except OSError:
        pass


# --- Main Loop ---
def main():
    if not HA_TOKEN:
        raise ValueError("HA_TOKEN environment variable is required.")

    def _term(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _term)

    init_timezone()
    conn = init_db()
    mqtt_client = setup_mqtt()
    fetch_home_coords()
    log.info("HA ML Predictor started.")

    last_corr_run = last_holiday_fetch = None
    driver_map, top_insights, holiday_dates = {}, [], {}

    try:
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

                member_ids = household_member_ids(persons)
                household_item = None
                if HOUSEHOLD_ENABLED:
                    try:
                        household_item = sync_household_history(conn, member_ids)
                    except Exception as e:
                        log.warning(f"Household history failed: {e}")
                        note_error(f"household: {e}")
                if household_item:
                    all_entity_map[HOUSEHOLD_EID] = household_item
                predict_items = targets + ([household_item] if household_item else [])

                acc_stats = evaluate_past_predictions(conn)
                now_utc = datetime.now(timezone.utc)

                if HOLIDAY_CALENDAR_ENTITY_ID and (
                    last_holiday_fetch is None
                    or (now_utc - last_holiday_fetch).total_seconds() >= HOLIDAY_REFRESH_HOURS * 3600
                ):
                    new_holidays = fetch_holiday_calendar_dates(days_back=max(30, TRAIN_WINDOW_DAYS + 2), days_forward=14)
                    if new_holidays is not None:
                        holiday_dates, last_holiday_fetch = new_holidays, now_utc
                        log.info(f"Loaded {len(holiday_dates)} holiday dates from {HOLIDAY_CALENDAR_ENTITY_ID}.")

                if last_corr_run is None or (now_utc - last_corr_run).total_seconds() >= CORRELATION_INTERVAL_HOURS * 3600:
                    try:
                        prune_old_records(conn)
                        driver_map, top_insights = run_correlation_analysis(conn, all_entity_map)
                    except Exception as e:
                        log.error(f"Correlation analysis failed, keeping previous results: {e}", exc_info=True)
                        note_error(f"correlation: {e}")
                    last_corr_run = now_utc

                publish_insights_sensor(mqtt_client, top_insights, context_entities, acc_stats, holiday_dates)

                results = {}
                for item in predict_items:
                    eid = item["entity_id"]
                    try:
                        pred_state, cert, meta = train_and_predict(
                            conn, item, driver_map.get(eid, []), all_entity_map, holiday_dates, workday_item, acc_stats)
                        results[eid] = (pred_state, cert, meta)
                        publish_ha_discovery(mqtt_client, item, pred_state, meta)
                        nc = meta.get("next_change_window")
                        log.info(
                            f"{eid} -> {pred_state} (cert: {cert:.2f}, acc_7d: {meta.get('accuracy_7d')}, "
                            f"base_7d: {meta.get('baseline_accuracy_7d')}, next: {nc}, "
                            f"anomaly: {meta.get('anomaly')}, 24h: {meta.get('timeline_24h')})"
                        )
                    except Exception as e:
                        log.error(f"Prediction failed for {eid}: {e}", exc_info=True)
                        note_error(f"{eid}: {e}")

                if household_item:
                    publish_household_counts(mqtt_client, results, set(member_ids))

                RUNTIME["loops"] += 1
                RUNTIME["loop_secs"] = round(time.time() - loop_started, 1)
                publish_diagnostics(mqtt_client, conn, {
                    "targets": len(predict_items), "context_entities": len(context_entities),
                    "household_members": len(member_ids) if household_item else 0,
                    "holiday_dates_loaded": len(holiday_dates),
                })

            except Exception as e:
                log.error(f"Error in main loop: {e}", exc_info=True)
                note_error(f"main loop: {e}")

            time.sleep(max(5.0, LOOP_SECS - (time.time() - loop_started)))
    finally:
        try:
            mqtt_client.publish(STATUS_TOPIC, "offline", retain=True).wait_for_publish(timeout=3)
            mqtt_client.loop_stop()
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
