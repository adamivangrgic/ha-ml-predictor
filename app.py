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
from sklearn.cluster import KMeans

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

# Optional pattern to identify bedroom presence sensors for sleep/wake-up estimation
BEDROOM_SENSOR_PATTERNS = [
    p.strip()
    for p in os.getenv("BEDROOM_SENSOR_PATTERNS", "*bedroom*,*soba*,*spavaca*,*bed*").split(",")
    if p.strip()
]

# Tier 2: Human-triggered context entities only (no automated lights/climate outputs)
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
        "*_predicted,*_expected_wake_up,*_battery*,*_rssi*,*_linkquality*,*_update*,*_uptime*,*_power*,*_energy*,climate.*,light.*,cover.*",
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
ANALYSIS_INTERVAL_HOURS = float(os.getenv("ANALYSIS_INTERVAL_HOURS", "24.0"))
DB_PATH = os.getenv("DB_PATH", "/data/history.db")

HEADERS = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
HOME_COORDS = (None, None)
LOCAL_TZ = datetime.now().astimezone().tzinfo

ACTIVE_STATES = {"on", "home", "open", "unlocked", "playing", "heating", "cooling", "true"}
INACTIVE_STATES = {"off", "not_home", "closed", "locked", "idle", "paused", "standby", "false"}


# --- Geospatial & State Helpers ---
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state_history (
            entity_id TEXT,
            timestamp TEXT,
            state TEXT,
            attributes TEXT,
            PRIMARY KEY (entity_id, timestamp)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS context_history (
            entity_id TEXT,
            timestamp TEXT,
            state TEXT,
            PRIMARY KEY (entity_id, timestamp)
        )
    """)
    # Feature #3: Prediction audit log for self-scoring accuracy
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT,
            made_at TEXT,
            target_at TEXT,
            predicted_state TEXT,
            confidence REAL,
            actual_state TEXT,
            is_correct INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_state_hist_eid ON state_history(entity_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ctx_hist_eid ON context_history(entity_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_eid ON prediction_audit(entity_id, target_at)")
    conn.commit()
    return conn


def get_last_timestamp(conn, table, entity_id):
    cur = conn.execute(f"SELECT MAX(timestamp) FROM {table} WHERE entity_id = ?", (entity_id,))
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def prune_old_records(conn):
    ctx_cutoff = (datetime.now(timezone.utc) - timedelta(days=CONTEXT_RETENTION_DAYS)).isoformat()
    cur1 = conn.execute("DELETE FROM context_history WHERE timestamp < ?", (ctx_cutoff,))
    audit_cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    cur2 = conn.execute("DELETE FROM prediction_audit WHERE made_at < ?", (audit_cutoff,))
    conn.commit()
    if cur1.rowcount > 0 or cur2.rowcount > 0:
        log.info(f"Pruned {cur1.rowcount} context rows and {cur2.rowcount} old audit rows.")


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


# --- Feature #3: Self-Scoring Accuracy & Auto-Calibration ---
def log_prediction_for_audit(conn, entity_id, now_dt, predicted_state, confidence):
    """Records concrete predictions (not 'unknown') so they can be scored once target_at passes."""
    if predicted_state in ("unknown", None):
        return
    target_dt = now_dt + timedelta(minutes=PREDICT_HORIZON_MINS)
    # Avoid duplicate logs within the same 5-minute window
    recent_cutoff = (now_dt - timedelta(minutes=4)).isoformat()
    cur = conn.execute(
        "SELECT 1 FROM prediction_audit WHERE entity_id = ? AND made_at >= ? LIMIT 1",
        (entity_id, recent_cutoff),
    )
    if cur.fetchone():
        return

    conn.execute(
        "INSERT INTO prediction_audit (entity_id, made_at, target_at, predicted_state, confidence) VALUES (?, ?, ?, ?, ?)",
        (entity_id, now_dt.isoformat(), target_dt.isoformat(), predicted_state, float(confidence)),
    )
    conn.commit()


def score_due_predictions(conn, live_entity_map):
    """Scores past predictions whose target_at timestamp has now arrived."""
    now_dt = datetime.now(timezone.utc)
    # Only score predictions whose target time is in the past (and not older than 2 hours)
    min_target = (now_dt - timedelta(hours=2)).isoformat()
    max_target = now_dt.isoformat()

    rows = conn.execute(
        "SELECT id, entity_id, target_at, predicted_state FROM prediction_audit "
        "WHERE actual_state IS NULL AND target_at BETWEEN ? AND ?",
        (min_target, max_target),
    ).fetchall()

    for row_id, eid, target_at, pred_state in rows:
        # Look up the state in state_history at or immediately before target_at
        hist_row = conn.execute(
            "SELECT state FROM state_history WHERE entity_id = ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
            (eid, target_at),
        ).fetchone()

        actual = hist_row[0] if hist_row else live_entity_map.get(eid, {}).get("state")
        if actual in ("unavailable", "unknown", None):
            continue

        # Normalize person states ('home' vs everything else as 'not_home')
        if eid.startswith("person."):
            norm_actual = "home" if actual == "home" else "not_home"
        else:
            norm_actual = actual

        is_correct = 1 if norm_actual == pred_state else 0
        conn.execute(
            "UPDATE prediction_audit SET actual_state = ?, is_correct = ? WHERE id = ?",
            (norm_actual, is_correct, row_id),
        )
    conn.commit()


def get_entity_accuracy_stats(conn, entity_id):
    """Returns (accuracy_7d_pct, evaluated_count, calibrated_threshold)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    row = conn.execute(
        "SELECT COUNT(*), AVG(is_correct) FROM prediction_audit "
        "WHERE entity_id = ? AND is_correct IS NOT NULL AND made_at >= ?",
        (entity_id, cutoff),
    ).fetchone()

    count = int(row[0] or 0)
    avg_acc = float(row[1]) if row[1] is not None else None

    # Auto-calibrate certainty threshold if historical accuracy is low
    effective_thresh = CERTAINTY_THRESHOLD
    if count >= 20 and avg_acc is not None and avg_acc < 0.75:
        # Boost required certainty proportionally up to 0.90
        penalty = (0.75 - avg_acc) * 0.6
        effective_thresh = min(0.90, round(CERTAINTY_THRESHOLD + penalty, 2))

    return (round(avg_acc * 100.0, 1) if avg_acc is not None else None), count, effective_thresh


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


# --- Aligned Multi-Entity Grid Builder ---
def build_aligned_grid(conn, days_back=CONTEXT_RETENTION_DAYS):
    """Builds a 5-minute binary state grid across all Tier 1 and Tier 2 entities."""
    now_dt = datetime.now(timezone.utc)
    cutoff_str = (now_dt - timedelta(days=days_back)).isoformat()

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
        return pd.DataFrame()

    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], format="ISO8601", utc=True)
    resample_rule = f"{RESAMPLE_MINS}min"

    series_dict = {}
    for eid, grp in df_all.groupby("entity_id"):
        grp = grp.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        is_p = eid.startswith("person.")
        s = grp.set_index("timestamp")["state"].apply(lambda x: state_to_binary(x, is_person=is_p))
        s.loc[now_dt] = s.iloc[-1]
        s_res = s.resample(resample_rule).ffill().dropna()
        if len(s_res) >= 48:
            series_dict[eid] = s_res

    if not series_dict:
        return pd.DataFrame()

    return pd.DataFrame(series_dict).ffill().bfill().dropna()


# --- Nightly Cross-Sensor Lagged Correlation Engine ---
def run_correlation_analysis(grid, all_entity_map):
    if grid.empty or len(grid.columns) < 2:
        return {}, []

    active_cols = [c for c in grid.columns if grid[c].nunique() >= 2]
    if len(active_cols) < 2:
        return {}, []

    target_eids = [
        eid for eid in active_cols if eid.startswith("person.") or any(
            fnmatch.fnmatch(eid, pat) for pat in BINARY_SENSOR_PATTERNS
        )
    ]

    # Strictly use positive lead lags (5m, 10m, 15m) so simultaneous automation echoes never trigger false links
    lags_mins = [5, 10, 15]
    driver_map = {}
    all_findings = []

    def friendly(eid):
        return all_entity_map.get(eid, {}).get("attributes", {}).get("friendly_name", eid)

    for target in target_eids:
        target_series = grid[target]
        candidates = []

        for driver in active_cols:
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

    log.info(f"Correlation analysis complete: {len(top_insights)} significant rules found.")
    return driver_map, top_insights


# --- Feature #1: Unsupervised Household Mode Clustering ---
def train_and_predict_house_mode(grid, all_entity_map):
    """
    Clusters historical sensor states + cyclical hour into 4 interpretable household modes:
    'away', 'sleeping', 'evening_relaxing', 'active_home'.
    """
    if grid.empty or len(grid) < 96:
        return "unknown", 0.0, {"reason": "insufficient_history"}

    local_idx = grid.index.tz_convert(LOCAL_TZ)
    hours = local_idx.hour + local_idx.minute / 60.0

    X_clust = grid.copy()
    X_clust["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    X_clust["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    n_clusters = min(4, max(2, len(X_clust) // 48))
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X_clust)

    person_cols = [c for c in grid.columns if c.startswith("person.")]
    bedroom_cols = [
        c for c in grid.columns
        if not c.startswith("person.") and any(fnmatch.fnmatch(c, pat) for pat in BEDROOM_SENSOR_PATTERNS)
    ]
    common_cols = [c for c in grid.columns if c not in person_cols and c not in bedroom_cols]

    # Auto-label each cluster centroid semantically
    cluster_names = {}
    cluster_profiles = {}
    used_names = set()

    for cid in range(n_clusters):
        mask = labels == cid
        sub = grid[mask]
        sub_hours = hours[mask]

        person_avg = float(sub[person_cols].mean().mean()) if person_cols else 0.5
        bed_avg = float(sub[bedroom_cols].mean().mean()) if bedroom_cols else 0.0
        common_avg = float(sub[common_cols].mean().mean()) if common_cols else 0.0
        # Overnight fraction (23:00 to 07:00)
        night_frac = float(((sub_hours >= 23.0) | (sub_hours <= 7.0)).mean())
        evening_frac = float(((sub_hours >= 18.0) & (sub_hours < 23.0)).mean())

        if person_cols and person_avg < 0.25 and "away" not in used_names:
            name = "away"
        elif (night_frac > 0.45 or (bed_avg > common_avg and night_frac > 0.25)) and "sleeping" not in used_names:
            name = "sleeping"
        elif evening_frac > 0.35 and "evening_relaxing" not in used_names:
            name = "evening_relaxing"
        elif "active_home" not in used_names:
            name = "active_home"
        else:
            name = f"mode_{cid + 1}"

        used_names.add(name)
        cluster_names[cid] = name

        # Summarize top active entities in this cluster
        means = sub.mean().sort_values(ascending=False)
        top_active = [
            f"{all_entity_map.get(k, {}).get('attributes', {}).get('friendly_name', k)} ({int(v * 100)}%)"
            for k, v in means.head(3).items()
            if v >= 0.3
        ]
        cluster_profiles[name] = ", ".join(top_active) if top_active else "Low activity"

    # Train a RandomForest forecaster to predict future household mode at +PREDICT_HORIZON_MINS
    shift_steps = max(1, PREDICT_HORIZON_MINS // RESAMPLE_MINS)
    y_mode = pd.Series(labels, index=X_clust.index).shift(-shift_steps).dropna().astype(int)
    X_train = X_clust.iloc[: len(y_mode)]

    if y_mode.nunique() < 2:
        only_mode = cluster_names[int(y_mode.iloc[0])]
        return only_mode, 1.0, {"current_mode": only_mode, "cluster_profiles": cluster_profiles}

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train, y_mode)

    # Current live row is the last row of X_clust
    live_X = X_clust.iloc[[-1]]
    curr_cid = int(kmeans.predict(live_X)[0])
    current_mode = cluster_names[curr_cid]

    probs = clf.predict_proba(live_X)[0]
    classes = list(clf.classes_)
    best_idx = int(np.argmax(probs))
    pred_cid = int(classes[best_idx])
    pred_cert = float(probs[best_idx])
    predicted_mode = cluster_names[pred_cid] if pred_cert >= CERTAINTY_THRESHOLD else "unknown"

    mode_probs = {cluster_names[int(c)]: round(float(p), 3) for c, p in zip(classes, probs)}

    # Build 24h mode timeline using a schedule-only classifier
    sched_X = pd.DataFrame({
        "hour_sin": X_train["hour_sin"],
        "hour_cos": X_train["hour_cos"],
        "dow_sin": np.sin(2 * np.pi * X_train.index.day_of_week / 7.0),
        "dow_cos": np.cos(2 * np.pi * X_train.index.day_of_week / 7.0),
    })
    sched_clf = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42)
    sched_clf.fit(sched_X, y_mode)

    now_dt = datetime.now(timezone.utc)
    slot_offsets = np.arange(48) * 30 + 15
    slot_ts = pd.to_datetime([now_dt + timedelta(minutes=int(m)) for m in slot_offsets], utc=True)
    s_hours = slot_ts.tz_convert(LOCAL_TZ).hour + slot_ts.tz_convert(LOCAL_TZ).minute / 60.0
    s_dows = slot_ts.day_of_week
    X_slots = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * s_hours / 24.0),
        "hour_cos": np.cos(2 * np.pi * s_hours / 24.0),
        "dow_sin": np.sin(2 * np.pi * s_dows / 7.0),
        "dow_cos": np.cos(2 * np.pi * s_dows / 7.0),
    })
    slot_preds = sched_clf.predict(X_slots)
    mode_glyphs = {"sleeping": "Z", "away": "·", "active_home": "█", "evening_relaxing": "▓"}
    chars = []
    for i, cid in enumerate(slot_preds):
        if i > 0 and i % 12 == 0:
            chars.append("|")
        m_name = cluster_names[int(cid)]
        chars.append(mode_glyphs.get(m_name, "?"))

    attrs = {
        "current_mode": current_mode,
        "certainty": round(pred_cert, 3),
        "horizon_mins": PREDICT_HORIZON_MINS,
        "mode_probabilities": mode_probs,
        "timeline_24h": "".join(chars),
        "timeline_legend": "Z=sleeping ·=away █=active_home ▓=evening_relaxing |=6h",
        "cluster_profiles": cluster_profiles,
        "last_updated": now_dt.astimezone(LOCAL_TZ).isoformat(),
    }
    return predicted_mode, pred_cert, attrs


# --- Feature #2: Sleep & Wake-Up Predictor ---
def estimate_sleep_schedule(conn, person_eid, all_entity_map, now_dt):
    """
    Infers expected_wake_up and expected_bedtime by combining person home presence
    with bedroom presence sensor patterns (or overnight stillness if no bedroom sensor exists).
    """
    bedroom_eids = [
        eid for eid in all_entity_map.keys()
        if not eid.startswith("person.") and any(fnmatch.fnmatch(eid, pat) for pat in BEDROOM_SENSOR_PATTERNS)
    ]

    # Pick the first matching bedroom sensor if available
    target_eid = bedroom_eids[0] if bedroom_eids else person_eid
    table = "state_history" if (
        target_eid.startswith("person.") or any(fnmatch.fnmatch(target_eid, pat) for pat in BINARY_SENSOR_PATTERNS)
    ) else "context_history"

    df = pd.read_sql_query(
        f"SELECT timestamp, state FROM {table} WHERE entity_id = ? ORDER BY timestamp ASC",
        conn,
        params=(target_eid,),
    )
    if df.empty:
        return None, None

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    if (now_dt - df["timestamp"].min()).total_seconds() < 36 * 3600:
        return None, None

    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    is_p = target_eid.startswith("person.")
    s = df.set_index("timestamp")["state"].apply(lambda x: state_to_binary(x, is_person=is_p))
    s.loc[now_dt] = s.iloc[-1]
    s_res = s.resample("15min").ffill().dropna()

    if s_res.nunique() < 2:
        return None, None

    local_idx = s_res.index.tz_convert(LOCAL_TZ)
    hours = local_idx.hour + local_idx.minute / 60.0
    dows = local_idx.day_of_week

    X = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hours / 24.0),
        "hour_cos": np.cos(2 * np.pi * hours / 24.0),
        "dow_sin": np.sin(2 * np.pi * dows / 7.0),
        "dow_cos": np.cos(2 * np.pi * dows / 7.0),
        "is_weekend": (dows >= 5).astype(int),
    })
    y = s_res.values.astype(int)

    clf = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42)
    clf.fit(X, y)

    # Forecast next 36 hours in 15-min steps
    steps = np.arange(1, 145)
    fut_ts = pd.to_datetime([now_dt + timedelta(minutes=15 * int(st)) for st in steps], utc=True)
    f_local = fut_ts.tz_convert(LOCAL_TZ)
    f_hours = f_local.hour + f_local.minute / 60.0
    f_dows = f_local.day_of_week

    X_fut = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * f_hours / 24.0),
        "hour_cos": np.cos(2 * np.pi * f_hours / 24.0),
        "dow_sin": np.sin(2 * np.pi * f_dows / 7.0),
        "dow_cos": np.cos(2 * np.pi * f_dows / 7.0),
        "is_weekend": (f_dows >= 5).astype(int),
    })
    preds = clf.predict(X_fut)

    expected_wake_up = None
    expected_bedtime = None

    # Scan transitions: leaving bedroom in morning window (05:00 - 12:30) = wake-up;
    # entering bedroom in night window (20:30 - 03:30) = bedtime
    for i in range(1, len(preds)):
        h = f_hours[i]
        ts_iso = f_local[i].isoformat()
        if preds[i - 1] == 1 and preds[i] == 0 and (5.0 <= h <= 12.5) and expected_wake_up is None:
            expected_wake_up = ts_iso
        elif preds[i - 1] == 0 and preds[i] == 1 and (h >= 20.5 or h <= 3.5) and expected_bedtime is None:
            expected_bedtime = ts_iso
        if expected_wake_up and expected_bedtime:
            break

    return expected_wake_up, expected_bedtime


# --- Feature Engineering & Main Entity ML ---
def get_driver_history_series(conn, driver_eid, lag_mins, resample_rule, index_like):
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


def build_features(df, is_person=False, driver_cols=None):
    df = df.copy()
    local_ts = df["timestamp"].dt.tz_convert(LOCAL_TZ)
    df["hour"] = local_ts.dt.hour + local_ts.dt.minute / 60.0
    df["dow"] = local_ts.dt.day_of_week
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
    local_ts = slot_timestamps.tz_convert(LOCAL_TZ)
    hours = local_ts.hour + local_ts.minute / 60.0
    dows = local_ts.day_of_week

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
    local_ts = fut_timestamps.tz_convert(LOCAL_TZ)
    hours = local_ts.hour + local_ts.minute / 60.0
    dows = local_ts.day_of_week

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
            next_on = local_ts[idx].isoformat()
        elif sim_state == 1 and pred == 0 and next_off is None:
            next_off = local_ts[idx].isoformat()

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

    acc_7d, eval_count, effective_threshold = get_entity_accuracy_stats(conn, eid)

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
        return const_state, 1.0, {
            "reason": "constant_historical_state",
            "expected_next_on": None,
            "expected_next_off": None,
            "timeline_24h": flat_line,
            "timeline_start": now_dt.astimezone(LOCAL_TZ).strftime("%H:%M"),
            "timeline_legend": f"█={positive_state} ·={negative_state} |=6h (30m/char)",
            "correlated_triggers": [d["summary"] for d in drivers],
            "routine_anomaly_score": 0,
            "is_routine_anomaly": False,
            "accuracy_7d": acc_7d,
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

    # Feature #4: Routine Anomaly Score (compares live state against pure calendar schedule expectation)
    sched_probs = schedule_model.predict_proba(now_df[schedule_cols])[0]
    sched_classes = list(schedule_model.classes_)
    sched_prob_pos = sched_probs[sched_classes.index(1)] if 1 in sched_classes else 0.5
    # Surprise magnitude: how strongly current live state contradicts usual schedule at this hour
    contradiction = abs(curr_bin - sched_prob_pos)
    duration_factor = min(1.0, mins_in_state / 45.0)  # Ramp up over 45m so brief blips don't alarm
    anomaly_score = int(round(contradiction * duration_factor * 100))

    next_on, next_off, timeline_24h = estimate_next_transitions(schedule_model, now_dt, curr_bin)

    meta = {
        "probability_active": round(float(prob_pos), 3),
        "certainty": round(float(max_prob), 3),
        "threshold": effective_threshold,
        "horizon_mins": PREDICT_HORIZON_MINS,
        "expected_next_on": next_on,
        "expected_next_off": next_off,
        "timeline_24h": timeline_24h,
        "timeline_start": now_dt.astimezone(LOCAL_TZ).strftime("%H:%M"),
        "timeline_legend": f"█={positive_state} ·={negative_state} |=6h (30m/char)",
        "correlated_triggers": [d["summary"] for d in drivers],
        "routine_anomaly_score": anomaly_score,
        "is_routine_anomaly": anomaly_score >= 70,
        "accuracy_7d": acc_7d,
        "evaluated_predictions_7d": eval_count,
        "mins_in_current_state": round(mins_in_state, 1),
        "recorded_hours": round(span_hours, 1),
        "last_updated": now_dt.astimezone(LOCAL_TZ).isoformat(),
    }

    if is_person:
        meta["distance_from_home_km"] = round(float(now_df["distance_from_home_km"].iloc[0]), 2)
        meta["home_approach_speed_kmh"] = round(float(now_df["home_approach_speed_kmh"].iloc[0]), 1)
        meta["eta_mins"] = round(float(now_df["eta_mins"].iloc[0]), 0)
        wake_up, bedtime = estimate_sleep_schedule(conn, eid, all_entity_map, now_dt)
        meta["expected_wake_up"] = wake_up
        meta["expected_bedtime"] = bedtime

    if max_prob < effective_threshold:
        return "unknown", max_prob, meta

    log_prediction_for_audit(conn, eid, now_dt, predicted_state, max_prob)
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

    # Feature #2: Publish dedicated timestamp sensor for person wake-up time
    if eid.startswith("person.") and attributes.get("expected_wake_up"):
        wake_uid = f"{object_slug}_expected_wake_up"
        wake_topic = f"{MQTT_PREFIX}/sensor/{wake_uid}"
        wake_payload = {
            "name": f"{friendly_name} Expected Wake Up",
            "default_entity_id": f"sensor.{wake_uid}",
            "unique_id": wake_uid,
            "device_class": "timestamp",
            "state_topic": f"{wake_topic}/state",
            "icon": "mdi:weather-sunset-up",
            "device": {
                "identifiers": ["ha_ml_predictor_service"],
                "name": "HA ML Predictor",
                "manufacturer": "Custom ML Service",
                "model": "RandomForest State Forecaster",
            },
        }
        client.publish(f"{wake_topic}/config", json.dumps(wake_payload), retain=True)
        client.publish(f"{wake_topic}/state", attributes["expected_wake_up"], retain=True)


def publish_house_mode_sensor(client, mode_state, attributes):
    unique_id = "house_mode_predicted"
    base_topic = f"{MQTT_PREFIX}/sensor/{unique_id}"
    discovery_payload = {
        "name": "House Mode Predicted",
        "default_entity_id": "sensor.house_mode_predicted",
        "unique_id": unique_id,
        "state_topic": f"{base_topic}/state",
        "json_attributes_topic": f"{base_topic}/attributes",
        "icon": "mdi:home-analytics",
        "device": {
            "identifiers": ["ha_ml_predictor_service"],
            "name": "HA ML Predictor",
            "manufacturer": "Custom ML Service",
            "model": "RandomForest State Forecaster",
        },
    }
    client.publish(f"{base_topic}/config", json.dumps(discovery_payload), retain=True)
    client.publish(f"{base_topic}/state", mode_state, retain=True)
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


# --- Main Loop ---
def main():
    if not HA_TOKEN:
        raise ValueError("HA_TOKEN environment variable is required.")

    conn = init_db()
    mqtt_client = setup_mqtt()
    fetch_home_coords()
    log.info("HA ML Predictor started.")

    last_analysis_run = None
    driver_map = {}
    top_insights = []
    cached_grid = pd.DataFrame()

    while True:
        try:
            targets, context_entities = get_entities()
            all_entity_map = {e["entity_id"]: e for e in (targets + context_entities)}

            sync_entity_group(conn, targets, is_context=False)
            sync_entity_group(conn, context_entities, is_context=True)

            # Feature #3: Score expired predictions against reality
            score_due_predictions(conn, all_entity_map)

            now_utc = datetime.now(timezone.utc)
            if last_analysis_run is None or (now_utc - last_analysis_run).total_seconds() >= ANALYSIS_INTERVAL_HOURS * 3600:
                prune_old_records(conn)
                cached_grid = build_aligned_grid(conn, days_back=CONTEXT_RETENTION_DAYS)
                driver_map, top_insights = run_correlation_analysis(cached_grid, all_entity_map)
                publish_insights_sensor(mqtt_client, top_insights, len(context_entities))
                last_analysis_run = now_utc

            # Feature #1: Update live row in grid and predict Household Mode
            if not cached_grid.empty:
                live_vals = {
                    col: state_to_binary(all_entity_map[col].get("state"), is_person=col.startswith("person."))
                    for col in cached_grid.columns
                    if col in all_entity_map
                }
                if live_vals:
                    live_idx = pd.DatetimeIndex([now_utc])
                    live_df = pd.DataFrame([live_vals], index=live_idx).reindex(columns=cached_grid.columns, fill_value=0)
                    updated_grid = pd.concat([cached_grid, live_df])
                    mode_pred, mode_cert, mode_attrs = train_and_predict_house_mode(updated_grid, all_entity_map)
                    publish_house_mode_sensor(mqtt_client, mode_pred, mode_attrs)
                    log.info(f"house_mode -> {mode_pred} (current: {mode_attrs.get('current_mode')}, cert: {mode_cert:.2f})")

            for item in targets:
                eid = item["entity_id"]
                drivers = driver_map.get(eid, [])
                pred_state, cert, meta = train_and_predict(conn, item, drivers, all_entity_map)
                publish_ha_discovery(mqtt_client, item, pred_state, meta)
                log.info(
                    f"{eid} -> {pred_state} "
                    f"(cert: {cert:.2f}, acc7d: {meta.get('accuracy_7d')}%, anomaly: {meta.get('routine_anomaly_score')})"
                )

        except Exception as e:
            log.error(f"Error in main loop: {e}", exc_info=True)

        time.sleep(300)


if __name__ == "__main__":
    main()
