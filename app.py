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

# Comma-separated wildcards for binary sensors (e.g., "binary_sensor.*_motion,binary_sensor.front_door")
BINARY_SENSOR_PATTERNS = [p.strip() for p in os.getenv("BINARY_SENSOR_PATTERNS", "").split(",") if p.strip()]
CERTAINTY_THRESHOLD = float(os.getenv("CERTAINTY_THRESHOLD", "0.70"))
PREDICT_HORIZON_MINS = int(os.getenv("PREDICT_HORIZON_MINS", "30"))
RESAMPLE_MINS = int(os.getenv("RESAMPLE_MINS", "5"))
INITIAL_HISTORY_DAYS = int(os.getenv("INITIAL_HISTORY_DAYS", "10"))
MIN_HISTORY_HOURS = int(os.getenv("MIN_HISTORY_HOURS", "24"))
DECAY_HALF_LIFE_DAYS = float(os.getenv("DECAY_HALF_LIFE_DAYS", "7.0"))
MAX_NEXT_STATES = int(os.getenv("MAX_NEXT_STATES", "5"))
DB_PATH = os.getenv("DB_PATH", "/data/history.db")

HEADERS = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
HOME_COORDS = (None, None)
LOCAL_TZ = datetime.now().astimezone().tzinfo


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


# --- Database Layer (Prevents HA Purge Loss) ---
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
    conn.commit()
    return conn


def get_last_timestamp(conn, entity_id):
    cur = conn.execute("SELECT MAX(timestamp) FROM state_history WHERE entity_id = ?", (entity_id,))
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def archive_states(conn, states_list):
    inserted = 0
    for entity_history in states_list:
        for entry in entity_history:
            entity_id = entry.get("entity_id")
            ts = entry.get("last_changed") or entry.get("last_updated")
            state = entry.get("state")
            attrs = json.dumps(entry.get("attributes", {}))
            if not entity_id or not ts or state in ("unavailable", "unknown", None):
                continue
            try:
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


def get_target_entities():
    resp = requests.get(f"{HA_URL}/api/states", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    all_states = resp.json()

    targets = []
    for item in all_states:
        eid = item["entity_id"]
        if eid.startswith("person."):
            targets.append(item)
        elif eid.startswith("binary_sensor."):
            if any(fnmatch.fnmatch(eid, pat) for pat in BINARY_SENSOR_PATTERNS):
                targets.append(item)
    return targets


def sync_history(conn, targets):
    for item in targets:
        eid = item["entity_id"]
        last_ts = get_last_timestamp(conn, eid)
        if last_ts:
            start_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        else:
            start_dt = datetime.now(timezone.utc) - timedelta(days=INITIAL_HISTORY_DAYS)

        url = f"{HA_URL}/api/history/period/{start_dt.isoformat()}"
        params = {"filter_entity_id": eid}
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 200:
                count = archive_states(conn, resp.json())
                if count > 0:
                    log.info(f"Archived {count} new state entries for {eid}")
        except Exception as e:
            log.warning(f"Failed syncing history for {eid}: {e}")

        # Ensure live state is recorded so steady-state entities maintain timeline continuity
        ts_now = item.get("last_changed") or datetime.now(timezone.utc).isoformat()
        state_now = item.get("state")
        if state_now not in ("unavailable", "unknown", None):
            conn.execute(
                "INSERT OR IGNORE INTO state_history (entity_id, timestamp, state, attributes) VALUES (?, ?, ?, ?)",
                (eid, ts_now, state_now, json.dumps(item.get("attributes", {}))),
            )
    conn.commit()


# --- Feature Engineering & ML ---
def build_features(df, is_person=False):
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
        feature_cols.extend(["distance_from_home_km", "gps_accuracy"])

    return df, feature_cols, schedule_cols


def estimate_next_transitions(schedule_model, now_dt, current_state_bin, positive_state, negative_state):
    """Vectorized 7-day scan using pure schedule features to find next ON/OFF and upcoming state sequence."""
    next_on = None
    next_off = None
    next_states = []

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
        if pred != sim_state:
            ts_local = fut_timestamps[idx].tz_convert(LOCAL_TZ).isoformat()
            state_label = positive_state if pred == 1 else negative_state
            state_conf = float(prob_pos[idx]) if pred == 1 else float(1.0 - prob_pos[idx])

            if sim_state == 0 and pred == 1 and next_on is None:
                next_on = ts_local
            elif sim_state == 1 and pred == 0 and next_off is None:
                next_off = ts_local

            if len(next_states) < MAX_NEXT_STATES:
                next_states.append({
                    "state": state_label,
                    "at": ts_local,
                    "confidence": round(state_conf, 2),
                })

            sim_state = pred

            if next_on and next_off and len(next_states) >= MAX_NEXT_STATES:
                break

    return next_on, next_off, next_states


def train_and_predict(conn, entity_item):
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
    df = df.set_index("timestamp").resample(resample_rule).ffill().dropna(subset=["state"]).reset_index()
    df["attr_dict"] = df["attributes"].apply(json.loads)
    df["current_state_bin"] = (df["state"] == positive_state).astype(int)
    df["mins_in_current_state"] = (df["timestamp"] - df["group_start_ts"]).dt.total_seconds() / 60.0

    shift_steps = max(1, PREDICT_HORIZON_MINS // RESAMPLE_MINS)
    df["future_target"] = df["current_state_bin"].shift(-shift_steps)
    train_df = df.dropna(subset=["future_target"]).copy()

    if train_df["future_target"].nunique() < 2:
        const_state = positive_state if train_df["future_target"].iloc[0] == 1 else negative_state
        return const_state, 1.0, {
            "reason": "constant_historical_state",
            "expected_next_on": None,
            "expected_next_off": None,
            "expected_next_states": [],
            "recorded_hours": round(span_hours, 1),
        }

    train_df, feature_cols, schedule_cols = build_features(train_df, is_person=is_person)
    X = train_df[feature_cols]
    y = train_df["future_target"].astype(int)

    age_days = (now_dt - train_df["timestamp"]).dt.total_seconds() / 86400.0
    sample_weights = np.exp(-np.log(2) * age_days / DECAY_HALF_LIFE_DAYS)

    # 1. Context-aware model for short-term state prediction
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y, sample_weight=sample_weights)

    # 2. Pure schedule model for next_on / next_off & transition sequence forecasting
    schedule_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    schedule_model.fit(train_df[schedule_cols], y, sample_weight=sample_weights)

    last_changed_str = entity_item.get("last_changed")
    last_changed_dt = datetime.fromisoformat(last_changed_str.replace("Z", "+00:00")) if last_changed_str else now_dt
    mins_in_state = max(0.0, (now_dt - last_changed_dt).total_seconds() / 60.0)
    curr_bin = 1 if entity_item.get("state") == positive_state else 0

    now_df = pd.DataFrame([{
        "timestamp": now_dt,
        "attr_dict": entity_item.get("attributes", {}),
        "current_state_bin": curr_bin,
        "mins_in_current_state": mins_in_state,
    }])
    now_df, _, _ = build_features(now_df, is_person=is_person)

    probs = model.predict_proba(now_df[feature_cols])[0]
    classes = list(model.classes_)
    prob_pos = probs[classes.index(1)] if 1 in classes else 0.0
    prob_neg = probs[classes.index(0)] if 0 in classes else 0.0

    max_prob = max(prob_pos, prob_neg)
    predicted_state = positive_state if prob_pos >= prob_neg else negative_state

    next_on, next_off, next_states = estimate_next_transitions(
        schedule_model, now_dt, curr_bin, positive_state, negative_state
    )

    meta = {
        "probability_active": round(float(prob_pos), 3),
        "certainty": round(float(max_prob), 3),
        "threshold": CERTAINTY_THRESHOLD,
        "horizon_mins": PREDICT_HORIZON_MINS,
        "expected_next_on": next_on,
        "expected_next_off": next_off,
        "expected_next_states": next_states,
        "mins_in_current_state": round(mins_in_state, 1),
        "recorded_hours": round(span_hours, 1),
        "last_updated": now_dt.astimezone(LOCAL_TZ).isoformat(),
    }

    if is_person:
        meta["distance_from_home_km"] = round(float(now_df["distance_from_home_km"].iloc[0]), 2)

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


# --- Main Loop ---
def main():
    if not HA_TOKEN:
        raise ValueError("HA_TOKEN environment variable is required.")

    conn = init_db()
    mqtt_client = setup_mqtt()
    fetch_home_coords()
    log.info("HA ML Predictor started.")

    while True:
        try:
            targets = get_target_entities()
            sync_history(conn, targets)

            for item in targets:
                pred_state, cert, meta = train_and_predict(conn, item)
                publish_ha_discovery(mqtt_client, item, pred_state, meta)
                log.info(
                    f"{item['entity_id']} -> {pred_state} "
                    f"(cert: {cert:.2f}, next_on: {meta.get('expected_next_on')}, next_off: {meta.get('expected_next_off')})"
                )

        except Exception as e:
            log.error(f"Error in main loop: {e}", exc_info=True)

        time.sleep(300)


if __name__ == "__main__":
    main()
