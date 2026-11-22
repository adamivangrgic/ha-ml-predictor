import os
import re
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

# Comma-separated wildcards for binary_sensors (e.g., "binary_sensor.*_motion,binary_sensor.front_door")
BINARY_SENSOR_PATTERNS = [p.strip() for p in os.getenv("BINARY_SENSOR_PATTERNS", "").split(",") if p.strip()]
CERTAINTY_THRESHOLD = float(os.getenv("CERTAINTY_THRESHOLD", "0.70"))
PREDICT_HORIZON_MINS = int(os.getenv("PREDICT_HORIZON_MINS", "30"))
INITIAL_HISTORY_DAYS = int(os.getenv("INITIAL_HISTORY_DAYS", "10"))
DB_PATH = os.getenv("DB_PATH", "/data/history.db")

HEADERS = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}


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
                conn.execute(
                    "INSERT OR IGNORE INTO state_history (entity_id, timestamp, state, attributes) VALUES (?, ?, ?, ?)",
                    (entity_id, ts, state, attrs),
                )
                inserted += 1
            except sqlite3.Error as e:
                log.error(f"DB Insert error: {e}")
    conn.commit()
    return inserted


# --- Home Assistant API ---
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


def sync_history(conn, entity_ids):
    for eid in entity_ids:
        last_ts = get_last_timestamp(conn, eid)
        if last_ts:
            start_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        else:
            start_dt = datetime.now(timezone.utc) - timedelta(days=INITIAL_HISTORY_DAYS)

        url = f"{HA_URL}/api/history/period/{start_dt.isoformat()}"
        params = {"filter_entity_id": eid, "significant_changes_only": "1"}
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 200:
                count = archive_states(conn, resp.json())
                if count > 0:
                    log.info(f"Archived {count} new state transitions for {eid}")
        except Exception as e:
            log.warning(f"Failed syncing history for {eid}: {e}")


# --- Feature Engineering & ML ---
def extract_features(df, is_person=False):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["dow"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Cyclical encoding so 23:59 is close to 00:01
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    feature_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]

    if is_person:
        # Extract useful attributes for person entities
        df["gps_accuracy"] = df["attr_dict"].apply(lambda a: float(a.get("gps_accuracy", 0) or 0))
        df["has_source"] = df["attr_dict"].apply(lambda a: 1 if a.get("source") else 0)
        feature_cols.extend(["gps_accuracy", "has_source"])

    return df, feature_cols


def train_and_predict(conn, entity_item):
    eid = entity_item["entity_id"]
    is_person = eid.startswith("person.")
    positive_state = "home" if is_person else "on"
    negative_state = "not_home" if is_person else "off"

    df = pd.read_sql_query(
        "SELECT timestamp, state, attributes FROM state_history WHERE entity_id = ? ORDER BY timestamp ASC",
        conn,
        params=(eid,),
    )

    if len(df) < 20:
        return "unknown", 0.0, {"reason": "insufficient_history", "samples": len(df)}

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df["attr_dict"] = df["attributes"].apply(json.loads)

    # Resample to 15-minute intervals to capture duration spent in each state
    df = df.set_index("timestamp").resample("15min").ffill().dropna().reset_index()
    if len(df) < 30:
        return "unknown", 0.0, {"reason": "insufficient_resampled_data", "samples": len(df)}

    df["target"] = (df["state"] == positive_state).astype(int)

    # Shift target backwards so current features predict future state (PREDICT_HORIZON_MINS)
    shift_steps = max(1, PREDICT_HORIZON_MINS // 15)
    df["future_target"] = df["target"].shift(-shift_steps)
    train_df = df.dropna(subset=["future_target"]).copy()

    if train_df["future_target"].nunique() < 2:
        return "unknown", 0.0, {"reason": "single_class_history"}

    train_df, feature_cols = extract_features(train_df, is_person=is_person)
    X = train_df[feature_cols]
    y = train_df["future_target"].astype(int)

    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y)

    # Build current observation row for inference
    now_df = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc),
        "attr_dict": entity_item.get("attributes", {})
    }])
    now_df, _ = extract_features(now_df, is_person=is_person)

    probs = model.predict_proba(now_df[feature_cols])[0]
    classes = list(model.classes_)
    prob_pos = probs[classes.index(1)] if 1 in classes else 0.0
    prob_neg = probs[classes.index(0)] if 0 in classes else 0.0

    max_prob = max(prob_pos, prob_neg)
    predicted_state = positive_state if prob_pos >= prob_neg else negative_state

    meta = {
        "probability_active": round(float(prob_pos), 3),
        "certainty": round(float(max_prob), 3),
        "threshold": CERTAINTY_THRESHOLD,
        "horizon_mins": PREDICT_HORIZON_MINS,
        "training_samples": len(train_df),
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

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
    friendly_name = entity_item.get("attributes", {}).get("friendly_name", eid)
    safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", eid)
    unique_id = f"ha_ml_pred_{safe_id}"

    base_topic = f"{MQTT_PREFIX}/sensor/{unique_id}"
    config_topic = f"{base_topic}/config"
    state_topic = f"{base_topic}/state"
    attr_topic = f"{base_topic}/attributes"

    discovery_payload = {
        "name": f"{friendly_name} Predicted ({PREDICT_HORIZON_MINS}m)",
        "unique_id": unique_id,
        "state_topic": state_topic,
        "json_attributes_topic": attr_topic,
        "icon": "mdi:crystal-ball" if eid.startswith("person.") else "mdi:chart-bell-curve-cumulative",
        "device": {
            "identifiers": ["ha_ml_predictor_service"],
            "name": "HA ML Predictor",
            "manufacturer": "Custom ML Service",
            "model": "RandomForest State Forecaster"
        }
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
    log.info("HA ML Predictor started.")

    while True:
        try:
            targets = get_target_entities()
            target_ids = [t["entity_id"] for t in targets]
            log.info(f"Tracking {len(target_ids)} entities: {target_ids}")

            sync_history(conn, target_ids)

            for item in targets:
                pred_state, cert, meta = train_and_predict(conn, item)
                publish_ha_discovery(mqtt_client, item, pred_state, meta)
                log.info(f"{item['entity_id']} -> Prediction: {pred_state} (certainty: {cert:.2f})")

        except Exception as e:
            log.error(f"Error in main loop: {e}", exc_info=True)

        time.sleep(300)  # Run every 5 minutes


if __name__ == "__main__":
    main()
