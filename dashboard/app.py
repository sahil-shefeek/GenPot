import streamlit as st
import pandas as pd
from pathlib import Path
import time

# --- Configuration ---
st.set_page_config(
    page_title="GenPot Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
)

# Robust pathing: Resolve logs relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "logs" / "honeypot.jsonl"


def load_data():
    """
    Loads the honeypot logs from the JSONL file.
    Returns an empty DataFrame if the file is missing or empty.
    """
    if not LOG_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_json(LOG_FILE, lines=True)
        if df.empty:
            return pd.DataFrame()

        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Sort by timestamp descending (newest first)
            df = df.sort_values(by="timestamp", ascending=False)

        return df
    except ValueError:
        # Handle cases where file exists but is empty or invalid JSON
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# --- UI Layout ---
st.title("🛡️ GenPot Live Threat Intelligence")

# Refresh button
if st.button("Refresh Data"):
    st.rerun()

# Load data
df = load_data()

if df.empty:
    st.warning("No data available yet. Waiting for attacks...")
else:
    # --- Top Row: Metrics ---
    col1, col2, col3 = st.columns(3)

    total_attacks = len(df)
    unique_attackers = df["ip"].nunique() if "ip" in df.columns else 0

    # Calculate average latency safely
    avg_latency = 0.0
    if "response_time_ms" in df.columns:
        avg_latency = df["response_time_ms"].mean()

    col1.metric("Total Attacks", total_attacks)
    col2.metric("Unique Attackers", unique_attackers)
    col3.metric("Avg Latency (ms)", f"{avg_latency:.2f}")

    st.divider()

    # --- Middle Row: Charts ---
    st.subheader("Top Attacked Endpoints")
    if "path" in df.columns:
        # Count attacks per path
        path_counts = df["path"].value_counts().reset_index()
        path_counts.columns = ["Path", "Count"]

        st.bar_chart(path_counts, x="Path", y="Count")
    else:
        st.info("No path data available.")

    st.divider()

    # --- Bottom Row: Raw Data ---
    st.subheader("Recent Activity")

    # Select specific columns to display if they exist
    cols_to_display = [
        "timestamp",
        "method",
        "path",
        "ip",
        "response_time_ms",
        "status_code",
    ]
    available_cols = [c for c in cols_to_display if c in df.columns]

    if available_cols:
        st.dataframe(
            df[available_cols].head(10),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(df.head(10), use_container_width=True)
