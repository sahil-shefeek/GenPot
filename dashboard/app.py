import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

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

        # Schema Safety: Ensure status_code exists and handle missing values
        if "status_code" not in df.columns:
            df["status_code"] = 0
        else:
            df["status_code"] = df["status_code"].fillna(0).astype(int)

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

    # --- Attack Trends ---
    st.subheader("Attack Trends")

    # 1. Time Series Chart (Attacks per Minute)
    if "timestamp" in df.columns:
        # Resample to 1 minute intervals and count attacks
        # We use a placeholder column 'ip' to count occurrences
        # fillna(0) ensures gaps in traffic appear as 0 instead of breaks/NaN
        attacks_per_min = (
            df.set_index("timestamp")
            .resample("1Min")["ip"]
            .count()
            .fillna(0)
            .reset_index(name="Count")
        )

        fig_ts = px.line(
            attacks_per_min, x="timestamp", y="Count", title="Attacks per Minute"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    col_trend1, col_trend2 = st.columns(2)

    with col_trend1:
        # 2. Status Code Distribution
        if "status_code" in df.columns:
            status_counts = df["status_code"].value_counts().reset_index()
            status_counts.columns = ["Status Code", "Count"]
            fig_pie = px.pie(
                status_counts,
                names="Status Code",
                values="Count",
                title="Status Code Distribution",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_trend2:
        # 3. HTTP Method Distribution
        if "method" in df.columns:
            method_counts = df["method"].value_counts().reset_index()
            method_counts.columns = ["Method", "Count"]
            fig_bar = px.bar(
                method_counts, x="Method", y="Count", title="HTTP Method Distribution"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

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
