import streamlit as st
import pandas as pd
import plotly.express as px
import json
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

SUSPICIOUS_KEYWORDS = [
    "/etc/passwd",
    ".env",
    "union select",
    "eval(",
    "git",
    "config",
    "admin",
]


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

        # Threat Tagging Logic
        def classify_threat(row):
            # content to check
            targets = [str(row.get("path", "")), str(row.get("body", ""))]
            content = " ".join(targets).lower()

            for keyword in SUSPICIOUS_KEYWORDS:
                if keyword in content:
                    return "CRITICAL"
            return "INFO"

        df["threat_level"] = df.apply(classify_threat, axis=1)

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

# Sidebar Filter
st.sidebar.header("Filters")
threat_filter = st.sidebar.radio("Filter by Threat Level", ["All", "Critical", "Info"])

# Apply Filter
if not df.empty and threat_filter != "All":
    df = df[df["threat_level"] == threat_filter.upper()]

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
    st.caption("Select a row to inspect payload details.")

    # Ensure we are working with the latest data (sorted by timestamp)
    # Use top 50 rows to allow deeper inspection without overwhelming the generic table
    display_df = df.head(50)

    # Select specific columns to display if they exist
    cols_to_display = [
        "timestamp",
        "threat_level",
        "method",
        "path",
        "ip",
        "response_time_ms",
        "status_code",
    ]
    available_cols = [c for c in cols_to_display if c in display_df.columns]

    # Show table with selection enabled
    event = st.dataframe(
        display_df[available_cols] if available_cols else display_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        column_config={
            "threat_level": st.column_config.TextColumn(
                "Threat Level", help="Based on heuristic keywords"
            ),
        },
    )

    # Payload Inspector
    if event.selection.rows:
        selected_index = event.selection.rows[0]
        selected_row = display_df.iloc[selected_index]

        st.divider()
        st.subheader("🔍 Interaction Details")

        col_details1, col_details2 = st.columns(2)

        with col_details1:
            st.markdown("**Request Body**")
            req_body = selected_row.get("body", None)

            if isinstance(req_body, dict):
                st.json(req_body)
            elif isinstance(req_body, str) and req_body.strip():
                try:
                    st.json(json.loads(req_body))
                except json.JSONDecodeError:
                    st.text(req_body)  # Display as plain text if not valid JSON
            elif req_body:
                st.write(req_body)  # Fallback
            else:
                st.info("Empty Payload")

        with col_details2:
            st.markdown("**Response Body**")
            res_body = selected_row.get("response", None)

            if isinstance(res_body, dict):
                st.json(res_body)
            elif isinstance(res_body, str) and res_body.strip():
                try:
                    st.json(json.loads(res_body))
                except json.JSONDecodeError:
                    st.text(res_body)
            elif res_body:
                st.write(res_body)
            else:
                st.info("Empty Response")
