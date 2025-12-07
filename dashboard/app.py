import streamlit as st
import pandas as pd
import plotly.express as px
import json
import sys

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
CACHE_FILE = BASE_DIR / "logs" / "analysis_cache.json"

# Add server directory to path for imports
sys.path.append(str(BASE_DIR))

try:
    from server.llm_client import generate_response
    from server.utils import clean_llm_response
except ImportError:
    st.error(
        "Failed to import server modules. Ensure you are running from the project root."
    )
    st.stop()

SUSPICIOUS_KEYWORDS = [
    "/etc/passwd",
    ".env",
    "union select",
    "eval(",
    "git",
    "config",
    "admin",
]


def load_cache():
    """Load the analysis cache from disk."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_to_cache(key, data):
    """Save a new analysis result to the cache."""
    cache = load_cache()
    cache[key] = data
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        st.error(f"Failed to save cache: {e}")


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

        # Extract User-Agent / Tool Signature
        def extract_tool(row):
            headers = row.get("headers", {})
            if not isinstance(headers, dict):
                return "Unknown"

            ua = headers.get("user-agent", "Unknown")
            if not ua or ua == "Unknown":
                return "Unknown"

            # Simple cleaning: "curl/7.68.0" -> "curl"
            clean_ua = str(ua).strip()
            return clean_ua.split("/")[0] if "/" in clean_ua else clean_ua

        df["tool_signature"] = df.apply(extract_tool, axis=1)

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

    # 4. Top Attack Tools (User-Agents)
    if "tool_signature" in df.columns:
        tool_counts = df["tool_signature"].value_counts().reset_index()
        tool_counts.columns = ["Tool", "Count"]
        fig_tools = px.bar(
            tool_counts, y="Tool", x="Count", title="Top Attack Tools", orientation="h"
        )
        # Invert y-axis to show top tools at the top
        fig_tools.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_tools, use_container_width=True)

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
        "tool_signature",
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
        # --- AI Forensic Analysis ---
        st.subheader("🤖 AI Forensic Analysis")

        # Create unique cache key
        ts_str = str(selected_row.get("timestamp", ""))
        src_ip = str(selected_row.get("ip", ""))
        path_str = str(selected_row.get("path", ""))
        cache_key = f"{ts_str}_{src_ip}_{path_str}"

        # Check cache
        cache = load_cache()
        cached_entry = cache.get(cache_key)

        # Helper to render the nice UI
        def render_analysis_ui(data, timestamp_str=None):
            # Parse main fields
            intent = data.get("intent", "Unknown Intent")
            severity = data.get("severity", "Unknown")
            category = data.get("category", "Unknown")
            explanation = data.get("explanation", "No explanation provided.")

            # Severity Color Mapping
            severity_colors = {
                "High": "red",
                "Medium": "orange",
                "Low": "green",
                "Unknown": "grey",
            }
            color = severity_colors.get(severity, "grey")

            # Layout: Header with Intent
            st.markdown(f"**Intent:** {intent}")

            # Meta-data columns
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"**Severity:** :{color}[{severity}]")
            with m2:
                st.markdown(f"**Category:** {category}")
            with m3:
                if timestamp_str:
                    st.caption(f"Analyzed: {timestamp_str}")

            st.info(explanation)

        if cached_entry:
            # Handle both old format (raw dict) and new format (wrapper)
            if "data" in cached_entry and "timestamp" in cached_entry:
                analysis_data = cached_entry["data"]
                analysis_ts = cached_entry["timestamp"]
            else:
                analysis_data = cached_entry
                analysis_ts = "Previously"

            render_analysis_ui(analysis_data, analysis_ts)

            if st.button("🔄 Re-analyze (Update Models)"):
                # Remove from cache and rerun to trigger formatting below (or just analyze immediately)
                del cache[cache_key]
                # Save the deletion to disk so the next run knows it's gone
                try:
                    with open(CACHE_FILE, "w") as f:
                        json.dump(cache, f, indent=2)
                except IOError:
                    pass  # Non-critical if delete fails, but good to try
                st.rerun()

        else:
            if st.button("🤖 Analyze with AI"):
                with st.spinner("Analyzing threat signature..."):
                    # Construct Prompt
                    body_content = selected_row.get("body", "No Body")
                    prompt = (
                        f"Act as a Tier 3 Security Analyst. Analyze this HTTP request.\n"
                        f"Input: {body_content}\n"
                        f"Path: {path_str}\n"
                        f"Output JSON with keys: 'intent' (short summary), 'severity' (Low/Med/High), "
                        f"'category' (e.g. SQLi, Recon), and 'explanation'."
                    )

                    try:
                        # Call LLM
                        raw_response = generate_response(prompt)
                        analysis_result = clean_llm_response(raw_response)

                        # Save to cache with timestamp
                        from datetime import datetime

                        entry = {
                            "data": analysis_result,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        save_to_cache(cache_key, entry)

                        # Display result immediately
                        render_analysis_ui(analysis_result, entry["timestamp"])
                        # Optional: st.rerun() to lock it in, but rendering directly is faster for the user

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
