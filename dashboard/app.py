import streamlit as st
import pandas as pd
import plotly.express as px
import json
import sys
import os
import requests
import time

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
    from server.llm_client import (
        generate_response,
        list_available_models,
        LLMRateLimitError,
    )
    from server.utils import clean_llm_response
    from server.config_manager import load_config, save_config
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


# --- Health Checks ---
if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ GOOGLE_API_KEY is missing. Gemini provider will fail.")

# Optional: Check Ollama availability once
# We can't easily ping without a client, so we rely on list_available_models returning empty
# or specific error handling in the UI if needed.


# --- UI Layout ---
# --- UI Layout ---
st.title("🛡️ GenPot Threat Intelligence")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Live Threat Feed", "Attack Simulator", "RAG Inspector"]
)

st.sidebar.divider()

# --- Configuration Sidebar ---
st.sidebar.header("⚙️ Model Configuration")
current_config = load_config()

st.sidebar.caption("Currently Active")
with st.sidebar.container(border=True):
    st.markdown("**🍯 Honeypot**")
    st.markdown(
        f"`{current_config.get('honeypot_provider', 'Unknown')}` / `{current_config.get('honeypot_model', 'Unknown')}`"
    )

    st.divider()

    st.markdown("**🕵️ Analyst**")
    st.markdown(
        f"`{current_config.get('analysis_provider', 'Unknown')}` / `{current_config.get('analysis_model', 'Unknown')}`"
    )

    st.divider()

    st.markdown("**⚔️ Generator**")
    st.markdown(
        f"`{current_config.get('generator_provider', 'Unknown')}` / `{current_config.get('generator_model', 'Unknown')}`"
    )

st.sidebar.divider()


def render_model_selector(
    label_prefix: str,
    provider_key: str,
    model_key: str,
    config_key_provider: str,
    config_key_model: str,
):
    st.sidebar.subheader(f"{label_prefix} LLM")

    # Provider Selection
    current_provider = current_config.get(config_key_provider, "gemini")
    provider = st.sidebar.selectbox(
        "Provider",
        ["gemini", "ollama"],
        index=0 if current_provider == "gemini" else 1,
        key=provider_key,
    )

    # Model Selection Logic
    available_models = list_available_models(provider)

    if not available_models:
        if provider == "ollama":
            st.sidebar.warning("⚠️ No Ollama models found. Is Ollama running?")
        else:
            st.sidebar.warning("⚠️ No models available.")
        # Fallback options
        model_options = ["Unavailable"]
        default_index = 0
        disabled = True
    else:
        model_options = available_models
        disabled = False
        saved_model = current_config.get(config_key_model)
        if saved_model in model_options:
            default_index = model_options.index(saved_model)
        else:
            default_index = 0

    model = st.sidebar.selectbox(
        "Model Name",
        model_options,
        index=default_index,
        disabled=disabled,
        key=model_key,
    )

    return provider, model


# Render Selectors
hp_provider, hp_model = render_model_selector(
    "Honeypot", "hp_provider", "hp_model", "honeypot_provider", "honeypot_model"
)
an_provider, an_model = render_model_selector(
    "Analyst", "an_provider", "an_model", "analysis_provider", "analysis_model"
)
gen_provider, gen_model = render_model_selector(
    "Test Generator",
    "gen_provider",
    "gen_model",
    "generator_provider",
    "generator_model",
)


# Determine if settings are dirty (unsaved changes)
has_changes = False


# Helper to safely get session state or default
def get_state(key, default):
    return st.session_state.get(key, default)


# Check Honeypot changes
if get_state(
    "hp_provider", current_config.get("honeypot_provider")
) != current_config.get("honeypot_provider"):
    has_changes = True
if get_state("hp_model", current_config.get("honeypot_model")) != current_config.get(
    "honeypot_model"
):
    has_changes = True

# Check Analyst changes
if get_state(
    "an_provider", current_config.get("analysis_provider")
) != current_config.get("analysis_provider"):
    has_changes = True
if get_state("an_model", current_config.get("analysis_model")) != current_config.get(
    "analysis_model"
):
    has_changes = True

# Check Generator changes
if get_state(
    "gen_provider", current_config.get("generator_provider")
) != current_config.get("generator_provider"):
    has_changes = True
if get_state("gen_model", current_config.get("generator_model")) != current_config.get(
    "generator_model"
):
    has_changes = True

# Save Button with Feedback
if has_changes:
    st.sidebar.warning("⚠️ You have unsaved changes.")
    if st.sidebar.button("Save Settings"):
        new_config = {
            "honeypot_provider": hp_provider,
            "honeypot_model": hp_model,
            "analysis_provider": an_provider,
            "analysis_model": an_model,
            "generator_provider": gen_provider,
            "generator_model": gen_model,
        }
        save_config(new_config)
        st.sidebar.success("Configuration updated!")
        st.rerun()
else:
    st.sidebar.button("Save Settings", disabled=True)

st.sidebar.divider()

# Sidebar: Refresh Button
if st.sidebar.button("Refresh Data"):
    st.rerun()


def render_live_feed():
    st.header("Live Threat Feed")
    # Load data
    df = load_data()

    # --- Main Area Filters ---
    col_filter1, col_filter2 = st.columns([2, 1])
    with col_filter1:
        threat_filter = st.radio(
            "Filter by Threat Level", ["All", "Critical", "Info"], horizontal=True
        )

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
            st.plotly_chart(fig_ts)

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
                st.plotly_chart(fig_pie)

        with col_trend2:
            # 3. HTTP Method Distribution
            if "method" in df.columns:
                method_counts = df["method"].value_counts().reset_index()
                method_counts.columns = ["Method", "Count"]
                fig_bar = px.bar(
                    method_counts,
                    x="Method",
                    y="Count",
                    title="HTTP Method Distribution",
                )
                st.plotly_chart(fig_bar)

        # 4. Top Attack Tools (User-Agents)
        if "tool_signature" in df.columns:
            tool_counts = df["tool_signature"].value_counts().reset_index()
            tool_counts.columns = ["Tool", "Count"]
            fig_tools = px.bar(
                tool_counts,
                y="Tool",
                x="Count",
                title="Top Attack Tools",
                orientation="h",
            )
            # Invert y-axis to show top tools at the top
            fig_tools.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_tools)

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
            width="stretch",
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time", format="YYYY-MM-DD HH:mm:ss"
                ),
                "threat_level": st.column_config.TextColumn(
                    "Threat Level", help="Based on heuristic keywords"
                ),
                "method": st.column_config.TextColumn("Method"),
                "path": st.column_config.TextColumn("Path"),
                "ip": st.column_config.TextColumn("Source IP"),
                "response_time_ms": st.column_config.NumberColumn(
                    "Latency (ms)", format="%.2f"
                ),
                "status_code": st.column_config.NumberColumn("Status"),
                "tool_signature": st.column_config.TextColumn("Tool/Agent"),
            },
        )

        # Payload Inspector
        if event.selection.rows:
            selected_index = event.selection.rows[0]
            selected_row = display_df.iloc[selected_index]

            st.divider()
            st.subheader("🔍 Interaction Details")

            # Display Provider/Model Metadata
            provider = selected_row.get("provider", "Unknown")
            model = selected_row.get("model", "Unknown")
            st.info(f"**Generated by:** {provider} / {model}")

            tab_req, tab_res, tab_rag, tab_ai = st.tabs(
                ["Request", "Response", "RAG & State", "AI Analysis"]
            )

            with tab_req:
                req_col1, req_col2, req_col3 = st.columns(3)
                req_col1.metric("Method", selected_row.get("method", "N/A"))
                req_col2.metric("Path", selected_row.get("path", "N/A"))
                req_col3.metric("Source IP", selected_row.get("ip", "N/A"))

                st.markdown("#### Request Headers")
                headers_data = selected_row.get("headers", {})
                with st.expander("View Headers", expanded=False):
                    if isinstance(headers_data, dict) and headers_data:
                        st.json(headers_data)
                    else:
                        st.write(headers_data if headers_data else "No Headers")

                st.markdown("#### Request Body")
                req_body = selected_row.get("body", None)
                if isinstance(req_body, dict):
                    st.json(req_body)
                elif isinstance(req_body, str) and req_body.strip():
                    try:
                        st.json(json.loads(req_body))
                    except json.JSONDecodeError:
                        st.code(req_body, language="text")
                elif req_body:
                    st.write(req_body)
                else:
                    st.info("Empty Request Payload")

            with tab_res:
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Status Code", selected_row.get("status_code", "N/A"))
                latency = selected_row.get("response_time_ms")
                res_col2.metric(
                    "Latency", f"{latency:.2f} ms" if pd.notnull(latency) else "N/A"
                )

                st.markdown("#### Generated Response")
                res_body = selected_row.get("response", None)
                if isinstance(res_body, dict):
                    st.json(res_body)
                elif isinstance(res_body, str) and res_body.strip():
                    try:
                        st.json(json.loads(res_body))
                    except json.JSONDecodeError:
                        st.code(res_body, language="text")
                elif res_body:
                    st.write(res_body)
                else:
                    st.info("No Response Body")

            with tab_rag:
                st.markdown("#### RAG Context")
                rag_query = selected_row.get("rag_query", "None")
                st.write(f"**Query used:** `{rag_query}`")

                rag_context = selected_row.get("context", "")
                with st.expander("View Retrieved Context", expanded=False):
                    if rag_context:
                        st.code(rag_context, language="yaml")
                    else:
                        st.info("No Context Retrieved")

                st.markdown("#### State Actions")
                state_actions = selected_row.get("state_actions", [])
                if state_actions and isinstance(state_actions, list):
                    for action in state_actions:
                        action_type = action.get("action", "UNKNOWN")
                        scope = action.get("scope", "")
                        key = action.get("key", "")
                        val = action.get("value", "")
                        if action_type == "SET":
                            st.success(f"**{action_type}** `{scope}.{key}` = `{val}`")
                        elif action_type == "DELETE":
                            st.error(f"**{action_type}** `{scope}.{key}`")
                        else:
                            st.info(f"**{action_type}**: {action}")
                else:
                    st.info("No State Actions taken.")

            with tab_ai:
                st.markdown("#### 🤖 AI Forensic Analysis")

                # Create unique cache key
                ts_str = str(selected_row.get("timestamp", ""))
                src_ip = str(selected_row.get("ip", ""))
                path_str = str(selected_row.get("path", ""))
                cache_key = f"{ts_str}_{src_ip}_{path_str}"

                cache = load_cache()
                cached_entry = cache.get(cache_key)

                # Helper to render the nice UI
                def render_analysis_ui(data, timestamp_str=None):
                    intent = data.get("intent", "Unknown Intent")
                    severity = data.get("severity", "Unknown")
                    category = data.get("category", "Unknown")
                    explanation = data.get("explanation", "No explanation provided.")

                    severity_colors = {
                        "High": "red",
                        "Medium": "orange",
                        "Low": "green",
                        "Unknown": "grey",
                    }
                    color = severity_colors.get(severity, "grey")

                    st.markdown(f"**Intent:** {intent}")

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
                    if "data" in cached_entry and "timestamp" in cached_entry:
                        analysis_data = cached_entry["data"]
                        analysis_ts = cached_entry["timestamp"]
                    else:
                        analysis_data = cached_entry
                        analysis_ts = "Previously"

                    render_analysis_ui(analysis_data, analysis_ts)

                    if st.button("🔄 Re-analyze (Update Models)"):
                        del cache[cache_key]
                        try:
                            with open(CACHE_FILE, "w") as f:
                                json.dump(cache, f, indent=2)
                        except IOError:
                            pass
                        st.rerun()

                else:
                    if st.button("🤖 Analyze with AI"):
                        with st.spinner("Analyzing threat signature..."):
                            body_content = selected_row.get("body", "No Body")
                            prompt = (
                                f"Act as a Tier 3 Security Analyst. Analyze this HTTP request.\n"
                                f"Input: {body_content}\n"
                                f"Path: {path_str}\n"
                                f"Output JSON with keys: 'intent' (short summary), 'severity' (Low/Med/High), "
                                f"'category' (e.g. SQLi, Recon), and 'explanation'."
                            )

                            try:
                                raw_response = generate_response(
                                    prompt,
                                    provider_type=current_config.get(
                                        "analysis_provider", "gemini"
                                    ),
                                    model_name=current_config.get(
                                        "analysis_model", "gemini-1.5-flash"
                                    ),
                                )
                                analysis_result = clean_llm_response(raw_response)

                                from datetime import datetime

                                entry = {
                                    "data": analysis_result,
                                    "timestamp": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                }
                                save_to_cache(cache_key, entry)
                                render_analysis_ui(analysis_result, entry["timestamp"])

                            except LLMRateLimitError as e:
                                st.error("⚠️ Gemini API Quota Exceeded!")
                                st.info(
                                    "Tip: You can switch to a local model (Ollama) in the 'Model Configuration' sidebar."
                                )
                                if e.retry_after:
                                    st.warning(
                                        f"Google suggests retrying in: {e.retry_after}"
                                    )

                            except Exception as e:
                                st.error(f"Analysis failed: {e}")


def render_attack_simulator():
    st.header("⚔️ Attack Simulator")
    st.caption("Generate synthetic attack scenarios using AI to test the honeypot.")

    # Try importing Generator
    try:
        from dashboard.test_generator import TestGenerator
    except ImportError:
        # Fallback for when running form different root
        try:
            from test_generator import TestGenerator
        except ImportError:
            st.error("Could not import TestGenerator. Please check file structure.")
            st.stop()

    # Inputs
    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        num_cases = st.number_input(
            "Number of Test Cases", min_value=1, max_value=10, value=3
        )

    if st.button("Generate Attack Scenarios"):
        with st.spinner(
            f"Generating {num_cases} scenarios with {current_config.get('generator_provider')}..."
        ):
            try:
                generator = TestGenerator()
                test_cases = generator.generate_test_cases(
                    n=num_cases,
                    provider=current_config.get("generator_provider", "gemini"),
                    model=current_config.get("generator_model", "gemini-2.5-flash"),
                )

                if test_cases:
                    # Convert to DataFrame for editing
                    st.session_state["test_cases_df"] = pd.DataFrame(test_cases)
                    st.success(f"Generated {len(test_cases)} attacks!")
                else:
                    st.error("Generation failed. Check logs or provider status.")

            except LLMRateLimitError as e:
                st.error("⚠️ Gemini API Quota Exceeded!")
                st.info(
                    "Tip: You can switch to a local model (Ollama) in the 'Model Configuration' sidebar to continue testing without limits."
                )
                if e.retry_after:
                    st.warning(f"Google suggests retrying in: {e.retry_after}")

            except Exception as e:
                st.error(f"Generation failed: {e}")

    # Editor Section
    if "test_cases_df" in st.session_state:
        st.divider()
        st.subheader("Review & Edit Scenarios")
        st.caption("Edit the attack details before launching (Mock launch).")

        edited_df = st.data_editor(
            st.session_state["test_cases_df"],
            key="attack_editor",
            num_rows="dynamic",
            column_config={
                "method": st.column_config.SelectboxColumn(
                    "HTTP Method",
                    options=[
                        "GET",
                        "POST",
                        "PUT",
                        "DELETE",
                        "HEAD",
                        "OPTIONS",
                        "PATCH",
                    ],
                    help="Select the HTTP method",
                    required=True,
                ),
                "body": st.column_config.TextColumn(
                    "Request Body",
                    help="JSON string or plain text payload",
                    width="large",
                ),
                "headers": st.column_config.TextColumn(
                    "Headers", help="JSON string of headers", width="medium"
                ),
                "description": st.column_config.TextColumn("Intent / Description"),
            },
        )

        # Capture edits (st.data_editor updates state automatically via key, but return val is useful too)
        # Verify valid JSON in body/headers if we wanted, but for now just showing it.

        st.divider()

        if st.button("🚀 Launch Attacks", type="primary"):
            st.write("Preparing to launch attacks against `http://localhost:8000`...")

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_cases = len(edited_df)

            for i, row in edited_df.iterrows():
                # 1. Parse Data
                method = row.get("method", "GET").upper()
                path = str(row.get("path", "/")).strip()
                if not path.startswith("/"):
                    path = "/" + path

                url = f"http://localhost:8000{path}"

                # Headers Parsing
                headers_raw = row.get("headers", "{}")
                headers = {}
                if isinstance(headers_raw, dict):
                    headers = headers_raw
                else:
                    try:
                        headers = json.loads(str(headers_raw))
                    except json.JSONDecodeError:
                        headers = {}  # Fallback

                # Body Parsing (send as data string or json depending on content,
                # but requests 'data' param handles string well)
                body_raw = row.get("body", "")
                data_payload = body_raw
                if body_raw is None:
                    data_payload = ""
                elif isinstance(body_raw, dict):
                    data_payload = json.dumps(body_raw)

                # 2. Execute Request
                status_text.text(
                    f"Sending {method} {path} (waiting for AI... logic can take >10s)..."
                )
                start_time = time.time()
                try:
                    response = requests.request(
                        method=method,
                        url=url,
                        headers=headers,
                        data=data_payload,
                        timeout=60,  # Increased timeout for LLM latency
                    )
                    latency_ms = (time.time() - start_time) * 1000
                    status_code = response.status_code
                    response_text = response.text[:500]  # Truncate for display
                except requests.exceptions.RequestException as e:
                    latency_ms = 0
                    status_code = -1  # Error code
                    response_text = str(e)

                # 3. Record Result
                results.append(
                    {
                        "Method": method,
                        "Path": path,
                        "Status": status_code,
                        "Latency (ms)": round(latency_ms, 2),
                        "Response Preview": response_text,
                    }
                )

                # Rate Limiting & Progress
                time.sleep(0.1)
                progress_bar.progress((i + 1) / total_cases)

            status_text.text("Execution Complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            # --- Results Visualization ---
            if results:
                st.subheader("Execution Results")

                results_df = pd.DataFrame(results)

                # Calculate Metrics
                success_count = len(results_df[results_df["Status"] == 200])
                failure_count = len(results_df[results_df["Status"] != 200])

                m1, m2 = st.columns(2)
                m1.metric("Successful (200 OK)", f"{success_count}/{total_cases}")
                m2.metric("Blocked/Error", f"{failure_count}/{total_cases}")

                # Display DataFrame with Status Badges logic
                # Streamlit doesn't have native "badges" in dataframe yet, but we can map status to emoji
                def status_emoji(code):
                    if code == 200:
                        return "✅ 200 OK"
                    if code == -1:
                        return "❌ Error"
                    if 400 <= code < 500:
                        return f"⚠️ {code} Client Error"
                    if 500 <= code < 600:
                        return f"🔥 {code} Server Error"
                    return f"❓ {code}"

                results_df["Status"] = results_df["Status"].apply(status_emoji)

                st.dataframe(results_df, use_container_width=True)

                st.info(
                    "Attacks sent! Go to the **Live Threat Feed** tab to see how the Honeypot logged and analyzed these requests."
                )


def render_rag_inspector():
    st.header("🔍 RAG Inspector")
    st.caption(
        "Test and inspect the Retrieval-Augmented Generation (RAG) system directly."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Simulated Request (e.g., POST /api/login)", value="GET /"
        )
    with col2:
        top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=3)

    if st.button("Run Inspection", type="primary"):
        with st.spinner("Querying RAG database..."):
            try:
                response = requests.post(
                    "http://localhost:8000/api/rag-inspect",
                    json={"query": query, "top_k": top_k},
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()

                m1, m2 = st.columns(2)
                m1.metric("Retrieval Latency", f"{data.get('latency_ms', 0)} ms")
                m2.metric("Chunks Retrieved", len(data.get("chunks", [])))

                for i, chunk in enumerate(data.get("chunks", [])):
                    with st.expander(
                        f"Rank {i + 1} - Index: {chunk['chunk_index']} | FAISS Distance: {chunk['faiss_distance']:.4f}"
                    ):
                        st.text(chunk["text"])
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to backend API: {e}")


# --- Main Routing ---
if page == "Live Threat Feed":
    render_live_feed()
elif page == "Attack Simulator":
    render_attack_simulator()
elif page == "RAG Inspector":
    render_rag_inspector()
