"""Streamlit dashboard for the Incident Management using Multi-Agent RAG."""
import os
import tempfile
import streamlit as st

from src.log_parser import extract_error_events, parse_log_lines
from src.incident_store import IncidentStore
from src.agent_graph import build_graph, analyze_event, CONFIDENCE_ESCALATION_THRESHOLD

INDEX_DIR = "faiss_incident_store"
DEFAULT_LOG_PATH = "data/sample_pipeline.log"

st.set_page_config(page_title="Incident Management using Multi-Agent RAG", layout="wide")


@st.cache_resource
def load_store():
    store = IncidentStore(INDEX_DIR)
    if not os.path.exists(os.path.join(INDEX_DIR, "faiss.index")):
        store.build_from_jsonl("data/incident_kb.jsonl")
    else:
        store.load()
    return store


@st.cache_resource
def load_graph(_store):
    return build_graph(_store)


st.title("🛰️ Incident Management using Multi-Agent RAG")
st.caption(
    "Multi-agent RAG system: parses pipeline logs, retrieves similar historical incidents, "
    "diagnoses root cause, and escalates when confidence is low."
)

with st.sidebar:
    st.header("Log source")
    source_choice = st.radio("Choose input", ["Use sample log", "Upload a log file"])

    log_path = DEFAULT_LOG_PATH
    if source_choice == "Upload a log file":
        uploaded = st.file_uploader("Upload pipeline log (.log/.txt)", type=["log", "txt"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
            tmp.write(uploaded.read())
            tmp.close()
            log_path = tmp.name
        else:
            st.info("Upload a file or switch to the sample log.")
            st.stop()

    st.divider()
    st.header("Knowledge base")
    store = load_store()
    st.metric("Incidents in KB", len(store.incidents))
    with st.expander("View knowledge base"):
        for inc in store.incidents:
            st.markdown(f"**{inc['id']}** ({inc['system']}, {inc['severity']}) — {inc['error_signature']}")

    st.divider()
    run_button = st.button("▶ Run analysis", type="primary", use_container_width=True)

if run_button:
    graph = load_graph(store)
    events = extract_error_events(log_path)

    if not events:
        parsed_lines = parse_log_lines(log_path)
        if not parsed_lines:
            st.error(
                "Couldn't recognize any lines in this log. Expected format:\n\n"
                "`YYYY-MM-DD HH:MM:SS  LEVEL  [source]  message`\n\n"
                "e.g. `2026-08-25 02:04:47 ERROR [spark_executor_3] Failed to connect...`\n\n"
                "This log may use a different timestamp, level, or source format than the parser expects."
            )
        else:
            st.warning(
                f"Recognized {len(parsed_lines)} log line(s) but none were ERROR level — "
                "no incidents to analyze."
            )
        st.stop()

    st.success(f"Found {len(events)} incident event(s). Confidence escalation threshold: {CONFIDENCE_ESCALATION_THRESHOLD}")

    escalated_count = 0
    progress = st.progress(0.0, text="Analyzing events...")

    results = []
    for i, event in enumerate(events):
        result = analyze_event(graph, event)
        results.append(result)
        progress.progress((i + 1) / len(events), text=f"Analyzing event {i + 1}/{len(events)}...")
    progress.empty()

    for i, result in enumerate(results, 1):
        event = result["error_event"]
        escalate = result["escalate"]
        if escalate:
            escalated_count += 1

        border_color = "🔴" if escalate else "🟢"
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"{border_color} Event {i} — {event['source']}")
                st.caption(f"{event['first_ts']} → {event['last_ts']}")
                for msg in event["messages"]:
                    st.code(msg, language="text")
            with col2:
                st.metric("Confidence", f"{result['confidence']:.2f}")
                if escalate:
                    st.error("ESCALATED to on-call")
                else:
                    st.success("Auto-analyzed")

            st.markdown(f"**Root cause hypothesis:** {result['root_cause']}")
            st.markdown(f"**Recommended action:** {result['recommended_action']}")

            reviewer_icon = "🔁" if result["revision_count"] > 0 else "✅"
            reviewer_line = f"{reviewer_icon} Reviewer agent: **{result['reviewer_verdict']}**"
            if result["revision_count"] > 0:
                reviewer_line += f" (requested {result['revision_count']} revision(s) before approving)"
            st.caption(reviewer_line)
            if result.get("reviewer_feedback"):
                st.caption(f"Last reviewer feedback: _{result['reviewer_feedback']}_")

            with st.expander("Retrieved similar incidents (RAG evidence)"):
                for inc in result["retrieved_incidents"]:
                    st.markdown(
                        f"- **{inc['id']}** ({inc['system']}, severity={inc['severity']}, "
                        f"distance={inc['distance']:.3f}) — {inc['error_signature']}\n"
                        f"  - *Historical root cause:* {inc['root_cause']}\n"
                        f"  - *Historical resolution:* {inc['resolution']}"
                    )

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total incidents", len(results))
    c2.metric("Auto-analyzed", len(results) - escalated_count)
    c3.metric("Escalated", escalated_count)
else:
    st.info("Configure a log source in the sidebar and click **Run analysis** to start.")
