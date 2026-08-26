"""Multi-agent incident analysis graph: Diagnosis -> Root Cause <-> Reviewer -> Report, with escalation routing."""
import os
import json
from typing import TypedDict, List, Dict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from src.incident_store import IncidentStore

load_dotenv()

CONFIDENCE_ESCALATION_THRESHOLD = 0.6
MAX_REVISIONS = 1


class AnalysisState(TypedDict):
    error_event: Dict
    retrieved_incidents: List[Dict]
    root_cause: str
    confidence: float
    recommended_action: str
    report: str
    escalate: bool
    reviewer_feedback: Optional[str]
    reviewer_verdict: str
    revision_count: int


def _get_llm(model_name: str = "openai/gpt-oss-120b"):
    return ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=model_name)


def build_graph(incident_store: IncidentStore, llm=None):
    llm = llm or _get_llm()

    def diagnosis_node(state: AnalysisState) -> AnalysisState:
        error_text = " ".join(state["error_event"]["messages"])
        retrieved = incident_store.search(error_text, top_k=3)
        return {**state, "retrieved_incidents": retrieved}

    def root_cause_node(state: AnalysisState) -> AnalysisState:
        error_event = state["error_event"]
        error_text = "\n".join(error_event["messages"])
        context = "\n\n".join(
            f"[{inc['id']}] system={inc['system']} severity={inc['severity']}\n"
            f"error_signature: {inc['error_signature']}\n"
            f"root_cause: {inc['root_cause']}\n"
            f"resolution: {inc['resolution']}"
            for inc in state["retrieved_incidents"]
        )
        revision_note = ""
        if state.get("reviewer_feedback"):
            revision_note = f"""
A reviewing agent already checked a previous version of your diagnosis and found it lacking:
"{state['reviewer_feedback']}"
Revise your diagnosis to address this feedback directly."""

        prompt = f"""You are a senior data pipeline on-call engineer diagnosing a production incident.

New error event (source: {error_event['source']}, first seen: {error_event['first_ts']}):
{error_text}

Similar historical incidents retrieved from the knowledge base:
{context}
{revision_note}

Based on the historical incidents, respond in strict JSON with these keys:
- "root_cause": your best hypothesis for the root cause (1-2 sentences)
- "recommended_action": concrete next step the on-call engineer should take (1-2 sentences)
- "confidence": a float between 0 and 1 for how confident you are, given how well the historical incidents match

Only output the JSON object, nothing else."""
        response = llm.invoke([prompt])
        raw = response.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {
                "root_cause": raw,
                "recommended_action": "Manual review needed — model did not return structured output.",
                "confidence": 0.3,
            }
        return {
            **state,
            "root_cause": parsed.get("root_cause", "Unknown"),
            "recommended_action": parsed.get("recommended_action", "Escalate to on-call."),
            "confidence": float(parsed.get("confidence", 0.3)),
        }

    def reviewer_node(state: AnalysisState) -> AnalysisState:
        error_event = state["error_event"]
        evidence_ids = ", ".join(inc["id"] for inc in state["retrieved_incidents"]) or "none"
        prompt = f"""You are a skeptical senior engineer reviewing a colleague's incident diagnosis before it ships.

Original error: {' '.join(error_event['messages'])}

Evidence used (historical incident IDs): {evidence_ids}

Colleague's diagnosis:
- root_cause: {state['root_cause']}
- recommended_action: {state['recommended_action']}
- self-reported confidence: {state['confidence']}

Judge only whether the root_cause is actually supported by the evidence IDs listed, and whether the
recommended_action is concrete and actionable (not vague like "investigate further"). Respond in strict
JSON with these keys:
- "verdict": "approve" or "revise"
- "feedback": one sentence explaining your verdict (empty string if approved)

Only output the JSON object, nothing else."""
        response = llm.invoke([prompt])
        raw = response.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fail closed: an unparseable review is inconclusive, not a pass.
            # Treating it as "revise" forces one retry instead of silently
            # letting a broken review look identical to an approval.
            parsed = {"verdict": "revise", "feedback": "Reviewer response was not valid JSON; treating as inconclusive."}

        verdict = parsed.get("verdict", "revise")
        return {
            **state,
            "reviewer_verdict": verdict,
            "reviewer_feedback": parsed.get("feedback", "") if verdict == "revise" else None,
        }

    def route_after_review(state: AnalysisState) -> str:
        if state["reviewer_verdict"] == "revise" and state["revision_count"] < MAX_REVISIONS:
            return "revise"
        return "proceed"

    def increment_revision(state: AnalysisState) -> AnalysisState:
        return {**state, "revision_count": state["revision_count"] + 1}

    def report_node(state: AnalysisState) -> AnalysisState:
        error_event = state["error_event"]
        escalate = state["confidence"] < CONFIDENCE_ESCALATION_THRESHOLD
        top_matches = ", ".join(inc["id"] for inc in state["retrieved_incidents"])
        lines = [
            f"INCIDENT REPORT — source: {error_event['source']}",
            f"Time window: {error_event['first_ts']} - {error_event['last_ts']}",
            f"Error: {error_event['messages'][0]}",
            f"Similar past incidents: {top_matches}",
            f"Root cause hypothesis: {state['root_cause']}",
            f"Recommended action: {state['recommended_action']}",
            f"Confidence: {state['confidence']:.2f}",
            f"Reviewer agent verdict: {state['reviewer_verdict']} "
            f"({state['revision_count']} revision(s) requested)",
            f"Status: {'ESCALATED to human on-call (low confidence)' if escalate else 'Auto-analyzed, no escalation needed'}",
        ]
        return {**state, "report": "\n".join(lines), "escalate": escalate}

    graph = StateGraph(AnalysisState)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("root_cause", root_cause_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("increment_revision", increment_revision)
    graph.add_node("report", report_node)

    graph.set_entry_point("diagnosis")
    graph.add_edge("diagnosis", "root_cause")
    graph.add_edge("root_cause", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"revise": "increment_revision", "proceed": "report"},
    )
    graph.add_edge("increment_revision", "root_cause")
    graph.add_edge("report", END)

    return graph.compile()


def analyze_event(compiled_graph, error_event: Dict) -> AnalysisState:
    initial_state: AnalysisState = {
        "error_event": error_event,
        "retrieved_incidents": [],
        "root_cause": "",
        "confidence": 0.0,
        "recommended_action": "",
        "report": "",
        "escalate": False,
        "reviewer_feedback": None,
        "reviewer_verdict": "",
        "revision_count": 0,
    }
    return compiled_graph.invoke(initial_state)
