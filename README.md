# Incident Management using Multi-Agent RAG

A multi-agent RAG system that watches data pipeline logs, retrieves similar historical
incidents, diagnoses the likely root cause, and drafts an incident report — escalating
to a human on-call when confidence is low.

Inspired by manual root-cause investigation workflows for distributed data pipelines
(Spark, Airflow, Kafka, HDFS, PostgreSQL).

## How it works

- **diagnosis_node** — embeds the new error and retrieves similar past incidents from a FAISS vector store (`data/incident_kb.jsonl`)
- **root_cause_node** — an LLM (Groq) reasons over the error + retrieved evidence, returning a root cause, recommended action, and confidence score
- **reviewer_node** — a second LLM agent reviews the diagnosis for support/specificity; on "revise" it sends feedback back to `root_cause_node` for one retry
- **report_node** — formats the final report and escalates to a human if confidence falls below 0.6

## Data

- `data/incident_kb.jsonl` — 15 synthetic past incidents (HDFS, Spark, Airflow, PostgreSQL, Kafka) used as RAG evidence
- `data/sample_pipeline.log` — a simulated log with injected failures to analyze

## Run it

```bash
pip install -r requirements.txt
python main.py
```

Requires a `GROQ_API_KEY` in `.env` (free tier at console.groq.com). Copy `.env.example` to `.env` and fill in your own key.

- `python main.py` — console output, builds the index automatically on first run
- `python build_index.py` — rebuild the index after editing `incident_kb.jsonl`
- `python -m streamlit run dashboard.py` — interactive web UI at `localhost:8501`

## Roadmap (v2 ideas)
- Add RAGAS-based evaluation harness to measure retrieval/diagnosis quality over time
