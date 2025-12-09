import streamlit as st
import sqlparse
from sql_metadata import Parser
import sqlglot
import sqlite3
import pandas as pd
import re
import json
import os

# New OpenAI client import (v1.0+)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ------------------------------
# UI SETUP
# ------------------------------
st.set_page_config(page_title="SQL Query Optimizer Chatbot | Kabir Puri", layout="wide")
st.title("⚡ SQL Query Optimizer Chatbot")
st.subheader("LLM-powered SQL explainer + index advisor + rewrite engine")

st.markdown(
    """
This tool analyzes your SQL query, suggests indexes, proposes safe rewrites, and optionally provides a polished optimization plan using OpenAI.
**AI features are enabled only if an OpenAI API key is configured in Streamlit Secrets.**
"""
)

# ---------- OPENAI SECRETS (no input box) ----------
use_openai = False
client = None

if OpenAI is None:
    st.warning("OpenAI client library not available. Ensure `openai>=1.0.0` is listed in requirements.txt.")
else:
    if "OPENAI_API_KEY" in st.secrets:
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            use_openai = True
            st.success("AI Mode: Enabled")
        except Exception as e:
            use_openai = False
            st.error(f"Failed to initialize OpenAI client: {e}")
    else:
        st.info("AI Mode: Disabled — add OPENAI_API_KEY in Streamlit Secrets to enable polished explanations.")

# ------------------------------
# INPUT SECTION
# ------------------------------
sql_input = st.text_area("Paste your SQL query here:", height=200)

uploaded_files = st.file_uploader(
    "Upload optional CSV or DDL (.sql) files to build a temporary SQLite database (optional for EXPLAIN).",
    type=["csv", "sql"],
    accept_multiple_files=True,
)

# ------------------------------
# FUNCTIONS
# ------------------------------
def parse_sql(user_sql):
    """Extract tables, columns, where columns, joins, etc."""
    try:
        parser = Parser(user_sql)
        tables = parser.tables
        columns = parser.columns
        where_cols = parser.columns_dict.get("where", [])
        return tables, columns, where_cols
    except Exception:
        return [], [], []


def recommend_indexes(tables, columns, where_cols):
    """Heuristic index suggestions."""
    suggestions = []
    index_recs = []

    suggestions.append("Avoid SELECT * in production; list required columns to reduce IO.")
    suggestions.append("Use LIMIT for exploratory queries to avoid large result sets.")
    suggestions.append("Prefer explicit JOIN types and ensure join/filter columns are indexed where appropriate.")

    # Basic heuristics for each table
    for t in tables:
        candidate_cols = []

        # add WHERE columns that reference this table (or generic filter columns)
        for c in where_cols:
            # c might be "table.col" or "col"
            if "." in c:
                table_part, col_part = c.split(".", 1)
                if table_part.lower() == t.lower():
                    candidate_cols.append(col_part)
            else:
                candidate_cols.append(c)

        # Add common id/time columns if present in parsed columns
        for cc in ["id", "user_id", "created_at", "updated_at"]:
            if cc in columns and cc not in candidate_cols:
                candidate_cols.append(cc)

        # Limit to unique top-3 columns
        candidate_cols = list(dict.fromkeys(candidate_cols))[:3]

        if candidate_cols:
            index_recs.append({"table": t, "columns": candidate_cols})
            suggestions.append(f"Consider index on `{t}({', '.join(candidate_cols)})` for filtering and joins.")

    return suggestions, index_recs


def run_sqlite_explain(user_sql, uploaded_files):
    """Load user files into SQLite and run EXPLAIN QUERY PLAN."""
    conn = sqlite3.connect("temp.sqlite")

    for file in uploaded_files:
        try:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file)
                tname = os.path.splitext(file.name)[0]
                df.to_sql(tname, conn, if_exists="replace", index=False)
            elif file.name.lower().endswith(".sql"):
                ddl = file.read().decode("utf-8")
                conn.executescript(ddl)
        except Exception as e:
            # continue on error for any single file
            st.warning(f"Failed to load {file.name}: {e}")

    try:
        cur = conn.cursor()
        cur.execute(f"EXPLAIN QUERY PLAN {user_sql}")
        rows = cur.fetchall()
        return rows
    except Exception as e:
        return [("EXPLAIN failed", str(e))]


def produce_openai_explanation(sql, suggestions, indexes):
    """Polished explanation using new OpenAI API surface."""
    if not use_openai or client is None:
        return "OpenAI not configured. To enable polished explanations, add OPENAI_API_KEY to Streamlit Secrets."

    prompt = f"""
You are a senior SQL performance engineer. Analyze this SQL query and provide:
1) A concise description of what the query does.
2) Top performance risks and why.
3) A prioritized optimization plan (3–6 steps).
4) Example CREATE INDEX statements where relevant.
5) A short validation checklist to confirm improvements.

SQL:
{sql}

Heuristic suggestions:
{json.dumps(suggestions)}

Index recommendations:
{json.dumps(indexes)}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        # New response structure: choices[0].message.content
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI error: {e}"


# ------------------------------
# ANALYSIS EXECUTION
# ------------------------------
if st.button("Analyze SQL"):
    if not sql_input.strip():
        st.error("Please paste an SQL query.")
        st.stop()

    st.header("🔍 SQL Analysis Results")

    # Normalize
    formatted_sql = sqlparse.format(sql_input, reindent=True, keyword_case="upper")
    st.subheader("Formatted Query")
    st.code(formatted_sql, language="sql")

    # Parse
    tables, columns, where_cols = parse_sql(sql_input)

    st.subheader("Detected Tables")
    st.write(tables)

    st.subheader("Detected Columns (sample)")
    st.write(columns[:50])

    st.subheader("WHERE Columns")
    st.write(where_cols)

    # Recommendations
    suggestions, index_recs = recommend_indexes(tables, columns, where_cols)

    st.subheader("💡 Optimization Suggestions")
    for s in suggestions:
        st.write("- " + s)

    st.subheader("📌 Index Recommendations")
    for rec in index_recs:
        st.code(
            f"CREATE INDEX idx_{rec['table']}_{'_'.join(rec['columns'])} ON {rec['table']}({', '.join(rec['columns'])});"
        )

    # Optional SQLite EXPLAIN
    if uploaded_files:
        st.subheader("🧪 SQLite EXPLAIN QUERY PLAN")
        rows = run_sqlite_explain(sql_input, uploaded_files)
        st.write(rows)

    # Optional OpenAI polishing
    if use_openai:
        st.subheader("✨ Polished Explanation (AI)")
        explanation = produce_openai_explanation(sql_input, suggestions, index_recs)
        st.write(explanation)
    else:
        st.info("Polished AI explanation unavailable. Set OPENAI_API_KEY in Streamlit Secrets to enable.")

    st.success("Analysis complete!")
