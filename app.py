import streamlit as st
import sqlparse
from sql_metadata import Parser
import sqlglot
import sqlite3
import pandas as pd
import re
import json
import os

# Optional OpenAI
use_openai = False
try:
    import openai
except:
    pass

# ------------------------------
# UI SETUP
# ------------------------------
st.set_page_config(page_title="SQL Query Optimizer Chatbot | Kabir Puri", layout="wide")
st.title("⚡ SQL Query Optimizer Chatbot")
st.subheader("LLM-powered SQL explainer + index advisor + rewrite engine")

st.markdown("""
This tool analyzes your SQL query, explains what it does, detects performance problems, 
suggests indexes, proposes rewrites, and optionally generates polished explanations using OpenAI.

**Created by: Kabir Puri**  
**Specialization:** SQL Optimization, AI/ML, Data Engineering
""")

# ------------------------------
# INPUT SECTION
# ------------------------------

sql_input = st.text_area("Paste your SQL query here:", height=200)

uploaded_files = st.file_uploader(
    "Upload optional CSV or DDL (.sql) files to build a temporary SQLite database (optional for EXPLAIN).",
    type=["csv", "sql"],
    accept_multiple_files=True
)

openai_key = st.text_input("OpenAI API Key (optional)", type="password")

if openai_key:
    try:
        openai.api_key = openai_key
        use_openai = True
        st.success("OpenAI enabled.")
    except Exception as e:
        st.error(f"Error initializing OpenAI: {e}")
        use_openai = False


# ------------------------------
# FUNCTIONS
# ------------------------------

def parse_sql(user_sql):
    """Extract tables, columns, where columns, joins, etc."""
    try:
        parser = Parser(user_sql)
        tables = parser.tables
        columns = parser.columns
        where_cols = parser.columns_dict.get('where', [])
        return tables, columns, where_cols
    except Exception:
        return [], [], []


def recommend_indexes(tables, columns, where_cols):
    """Heuristic index suggestions."""
    suggestions = []
    index_recs = []

    # Basic suggestions
    suggestions.append("Avoid SELECT *; specify only required columns.")
    suggestions.append("Add LIMIT for exploration queries.")
    suggestions.append("Ensure JOIN conditions use indexed columns.")

    for t in tables:
        candidate_cols = []

        # WHERE columns
        for c in where_cols:
            candidate_cols.append(c.split('.')[-1])

        # ususal suspects
        for c in ["id", "user_id", "created_at"]:
            if c in columns:
                candidate_cols.append(c)

        candidate_cols = list(dict.fromkeys(candidate_cols))[:3]

        if candidate_cols:
            index_recs.append({"table": t, "columns": candidate_cols})
            suggestions.append(
                f"Consider index on `{t}({', '.join(candidate_cols)})` for filtering."
            )

    return suggestions, index_recs


def run_sqlite_explain(user_sql, uploaded_files):
    """Load user files into SQLite and run EXPLAIN QUERY PLAN."""
    conn = sqlite3.connect("temp.sqlite")

    for file in uploaded_files:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
            tname = os.path.splitext(file.name)[0]
            df.to_sql(tname, conn, if_exists='replace', index=False)

        elif file.name.endswith(".sql"):
            ddl = file.read().decode("utf-8")
            conn.executescript(ddl)

    try:
        cur = conn.cursor()
        cur.execute(f"EXPLAIN QUERY PLAN {user_sql}")
        rows = cur.fetchall()
        return rows
    except Exception as e:
        return [("EXPLAIN failed", str(e))]


def produce_openai_explanation(sql, suggestions, indexes):
    """Polished explanation with OpenAI."""
    prompt = f"""
You are a senior SQL performance engineer. Analyze the user SQL and provide:

1. What the query is doing.
2. The top performance risks.
3. 3–6 prioritized optimization steps.
4. Example CREATE INDEX statements.
5. A validation checklist.

SQL:
{sql}

Suggestions:
{json.dumps(suggestions)}

Index recommendations:
{json.dumps(indexes)}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response["choices"][0]["message"]["content"]
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
    formatted_sql = sqlparse.format(sql_input, reindent=True, keyword_case='upper')
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
        st.code(f"CREATE INDEX idx_{rec['table']}_{'_'.join(rec['columns'])} "
                f"ON {rec['table']}({', '.join(rec['columns'])});")

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

    st.success("Analysis complete!")


