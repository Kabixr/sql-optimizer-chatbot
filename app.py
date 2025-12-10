import streamlit as st
import sqlparse
from sql_metadata import Parser
import sqlglot
import sqlite3
import pandas as pd
import re
import json
import os

# External DB connectors (import lazily inside functions where possible)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ------------------------------
# UI SETUP
# ------------------------------
st.set_page_config(page_title="SQL Query Optimizer Chatbot | Kabir Puri", layout="wide")
st.title("⚡ SQL Query Optimizer Chatbot")

# Signature
st.markdown("""
<div style='padding:8px 0; font-size:22px; font-weight:700;
            color:#f5c542; letter-spacing:1px;'>
✨ Created by <span style='color:#ffd86b;'>Kabir Puri</span>
</div>
""", unsafe_allow_html=True)

st.subheader("LLM-powered SQL explainer • index advisor • multi-engine EXPLAIN")

st.markdown(
    """
This tool analyzes your SQL query, suggests indexes, proposes safe rewrites,  
and optionally provides a polished optimization plan using OpenAI.

It now supports multiple SQL engines for EXPLAIN:

- SQLite (demo, CSV/DDL upload)  
- PostgreSQL  
- MySQL  
- Snowflake  
- BigQuery
"""
)

# ---------- OPENAI SECRETS ----------
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
# ENGINE SELECTION & CONNECTION
# ------------------------------
st.markdown("### 🔌 Execution Engine for `EXPLAIN`")

engine = st.selectbox(
    "Choose SQL engine for EXPLAIN / plan analysis:",
    ["SQLite (local demo)", "PostgreSQL", "MySQL", "Snowflake", "BigQuery"],
)

st.caption("Note: SQL parsing & heuristics work for all dialects; EXPLAIN runs only on the selected engine.")

conn_params = {}

if engine == "SQLite (local demo)":
    st.markdown("Using an in-memory SQLite database built from uploaded CSV/DDL files.")
else:
    st.markdown("Provide connection details (recommended: store in `st.secrets` in production).")

if engine == "PostgreSQL":
    conn_params["pg_dsn"] = st.text_input(
        "PostgreSQL DSN / connection string",
        help="Example: postgres://user:password@host:5432/dbname",
        value=st.secrets.get("PG_DSN", "") if "PG_DSN" in st.secrets else "",
    )

elif engine == "MySQL":
    conn_params["mysql_host"] = st.text_input("MySQL Host", value=st.secrets.get("MYSQL_HOST", "") if "MYSQL_HOST" in st.secrets else "localhost")
    conn_params["mysql_port"] = st.number_input("MySQL Port", value=int(st.secrets.get("MYSQL_PORT", 3306)) if "MYSQL_PORT" in st.secrets else 3306)
    conn_params["mysql_user"] = st.text_input("MySQL User", value=st.secrets.get("MYSQL_USER", "") if "MYSQL_USER" in st.secrets else "")
    conn_params["mysql_password"] = st.text_input("MySQL Password", type="password", value=st.secrets.get("MYSQL_PASSWORD", "") if "MYSQL_PASSWORD" in st.secrets else "")
    conn_params["mysql_db"] = st.text_input("MySQL Database", value=st.secrets.get("MYSQL_DB", "") if "MYSQL_DB" in st.secrets else "")

elif engine == "Snowflake":
    conn_params["sf_account"] = st.text_input("Snowflake Account", value=st.secrets.get("SF_ACCOUNT", "") if "SF_ACCOUNT" in st.secrets else "")
    conn_params["sf_user"] = st.text_input("Snowflake User", value=st.secrets.get("SF_USER", "") if "SF_USER" in st.secrets else "")
    conn_params["sf_password"] = st.text_input("Snowflake Password", type="password", value=st.secrets.get("SF_PASSWORD", "") if "SF_PASSWORD" in st.secrets else "")
    conn_params["sf_warehouse"] = st.text_input("Snowflake Warehouse", value=st.secrets.get("SF_WAREHOUSE", "") if "SF_WAREHOUSE" in st.secrets else "")
    conn_params["sf_database"] = st.text_input("Snowflake Database", value=st.secrets.get("SF_DATABASE", "") if "SF_DATABASE" in st.secrets else "")
    conn_params["sf_schema"] = st.text_input("Snowflake Schema", value=st.secrets.get("SF_SCHEMA", "") if "SF_SCHEMA" in st.secrets else "")

elif engine == "BigQuery":
    conn_params["bq_project"] = st.text_input("BigQuery Project ID", value=st.secrets.get("BQ_PROJECT", "") if "BQ_PROJECT" in st.secrets else "")
    conn_params["bq_dataset"] = st.text_input("(Optional) Default Dataset", value=st.secrets.get("BQ_DATASET", "") if "BQ_DATASET" in st.secrets else "")
    st.caption("Authentication is expected via service account / environment (e.g., JSON key in GCP or Streamlit secrets).")

# ------------------------------
# INPUT SECTION
# ------------------------------
sql_input = st.text_area("Paste your SQL query here:", height=220)

uploaded_files = st.file_uploader(
    "Upload optional CSV or DDL (.sql) files to build a temporary SQLite database (only used in SQLite mode).",
    type=["csv", "sql"],
    accept_multiple_files=True,
)

# ------------------------------
# CORE FUNCTIONS
# ------------------------------
def parse_sql(user_sql):
    """Extract tables, columns, where columns, etc."""
    try:
        parser = Parser(user_sql)
        tables = parser.tables
        columns = parser.columns
        where_cols = parser.columns_dict.get("where", [])
        return tables, columns, where_cols
    except Exception:
        return [], [], []


def recommend_indexes(tables, columns, where_cols):
    """
    Heuristic index suggestions.
    - Handles nested list/tuple results from parsers.
    - Normalizes values to strings and deduplicates while preserving order.
    """
    suggestions = []
    index_recs = []

    # Generic useful suggestions
    suggestions.append("Avoid SELECT * in production; list required columns to reduce IO.")
    suggestions.append("Use LIMIT for exploratory queries to avoid large result sets.")
    suggestions.append("Prefer explicit JOIN types and ensure join/filter columns are indexed where appropriate.")

    # Helper: normalize a value (list/tuple/set/other) -> flat list of strings
    def normalize_to_list(val):
        out = []
        if val is None:
            return out
        if isinstance(val, (list, tuple, set)):
            for v in val:
                if v is None:
                    continue
                out.append(str(v))
        else:
            out.append(str(val))
        return out

    for t in tables:
        candidate_cols = []

        # Add WHERE/filter columns that reference this table (or generic filters)
        for c in where_cols:
            normalized = normalize_to_list(c)
            for nc in normalized:
                # if parser returned "table.col" form
                if "." in nc:
                    table_part, col_part = nc.split(".", 1)
                    if table_part.lower() == str(t).lower():
                        candidate_cols.append(col_part)
                else:
                    candidate_cols.append(nc)

        # Add common id/time columns if present in parsed columns
        for cc in ["id", "user_id", "created_at", "updated_at"]:
            # columns may be list of strings; ensure we compare strings
            if any(str(col).lower() == cc for col in columns) and cc not in candidate_cols:
                candidate_cols.append(cc)

        # Flatten any nested items (safety) and convert everything to strings
        flat_cols = []
        for itm in candidate_cols:
            flat_cols.extend(normalize_to_list(itm))

        # Remove empty strings and deduplicate while preserving order
        seen = set()
        deduped = []
        for x in flat_cols:
            x_str = x.strip()
            if not x_str:
                continue
            if x_str not in seen:
                seen.add(x_str)
                deduped.append(x_str)

        # Limit to top-3 candidates
        candidate_cols = deduped[:3]

        if candidate_cols:
            index_recs.append({"table": t, "columns": candidate_cols})
            suggestions.append(
                f"Consider index on `{t}({', '.join(candidate_cols)})` for filtering and joins."
            )

    return suggestions, index_recs


def run_sqlite_explain(user_sql, uploaded_files):
    """SQLite: load user files into temp DB and run EXPLAIN QUERY PLAN."""
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
            st.warning(f"Failed to load {file.name}: {e}")

    try:
        cur = conn.cursor()
        cur.execute(f"EXPLAIN QUERY PLAN {user_sql}")
        rows = cur.fetchall()
        return rows
    except Exception as e:
        return [("EXPLAIN failed", str(e))]


def run_pg_explain(user_sql, dsn):
    import psycopg2
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute("EXPLAIN (FORMAT JSON) " + user_sql)
        rows = cur.fetchall()
        return rows
    except Exception as e:
        return [("PostgreSQL EXPLAIN error", str(e))]
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_mysql_explain(user_sql, params):
    import mysql.connector
    try:
        conn = mysql.connector.connect(
            host=params["mysql_host"],
            port=params["mysql_port"],
            user=params["mysql_user"],
            password=params["mysql_password"],
            database=params["mysql_db"],
        )
        cur = conn.cursor()
        cur.execute("EXPLAIN " + user_sql)
        rows = cur.fetchall()
        return rows
    except Exception as e:
        return [("MySQL EXPLAIN error", str(e))]
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_snowflake_explain(user_sql, params):
    import snowflake.connector
    try:
        ctx = snowflake.connector.connect(
            user=params["sf_user"],
            password=params["sf_password"],
            account=params["sf_account"],
            warehouse=params["sf_warehouse"],
            database=params["sf_database"],
            schema=params["sf_schema"],
        )
        cur = ctx.cursor()
        cur.execute("EXPLAIN USING TEXT " + user_sql)
        rows = cur.fetchall()
        return rows
    except Exception as e:
        return [("Snowflake EXPLAIN error", str(e))]
    finally:
        try:
            cur.close()
            ctx.close()
        except Exception:
            pass


def run_bigquery_explain(user_sql, params):
    from google.cloud import bigquery
    from google.cloud.bigquery import QueryJobConfig

    try:
        client = bigquery.Client(project=params["bq_project"] or None)
        job_config = QueryJobConfig(dry_run=True, use_query_cache=False)
        job = client.query(user_sql, job_config=job_config)
        # For dry runs, job is not executed; we get statistics instead
        stats = {
            "total_bytes_processed": job.total_bytes_processed,
            "slot_millis": job.slot_millis,
        }
        return [("BigQuery dry-run statistics", json.dumps(stats, indent=2))]
    except Exception as e:
        return [("BigQuery explain/dry-run error", str(e))]


def run_explain_plan(engine, sql, uploaded_files, params):
    if engine == "SQLite (local demo)":
        return run_sqlite_explain(sql, uploaded_files)
    elif engine == "PostgreSQL":
        if not params.get("pg_dsn"):
            return [("Config error", "No PostgreSQL DSN provided.")]
        return run_pg_explain(sql, params["pg_dsn"])
    elif engine == "MySQL":
        for key in ["mysql_host", "mysql_user", "mysql_db"]:
            if not params.get(key):
                return [("Config error", f"Missing MySQL param: {key}")]
        return run_mysql_explain(sql, params)
    elif engine == "Snowflake":
        for key in ["sf_account", "sf_user", "sf_password", "sf_warehouse", "sf_database", "sf_schema"]:
            if not params.get(key):
                return [("Config error", f"Missing Snowflake param: {key}")]
        return run_snowflake_explain(sql, params)
    elif engine == "BigQuery":
        if not params.get("bq_project"):
            return [("Config error", "Missing BigQuery project id.")]
        return run_bigquery_explain(sql, params)
    else:
        return [("Engine error", "Unknown engine.")]


def produce_openai_explanation(sql, suggestions, indexes):
    """Polished explanation using new OpenAI API."""
    if not use_openai or client is None:
        return "OpenAI not configured. Add OPENAI_API_KEY to Streamlit Secrets."

    prompt = f"""
You are a senior SQL performance engineer. Analyze this SQL query and provide:
1) A concise description of what the query does.
2) Top performance risks and why.
3) A prioritized optimization plan (3–6 steps).
4) Example CREATE INDEX statements where relevant.
5) A short validation checklist.

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

    formatted_sql = sqlparse.format(sql_input, reindent=True, keyword_case="upper")
    st.subheader("Formatted Query")
    st.code(formatted_sql, language="sql")

    tables, columns, where_cols = parse_sql(sql_input)

    st.subheader("Detected Tables")
    st.write(tables)

    st.subheader("Detected Columns (sample)")
    st.write(columns[:50])

    st.subheader("WHERE Columns")
    st.write(where_cols)

    suggestions, index_recs = recommend_indexes(tables, columns, where_cols)

    st.subheader("💡 Optimization Suggestions")
    for s in suggestions:
        st.write("- " + s)

    st.subheader("📌 Index Recommendations")
    for rec in index_recs:
        st.code(
            f"CREATE INDEX idx_{rec['table']}_{'_'.join(rec['columns'])} ON {rec['table']}({', '.join(rec['columns'])});"
        )

    # EXPLAIN / plan section
    st.subheader(f"🧪 Execution Plan / EXPLAIN ({engine})")
    if engine == "SQLite (local demo)" and not uploaded_files:
        st.info("Upload CSV/DDL files that represent the tables in your query to get a more realistic EXPLAIN.")
    rows = run_explain_plan(engine, sql_input, uploaded_files, conn_params)
    st.write(rows)

    # Optional OpenAI polishing
    if use_openai:
        st.subheader("✨ Polished Explanation (AI)")
        explanation = produce_openai_explanation(sql_input, suggestions, index_recs)
        st.write(explanation)
    else:
        st.info("Polished AI explanation unavailable — add OPENAI_API_KEY in Streamlit Secrets to enable.")

    st.success("Analysis complete!")
