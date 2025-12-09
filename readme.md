# ⚡ SQL Query Optimizer Chatbot  
AI-powered SQL explainer + index advisor + optimization assistant  
Created by **Kabir Puri** (Data Analytics & AI/ML Professional)

---

## 🔥 Live Demo
Streamlit Cloud:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sql-optimizer-chatbot-d5khy9cxxdfsd7bx5btmtk.streamlit.app/)

---

## 🚀 Try It in Google Colab  
Run the full notebook with one click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kabixr/sql-optimizer-chatbot/blob/main/sql_optimizer_demo.ipynb)


---

## 🎯 What this chatbot does
- Parses SQL (SELECT, JOIN, GROUP BY, WHERE, HAVING)  
- Suggests optimal indexes  
- Detects performance bottlenecks  
- Provides query rewrites  
- Generates `CREATE INDEX` statements  
- Explains queries in plain English  
- Optional OpenAI-powered optimization guide  
- Optional EXPLAIN QUERY PLAN using SQLite  

---

## 🛠️ Tech Stack
- Python  
- Streamlit  
- SQLGlot  
- sqlparse  
- sql_metadata  
- SQLite  
- OpenAI (optional)

---

## 📦 Installation
```bash
pip install -r requirements.txt
streamlit run app.py
