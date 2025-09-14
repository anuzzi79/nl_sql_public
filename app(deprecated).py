import os
import re
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import sqlparse
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

from dotenv import load_dotenv
import requests

# =========================
# Setup / .env
# =========================
load_dotenv()


def env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and (v is None or str(v).strip() == ""):
        raise RuntimeError(f"ENV mancante: {name}")
    return v


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nl2sql")

# --- LLM config ---
# 'openai' | 'ollama' | 'none'
LLM_PROVIDER = env("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = env("OPENAI_API_KEY", "")
OLLAMA_MODEL = env("OLLAMA_MODEL", "llama3.1:8b")
LLM_MODEL = env("LLM_MODEL", "gpt-4o-mini")
EMB_MODEL = env("EMB_MODEL", "text-embedding-3-small")
MAX_ROWS = int(env("MAX_ROWS", "1000"))

# --- Schema cache config ---
SCHEMA_CACHE_TTL = int(env("SCHEMA_CACHE_TTL", "300"))  # sec
SCHEMA_PREVIEW_ROWS = int(env("SCHEMA_PREVIEW_ROWS", "0"))

# --- DB config ---
DB_URI = os.getenv("DB_URI")
if DB_URI and DB_URI.strip():
    db_url = DB_URI
else:
    DB_USER = env("DB_USER", required=True)
    DB_PASSWORD = env("DB_PASSWORD", required=True)
    DB_HOST = env("DB_HOST", required=True)
    DB_PORT = int(env("DB_PORT", "3306"))
    DB_NAME = env("DB_NAME", required=True)
    DB_CHARSET = env("DB_CHARSET", "utf8mb4")
    db_url = URL.create(
        drivername="mysql+pymysql",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        query={"charset": DB_CHARSET},
    )

engine: Engine = create_engine(db_url, pool_recycle=3600, pool_pre_ping=True)
try:
    masked = str(db_url) if isinstance(db_url, str) else str(
        db_url).replace(db_url.password or "", "*****")
    masked = re.sub(r":([^:@/]+)@", r":*****@", masked)
    logger.info(f"[DB] URL: {masked}")
except Exception:
    pass

app = FastAPI(title="NL2SQL Agent", version="3.0")

# =========================
# MODELLI
# =========================


class NLQuery(BaseModel):
    question: str
    limit: Optional[int] = None
    language: Optional[str] = None  # 'it'|'pt'|'en'


class AgentQuery(BaseModel):
    question: str
    limit: Optional[int] = None
    language: Optional[str] = None
    max_steps: Optional[int] = 6


# =========================
# UTILS SQL
# =========================
SELECT_ONLY = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)
SQL_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|REPLACE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE)\b", re.IGNORECASE)


def is_safe_sql(sql: str) -> None:
    parsed = sqlparse.parse(sql)
    if len(parsed) != 1:
        raise HTTPException(
            status_code=400, detail="Consenti un solo statement SQL.")
    st = parsed[0]
    if st.get_type() != "SELECT" and not SELECT_ONLY.match(str(st)):
        raise HTTPException(
            status_code=400, detail="Sono permesse solo SELECT.")
    if SQL_FORBIDDEN.search(str(st)):
        raise HTTPException(
            status_code=400, detail="Sono permesse solo SELECT.")


def enforce_limit(sql: str, limit: int) -> str:
    return sql if re.search(r"\blimit\s+\d+\b", sql, flags=re.IGNORECASE) else f"{sql.rstrip(';')} LIMIT {limit}"


def fetchall_dict(sql: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    with engine.connect() as conn:
        res = conn.execute(text(sql), params or {})
        return [dict(r._mapping) for r in res]


def extract_sql_from_text(s: str) -> str:
    m = re.findall(r"```sql\s*(.*?)```", s, flags=re.IGNORECASE | re.DOTALL)
    return (m[0].strip() if m else s.strip())


# =========================
# SCHEMA + GRAFO JOIN (cache)
# =========================
_schema_cache: Optional[Dict[str, Any]] = None
_schema_cache_ts: float = 0.0
_schema_graph: Optional[Dict[str, Any]] = None
_schema_graph_ts: float = 0.0


def get_schema_snapshot(max_cols: int = 2000, preview_rows: int = 0) -> Dict[str, Any]:
    schema: Dict[str, Any] = {}
    tables = fetchall_dict("""
        SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = DATABASE() ORDER BY TABLE_NAME
    """)
    for t in tables:
        tbl = t["TABLE_NAME"]
        cols = fetchall_dict("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY, COLUMN_COMMENT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :tbl
            ORDER BY ORDINAL_POSITION
        """, {"tbl": tbl})
        preview = []
        if preview_rows > 0:
            try:
                preview = fetchall_dict(
                    f"SELECT * FROM `{tbl}` LIMIT {int(preview_rows)}")
            except Exception:
                preview = []
        schema[tbl] = {"columns": cols[:max_cols], "preview": preview}
    return schema


def get_schema_snapshot_cached() -> Dict[str, Any]:
    global _schema_cache, _schema_cache_ts
    now = time.time()
    if _schema_cache is None or (now - _schema_cache_ts) > SCHEMA_CACHE_TTL:
        t0 = time.time()
        _schema_cache = get_schema_snapshot(preview_rows=0)
        _schema_cache_ts = time.time()
        logger.info(f"[SCHEMA] aggiornato in {(_schema_cache_ts - t0):.2f}s")
    return _schema_cache


def build_schema_graph() -> Dict[str, Any]:
    """FK reali + euristica *_id → tabella."""
    schema = get_schema_snapshot_cached()
    # FK reali
    fks = fetchall_dict("""
        SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME AS REF_TABLE, REFERENCED_COLUMN_NAME AS REF_COL
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL
    """)
    fk_edges = [(r["TABLE_NAME"], r["COLUMN_NAME"],
                 r["REF_TABLE"], r["REF_COL"]) for r in fks]

    # euristiche *_id
    table_names = set(schema.keys())
    heur_edges = []
    for t, meta in schema.items():
        for c in meta["columns"]:
            cn = c["COLUMN_NAME"]
            if cn.endswith("_id"):
                base = cn[:-3]
                candidates = [base, base + "s", base +
                              "es", base + "_type", base + "_header"]
                for cand in candidates:
                    if cand in table_names:
                        heur_edges.append((t, cn, cand, "id"))
                        break

    return {"schema": schema, "fk_edges": fk_edges, "heur_edges": heur_edges}


def get_schema_graph_cached() -> Dict[str, Any]:
    global _schema_graph, _schema_graph_ts
    now = time.time()
    if _schema_graph is None or (now - _schema_graph_ts) > SCHEMA_CACHE_TTL:
        _schema_graph = build_schema_graph()
        _schema_graph_ts = time.time()
        logger.info("[GRAPH] join graph ricostruito")
    return _schema_graph


# =========================
# EMBEDDINGS: indice semantico (tabelle/colonne)
# =========================
_sem_index: Optional[List[Dict[str, Any]]] = None
_sem_index_ts: float = 0.0


def _embed(texts: List[str]) -> np.ndarray:
    if LLM_PROVIDER != "openai":
        raise HTTPException(
            status_code=500, detail="Embeddings disponibili solo con OPENAI per ora.")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY mancante")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0


def build_semantic_index() -> List[Dict[str, Any]]:
    g = get_schema_graph_cached()
    schema = g["schema"]
    entries = []
    for t, meta in schema.items():
        entries.append({"type": "table", "table": t,
                       "column": None, "label": t, "text": t})
        for c in meta["columns"]:
            col = c["COLUMN_NAME"]
            dt = c["DATA_TYPE"]
            comment = c.get("COLUMN_COMMENT") or ""
            text = f"{t}.{col} ({dt}) {comment}".strip()
            label = f"{t}.{col}"
            entries.append({"type": "column", "table": t,
                           "column": col, "label": label, "text": text})
    vecs = _embed([e["text"] for e in entries])
    for e, v in zip(entries, vecs):
        e["vec"] = v
    return entries


def get_semantic_index_cached() -> List[Dict[str, Any]]:
    global _sem_index, _sem_index_ts
    now = time.time()
    if _sem_index is None or (now - _sem_index_ts) > SCHEMA_CACHE_TTL:
        _sem_index = build_semantic_index()
        _sem_index_ts = time.time()
        logger.info("[SEMI] semantic index rebuild")
    return _sem_index

# =========================
# LLM CLIENT
# =========================


def llm_chat(messages: List[Dict[str, str]], temperature=0.1, max_tokens=600, timeout=40) -> str:
    if LLM_PROVIDER == "none":
        raise HTTPException(
            status_code=400, detail="LLM disattivato (LLM_PROVIDER=none).")
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise HTTPException(
                status_code=500, detail="OPENAI_API_KEY mancante")
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages,
            temperature=temperature, max_tokens=max_tokens, timeout=timeout,
        )
        return resp.choices[0].message.content or ""
    elif LLM_PROVIDER == "ollama":
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    else:
        raise HTTPException(
            status_code=500, detail=f"LLM_PROVIDER non supportato: {LLM_PROVIDER}")

# =========================
# TOOLS generici (per l’agente)
# =========================


def tool_list_tables() -> List[str]:
    rows = fetchall_dict("""
        SELECT TABLE_NAME AS name FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = DATABASE() ORDER BY 1
    """)
    return [r["name"] for r in rows]


def tool_table_schema(table: str) -> Dict[str, Any]:
    cols = fetchall_dict("""
        SELECT c.COLUMN_NAME, c.DATA_TYPE, c.IS_NULLABLE, c.COLUMN_KEY,
               k.REFERENCED_TABLE_NAME AS fk_table, k.REFERENCED_COLUMN_NAME AS fk_column
        FROM INFORMATION_SCHEMA.COLUMNS c
        LEFT JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
          ON k.TABLE_SCHEMA=c.TABLE_SCHEMA AND k.TABLE_NAME=c.TABLE_NAME AND k.COLUMN_NAME=c.COLUMN_NAME
        WHERE c.TABLE_SCHEMA=DATABASE() AND c.TABLE_NAME=:tbl
        ORDER BY c.ORDINAL_POSITION
    """, {"tbl": table})
    return {"table": table, "columns": cols}


def tool_sample_rows(table: str, limit: int = 3) -> List[Dict[str, Any]]:
    try:
        return fetchall_dict(f"SELECT * FROM `{table}` LIMIT {int(limit)}")
    except Exception as e:
        return [{"error": str(e)}]


def tool_execute_sql(sql: str, limit: int) -> Tuple[str, List[Dict[str, Any]]]:
    # se arrivano più SELECT separate da ';', prendo l'ultima
    stmts = [s.strip() for s in sql.split(";") if s.strip()]
    if not stmts:
        raise HTTPException(status_code=400, detail="SQL vuoto.")
    last = stmts[-1]
    is_safe_sql(last)
    last = enforce_limit(last, limit)
    rows = fetchall_dict(last)
    return last, rows


def tool_semantic_search(query: str, scope: str = "column", top_k: int = 8) -> List[Dict[str, Any]]:
    """scope: 'column' | 'table' | 'both'"""
    idx = get_semantic_index_cached()
    qv = _embed([query])[0]
    scored = []
    for e in idx:
        if scope == "column" and e["type"] != "column":
            continue
        if scope == "table" and e["type"] != "table":
            continue
        sim = _cos(qv, e["vec"])
        scored.append({"type": e["type"], "table": e["table"],
                      "column": e["column"], "label": e["label"], "score": round(sim, 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def tool_join_suggestions(tables: List[str]) -> List[Dict[str, Any]]:
    """Suggerisce ON tra tabelle in base a FK e euristiche."""
    g = get_schema_graph_cached()
    edges = g["fk_edges"] + g["heur_edges"]
    out = []
    for (t, col, ref_t, ref_c) in edges:
        if t in tables and ref_t in tables:
            out.append({"left": t, "right": ref_t,
                       "on": f"`{t}`.`{col}` = `{ref_t}`.`{ref_c}`"})
    # dedup
    seen = set()
    unique = []
    for r in out:
        key = (r["left"], r["right"], r["on"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# =========================
# AGENTE (ReAct-like)
# =========================
AGENT_SYSTEM_PROMPT = """
Sei un agente NL→SQL per MariaDB/MySQL. Usa passi di ragionamento + azione.
Disponi di questi strumenti:
- list_tables()
- table_schema(table)
- sample_rows(table, limit=3)
- semantic_search(query, scope='column'|'table'|'both', top_k=8)
- join_suggestions(tables:list[str])
- execute_sql(sql)

Regole:
- Rispondi **sempre** con UN SOLO oggetto JSON valido, senza testo extra.
  {"thought":"...", "action":"<uno degli strumenti o 'final'>", "args":{...}}
- Strategia tipica:
  1) semantic_search per capire quali colonne/tabelle mappano i termini naturali richiesti.
  2) table_schema per confermare le colonne e tipi.
  3) join_suggestions per ottenere le condizioni ON quando servono più tabelle.
  4) compose SQL (SELECT only, con backtick, alias chiari, filtra state=1 se ha senso).
  5) execute_sql e poi "final" con risposta + SQL usato riassunto.
- Evita CTE; usa JOIN espliciti.
- Se 0 righe, rilassa i filtri in un nuovo passo.
"""


def _parse_agent_json(s: str) -> Dict[str, Any]:
    m = re.search(r"```json\s*(\{.*?\})\s*```", s,
                  flags=re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end+1])
    raise HTTPException(
        status_code=400, detail=f"Agente: risposta non-JSON:\n{s}")


def run_agent(question: str, limit: int, language: str, max_steps: int = 6):
    trace: List[Dict[str, Any]] = []
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"[lingua={language}] {question}"}
    ]
    final_answer = None
    final_sql = None
    final_rows: List[Dict[str, Any]] = []

    for step in range(max_steps):
        content = llm_chat(messages, temperature=0.1, max_tokens=800)
        obj = _parse_agent_json(content)
        action = obj.get("action")
        thought = obj.get("thought")
        args = obj.get("args", {}) or {}

        if action == "list_tables":
            result = tool_list_tables()
            trace.append({"step": step+1, "thought": thought,
                         "action": action, "args": args, "result": result})
            messages += [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                         {"role": "user", "content": f"OUTPUT: {json.dumps(result, ensure_ascii=False)}"}]
            continue

        if action == "table_schema":
            tbl = str(args.get("table", "")).strip()
            if not tbl:
                raise HTTPException(
                    status_code=400, detail="args.table mancante")
            result = tool_table_schema(tbl)
            trace.append({"step": step+1, "thought": thought,
                         "action": action, "args": args, "result": result})
            messages += [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                         {"role": "user", "content": f"OUTPUT: {json.dumps(result, ensure_ascii=False)}"}]
            continue

        if action == "sample_rows":
            tbl = str(args.get("table", "")).strip()
            lim = int(args.get("limit", 3) or 3)
            if not tbl:
                raise HTTPException(
                    status_code=400, detail="args.table mancante")
            result = tool_sample_rows(tbl, lim)
            trace.append({"step": step+1, "thought": thought,
                         "action": action, "args": args, "result": result})
            messages += [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                         {"role": "user", "content": f"OUTPUT: {json.dumps(result, ensure_ascii=False)}"}]
            continue

        if action == "semantic_search":
            q = str(args.get("query", "")).strip()
            scope = str(args.get("scope", "column"))
            top_k = int(args.get("top_k", 8))
            if not q:
                raise HTTPException(
                    status_code=400, detail="args.query mancante")
            result = tool_semantic_search(q, scope=scope, top_k=top_k)
            trace.append({"step": step+1, "thought": thought,
                         "action": action, "args": args, "result": result})
            messages += [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                         {"role": "user", "content": f"OUTPUT: {json.dumps(result, ensure_ascii=False)}"}]
            continue

        if action == "join_suggestions":
            tables = args.get("tables") or []
            if not isinstance(tables, list) or not tables:
                raise HTTPException(
                    status_code=400, detail="args.tables deve essere lista non vuota")
            result = tool_join_suggestions([str(t) for t in tables])
            trace.append({"step": step+1, "thought": thought,
                         "action": action, "args": args, "result": result})
            messages += [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                         {"role": "user", "content": f"OUTPUT: {json.dumps(result, ensure_ascii=False)}"}]
            continue

        if action == "execute_sql":
            sql = str(args.get("sql", "")).strip()
            if not sql:
                raise HTTPException(
                    status_code=400, detail="args.sql mancante")
            safe_sql, rows = tool_execute_sql(sql, limit)
            trace.append({"step": step+1, "thought": thought, "action": action,
                         "args": {"sql": safe_sql}, "result": rows[:5]})
            messages += [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                         {"role": "user", "content": f"OUTPUT: rows={len(rows)} (LIMIT={limit}). SQL_USED: {safe_sql[:500]}"}]
            final_sql, final_rows = safe_sql, rows
            continue

        if action == "final":
            final_answer = obj.get("final_answer") or "(nessuna risposta)"
            trace.append({"step": step+1, "thought": thought,
                         "action": action, "args": args, "result": final_answer})
            break

        raise HTTPException(
            status_code=400, detail=f"Azione sconosciuta: {action}")

    if final_answer is None:
        raise HTTPException(
            status_code=400, detail="Limite passi raggiunto senza final_answer.")
    return {"trace": trace, "final_answer": final_answer, "final_sql": final_sql, "rows": final_rows}

# =========================
# STARTUP warm-up
# =========================


@app.on_event("startup")
def _warmup():
    try:
        _ = get_schema_snapshot_cached()
        _ = get_schema_graph_cached()
        if LLM_PROVIDER == "openai":
            _ = get_semantic_index_cached()
        logger.info("[WARMUP] ok")
    except Exception as e:
        logger.warning(f"[WARMUP] {e}")

# =========================
# ENDPOINTS
# =========================


@app.get("/health")
def health():
    return {"ok": True, "provider": LLM_PROVIDER,
            "model": LLM_MODEL if LLM_PROVIDER == "openai" else OLLAMA_MODEL,
            "embeddings": EMB_MODEL if LLM_PROVIDER == "openai" else None}


@app.get("/schema")
def schema():
    return get_schema_snapshot_cached()


@app.post("/query")
def query(q: NLQuery):
    t0 = time.time()
    limit = min(MAX_ROWS, q.limit or MAX_ROWS)
    try:
        schema = get_schema_snapshot_cached()
        # one-shot “traduttore” come fallback
        brief = []
        for tbl, meta in schema.items():
            cols = ", ".join([c["COLUMN_NAME"] for c in meta["columns"]])
            brief.append(f"- {tbl}({cols})")
        schema_text = "\n".join(brief)
        lang_hint = {"it": "Rispondi in italiano.", "pt": "Responda em português.",
                     "en": "Answer in English."}.get(q.language or "it", "Rispondi in italiano.")
        prompt = f"""{lang_hint}
Genera SQL MariaDB/MySQL **solo SELECT** (niente CTE). Usa backtick `t`.`col`. Aggiungi SEMPRE LIMIT {limit}.
Schema:
{schema_text}

Domanda:
{q.question}
"""
        raw = llm_chat([{"role": "system", "content": "Sei un traduttore NL→SQL."},
                        {"role": "user", "content": prompt}], temperature=0.1, max_tokens=400)
        sql = extract_sql_from_text(raw)
        if not sql:
            raise HTTPException(
                status_code=400, detail="Il modello non ha prodotto SQL.")
        # se arrivano più SELECT, uso l’ultima
        s = [x.strip() for x in sql.split(";") if x.strip()][-1]
        is_safe_sql(s)
        s = enforce_limit(s, limit)
        rows = fetchall_dict(s)
        return {"sql": s, "rows": rows, "meta": {"fastpath": False, "elapsed_s": round(time.time()-t0, 3)}}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /query")
        raise HTTPException(status_code=500, detail=f"Errore interno: {e}")


@app.post("/agent")
def agent_endpoint(q: AgentQuery):
    t0 = time.time()
    data = run_agent(q.question, limit=min(MAX_ROWS, q.limit or MAX_ROWS),
                     language=(q.language or "it").lower(), max_steps=q.max_steps or 6)
    data["meta"] = {"elapsed_s": round(
        time.time()-t0, 3), "max_rows": min(MAX_ROWS, q.limit or MAX_ROWS)}
    return data
