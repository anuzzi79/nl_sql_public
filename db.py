# db.py
import os
import re
from contextlib import contextmanager
from typing import Dict, List, Tuple

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine, URL
from sqlalchemy.pool import QueuePool

# Solo comandi de leitura permitidos
READ_ONLY_ALLOWED = ("SELECT", "SHOW", "DESCRIBE", "EXPLAIN")


def _get_env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    return v if v is not None else default


def _build_url() -> URL:
    """
    Usa URL.create para escapar credenciais com caracteres especiais.
    Suporta DB_NAME vazio (conexão sem schema default).
    """
    user = os.environ["DB_USER"]
    pwd = os.environ["DB_PASSWORD"]
    host = _get_env("DB_HOST", "localhost")
    port = int(_get_env("DB_PORT", "3306") or "3306")
    dbname = _get_env("DB_NAME", "")
    charset = _get_env("DB_CHARSET", "utf8mb4")

    # Se DB_NAME vier vazio, passamos None para 'database' (SQLAlchemy gera ...@host:port/?charset=...)
    database = dbname if dbname.strip() else None

    return URL.create(
        drivername="mysql+pymysql",
        username=user,
        password=pwd,         # pode conter @ : / ! etc.
        host=host,
        port=port,
        database=database,
        query={"charset": charset},
    )


def create_db_engine() -> Engine:
    """
    Cria o engine com parâmetros de pool e timeout razoáveis.
    """
    url = _build_url()
    engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=5,
        pool_recycle=1800,
        pool_pre_ping=True,
        isolation_level="READ COMMITTED",
        future=True,
        connect_args={
            "connect_timeout": 10,  # timeout de conexão em segundos
        },
    )
    return engine


def is_readonly_sql(sql: str) -> bool:
    # Verifica cada statement separado por ';'
    stmts = [s.strip() for s in sql.strip().split(";") if s.strip()]
    if not stmts:
        return False
    for stmt in stmts:
        first = re.split(r"\s+", stmt, maxsplit=1)[0].upper()
        if first not in READ_ONLY_ALLOWED:
            return False
        # Bloqueia DDL/DML em qualquer lugar do texto
        if re.search(
            r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|TRUNCATE|REPLACE|RENAME|GRANT|REVOKE)\b",
            stmt,
            re.I,
        ):
            return False
    return True


def ensure_limit(sql: str, default_limit: int = 50) -> str:
    """
    Se for SELECT sem LIMIT, adiciona LIMIT default.
    """
    s = sql.strip().rstrip(";")
    if re.match(r"(?is)^\s*SELECT\b", s):
        if re.search(r"\bLIMIT\s+\d+", s, flags=re.I):
            return s + ";"
        return f"{s}\nLIMIT {default_limit};"
    return s + ";"


@contextmanager
def read_only_conn(engine: Engine):
    """
    Abre conexão e força sessão read-only quando possível.
    """
    with engine.connect() as conn:
        try:
            conn.exec_driver_sql("SET SESSION TRANSACTION READ ONLY;")
        except Exception:
            pass
        try:
            yield conn
        finally:
            # rollback defensivo
            try:
                conn.rollback()
            except Exception:
                pass


def run_sql(engine: Engine, sql: str, max_rows: int = 100) -> Tuple[List[Dict], List[str]]:
    if not is_readonly_sql(sql):
        raise ValueError(
            "Blocked: only SELECT/EXPLAIN/SHOW/DESCRIBE are allowed.")
    safe_sql = ensure_limit(sql, default_limit=max_rows)
    with read_only_conn(engine) as conn:
        result = conn.execute(text(safe_sql))
        cols = list(result.keys())
        rows = [dict(zip(cols, r)) for r in result.fetchall()]
        return rows, cols


def schema_digest(engine: Engine, max_cols: int = 18) -> str:
    """
    Monta um resumo do schema: tabelas, colunas (até max_cols) e FKs.
    Se a conexão falhar (ex.: DB_NAME inválido ou ausente), retorna uma mensagem amigável.
    """
    try:
        insp = inspect(engine)
        tables = sorted(insp.get_table_names())
    except Exception as e:
        return f"(schema unavailable: {e})"

    out_lines: List[str] = []
    for t in tables:
        cols = insp.get_columns(t)
        col_names = [c["name"] for c in cols]
        if len(col_names) > max_cols:
            col_show = ", ".join(col_names[:max_cols]) + ", …"
        else:
            col_show = ", ".join(col_names) if col_names else "(no cols?)"
        out_lines.append(f"- {t}: {col_show}")

        # Foreign keys
        fks = insp.get_foreign_keys(t)
        for fk in fks:
            if not fk.get("referred_table"):
                continue
            local_cols = ", ".join(fk.get("constrained_columns", []))
            ref_table = fk.get("referred_table", "?")
            ref_cols = ", ".join(fk.get("referred_columns", []))
            out_lines.append(
                f"    ↳ FK: {t}.{local_cols} → {ref_table}.{ref_cols}")

    return "\n".join(out_lines)


def list_tables(engine: Engine) -> List[str]:
    insp = inspect(engine)
    return sorted(insp.get_table_names())


def describe_table(engine: Engine, table_name: str) -> List[Dict]:
    insp = inspect(engine)
    cols = insp.get_columns(table_name)
    return [
        {
            "name": c["name"],
            "type": str(c["type"]),
            "nullable": c.get("nullable", True),
            "default": c.get("default"),
        }
        for c in cols
    ]


def sample_table(engine: Engine, table_name: str, limit: int = 20) -> Tuple[List[Dict], List[str]]:
    sql = f"SELECT * FROM `{table_name}` LIMIT {limit};"
    return run_sql(engine, sql, max_rows=limit)
