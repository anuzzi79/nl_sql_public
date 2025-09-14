# tools_sql.py
import json
from typing import Any, Dict, List, Tuple

from db import run_sql, list_tables, describe_table, sample_table, schema_digest

# Tool wrappers used by the agent loop
def tool_list_tables(engine) -> Dict[str, Any]:
    return {"tables": list_tables(engine)}

def tool_describe_table(engine, table_name: str) -> Dict[str, Any]:
    return {"table": table_name, "columns": describe_table(engine, table_name)}

def tool_get_schema_summary(engine) -> Dict[str, Any]:
    return {"schema": schema_digest(engine)}

def tool_sample_rows(engine, table_name: str, limit: int = 20) -> Dict[str, Any]:
    rows, cols = sample_table(engine, table_name, limit)
    return {"table": table_name, "columns": cols, "rows": rows}

def tool_run_sql(engine, sql: str, max_rows: int = 100) -> Dict[str, Any]:
    rows, cols = run_sql(engine, sql, max_rows=max_rows)
    return {"columns": cols, "rows": rows, "row_count": len(rows), "sql_executed": sql}
