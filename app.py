# app.py
import os
import uuid
from typing import List, Dict, Any, Generator, Tuple

# --- JSON compat: usa orjson se disponibile; altrimenti json nativo ---
try:
    import orjson as _json

    def j_loads(s: str) -> dict:
        return _json.loads(s)

    def j_dumps(o: dict) -> str:
        return _json.dumps(o).decode("utf-8")
except Exception:
    import json as _json

    def j_loads(s: str) -> dict:
        return _json.loads(s)

    def j_dumps(o: dict) -> str:
        return _json.dumps(o, ensure_ascii=False)

import gradio as gr
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
from openai import OpenAI

from db import create_db_engine, schema_digest
from tools_sql import (
    tool_list_tables,
    tool_describe_table,
    tool_get_schema_summary,
    tool_sample_rows,
    tool_run_sql,
)
from prompts import SYSTEM_PROMPT, SCHEMA_PREAMBLE

# =========================
# Setup (carica .env, inizializza client/engine con valori attuali)
# =========================
load_dotenv()

# Variabili globali mutabili (aggiornabili dal pannello Settings)
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""


def _make_client(provider: str, api_key: str) -> OpenAI:
    """
    Crea il client LLM. Per DeepSeek usiamo l'endpoint OpenAI-compatibile.
    """
    provider = (provider or "openai").lower()
    key = (api_key or "").strip()
    if provider == "openai":
        if not key:
            raise RuntimeError("OPENAI_API_KEY non impostata.")
        return OpenAI(api_key=key)
    elif provider == "deepseek":
        if not key:
            raise RuntimeError("Chiave DeepSeek non impostata.")
        # IMPORTANTE: includere /v1 nell'endpoint
        return OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
    else:
        raise RuntimeError(f"LLM_PROVIDER non supportato: {provider}")


def _make_engine() -> Any:
    # create_db_engine legge le variabili d'ambiente correnti
    return create_db_engine()


# Iniziali
client = _make_client(LLM_PROVIDER, OPENAI_API_KEY)
engine = _make_engine()

# Non carichiamo lo schema al bootstrap per evitare blocchi.
SCHEMA = "(schema will be loaded on demand via tools)"

# Linee guida all'agente: consigli (non limiti rigidi)
AGENT_GUIDELINES = """
You are a SQL assistant with function tools. Be decisive and economical in tool usage.

Strategy patterns:
- If the user asks for projects linked to a person/email:
  1) SELECT the user_id from the user table (by email).
  2) Use a linking table (e.g., project_user/user_project/members) if present.
  3) SELECT projects via JOIN on that user_id. Prefer a single final query.
- Reuse previously fetched metadata when possible to avoid unnecessary calls.
- Prefer composing and running a clear SELECT over many metadata calls when feasible.
- If schema names are unclear, infer likely names from prior results before requesting more metadata.

Output: after tools, return a concise final answer with the relevant rows or a short explanation if no rows.
"""

SYSTEM_FULL = SYSTEM_PROMPT + "\n\n" + \
    SCHEMA_PREAMBLE.format(schema_digest=SCHEMA) + "\n\n" + AGENT_GUIDELINES

# =========================
# OpenAI Tool Definitions
# =========================
TOOLS = [
    {"type": "function", "function": {
        "name": "list_tables",
        "description": "List all tables in the FG database.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "describe_table",
        "description": "Describe columns of a given table.",
        "parameters": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]},
    }},
    {"type": "function", "function": {
        "name": "get_schema_summary",
        "description": "Return a short human-friendly schema digest.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "sample_rows",
        "description": "Preview N rows from a table.",
        "parameters": {"type": "object",
                       "properties": {"table_name": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
                       "required": ["table_name"]},
    }},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run a READ-ONLY SQL statement (SELECT/SHOW/DESCRIBE/EXPLAIN only).",
        "parameters": {"type": "object",
                       "properties": {"sql": {"type": "string"}, "max_rows": {"type": "integer", "default": 100}},
                       "required": ["sql"]},
    }},
]

# =========================
# Agent + LLM
# =========================


def call_tool(tool_call) -> Dict[str, Any]:
    name = tool_call.function.name
    args = {}
    if getattr(tool_call.function, "arguments", None):
        try:
            args = j_loads(tool_call.function.arguments)
        except Exception:
            args = {}
    if name == "list_tables":
        return tool_list_tables(engine)
    if name == "describe_table":
        return tool_describe_table(engine, args["table_name"])
    if name == "get_schema_summary":
        return tool_get_schema_summary(engine)
    if name == "sample_rows":
        return tool_sample_rows(engine, args["table_name"], args.get("limit", 20))
    if name == "run_sql":
        return tool_run_sql(engine, args["sql"], args.get("max_rows", 100))
    return {"error": f"Unknown tool {name}"}


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
def llm_chat(messages: List[Dict[str, str]], stream: bool = True):
    """
    Chiama il modello dal pannello Settings (.env runtime); se non disponibile, fallback al modello
    di default per il provider corrente.
    """
    global client, MODEL, LLM_PROVIDER
    try:
        return client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            stream=stream,
        )
    except Exception as e:
        msg = str(e).lower()
        # fallback per "model not found/insufficient/etc." mantenendo il provider corrente
        if any(x in msg for x in ["not found", "does not exist", "inactive", "insufficient", "unsupported model"]):
            fallback_model = "deepseek-chat" if (
                LLM_PROVIDER or "").lower() == "deepseek" else "gpt-4o-mini"
            return client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
                stream=stream,
            )
        raise

# ---------- Localizzazione semplice (PT/IT/EN) ----------


def _detect_lang(text: str, fallback: str = "en") -> str:
    t = (text or "").lower()
    pt_hits = any(w in t for w in [
        "você", "voce", "por favor", "obrigado", "tabela", "tabelas", "consultar", "cruzar", "poderia", "deseja",
        "me diga", "qual", "projeto", "projetos", "associado", "associados", "usuário", "usuario", "usuarios",
        "está", "esta"
    ])
    it_hits = any(w in t for w in [
        "per favore", "grazie", "tabelle", "interrogare", "incrociare", "potresti", "puoi", "progetti",
        "specifico", "specifica", "vorrei", "dimmi", "quale", "progetto", "associato", "utente", "tabella"
    ])
    if pt_hits and not it_hits:
        return "pt"
    if it_hits and not pt_hits:
        return "it"
    return fallback


def _t(lang: str, key: str, **kwargs) -> str:
    msgs = {
        "consulting_db": {
            "pt": "🔎 Consultando o banco de dados…",
            "it": "🔎 Consulto il database…",
            "en": "🔎 Querying the database…",
        },
        "exec_tool": {
            "pt": "• Executando `{name}` ({i}/{n})…",
            "it": "• Eseguo `{name}` ({i}/{n})…",
            "en": "• Running `{name}` ({i}/{n})…",
        },
        "help": {
            "pt": "❓ **Você pode ser mais específico?** Poderia citar as *tabelas* que deseja consultar ou cruzar? Se tiver outras informações para meu entendimento, por favor me diga.",
            "it": "❓ **Potresti essere più specifico?** Puoi citare le *tabelle* che vuoi interrogare o incrociare? Se hai altre informazioni per la mia comprensione, per favore dimmelo.",
            "en": "❓ **Could you be more specific?** Please name the *tables* you want me to query or join? If you have other information to help me understand, please tell me.",
        },
    }
    text = msgs[key][lang if lang in ("pt", "it", "en") else "en"]
    return text.format(**kwargs)

# ---------- Agente streaming ----------


def _should_extend_budget(user_msg: str) -> bool:
    lm = (user_msg or "").lower()
    return any(x in lm for x in ["continua", "não pare", "nao pare", "non fermarti", "continue", "keep going", "go on"])


def agent_generator(user_msg: str, history, state: Dict[str, Any]) -> Generator[str, None, None]:
    """
    Streaming con UNA sola bolla che cresce.
    Strategia:
    1) Prova autonoma nel primo round.
    2) Se nel 1° round non ha eseguito almeno una run_sql con successo → chiede aiuto all'utente (nella lingua dell'utente).
    """
    # Stato sessione
    if "session_id" not in state:
        state["session_id"] = str(uuid.uuid4())
    if "messages" not in state:
        state["messages"] = []
    if "asked_help_once" not in state:
        state["asked_help_once"] = False
    if "lang" not in state:
        state["lang"] = "en"

    # Atualiza língua com base na última mensagem do usuário
    state["lang"] = _detect_lang(user_msg, fallback=state.get("lang", "en"))

    # Conversazione iniziale
    messages = [{"role": "system", "content": SYSTEM_FULL}] + state["messages"]
    messages.append({"role": "user", "content": user_msg})

    # Placeholder per streaming
    display = "…"
    yield display

    # Limite iterazioni round — 12 default, 36 se l'utente chiede di continuare
    MAX_TOOL_LOOPS = 36 if _should_extend_budget(user_msg) else 12
    loops = 0

    while True:
        # Chiamata LLM (streaming)
        try:
            stream_resp = llm_chat(messages, stream=True)
        except RetryError as re:
            yield f"❌ LLM retry fallito ({LLM_PROVIDER}): {re.last_attempt.exception()}"
            return
        except Exception as e:
            yield f"❌ Errore chiamata modello: {e}"
            return

        saw_tool = False
        acc_round = ""

        try:
            for chunk in stream_resp:
                delta = chunk.choices[0].delta
                # Mostra "Consulto..." solo UNA volta per round (localizzato)
                if delta and delta.tool_calls:
                    if not saw_tool:
                        saw_tool = True
                        if acc_round.strip():
                            display = acc_round
                        display = (display.rstrip() + "\n\n" +
                                   _t(state["lang"], "consulting_db"))
                        yield display
                    continue
                if delta and delta.content:
                    acc_round += delta.content
                    display = acc_round
                    yield display
        except Exception as e:
            yield f"⚠️ Errore durante lo streaming: {e}"
            return

        if saw_tool:
            # Esecuzione tool
            try:
                full_resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                    stream=False,
                )
                inter = (full_resp.choices[0].message.content or "").strip()
                if inter:
                    display = (display.rstrip() +
                               ("\n\n" if display.strip() else "") + inter)
                    yield display

                tool_calls = full_resp.choices[0].message.tool_calls or []
                messages.append({
                    "role": "assistant",
                    "content": inter,
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                })

                ran_successful_sql = False

                for idx, tc in enumerate(tool_calls, start=1):
                    name = tc.function.name
                    # UI status localizzato
                    display += "\n\n" + \
                        _t(state["lang"], "exec_tool",
                           name=name, i=idx, n=len(tool_calls))
                    yield display

                    try:
                        result = call_tool(tc)
                        done_line = " fatto."
                        if name == "run_sql" and isinstance(result, dict) and not result.get("error"):
                            ran_successful_sql = True
                    except Exception as e:
                        result = {"error": f"Tool '{name}' failed: {e}"}
                        done_line = " errore."

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": j_dumps(result),
                    })

                    display += done_line
                    yield display

                loops += 1

                # 1° round senza run_sql riuscita → domanda di aiuto (lingua utente)
                if loops == 1 and not ran_successful_sql and not state.get("asked_help_once", False):
                    help_text = _t(state["lang"], "help")
                    display += "\n\n" + help_text
                    yield display
                    state["asked_help_once"] = True
                    state["messages"].append(
                        {"role": "user", "content": user_msg})
                    state["messages"].append(
                        {"role": "assistant", "content": display})
                    return

                # Fine silenziosa al raggiungimento del limite round
                if loops >= MAX_TOOL_LOOPS:
                    state["messages"].append(
                        {"role": "user", "content": user_msg})
                    state["messages"].append(
                        {"role": "assistant", "content": display})
                    return

                continue

            except RetryError as re:
                yield f"❌ LLM retry fallito ({LLM_PROVIDER}): {re.last_attempt.exception()}"
                return
            except Exception as e:
                yield f"❌ Errore nell'esecuzione strumenti: {e}"
                return

        else:
            final_text = acc_round if acc_round.strip() else display
            state["messages"].append({"role": "user", "content": user_msg})
            state["messages"].append(
                {"role": "assistant", "content": final_text})
            return


# =========================
# Helpers: scrittura .env e applicazione impostazioni
# =========================
ENV_KEYS = [
    "DB_USER",
    "DB_PASSWORD",
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_CHARSET",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "OPENAI_API_KEY",  # usiamo questa anche per DeepSeek (UI = "LLM Key")
]


def write_env_file(path: str, values: Dict[str, str]) -> None:
    """Sovrascrive il file .env con i valori passati (solo le chiavi conosciute)."""
    lines = []
    for k in ENV_KEYS:
        v = values.get(k, "")
        lines.append(f"{k}={v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# --------- euristica per riconoscere DeepSeek vs OpenAI ----------


def _infer_provider_from_key(key: str) -> str:
    """
    Migliorata:
    - 'sk-proj-'  => OpenAI (chiavi progetto OpenAI)
    - 'sk-' (senza 'sk-proj-') => DeepSeek (default)
    - contiene 'deepseek' o prefisso 'dsk-' => DeepSeek
    """
    k = (key or "").strip().lower()
    if not k:
        return ""
    if k.startswith("sk-proj-"):
        return "openai"
    if k.startswith("sk-") and not k.startswith("sk-proj-"):
        return "deepseek"
    if "deepseek" in k or k.startswith("dsk-"):
        return "deepseek"
    return ""


def apply_settings(
    db_user: str,
    db_password: str,
    db_host: str,
    db_port: str,
    db_name: str,
    db_charset: str,
    llm_provider: str,
    llm_model: str,
    openai_api_key: str,  # UI: "LLM Key"
) -> Tuple[str]:
    """
    Aggiorna le variabili d'ambiente, salva su .env, ricrea client LLM ed engine DB.
    Ritorna un messaggio di stato per l'UI.
    """
    global MODEL, LLM_PROVIDER, OPENAI_API_KEY, client, engine

    # Backup per eventuale rollback
    old_env = dict(os.environ)
    old_MODEL = MODEL
    old_PROVIDER = LLM_PROVIDER
    old_API = OPENAI_API_KEY
    old_client = client
    old_engine = engine

    # Normalizzazioni minime
    db_port = str(db_port or "").strip() or "3306"
    db_charset = (db_charset or "utf8mb4").strip()
    openai_api_key = (openai_api_key or "").strip()

    # AUTO-DETECT provider dal contenuto della key (fallback se UI non aggiorna)
    prov_guess = _infer_provider_from_key(openai_api_key)
    if prov_guess:
        llm_provider = prov_guess

    # Default modello coerente con provider
    if (llm_provider or "").lower() == "deepseek":
        if llm_model not in ("deepseek-reasoner", "deepseek-chat"):
            llm_model = "deepseek-reasoner"
    else:
        llm_provider = "openai"
        if llm_model not in ("gpt-4o-mini", "gpt-4o"):
            llm_model = "gpt-4o-mini"

    # Aggiorna env in-process
    updates = {
        "DB_USER": db_user or "",
        "DB_PASSWORD": db_password or "",
        "DB_HOST": db_host or "",
        "DB_PORT": db_port,
        "DB_NAME": db_name or "",
        "DB_CHARSET": db_charset,
        "LLM_PROVIDER": llm_provider,
        "LLM_MODEL": llm_model,
        "OPENAI_API_KEY": openai_api_key or "",
    }
    os.environ.update(updates)

    try:
        new_client = _make_client(llm_provider, openai_api_key)
        new_engine = _make_engine()

        # Prova un digest leggero dello schema (opzionale ma utile)
        try:
            _ = schema_digest(new_engine)
        except Exception:
            pass

        # Persisti su file .env
        write_env_file(".env", updates)

        # Commit delle nuove istanze globali
        MODEL = llm_model
        LLM_PROVIDER = llm_provider
        OPENAI_API_KEY = openai_api_key
        client = new_client
        engine = new_engine

        return ((
            "✅ Impostazioni applicate.\n\n"
            f"- DB: {db_user}@{db_host}:{db_port}/{db_name} (charset={db_charset})\n"
            f"- LLM_PROVIDER: {llm_provider}\n"
            f"- LLM_MODEL: {llm_model}\n"
            f"- LLM Key: {'••••••••' if openai_api_key else '(vuoto)'}\n"
            "Il file `.env` è stato aggiornato e l'istanza è ora attiva con i nuovi parametri."
        ),)

    except Exception as e:
        # Rollback
        os.environ.clear()
        os.environ.update(old_env)
        MODEL = old_MODEL
        LLM_PROVIDER = old_PROVIDER
        OPENAI_API_KEY = old_API
        client = old_client
        engine = old_engine
        return ((f"❌ Errore nell'applicare le impostazioni: {e}",),)


# =========================
# UI (Gradio, Dark) — Setup in testata, Chat
# =========================
DARK_CSS = """
html, body, .gradio-container { height: 100%; }
.gradio-container {max-width: 1024px !important;}
:root, .dark {
  --background-fill-primary: #0b0f19;
  --background-fill-secondary: #0e1525;
  --color-accent: #9ecbff;
  --color-text: #e6edf3;
  --border-color-primary: #1f2a44;
}
.gradio-container, body { background: var(--background-fill-primary); color: var(--color-text); }
button, .button { border-radius: 14px !important; }

/* Compatta sezioni */
.gr-box, .gr-panel, .gr-group { padding-top: 6px !important; padding-bottom: 6px !important; }
label { font-size: 12px !important; margin-bottom: 2px !important; }
.gr-row, .grid { gap: 6px !important; }
input[type="text"], input[type="password"], .gr-text-input input, .gr-textbox textarea {
  padding: 6px 8px !important; font-size: 14px !important; min-height: 34px !important;
}

/* Chat area: altezza dinamica per tenere il prompt a vista */
#chat_wrap { min-height: calc(100vh - 260px); }
"""


def on_llm_key_change(llm_key: str, current_provider: str):
    """
    Quando l'utente inserisce una LLM Key:
    - deduciamo il provider; impostiamo provider/model e li rendiamo read-only
    """
    key = (llm_key or "").strip()
    if key:
        prov = _infer_provider_from_key(key) or (current_provider or "openai")
        if prov == "deepseek":
            model_val = "deepseek-reasoner"
            model_choices = ["deepseek-reasoner", "deepseek-chat"]
        else:
            prov = "openai"
            model_val = "gpt-4o-mini"
            model_choices = ["gpt-4o-mini", "gpt-4o"]
        return (
            # provider read-only
            gr.update(value=prov, interactive=False),
            gr.update(value=model_val, choices=model_choices,
                      interactive=False),  # model read-only
        )
    else:
        # chiave vuota -> sblocca i campi
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
        )


def on_provider_change(provider: str):
    """
    Se la chiave è vuota e l'utente cambia provider manualmente, aggiorniamo le scelte del modello.
    """
    if provider == "deepseek":
        return gr.update(choices=["deepseek-reasoner", "deepseek-chat"], value="deepseek-reasoner", interactive=True)
    return gr.update(choices=["gpt-4o-mini", "gpt-4o"], value="gpt-4o-mini", interactive=True)


with gr.Blocks(title="FG Data Agent", css=DARK_CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FG Data Agent")

    # ----- SETTINGS IN TESTATA -----
    with gr.Accordion("Do here your dataset and LLM set-up", open=False):
        with gr.Row():
            db_user_in = gr.Textbox(label="DB User", value=os.environ.get(
                "DB_USER", ""), placeholder="es. antonio.nuzzi", lines=1, scale=1)
            db_password_in = gr.Textbox(label="DB Pass", value=os.environ.get(
                "DB_PASSWORD", ""), type="password", placeholder="password DB", lines=1, scale=1)
        with gr.Row():
            db_host_in = gr.Textbox(label="Host", value=os.environ.get(
                "DB_HOST", ""), placeholder="e.g. 127.0.0.1", lines=1, scale=1)
            db_port_in = gr.Textbox(label="Port", value=os.environ.get(
                "DB_PORT", "3306"), placeholder="3306", lines=1, scale=1)
        with gr.Row():
            db_name_in = gr.Textbox(label="Database", value=os.environ.get(
                "DB_NAME", ""), placeholder="nome DB", lines=1, scale=1)
            with gr.Accordion("Advanced", open=False):
                db_charset_in = gr.Textbox(label="Charset", value=os.environ.get(
                    "DB_CHARSET", "utf8mb4"), placeholder="utf8mb4", lines=1)

        gr.Markdown("---")

        # Provider/Model a tendina; diventano read-only quando la LLM Key è valorizzata
        with gr.Row():
            llm_provider_in = gr.Dropdown(
                label="LLM Provider",
                choices=["openai", "deepseek"],
                value=os.environ.get("LLM_PROVIDER", "openai"),
                interactive=True,
                scale=1,
            )
            llm_model_in = gr.Dropdown(
                label="LLM Model",
                choices=(["gpt-4o-mini", "gpt-4o"]
                         if os.environ.get("LLM_PROVIDER", "openai") == "openai"
                         else ["deepseek-reasoner", "deepseek-chat"]),
                value=(os.environ.get("LLM_MODEL", "gpt-4o-mini")
                       if os.environ.get("LLM_PROVIDER", "openai") == "openai"
                       else os.environ.get("LLM_MODEL", "deepseek-reasoner")),
                interactive=True,
                scale=1,
            )

        # Campo chiave unificato
        llm_key_in = gr.Textbox(
            label="LLM Key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            placeholder="incolla qui la chiave OpenAI o DeepSeek",
            lines=1,
        )

        # Reattività: chiave -> blocca Provider/Model e imposta default coerenti
        llm_key_in.change(
            fn=on_llm_key_change,
            inputs=[llm_key_in, llm_provider_in],
            outputs=[llm_provider_in, llm_model_in],
        )
        # Se la chiave è vuota e cambi provider manualmente, aggiorna i modelli disponibili
        llm_provider_in.change(
            fn=on_provider_change,
            inputs=[llm_provider_in],
            outputs=[llm_model_in],
        )

        with gr.Row():
            apply_btn = gr.Button("💾 Apply & Save (.env)", scale=1)
        status_out = gr.Markdown(
            "_(Le impostazioni correnti sono lette da `.env` all'avvio.)_")
        apply_btn.click(
            fn=apply_settings,
            inputs=[
                db_user_in,
                db_password_in,
                db_host_in,
                db_port_in,
                db_name_in,
                db_charset_in,
                llm_provider_in,
                llm_model_in,
                llm_key_in,  # UI: LLM Key -> OPENAI_API_KEY
            ],
            outputs=[status_out],
        )

    state = gr.State({})

    # ----- CHAT -----
    with gr.Column(elem_id="chat_wrap"):
        chat = gr.ChatInterface(
            fn=agent_generator,
            additional_inputs=[state],
            autofocus=True,
            submit_btn="Enviar",
            stop_btn="Parar",
            fill_height=True,  # assicura che l'input resti in vista
        )

if __name__ == "__main__":
    # Avvio pubblico
    port = int(os.environ.get("PORT", "7860"))
    demo.queue().launch(server_name="0.0.0.0",
                        server_port=port, share=True, show_error=True)
