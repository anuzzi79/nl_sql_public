#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL→SQL Client (Tk • dark) con modalità Query/Agent
- Query: POST /query  → {sql, rows, meta}
- Agent: POST /agent  → {trace, final_answer, final_sql, rows, meta}
Requisiti: pip install requests
Avvio: python nl2sql_tk.py
"""

import json
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List, Dict, Any, Optional
import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_LANGUAGE = "it"        # 'it' | 'pt' | 'en'
DEFAULT_LIMIT = 500
DEFAULT_MODE = "Query"         # "Query" | "Agent"


class NL2SQLApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("NL→SQL Client (Tk • dark)")
        self.master.geometry("1200x740")
        self.master.minsize(980, 580)

        self._apply_dark_theme()

        # ---- State
        self.base_url_var = tk.StringVar(value=DEFAULT_BASE_URL)
        self.limit_var = tk.IntVar(value=DEFAULT_LIMIT)
        self.lang_var = tk.StringVar(value=DEFAULT_LANGUAGE)
        self.mode_var = tk.StringVar(value=DEFAULT_MODE)  # Query | Agent
        self.status_var = tk.StringVar(value="Pronto.")
        self.last_rows: List[Dict[str, Any]] = []

        # ---- Top bar
        top = ttk.Frame(self.master, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Base URL:").grid(row=0, column=0, sticky="w")
        self.url_entry = ttk.Entry(
            top, textvariable=self.base_url_var, width=38)
        self.url_entry.grid(row=0, column=1, sticky="we", padx=(6, 16))

        ttk.Label(top, text="Lingua:").grid(row=0, column=2, sticky="w")
        self.lang_combo = ttk.Combobox(top, textvariable=self.lang_var, width=6, state="readonly",
                                       values=("it", "pt", "en"))
        self.lang_combo.grid(row=0, column=3, sticky="w", padx=(6, 16))

        ttk.Label(top, text="Limit:").grid(row=0, column=4, sticky="w")
        self.limit_spin = ttk.Spinbox(
            top, from_=1, to=1_000_000, textvariable=self.limit_var, width=8)
        self.limit_spin.grid(row=0, column=5, sticky="w", padx=(6, 16))

        ttk.Label(top, text="Modalità:").grid(row=0, column=6, sticky="w")
        self.mode_combo = ttk.Combobox(top, textvariable=self.mode_var, width=8, state="readonly",
                                       values=("Query", "Agent"))
        self.mode_combo.grid(row=0, column=7, sticky="w", padx=(6, 16))

        self.run_btn = ttk.Button(
            top, text="Esegui (Ctrl+Invio)", command=self.run)
        self.clear_btn = ttk.Button(
            top, text="Pulisci", command=self.clear_all)
        self.export_btn = ttk.Button(
            top, text="Esporta CSV", command=self.export_csv)

        self.run_btn.grid(row=0, column=8, padx=(0, 8))
        self.clear_btn.grid(row=0, column=9, padx=(0, 8))
        self.export_btn.grid(row=0, column=10)

        top.grid_columnconfigure(1, weight=1)

        # ---- Prompt
        qframe = ttk.LabelFrame(
            self.master, text="Domanda in linguaggio naturale", padding=10)
        qframe.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))

        self.question = tk.Text(
            qframe, height=4, wrap="word", bd=0, highlightthickness=0)
        self.question.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.question.insert(
            "1.0", "Dammi tutte le observation del progetto 9203 con state = 1")

        qs = ttk.Scrollbar(qframe, orient="vertical",
                           command=self.question.yview)
        self.question.configure(yscrollcommand=qs.set)
        qs.pack(side=tk.RIGHT, fill=tk.Y)

        # ---- Notebook
        nb = ttk.Notebook(self.master)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.nb = nb

        # Risultati
        self.results_tab = ttk.Frame(nb)
        nb.add(self.results_tab, text="Risultati")

        tree_container = ttk.Frame(self.results_tab, padding=4)
        tree_container.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(tree_container, show="headings")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.vsb = ttk.Scrollbar(
            tree_container, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(
            tree_container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.vsb.set,
                            xscrollcommand=self.hsb.set)
        self.vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # SQL generato / usato
        self.sql_tab = ttk.Frame(nb)
        nb.add(self.sql_tab, text="SQL")

        self.sql_text = tk.Text(self.sql_tab, wrap="none",
                                bd=0, highlightthickness=0, height=8)
        self.sql_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sql_vsb = ttk.Scrollbar(
            self.sql_tab, orient="vertical", command=self.sql_text.yview)
        sql_hsb = ttk.Scrollbar(
            self.sql_tab, orient="horizontal", command=self.sql_text.xview)
        self.sql_text.configure(
            yscrollcommand=sql_vsb.set, xscrollcommand=sql_hsb.set, state="disabled")
        sql_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        sql_hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # Agente (trace + finale)
        self.agent_tab = ttk.Frame(nb)
        nb.add(self.agent_tab, text="Agente")

        self.agent_text = tk.Text(
            self.agent_tab, wrap="word", bd=0, highlightthickness=0)
        self.agent_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ag_vsb = ttk.Scrollbar(
            self.agent_tab, orient="vertical", command=self.agent_text.yview)
        self.agent_text.configure(yscrollcommand=ag_vsb.set, state="disabled")
        ag_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # ---- Status bar
        status = ttk.Frame(self.master)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(
            status, textvariable=self.status_var, anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X,
                               expand=True, padx=10, pady=(0, 8))

        # ---- Bindings
        self.master.bind("<Control-Return>", lambda _e=None: self.run())

        # Health check
        threading.Thread(target=self._health_check, daemon=True).start()

    # ---------- Theme ----------
    def _apply_dark_theme(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        bg = "#111318"
        bg2 = "#171a21"
        panel = "#0c0f14"
        fg = "#e6e6e6"
        fg_dim = "#bfbfbf"
        sel = "#2b5fd9"

        self.master.configure(bg=bg)
        style.configure(".", background=bg, foreground=fg,
                        fieldbackground=bg2, borderwidth=0)

        style.configure("TFrame", background=panel)
        style.configure("TLabel", background=panel, foreground=fg)
        style.configure("TLabelframe", background=panel, foreground=fg)
        style.configure("TLabelframe.Label", background=panel, foreground=fg)
        for el in ("TEntry", "TCombobox", "TSpinbox"):
            style.configure(el, fieldbackground=bg2,
                            foreground=fg, insertcolor=fg)
        style.configure("TButton", background=bg2, foreground=fg)
        style.map("TButton", background=[
                  ("active", "#1f2430")], foreground=[("disabled", fg_dim)])
        style.configure("TNotebook", background=panel, borderwidth=0)
        style.configure("TNotebook.Tab", background=bg2,
                        foreground=fg, padding=(12, 6))
        style.map("TNotebook.Tab", background=[("selected", "#1b1f27")])
        style.configure("Treeview", background=bg2, foreground=fg,
                        fieldbackground=bg2, rowheight=24, borderwidth=0)
        style.map("Treeview", background=[("selected", sel)], foreground=[
                  ("selected", "#ffffff")])

    # ---------- Helpers ----------
    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def clear_all(self):
        # grid
        for c in self.tree.get_children():
            self.tree.delete(c)
        self.tree["columns"] = ()
        # sql text
        self.sql_text.configure(state="normal")
        self.sql_text.delete("1.0", tk.END)
        self.sql_text.configure(state="disabled")
        # agent text
        self.agent_text.configure(state="normal")
        self.agent_text.delete("1.0", tk.END)
        self.agent_text.configure(state="disabled")
        self.last_rows = []
        self._set_status("Pulito.")

    def _health_check(self):
        try:
            r = requests.get(self.base_url_var.get().rstrip(
                "/") + "/health", timeout=4)
            if r.ok:
                self._set_status("Connesso all'agent ✔")
            else:
                self._set_status("Agent non raggiungibile (health != 200)")
        except Exception:
            self._set_status(
                "Agent non raggiungibile. Avvia uvicorn o verifica l'URL.")

    # ---------- Run ----------
    def run(self):
        question = self.question.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning(
                "Attenzione", "Scrivi una domanda in linguaggio naturale.")
            return

        base_url = self.base_url_var.get().rstrip("/")
        mode = self.mode_var.get()  # Query | Agent
        limit = max(1, int(self.limit_var.get() or DEFAULT_LIMIT))
        language = self.lang_var.get()

        self.run_btn.config(state="disabled")
        self._set_status("Esecuzione…")
        t0 = time.time()

        def worker():
            try:
                if mode == "Agent":
                    payload = {"question": question,
                               "limit": limit, "language": language}
                    r = requests.post(base_url + "/agent",
                                      data=json.dumps(payload),
                                      headers={
                                          "Content-Type": "application/json"},
                                      timeout=120)
                    if not r.ok:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
                    data = r.json()
                    self.master.after(0, lambda: self._apply_agent(data, t0))
                else:
                    payload = {"question": question,
                               "limit": limit, "language": language}
                    r = requests.post(base_url + "/query",
                                      data=json.dumps(payload),
                                      headers={
                                          "Content-Type": "application/json"},
                                      timeout=60)
                    if not r.ok:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
                    data = r.json()
                    self.master.after(0, lambda: self._apply_query(data, t0))
            except Exception as exc:
                self.master.after(0, lambda exc=exc: self._on_error(exc))
            finally:
                self.master.after(
                    0, lambda: self.run_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Apply results ----------
    def _apply_query(self, data: Dict[str, Any], t0: float):
        sql = data.get("sql") or ""
        rows = data.get("rows") or []
        meta = data.get("meta") or {}

        # SQL
        self.sql_text.configure(state="normal")
        self.sql_text.delete("1.0", tk.END)
        self.sql_text.insert("1.0", sql or "-- (nessun SQL)")
        self.sql_text.configure(state="disabled")

        # Grid
        self._fill_tree(rows)
        self.last_rows = rows

        dt = time.time() - t0
        self._set_status(
            f"OK (Query): {len(rows)} righe in {dt:.2f}s. fastpath={meta.get('fastpath')}")
        self.nb.select(self.results_tab if rows else self.sql_tab)

    def _apply_agent(self, data: Dict[str, Any], t0: float):
        final_sql = data.get("final_sql") or ""
        rows = data.get("rows") or []
        final_ans = data.get("final_answer") or "(nessuna risposta)"
        trace = data.get("trace") or []
        meta = data.get("meta") or {}

        # SQL usato
        self.sql_text.configure(state="normal")
        self.sql_text.delete("1.0", tk.END)
        self.sql_text.insert(
            "1.0", final_sql or "-- (l'agente non ha eseguito SQL)")
        self.sql_text.configure(state="disabled")

        # Risultati
        self._fill_tree(rows)
        self.last_rows = rows

        # Trace + finale
        self.agent_text.configure(state="normal")
        self.agent_text.delete("1.0", tk.END)
        self.agent_text.insert("1.0", "== Risposta finale ==\n")
        self.agent_text.insert(tk.END, final_ans.strip() + "\n\n")
        self.agent_text.insert(tk.END, "== Trace ==\n")
        for step in trace:
            s = step.get("step")
            action = step.get("action")
            thought = step.get("thought") or ""
            args = step.get("args")
            self.agent_text.insert(tk.END, f"[{s}] action={action}\n")
            if thought:
                self.agent_text.insert(tk.END, f"  thought: {thought}\n")
            if args:
                self.agent_text.insert(
                    tk.END, f"  args: {json.dumps(args, ensure_ascii=False)}\n")
            # non stampo interi result voluminosi, solo hint
            if action == "execute_sql":
                self.agent_text.insert(
                    tk.END, f"  result: rows={len(step.get('result') or [])}\n")
            else:
                # mostro max 1-2 elementi sintetici
                r = step.get("result")
                if isinstance(r, list) and r and isinstance(r[0], (str, int, float, dict)):
                    self.agent_text.insert(
                        tk.END, f"  result: {str(r)[:300]}...\n")
                elif isinstance(r, dict):
                    self.agent_text.insert(
                        tk.END, f"  result: {json.dumps(r, ensure_ascii=False)[:300]}...\n")
            self.agent_text.insert(tk.END, "\n")
        self.agent_text.configure(state="disabled")

        dt = time.time() - t0
        self._set_status(
            f"OK (Agent): {len(rows)} righe in {dt:.2f}s. meta={meta}")
        # Se ci sono righe → risultati, altrimenti tab agente
        self.nb.select(self.results_tab if rows else self.agent_tab)

    # ---------- Grid ----------
    def _fill_tree(self, rows: List[Dict[str, Any]]):
        for c in self.tree.get_children():
            self.tree.delete(c)

        if not rows:
            self.tree["columns"] = ("_info",)
            self.tree.heading("_info", text="(nessun risultato)")
            self.tree.column("_info", width=300, anchor="w")
            return

        # Ordine colonne stabilizzato
        cols_order = list(rows[0].keys())
        seen = set(cols_order)
        for r in rows[1:]:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    cols_order.append(k)

        self.tree["columns"] = cols_order
        for c in cols_order:
            self.tree.heading(c, text=c)
            w = 160
            cl = c.lower()
            if cl in ("id", "rn", "state", "status", "seq_num", "priority"):
                w = 90
            elif cl in ("project_id", "subtask_id", "eq_type_id"):
                w = 110
            self.tree.column(c, width=w, anchor="w", stretch=True)

        for r in rows:
            values = [self._fmt_cell(r.get(c)) for c in cols_order]
            self.tree.insert("", "end", values=values)

    @staticmethod
    def _fmt_cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        return str(v)

    # ---------- CSV ----------
    def export_csv(self):
        if not self.last_rows:
            messagebox.showinfo("CSV", "Nessun dato da esportare.")
            return
        fpath = filedialog.asksaveasfilename(
            title="Salva CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Tutti i file", "*.*")]
        )
        if not fpath:
            return
        cols = self.tree["columns"]
        try:
            import csv
            with open(fpath, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for r in self.last_rows:
                    w.writerow([r.get(c, "") if r.get(c, "")
                               is not None else "" for c in cols])
            messagebox.showinfo("CSV", f"Esportato: {fpath}")
        except Exception as e:
            messagebox.showerror("CSV", f"Errore: {e}")

    # ---------- Error ----------
    def _on_error(self, exc: Exception):
        self._set_status("Errore.")
        messagebox.showerror("Errore", f"{exc}")


def main():
    root = tk.Tk()
    app = NL2SQLApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
