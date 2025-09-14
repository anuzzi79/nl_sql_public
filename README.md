# FG Data Agent (Gradio, Dark, Streaming)

Um "ChatGPT" focado no **dataset da Facility Grid**, com:
- Respostas **streaming**
- Memória de conversa por sessão
- Ferramentas de compreensão do schema
- Execução de **SQL somente leitura** (SELECT/SHOW/DESCRIBE/EXPLAIN)
- Interface **Gradio** em **tema escuro**

## Requisitos
- Python 3.10+
- Acesso de rede ao MySQL/MariaDB do FG (usuário read-only recomendado)
- Chave da OpenAI

## Instalação
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows (PowerShell)  |  source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
