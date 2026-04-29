# MI Paraphraser

Fully **local, offline** desktop GUI that rewrites academic dental-research
text in the voice of Dr. Maria Ibrahim. Section-aware: the syntax,
transitional markers, and lexicon adapt to whether the input is an
Introduction, Materials & Methods, Results, or Conclusion paragraph. Words
and phrases the model alters are highlighted in the output pane.

No API key. No internet. All inference runs locally via [Ollama](https://ollama.com).

## Requirements

- Python 3.10+
- Tkinter (bundled with python.org Python on macOS/Windows; on Debian/Ubuntu:
  `sudo apt install python3-tk`)
- [Ollama](https://ollama.com/download) installed and running on the same
  machine
- At least one model pulled locally (the app discovers and lists every
  model installed on your Ollama instance)

## Setup

### 1. Install Ollama

Follow the instructions at https://ollama.com/download for your OS.

After install, the Ollama daemon usually starts automatically. If not:

```bash
ollama serve
```

It listens on `http://localhost:11434`.

### 2. Pull a model

Pull any Ollama-compatible model. For this dense academic paraphrasing
task, models with **≥ 13B parameters** are flagged as recommended in the
UI (e.g. `qwen2.5:32b`, `llama3.1:70b`, `mixtral:8x7b`). Smaller models
work but tend to drift from the strict style contract.

```bash
ollama pull qwen2.5:32b
```

The app reads `GET /api/tags` from your local Ollama and populates the
**Ollama model** dropdown with every installed model. Click **Refresh**
in the UI after pulling a new one.

### 3. Install Python deps

```bash
git clone <this-repo>
cd MI_Paraphraser-
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Usage

1. Pick a section from the **Section** dropdown
   (Introduction / Materials & Methods / Results / Conclusion).
2. Pick a model from the **Ollama model** dropdown (lists every model
   installed on your Ollama daemon; recommended models are tagged
   `(recommended)` and a status label flags whether the current
   selection is suitable). Click **Refresh** if you've just pulled a
   new model.
3. Paste the source text into the upper **Source Text** box.
4. Click **Execute Paraphrase**.
5. The rewritten text appears in the lower box. Spans the model changed
   relative to the source are highlighted in yellow; the `<changed>` tags
   themselves are stripped before display.

## Troubleshooting

- **"Could not reach Ollama at http://localhost:11434"** — the Ollama
  daemon is not running. Start it with `ollama serve` (or open the Ollama
  desktop app on macOS/Windows).
- **"model 'X' is not available locally"** — pull it first:
  `ollama pull <model>`.
- **First call is very slow** — Ollama loads the model into memory on the
  first request; subsequent calls are much faster.

## Project layout

```
app.py              # Tkinter GUI, Ollama HTTP call, tag parser
requirements.txt    # requests
README.md           # this file
```

## Notes

- The full target-author profile and backend prompt logic live in the
  `SYSTEM_PROMPT` constant at the top of `app.py`. Edit it there to tune the
  voice or the output contract.
- The Ollama endpoint URL is hard-coded to `http://localhost:11434/api/generate`
  via the `OLLAMA_URL` constant in `app.py`. Change it if you run Ollama on
  a non-default host or port.
- API calls run on a background thread so the GUI stays responsive.
