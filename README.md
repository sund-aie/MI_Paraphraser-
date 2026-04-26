# MI Paraphraser

Local desktop GUI that rewrites academic dental-research text in the voice of
Dr. Maria Ibrahim. Section-aware: the syntax, transitional markers, and
lexicon adapt to whether the input is an Introduction, Materials & Methods,
Results, or Conclusion paragraph. Words and phrases the model alters are
highlighted in the output pane.

## Requirements

- Python 3.10+
- An Anthropic API key
- Tkinter (bundled with python.org Python on macOS/Windows; on Debian/Ubuntu:
  `sudo apt install python3-tk`)

## Setup

```bash
git clone <this-repo>
cd MI_Paraphraser-
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure your API key

The app reads `ANTHROPIC_API_KEY` from the environment.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

## Run

```bash
python app.py
```

## Usage

1. Pick a section from the **Section** dropdown
   (Introduction / Materials & Methods / Results / Conclusion).
2. Paste the source text into the upper **Source Text** box.
3. Click **Execute Paraphrase**.
4. The rewritten text appears in the lower box. Spans the model changed
   relative to the source are highlighted in yellow; the `<changed>` tags
   themselves are stripped before display.

## Project layout

```
app.py              # Tkinter GUI, Anthropic API call, tag parser
requirements.txt    # anthropic SDK
README.md           # this file
```

## Notes

- The full target-author profile and backend prompt logic live in the
  `SYSTEM_PROMPT` constant at the top of `app.py`. Edit it there to tune the
  voice or the output contract.
- The model is set to `claude-opus-4-7`. Change `MODEL_ID` in `app.py` to
  swap models.
- API calls run on a background thread so the GUI stays responsive.
