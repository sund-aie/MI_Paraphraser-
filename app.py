"""MI Paraphraser — fully local desktop GUI for section-aware academic paraphrasing.

Tkinter front-end + local Ollama back-end. No internet connection or API key
required. The model is instructed to wrap every altered word/phrase in
<changed>...</changed> tags; the UI strips the tags and highlights the
content in the output pane.
"""

from __future__ import annotations

import json
import re
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import requests


SECTIONS = ["Introduction", "Materials & Methods", "Results", "Conclusion"]
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
REQUEST_TIMEOUT = 300  # seconds; local inference can be slow on first call
TAGS_TIMEOUT = 10

# Models with at least this many billion parameters are flagged as
# recommended for the dense, instruction-heavy paraphrasing task.
RECOMMENDED_MIN_PARAMS_B = 13.0
RECOMMENDED_SUFFIX = " (recommended)"

# ---------------------------------------------------------------------------
# Theme — sunflower palette
# ---------------------------------------------------------------------------

PETAL = "#F5B82E"          # sunflower yellow (header / accents)
PETAL_DEEP = "#E89B0E"     # darker yellow (secondary buttons)
PETAL_HOVER = "#C97F00"    # button hover
SEED = "#3D2914"           # sunflower-center brown (titles, primary button)
SEED_HOVER = "#5C3D1F"
STEM = "#5C7F2E"           # leaf green (positive accents)
WARN = "#B85C00"           # warm warning brown
CREAM = "#FFF7E0"          # app background
PAPER = "#FFFFFF"          # card background
INK = "#2B1810"            # body text
INK_SOFT = "#7A6346"       # secondary / label text
RULE = "#EAD9A8"           # subtle separators
HIGHLIGHT_BG = "#FFE082"   # <changed> highlight
HIGHLIGHT_FG = "#3D2914"
DISABLED_BG = "#E8DCB7"
DISABLED_FG = "#9C8770"

UI_FONT = "Segoe UI"  # falls back to system default on platforms without it

SYSTEM_PROMPT = r"""<target_author_profile>
<author_identity>Dr. Maria Ibrahim, Assistant Professor, Preventive Dental Sciences (Pediatric Dentistry & Dental Biomaterials), Imam Abdulrahman Bin Faisal University</author_identity>

<core_stylistic_rules>
- The model must dynamically shift its syntax based on the [Section Variable] selected by the user in the UI dropdown:
  - IF [Introduction]: Use short, declarative sentences (10-15 words). Apply the Swales CARS model to establish global epidemiological significance ("Dental caries is considered the most common...") before narrowing to the niche ("Combining X is a promising approach...").
  - IF [Materials & Methods]: Use exclusively agentless passive voice ("Forty-five extracted premolars were randomly assigned"). Eliminate the researcher ("We"). Create multi-clausal, highly dense sentences detailing exact physical parameters, chronological steps, and material names. Nominalize chemical processes (e.g., use "The demineralization resistance" instead of "The materials resisted demineralization").
  - IF [Results]: Use comparative, deterministic phrasing. Thematically front the material being tested (e.g., "Zirconia crowns exhibited..."). You MUST inject statistical proofs directly following qualitative claims (e.g., "exhibited highest flexural strength (p=0.0013)").
  - IF [Conclusion]: Use bi-directional structures. Look backward to summarize the empirical data, then look forward using epistemic hedging (e.g., "These findings suggest the potential use...", "may provide comparable...") to protect against clinical absolutes.
</core_stylistic_rules>

<lexical_fingerprint>
- Integrate dual-lexicon terminology bridging polymer science and pediatric pathology.
- Mandatory Biomaterials Lexicon: "resin-based sealants (RBS)", "remineralizing additives", "microleakage", "demineralization resistance", "bioactive composites", "thermocycling", "elastic modulus", "flexural strength", "nanoparticles".
- Mandatory Pediatric/Clinical Lexicon: "primary dentition", "secondary caries", "parental satisfaction", "pediatric clinical care", "gingival and periodontal health".
- Strict Transitional Markers: Use "Conversely," ONLY to pivot to a negative finding or material failure. Use "Furthermore," or "In addition," for additive layering. Use "Consequently," or "Thus," to bridge to a clinical action.
- Eliminate ALL colloquial LLM filler ("delve," "crucial," "testament to," "multifaceted").
- "Microleakage" must be treated lexically as the primary antagonistic force; "bioactive" is treated as the primary protagonist.
</lexical_fingerprint>

<representative_excerpts>
[EXCERPT 1 - RESULTS]: "Helioseal, Rainbow Flow, and BioCoat exhibited no microleakage and demonstrated complete retention (100%) post-thermocycling. Conversely, Beautifil Kids SA showed the highest level of microleakage, with significant differences noted (p=0.0013)..."
[EXCERPT 2 - CONCLUSION]: "The incorporation of remineralizing additives into sealants has been considered as a feasible way to prevent caries by potential remineralization through ions release. These findings suggest the potential use of these bioactive formulations as an approach to avoid the occurrence of secondary caries..."
[EXCERPT 3 - METHODS]: "Forty-five extracted, caries-free premolars were randomly assigned into five groups (n=9) according to the sealant material: two conventional resin-based sealants; Helioseal, one flowable composite; Rainbow Flow, and three bioactive sealants... Sealants were applied under standardized conditions and subjected to 10,000 thermocycles (5 °C–55 °C, dwell time 30 s)."
[EXCERPT 4 - INTRODUCTION]: "Dental caries in children is a leading worldwide concern for oral health. Combining calcium phosphate nanoparticles into sealants is a promising approach..."
[EXCERPT 5 - REVIEW CONCLUSION]: "Bioactive restorative materials demonstrated good marginal adaptation, high fracture resistance, positive gingival and periodontal health, high parental acceptance, and a reduced need for behavior management techniques. Thus, they represent a promising alternative in pediatric dentistry."
</representative_excerpts>
</target_author_profile>

<llm_backend_prompt_logic>
The backend must send the following strict operational mandates to the API for the actual paraphrasing task:
1. Re-write the input text to perfectly match the <target_author_profile> rules for the specific [Section Variable] selected by the user.
2. Maximize burstiness (variation in sentence length and structure) and lexical perplexity (specific, domain-precise word choice over generic phrasing) so the prose reads with the rhythm of an expert human author rather than uniform LLM cadence.
3. Isolate every altered, replaced, or new word/phrase and wrap it strictly in <changed> and </changed> tags. Do not wrap punctuation unless it was structurally altered.
4. Output ONLY the tagged text. Do not include any pleasantries, conversational filler, or Markdown formatting outside of the XML tags.
</llm_backend_prompt_logic>
"""


# ---------------------------------------------------------------------------
# Ollama call (local, offline)
# ---------------------------------------------------------------------------

def call_paraphraser(model: str, section: str, source_text: str) -> str:
    user_message = (
        f"[Section Variable]: {section}\n"
        f"[Source Text]:\n{source_text}"
    )
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": user_message,
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. "
            "Is the Ollama daemon running? Start it with `ollama serve`."
        ) from exc

    if response.status_code == 404:
        raise RuntimeError(
            f"Ollama reports model '{model}' is not available locally. "
            f"Pull it first: `ollama pull {model}`."
        )
    response.raise_for_status()

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned non-JSON response: {response.text[:200]}") from exc

    return data.get("response", "")


def fetch_installed_models() -> list[dict]:
    """Return raw model entries from Ollama's /api/tags. Empty on failure."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=TAGS_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return []
    try:
        return response.json().get("models", []) or []
    except json.JSONDecodeError:
        return []


def _parse_param_size_b(size_str: str) -> float:
    if not size_str:
        return 0.0
    cleaned = size_str.strip().rstrip("Bb").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def is_recommended(model_info: dict) -> bool:
    details = model_info.get("details") or {}
    return _parse_param_size_b(details.get("parameter_size", "")) >= RECOMMENDED_MIN_PARAMS_B


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------

CHANGED_TAG = re.compile(r"<changed>(.*?)</changed>", re.DOTALL)


def split_changed_spans(raw: str) -> list[tuple[str, bool]]:
    """Split the model output into (text, is_changed) segments."""
    segments: list[tuple[str, bool]] = []
    cursor = 0
    for match in CHANGED_TAG.finditer(raw):
        if match.start() > cursor:
            segments.append((raw[cursor:match.start()], False))
        segments.append((match.group(1), True))
        cursor = match.end()
    if cursor < len(raw):
        segments.append((raw[cursor:], False))
    return segments


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class ParaphraserApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MI Paraphraser — Dental Research Style Engine")
        self.geometry("1200x820")
        self.minsize(960, 640)

        self._apply_theme()
        self._build_layout()
        self._refresh_models()

    def _apply_theme(self) -> None:
        self.configure(bg=CREAM)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Frames
        style.configure("App.TFrame", background=CREAM)
        style.configure("Header.TFrame", background=PETAL)
        style.configure("Card.TFrame", background=PAPER)
        style.configure("CardOuter.TFrame", background=RULE)
        style.configure("Status.TFrame", background=CREAM)

        # Labels
        style.configure(
            "Title.TLabel",
            background=PETAL, foreground=SEED,
            font=(UI_FONT, 24, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=PETAL, foreground=SEED,
            font=(UI_FONT, 11),
        )
        style.configure(
            "FieldLabel.TLabel",
            background=PAPER, foreground=INK_SOFT,
            font=(UI_FONT, 9, "bold"),
        )
        style.configure(
            "Section.TLabel",
            background=CREAM, foreground=INK,
            font=(UI_FONT, 11, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=CREAM, foreground=INK_SOFT,
            font=(UI_FONT, 9),
        )
        style.configure(
            "Recommend.TLabel",
            background=PAPER, foreground=STEM,
            font=(UI_FONT, 10, "bold"),
        )

        # Secondary button (Refresh)
        style.configure(
            "TButton",
            padding=(14, 8),
            background=PETAL_DEEP,
            foreground=SEED,
            font=(UI_FONT, 10, "bold"),
            relief="flat", borderwidth=0,
            focusthickness=0,
        )
        style.map(
            "TButton",
            background=[("active", PETAL_HOVER), ("disabled", DISABLED_BG)],
            foreground=[("disabled", DISABLED_FG)],
        )

        # Primary button (Execute)
        style.configure(
            "Primary.TButton",
            padding=(22, 10),
            background=SEED,
            foreground=PETAL,
            font=(UI_FONT, 11, "bold"),
            relief="flat", borderwidth=0,
            focusthickness=0,
        )
        style.map(
            "Primary.TButton",
            background=[("active", SEED_HOVER), ("disabled", DISABLED_BG)],
            foreground=[("active", PETAL), ("disabled", DISABLED_FG)],
        )

        # Combobox
        style.configure(
            "TCombobox",
            fieldbackground=PAPER,
            background=PAPER,
            foreground=INK,
            arrowcolor=SEED,
            selectbackground=PETAL,
            selectforeground=SEED,
            bordercolor=RULE,
            lightcolor=RULE,
            darkcolor=RULE,
            padding=6,
            relief="flat",
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PAPER), ("disabled", DISABLED_BG)],
            foreground=[("disabled", DISABLED_FG)],
            bordercolor=[("focus", PETAL_DEEP)],
        )
        # Style the dropdown listbox attached to comboboxes
        self.option_add("*TCombobox*Listbox.background", PAPER)
        self.option_add("*TCombobox*Listbox.foreground", INK)
        self.option_add("*TCombobox*Listbox.selectBackground", PETAL)
        self.option_add("*TCombobox*Listbox.selectForeground", SEED)
        self.option_add("*TCombobox*Listbox.font", (UI_FONT, 10))
        self.option_add("*TCombobox*Listbox.borderWidth", 0)

    def _build_layout(self) -> None:
        # ── Header bar ────────────────────────────────────────────────
        header = ttk.Frame(self, style="Header.TFrame", padding=(28, 22, 28, 22))
        header.pack(fill="x")
        ttk.Label(header, text="MI Paraphraser", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            header,
            text="Dental Research Style Engine — local, offline, section-aware",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # Thin brown rule under the header
        tk.Frame(self, height=3, bg=SEED).pack(fill="x")

        # ── Status bar (packed first so it pins to the bottom) ────────
        statusbar = ttk.Frame(self, style="Status.TFrame", padding=(24, 8, 24, 12))
        statusbar.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            statusbar, textvariable=self.status_var, style="Status.TLabel"
        ).pack(side="left")

        # ── Control card ──────────────────────────────────────────────
        control_wrap = ttk.Frame(self, style="App.TFrame", padding=(24, 18, 24, 8))
        control_wrap.pack(fill="x")
        # 1px tan outer frame fakes a card border without a real shadow
        card_outer = ttk.Frame(control_wrap, style="CardOuter.TFrame", padding=1)
        card_outer.pack(fill="x")
        controls = ttk.Frame(card_outer, style="Card.TFrame", padding=(20, 16, 20, 16))
        controls.pack(fill="x")

        ttk.Label(controls, text="SECTION", style="FieldLabel.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.section_var = tk.StringVar(value=SECTIONS[0])
        self.section_dropdown = ttk.Combobox(
            controls,
            textvariable=self.section_var,
            values=SECTIONS,
            state="readonly",
            width=22,
        )
        self.section_dropdown.grid(
            row=1, column=0, sticky="w", padx=(0, 22), pady=(4, 0)
        )

        ttk.Label(controls, text="OLLAMA MODEL", style="FieldLabel.TLabel").grid(
            row=0, column=1, sticky="w", padx=(0, 8)
        )
        self.model_var = tk.StringVar(value="")
        self.model_dropdown = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            state="readonly",
            width=32,
        )
        self.model_dropdown.grid(
            row=1, column=1, sticky="w", padx=(0, 8), pady=(4, 0)
        )
        self.model_dropdown.bind("<<ComboboxSelected>>", self._on_model_select)

        self.refresh_button = ttk.Button(
            controls, text="↻  Refresh", command=self._refresh_models
        )
        self.refresh_button.grid(
            row=1, column=2, sticky="w", padx=(0, 22), pady=(4, 0)
        )

        self.recommendation_var = tk.StringVar(value="")
        self.recommendation_label = ttk.Label(
            controls,
            textvariable=self.recommendation_var,
            style="Recommend.TLabel",
        )
        self.recommendation_label.grid(
            row=1, column=3, sticky="w", padx=(0, 22), pady=(4, 0)
        )

        self.execute_button = ttk.Button(
            controls,
            text="Execute Paraphrase",
            command=self.on_execute,
            style="Primary.TButton",
        )
        self.execute_button.grid(row=1, column=4, sticky="e", pady=(4, 0))
        controls.columnconfigure(4, weight=1)  # push primary action to the right

        # ── Body: source / output text cards ──────────────────────────
        body = ttk.Frame(self, style="App.TFrame", padding=(24, 12, 24, 14))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        body.rowconfigure(3, weight=1)

        ttk.Label(body, text="Source Text", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(2, 6)
        )
        self.input_box = scrolledtext.ScrolledText(
            body,
            wrap="word",
            font=(UI_FONT, 11),
            bg=PAPER, fg=INK,
            insertbackground=SEED,
            relief="flat", borderwidth=0,
            highlightthickness=1,
            highlightbackground=RULE,
            highlightcolor=PETAL_DEEP,
            padx=14, pady=12,
            height=12,
        )
        self.input_box.grid(row=1, column=0, sticky="nsew", pady=(0, 16))

        ttk.Label(
            body,
            text="Paraphrased Output  ·  changes highlighted",
            style="Section.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(0, 6))
        self.output_box = scrolledtext.ScrolledText(
            body,
            wrap="word",
            font=(UI_FONT, 11),
            bg=PAPER, fg=INK,
            insertbackground=SEED,
            relief="flat", borderwidth=0,
            highlightthickness=1,
            highlightbackground=RULE,
            highlightcolor=PETAL_DEEP,
            padx=14, pady=12,
            height=12,
            state="disabled",
        )
        self.output_box.grid(row=3, column=0, sticky="nsew")
        self.output_box.tag_configure(
            "changed", background=HIGHLIGHT_BG, foreground=HIGHLIGHT_FG
        )

    # --------------------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------------------

    def on_execute(self) -> None:
        source_text = self.input_box.get("1.0", "end").strip()
        if not source_text:
            messagebox.showwarning("Empty input", "Please paste source text first.")
            return
        model = self._selected_model()
        if not model:
            messagebox.showwarning(
                "No model selected",
                "Pull a model with `ollama pull <name>` and click Refresh.",
            )
            return

        section = self.section_var.get()
        self.execute_button.configure(state="disabled")
        self.status_var.set(f"Querying Ollama ({model}) for {section}…")

        thread = threading.Thread(
            target=self._run_request,
            args=(model, section, source_text),
            daemon=True,
        )
        thread.start()

    def _run_request(self, model: str, section: str, source_text: str) -> None:
        try:
            raw = call_paraphraser(model, section, source_text)
            self.after(0, self._render_output, raw)
        except Exception as exc:  # noqa: BLE001 — surface any backend error
            self.after(0, self._render_error, exc)

    def _render_output(self, raw: str) -> None:
        segments = split_changed_spans(raw)
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", "end")
        for text, is_changed in segments:
            if is_changed:
                self.output_box.insert("end", text, ("changed",))
            else:
                self.output_box.insert("end", text)
        self.output_box.configure(state="disabled")
        self.status_var.set("Done.")
        self.execute_button.configure(state="normal")

    def _render_error(self, exc: BaseException) -> None:
        self.status_var.set("Error.")
        self.execute_button.configure(state="normal")
        messagebox.showerror("Ollama error", f"{type(exc).__name__}: {exc}")

    # --------------------------------------------------------------------
    # Model list / recommendation
    # --------------------------------------------------------------------

    def _refresh_models(self) -> None:
        self.status_var.set("Fetching installed models from Ollama…")
        self.update_idletasks()
        models = fetch_installed_models()
        if not models:
            self.model_dropdown.configure(values=[])
            self.model_var.set("")
            self.recommendation_var.set("")
            self.recommendation_label.configure(foreground="#555")
            self.execute_button.configure(state="disabled")
            self.status_var.set(
                "No models found. Is Ollama running? Try `ollama pull <model>`."
            )
            return

        decorated: list[tuple[bool, str, str]] = []
        for entry in models:
            name = entry.get("name", "")
            if not name:
                continue
            rec = is_recommended(entry)
            display = f"{name}{RECOMMENDED_SUFFIX}" if rec else name
            # Sort key: recommended first, then name asc
            decorated.append((not rec, name.lower(), display))
        decorated.sort()
        values = [display for _, _, display in decorated]

        self.model_dropdown.configure(values=values)
        if self.model_var.get() not in values:
            self.model_var.set(values[0])
        self._on_model_select()
        self.execute_button.configure(state="normal")
        self.status_var.set(f"{len(values)} model(s) available.")

    def _on_model_select(self, _event: object | None = None) -> None:
        raw = self.model_var.get().strip()
        if raw.endswith(RECOMMENDED_SUFFIX):
            self.recommendation_var.set("✓ Recommended for this task")
            self.recommendation_label.configure(foreground=STEM)
        elif raw:
            self.recommendation_var.set(
                "⚠ Smaller model — paraphrasing quality may be limited"
            )
            self.recommendation_label.configure(foreground=WARN)
        else:
            self.recommendation_var.set("")
            self.recommendation_label.configure(foreground=INK_SOFT)

    def _selected_model(self) -> str:
        raw = self.model_var.get().strip()
        if raw.endswith(RECOMMENDED_SUFFIX):
            raw = raw[: -len(RECOMMENDED_SUFFIX)]
        return raw


def main() -> None:
    ParaphraserApp().mainloop()


if __name__ == "__main__":
    main()
