"""MI Paraphraser — desktop GUI for section-aware academic paraphrasing.

Tkinter front-end + Anthropic Claude back-end. The model is instructed to wrap
every altered word/phrase in <changed>...</changed> tags; the UI strips the
tags and highlights the content in the output pane.
"""

from __future__ import annotations

import os
import re
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from anthropic import Anthropic


SECTIONS = ["Introduction", "Materials & Methods", "Results", "Conclusion"]
MODEL_ID = "claude-opus-4-7"
MAX_TOKENS = 4096

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
ROLE: You are a paraphrasing engine that rewrites the user's input text to match the voice, syntax, and lexicon defined in <target_author_profile>.

INPUTS: Each user turn supplies a [Section Variable] and a [Source Text]. The Section Variable is one of: Introduction, Materials & Methods, Results, Conclusion.

PROCEDURE:
1. Identify the Section Variable and apply the corresponding rule from <core_stylistic_rules>.
2. Rewrite the [Source Text] preserving its empirical meaning, numerical values, citations, and statistical results. Never fabricate data, p-values, sample sizes, or citations.
3. Substitute generic vocabulary with terms from the Mandatory Biomaterials and Pediatric/Clinical Lexicons where semantically appropriate. Honor the transitional-marker rules and the "microleakage = antagonist / bioactive = protagonist" framing.
4. Strip all colloquial LLM filler.

OUTPUT CONTRACT (strict):
- Return ONLY the rewritten text. No preface, no explanation, no headings, no Markdown fences.
- Wrap every word, phrase, or punctuation token that differs from the source in <changed>...</changed> tags. Tokens identical to the source remain untagged. Do not nest tags. Do not tag whole paragraphs wholesale; tag at the smallest contiguous span that captures the change.
- Do not emit any tag other than <changed>. Do not echo the [Section Variable] or [Source Text] labels.
</llm_backend_prompt_logic>
"""


# ---------------------------------------------------------------------------
# Anthropic call
# ---------------------------------------------------------------------------

def call_paraphraser(section: str, source_text: str) -> str:
    client = Anthropic()
    user_message = (
        f"[Section Variable]: {section}\n"
        f"[Source Text]:\n{source_text}"
    )
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


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
        self.geometry("1100x780")
        self.minsize(900, 600)

        self._build_layout()

    def _build_layout(self) -> None:
        controls = ttk.Frame(self, padding=(12, 12, 12, 6))
        controls.pack(fill="x")

        ttk.Label(controls, text="Section:").pack(side="left")
        self.section_var = tk.StringVar(value=SECTIONS[0])
        self.section_dropdown = ttk.Combobox(
            controls,
            textvariable=self.section_var,
            values=SECTIONS,
            state="readonly",
            width=22,
        )
        self.section_dropdown.pack(side="left", padx=(6, 18))

        self.execute_button = ttk.Button(
            controls, text="Execute Paraphrase", command=self.on_execute
        )
        self.execute_button.pack(side="left")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(controls, textvariable=self.status_var, foreground="#555").pack(
            side="right"
        )

        body = ttk.Frame(self, padding=(12, 6, 12, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        body.rowconfigure(3, weight=1)

        ttk.Label(body, text="Source Text").grid(row=0, column=0, sticky="w")
        self.input_box = scrolledtext.ScrolledText(
            body, wrap="word", font=("TkDefaultFont", 11), height=12
        )
        self.input_box.grid(row=1, column=0, sticky="nsew", pady=(2, 10))

        ttk.Label(body, text="Paraphrased Output (changes highlighted)").grid(
            row=2, column=0, sticky="w"
        )
        self.output_box = scrolledtext.ScrolledText(
            body,
            wrap="word",
            font=("TkDefaultFont", 11),
            height=12,
            state="disabled",
        )
        self.output_box.grid(row=3, column=0, sticky="nsew", pady=(2, 0))
        self.output_box.tag_configure(
            "changed", background="#FFF59D", foreground="#000000"
        )

    # --------------------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------------------

    def on_execute(self) -> None:
        source_text = self.input_box.get("1.0", "end").strip()
        if not source_text:
            messagebox.showwarning("Empty input", "Please paste source text first.")
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            messagebox.showerror(
                "Missing API key",
                "ANTHROPIC_API_KEY is not set in the environment.",
            )
            return

        section = self.section_var.get()
        self.execute_button.configure(state="disabled")
        self.status_var.set(f"Calling Claude ({MODEL_ID}) for {section}…")

        thread = threading.Thread(
            target=self._run_request, args=(section, source_text), daemon=True
        )
        thread.start()

    def _run_request(self, section: str, source_text: str) -> None:
        try:
            raw = call_paraphraser(section, source_text)
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
        messagebox.showerror("API error", f"{type(exc).__name__}: {exc}")


def main() -> None:
    ParaphraserApp().mainloop()


if __name__ == "__main__":
    main()
