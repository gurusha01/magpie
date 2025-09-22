# MPI: Few-Shot Prompting Based Pipeline

## Overview
- **Generates** JSON datapoints from scenario prompts via OpenAI Chat Completions.  
- **Evaluates** them with a rubric: structure, consistency, privacy, quantifiability, originality, completeness.  
- **Scores & gates** results deterministically and adds **static checks**.  
- **Writes** generated data, eval reports, and any errors to disk.

> **Entry point:** `python your_script.py` (rename if needed)

---

## Features
- Few-shot **sample loading** from `sample/<category>/*.json`
- **Prompt injection** from `prompts.yaml` (read as plain text)
- Robust **JSON coercion** from model output (strips code fences, fixes trailing commas)
- Deterministic **scoring** with weights & pass threshold
- **Static checks:** list enforcement for `/agents`, `/tasks`; duplicate `id` detection
- Clear **on-disk outputs** (success, parse errors, crashes)

---

## Requirements
- **Python:** 3.10+
- **Packages:**  
  ```txt
  openai>=1.0.0
  ```

Install:
```bash
pip install -r requirements.txt
```

---

## Quickstart

### 1) Environment
```bash
export OPENAI_API_KEY=sk-********************************
# Optional:
export MODEL_GEN=gpt-4o-mini
export MODEL_EVAL=gpt-4o-mini
export EVAL_PASS_THRESHOLD=80
```

### 2) Inputs
- **prompts.yaml** — system instructions (read as plain text).  
- **sample/** — optional few-shot JSON examples:
  ```
  sample/
    try/
      example1.json
      example2.json
    economic/
      ...
  ```

### 3) Run
```bash
python your_script.py
```

The script iterates the `scenarios` mapping in `__main__` and performs:
**generate → parse → static checks → evaluate → write reports**.

---

## Directory Layout
```
.
├─ prompts.yaml
├─ sample/
│  └─ <category>/*.json
├─ data/
│  ├─ data/<category>/<key>.json        # generated datapoints
│  ├─ eval/<category>/<key>.json        # evaluation reports
│  ├─ <category>/<key>.error.json       # generator parse errors
│  └─ <category>/<key>.crash.json       # unhandled exceptions
└─ your_script.py
```

---

## Configuration

### Environment variables
- `OPENAI_API_KEY` *(required)*  
- `MODEL_GEN` *(default: `gpt-4o-mini`)*  
- `MODEL_EVAL` *(default: `gpt-4o-mini`)*  
- `EVAL_PASS_THRESHOLD` *(default: `80`)* — pass if `overall_score ≥ threshold` **and** no `severity="error"` issues

### Tuning (code constants)
- Few-shot cap: `max_chars=12000`, `max_files=6` in `load_sample_block`
- Generator temperature: `0.6`
- Evaluator temperature: `0.0` (deterministic)

---

## Message Contracts

### Generator
- **System:** `prompts.yaml` text  
- **System (optional):** `REFERENCE SAMPLES:
<concatenated sample JSONs>`
- **User:**
  ```
  Scenario:
  <scenario_line>
  Output JSON only.
  ```

**Expected output:** Valid JSON.  
`coerce_json()` tolerates code fences, trailing commas, and leading/trailing text.  
On parse failure → `data/<category>/<key>.error.json` with `raw_text` and error.

### Evaluator
- **System:** `prompts.yaml`  
- **System:** `SAMPLES:` + samples or `(none)`  
- **System:** rubric (added by code; includes pass rule & output schema)  
- **User:** the generated datapoint JSON

**Expected output shape (canonical):**
```json
{
  "scores": {
    "structure": 0,
    "consistency": 0,
    "privacy": 0,
    "quantifiability": 0,
    "originality": 0,
    "completeness": 0
  },
  "overall_score": 0,
  "passes": false,
  "issues": [
    { "path": "/...", "severity": "info|warn|error", "reason": "..." }
  ]
}
```

---

## Scoring

### Weights (sum = 17)
- structure **4**
- consistency **3**
- privacy **3**
- quantifiability **3**
- originality **2**
- completeness **2**

**Overall score** = weighted (0–5) × 20 → integer **0–100**.  
**Pass** iff `overall_score ≥ EVAL_PASS_THRESHOLD` **and** no `error` issues.

---

## Static Checks (pre-eval)
- `/agents` must be a **list** if present  
- `/tasks` must be a **list** if present  
- All `id` values across the document must be **unique**  
Findings are appended to evaluator `issues`.

---

## Example Run
```bash
python your_script.py
# try::Start
# try/gifting: Generating
# try/gifting::EVAL PASS (score=84) -> data/eval/try/gifting.json
# ...
# try::Done
```

Artifacts:
- `data/data/try/gifting.json` (datapoint)  
- `data/eval/try/gifting.json` (report)

---

## Add Your Own Scenarios
Edit `scenarios` in `__main__`:
```python
scenarios = {
  "my_category": {
    "my_key": "One-line scenario with hidden/private factors"
  }
}
```
Optionally add few-shot JSONs under `sample/my_category/`.

---

## Prompts
- `prompts.yaml` is injected **as text** (not parsed as YAML data).
- Keep instructions clear on **output JSON schema** and **privacy constraints**.
- Evaluator rubric is appended automatically.

---

## Troubleshooting
- **`OPENAI_API_KEY is not set`** → export the key before running  
- **Parse failures** → inspect `*.error.json` (`raw_text`), tighten prompts, reduce temperature, add format-strict samples  
- **Unstable/low scores** → keep evaluator at `eval_temperature=0.0`, refine rubric text in `prompts.yaml`  
- **Duplicate IDs / type errors** → strengthen generator prompt to enforce shapes & uniqueness

---

## Security & Privacy
- Use only synthetic or anonymized data in scenarios.  
- Review `data/` outputs before sharing.

---

## Extending
- Swap models via `MODEL_GEN` / `MODEL_EVAL`  
- Add richer static checks (e.g., JSON Schema)  
- Emit CSV summaries from `data/eval/` for dashboards  
- Add CLI args (e.g., `--category`, `--key`, `--prompt`, `--models`)

---

## License
MIT (or your choice)

## Acknowledgements
- OpenAI API for chat completion models


