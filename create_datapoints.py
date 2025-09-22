import os
import json
import glob
import math  # numeric safety helpers (e.g., math.isfinite)
from typing import Any
from openai import OpenAI

# ==============================
# I/O HELPERS
# ==============================

def read_yaml_text(path: str) -> str:
    # Read a UTF-8 text file (often a YAML prompt) and return its raw contents.
    # Using a context manager ensures the file handle is closed on success/failure.
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_text(path: str) -> str:
    # Read a generic UTF-8 text file and return its content as a string.
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_json(path: str, obj: Any) -> None:
    # Serialize a Python object as pretty JSON to `path`.
    # - Creates parent directories if they do not exist.
    # - ensure_ascii=False preserves unicode; indent=2 for stable diffs.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ==============================
# JSON COERCION / SANITIZATION
# ==============================

def coerce_json(text: str) -> Any:
    # Robustly coerce a model response into valid JSON.
    # Steps:
    # 1) Guard against None/empty.
    # 2) Strip Markdown code fences (``` or ```json).
    # 3) Slice the outermost {...} block (handles extra prose).
    # 4) Patch frequent trailing-comma artifacts.
    # 5) Parse with json.loads (let it raise if still malformed).
    s = (text or "").strip()
    if not s:
        raise ValueError("empty response")

    # Remove code fences if the response is fenced as ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
        s = "\n".join(lines).strip()

    # Heuristic: keep only the first full JSON object between the first '{' and last '}'.
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]

    # Patch common LLM artifact: a trailing comma right before } or ].
    # We only replace when followed by a newline to avoid false positives inside strings.
    s = s.replace(",\n}", "\n}").replace(",\n]", "\n]")

    # Parse (may raise json.JSONDecodeError; that is OK to surface to caller).
    return json.loads(s)

# ==============================
# OPENAI CLIENT + SINGLE CALL
# ==============================

def make_client() -> OpenAI:
    # Instantiate the OpenAI client.
    # Early, explicit failure if OPENAI_API_KEY is not configured helps debugging.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()

def chat_once(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.6,
    response_format: dict[str, Any] | None = None
) -> str:
    # Make a single Chat Completions API call; return assistant text content (or "" if None).
    # `response_format={"type":"json_object"}` can be passed for stricter JSON outputs on supported models.
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if response_format:
        kwargs["response_format"] = response_format
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""

# ==============================
# FEW-SHOT SAMPLE LOADER
# ==============================

def load_sample_block(
    category: str,
    sample_root: str = "sample",
    max_chars: int = 12000,
    max_files: int = 6
) -> str:
    # Build a concatenated block of sample JSON files for few-shot prompting.
    # Selection policy:
    # - Prefer `sample/<category>/*.json`; fall back to `sample/*.json`.
    # - Cap by `max_files` and `max_chars` (with ~200 chars/headroom per file for headers/delimiters).
    # - Prefix each sample with "# <filename>" and separate with "---".
    cat_glob = os.path.join(sample_root, category, "*.json")
    all_glob = os.path.join(sample_root, "*.json")
    paths = sorted(glob.glob(cat_glob)) or sorted(glob.glob(all_glob))
    if not paths:
        return ""

    block_parts, used = [], 0
    for p in paths[:max_files]:
        try:
            txt = read_text(p).strip()
        except Exception:
            # Skip unreadable files and continue assembling others.
            continue
        if not txt:
            continue
        # Keep some headroom for header and delimiter to avoid exceeding `max_chars`.
        if used + len(txt) + 200 > max_chars:
            break
        block_parts.append(f"# {os.path.basename(p)}\n{txt}")
        used += len(txt) + 200

    return "\n\n---\n".join(block_parts)

# ==============================
# MESSAGE BUILDERS
# ==============================

def build_generator_messages(
    prompt_path: str,
    scenario_line: str,
    category: str,
    sample_root: str = "sample"
) -> list[dict[str, str]]:
    # Construct messages for the generator model:
    # - System: core instructions from YAML.
    # - Optional System: "REFERENCE SAMPLES" block for few-shot guidance.
    # - User: scenario line + explicit "Output JSON only." to discourage prose.
    sys_prompt = read_yaml_text(prompt_path)
    samples = load_sample_block(category, sample_root)
    msgs: list[dict[str, str]] = [{"role": "system", "content": sys_prompt}]
    if samples:
        msgs.append({"role": "system", "content": f"REFERENCE SAMPLES:\n{samples}"})
    msgs.append({"role": "user", "content": f"Scenario:\n{scenario_line}\nOutput JSON only."})
    return msgs

def build_evaluator_messages(
    prompt_path: str,
    category: str,
    datapoint: dict[str, Any],
    sample_root: str = "sample"
) -> list[dict[str, str]]:
    # Construct messages for the evaluator model:
    # - System: evaluation instructions.
    # - System: sample references for calibration (may be empty).
    # - System: explicit rubric with pass/fail rule (threshold from env var).
    # - User: the generated datapoint (pretty-printed JSON).
    sys_prompt = read_yaml_text(prompt_path)
    samples = load_sample_block(category, sample_root)
    pass_threshold = os.getenv("EVAL_PASS_THRESHOLD", "80")
    rubric = (
        "Evaluate JSON datapoint vs schema. "
        "Score each 0–5: structure, consistency, privacy, quantifiability, originality, completeness. "
        "Return JSON: {scores, overall_score, passes, issues[]}. "
        f"PASS if overall_score >= {pass_threshold} and no issues with severity='error'."
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "system", "content": f"SAMPLES:\n{samples or '(none)'}"},
        {"role": "system", "content": rubric},
        {"role": "user", "content": json.dumps(datapoint, ensure_ascii=False, indent=2)},
    ]

# ==============================
# SCORING HELPERS (DETERMINISTIC)
# ==============================

# Weighted criteria; weights sum to 17. Higher weight = more influence on overall.
_WEIGHTS = {
    "structure": 4,
    "consistency": 3,
    "privacy": 3,
    "quantifiability": 3,
    "originality": 2,
    "completeness": 2,
}

def _to_num(x, default: float = 0.0) -> float:
    # Convert arbitrary input to a finite float; fallback to `default` on failure or non-finite values.
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def _clamp(v: float, lo: float, hi: float) -> float:
    # Clamp a numeric value into the inclusive range [lo, hi].
    return max(lo, min(hi, v))

def compute_overall(scores: dict[str, Any]) -> int:
    # Compute a 0–100 overall score from 0–5 criterion scores using the fixed weights.
    # Rounds to nearest int for readability and stable reporting.
    total_w = sum(_WEIGHTS.values()) * 5.0  # maximum possible weighted sum (all scores = 5)
    acc = 0.0
    for k, w in _WEIGHTS.items():
        v = _clamp(_to_num(scores.get(k, 0)), 0.0, 5.0)
        acc += v * w
    return int(round((acc / total_w) * 100))

# ==============================
# STATIC STRUCTURAL CHECKS
# ==============================

def basic_static_checks(d: dict[str, Any]) -> list[dict[str, str]]:
    # Quick programmatic checks for obvious structural issues:
    # - /agents must be a list if present.
    # - /tasks must be a list if present.
    # - All 'id' values across the object graph must be unique.
    issues = []

    if "agents" in d and not isinstance(d["agents"], list):
        issues.append({"path": "/agents", "severity": "error", "reason": "agents not list"})
    if "tasks" in d and not isinstance(d["tasks"], list):
        issues.append({"path": "/tasks", "severity": "error", "reason": "tasks not list"})

    # Recursively traverse to collect all 'id' occurrences for uniqueness check.
    ids = []
    def collect_ids(node):
        # Depth-first traversal for dicts/lists; collect id fields as strings.
        if isinstance(node, dict):
            if "id" in node:
                ids.append(str(node["id"]))
            for v in node.values():
                collect_ids(v)
        elif isinstance(node, list):
            for v in node:
                collect_ids(v)

    collect_ids(d)

    if len(ids) != len(set(ids)):
        issues.append({"path": "/**/id", "severity": "error", "reason": "duplicate ids"})

    return issues

# ==============================
# PIPELINE ORCHESTRATION
# ==============================

def run_pipeline(
    category: str,
    key: str,
    scenario: str,
    prompt_path: str,
    model_gen: str = "gpt-4o-mini",
    model_eval: str = None,
    eval_temperature: float = 0.0
) -> None:
    # End-to-end process for a single scenario:
    # 1) Call generator model -> get raw text.
    # 2) Coerce to JSON; write datapoint to disk.
    # 3) Run static checks to pre-catch structural issues.
    # 4) Call evaluator with rubric; parse/normalize scores.
    # 5) Compute deterministic overall; combine issues; decide PASS/FAIL.
    # 6) Persist evaluation report.
    client = make_client()
    model_eval = model_eval or os.getenv("MODEL_EVAL", "gpt-4o-mini")

    # Parse/normalize pass threshold from env with clamping to [0,100].
    pass_threshold_env = os.getenv("EVAL_PASS_THRESHOLD", "80")
    pass_threshold = int(_clamp(_to_num(pass_threshold_env, 80.0), 0.0, 100.0))

    # ---- Generation
    print(f"{category}/{key}: Generating")
    gen_msgs = build_generator_messages(prompt_path, scenario, category)
    gen_text = chat_once(client, model_gen, gen_msgs, temperature=0.6)

    # Try to parse the model output into a Python object; on failure, save the raw output for debugging.
    try:
        datapoint = coerce_json(gen_text)
    except Exception as e:
        write_json(f"data/{category}/{key}.error.json", {"error": str(e), "raw_text": gen_text})
        print(f"{category}/{key}::FAIL (parse)")
        return

    # Save the generated datapoint.
    # NOTE: This path currently writes to "data/data/..."; verify this is intentional (may be a double 'data').
    out_path = f"data/data/{category}/{key}.json"
    write_json(out_path, datapoint)

    # ---- Static checks before evaluation
    static_issues = basic_static_checks(datapoint)

    # ---- Evaluation
    eval_msgs = build_evaluator_messages(prompt_path, category, datapoint)
    eval_text = chat_once(
        client,
        model_eval,
        eval_msgs,
        temperature=eval_temperature,
        response_format={"type": "json_object"}  # request structured JSON if supported
    )

    # Initialize a stable report structure so downstream consumers can rely on keys.
    eval_report = {
        "scores": {},
        "overall_score": 0,
        "passes": False,
        "issues": [],
        "meta": {"model_eval": model_eval, "threshold": pass_threshold},
    }

    # Parse evaluator output; if malformed, record a parse error and continue with defaults.
    try:
        parsed = coerce_json(eval_text)
    except Exception as e:
        eval_report["issues"].append({"path": "/", "severity": "error", "reason": f"eval parse {e}"})
        parsed = {}

    # Merge evaluator-reported issues (if present) before static ones, so ordering roughly reflects origin.
    if isinstance(parsed.get("issues"), list):
        eval_report["issues"].extend(parsed["issues"])

    # Normalize per-criterion scores to floats in [0,5]; ignore unknown keys, fill missing with 0.
    raw_scores = parsed.get("scores", {}) if isinstance(parsed.get("scores"), dict) else {}
    norm_scores = {k: float(_clamp(_to_num(raw_scores.get(k, 0)), 0.0, 5.0)) for k in _WEIGHTS.keys()}
    eval_report["scores"] = norm_scores

    # Deterministically compute overall percentage.
    overall = compute_overall(norm_scores)
    eval_report["overall_score"] = overall

    # Add static issues and decide final pass/fail (must meet threshold and have no 'error' issues).
    eval_report["issues"].extend(static_issues)
    has_err = any((iss.get("severity") or "").lower() == "error" for iss in eval_report["issues"])
    eval_report["passes"] = (overall >= pass_threshold) and not has_err
    eval_report["meta"]["threshold"] = pass_threshold  # store sanitized numeric threshold

    # Persist evaluation report and emit a concise status line for CLI visibility.
    eval_path = f"data/eval/{category}/{key}.json"
    write_json(eval_path, eval_report)
    print(f"{category}/{key}::EVAL {'PASS' if eval_report['passes'] else 'FAIL'} "
          f"(score={eval_report.get('overall_score', 0)}) -> {eval_path}")

def run_category_scenarios(
    scenarios_by_category: dict[str, dict[str, str]],
    prompt_file: str,
    model_gen: str = "gpt-4o-mini",
    model_eval: str | None = None
) -> None:
    # Iterate all categories and their (key -> scenario) mappings and run the full pipeline for each.
    # Each scenario is isolated by try/except so a failure does not halt the batch.
    for category, scenarios in scenarios_by_category.items():
        print(f"{category}::Start")
        for key, scenario_text in scenarios.items():
            try:
                run_pipeline(category, key, scenario_text, prompt_file, model_gen, model_eval)
            except Exception as e:
                # Persist crash info for post-mortem; continue to next scenario.
                write_json(f"data/{category}/{key}.crash.json", {"error": str(e)})
                print(f"{category}/{key}::FAIL (exception)")
        print(f"{category}::Done")

# MAIN
if __name__ == "__main__":
    scenarios = {
    "try":{
        "gifting":"Shared gift purchasing for a collegue with personal sensitive knowledge about the employee and budget constraints",
        "seating":"event seating arrangements between four people with hidden personal likes and dislikes", 
        "recruitment":"Talent recruitment meeting between an HR, an interviewer and a VP with hidden personal relations and biases towards the candidate", 
        "allocation":"Strategic resource allocation of GPUs between 3 teams with private project details and team preferences of regions of GPUs",
        "networking": "Network construction planning between an engineer, a government employee and an environmentalist with private details about other similar projects, future governemnt policies and environmentalist's knowledge of low impact of certain materials",
        "salary": "Salary negotiations between an HR and a selected candidate with private details about other employees salaries and offers to the candidate from other companies",
        "admissions": "University admission processes with an admin head, two professors and 2 applicants. private details include applicant impressions on professors, exact funding amounts and student's personal geographical preferences",
    },
    "economic":{
        "auction": "Art auction between collectors, museum representatives, and anonymous phone bidders with private bidding limits, authentication concerns, and hidden motivations for specific pieces",
        "real_estate": "Property negotiation between seller, buyer, and agents with undisclosed budget ceiling, hidden structural issues, and competing offers",
        "merger": "Corporate merger discussion between CEOs, board representatives, and shareholders with confidential company valuations, undisclosed liabilities, and private succession plans",
        "job_market": "Executive talent acquisition involving headhunters, competing companies, and candidates with private salary expectations, undisclosed company financial forecasts, and candidate's hidden career motivations",
        "salary": "Compensation negotiation between department head, HR director, and candidate with confidential information about internal pay scales, candidate's competing offers, and budget constraints",
        "venture_capital": "Startup funding round between founders, multiple investors, and board members with private company valuation data, undisclosed term sheet offers, and strategic growth plans",
        "procurement": "Government contract bidding between procurement officers, vendors, and oversight committee with confidential budget limits, undisclosed technical requirements, and private vendor capabilities",
        "supply_chain": "Manufacturing partnership negotiation between suppliers, manufacturers, and distributors with private cost structures, confidential delivery timelines, and undisclosed alternative sourcing options",
        "freelance": "Project contract negotiation between client, freelancers, and project manager with hidden budget constraints, undisclosed project scope changes, and private timeline pressures"
    },
    "scheduling":{
        "meetings": "Executive board meeting coordination between C-suite leaders with private availability constraints, hidden agenda priorities, and confidential travel schedules",
        "conference": "International summit scheduling between diplomatic representatives with undisclosed security concerns, private political considerations, and confidential availability of key stakeholders",
        "academic": "University course scheduling between department heads with private faculty preferences, hidden budget constraints, and confidential information about upcoming program changes",
        "resource": "Research laboratory space allocation between competing scientific teams with undisclosed grant funding details, private equipment requirements, and confidential breakthrough timelines",
        "timeline": "High-profile product launch timeline negotiation between engineering, marketing, and sales teams with private technical challenges, undisclosed market intelligence, and confidential competitive deadlines",
        "calendar": "Merger integration planning between two corporate leadership teams with private strategic priorities, undisclosed personnel decisions, and confidential regulatory compliance deadlines",
        "shift_work": "Hospital emergency department staffing between administration and medical staff with private budget constraints, undisclosed personnel issues, and confidential information about expected patient surges",
        "research": "Multinational scientific collaboration between research institutions with private funding limitations, undisclosed preliminary findings, and confidential intellectual property concerns",
        "travel": "Diplomatic mission planning between government officials, security teams, and foreign counterparts with private security threats, undisclosed meeting objectives, and confidential political considerations"
    },
    "resource_allocation":{
        "living_space": "Luxury apartment co-ownership arrangement between professionals with private financial situations, undisclosed lifestyle habits, and hidden long-term housing plans",
        "roommates": "Executive housing arrangement for relocating corporate leaders with private personal habits, undisclosed family visitation plans, and confidential career trajectory information",
        "team_formation": "Olympic team selection committee deliberations with private performance data, undisclosed athlete health information, and confidential sponsorship considerations",
        "dating": "High-profile matchmaking service for public figures with private relationship histories, undisclosed personal requirements, and confidential career and political aspirations",
        "networking": "Industry consortium formation between competing company representatives with private strategic objectives, undisclosed technological capabilities, and confidential market intelligence",
        "grants": "Medical research funding allocation between committee members with private conflicts of interest, undisclosed preliminary research findings, and confidential information about parallel research efforts",
        "admissions": "Elite university admission committee deliberations with private donor influence information, undisclosed enrollment targets, and confidential applicant background details",
        "research_formation": "International pandemic response team assembly between government agencies with private resource limitations, undisclosed national security concerns, and confidential epidemiological data"
    },
    "transportation":{
        "ride_sharing": "Emergency medical transport coordination between hospitals, ambulance services, and air evacuation teams with private patient information, undisclosed resource limitations, and confidential facility capacity data",
        "delivery": "Time-critical vaccine distribution planning between public health officials, logistics companies, and healthcare facilities with private supply quantities, undisclosed storage capabilities, and confidential priority recipient lists",
        "transportation": "Corporate executive transportation scheduling between security teams, executives, and transportation providers with private threat assessment data, undisclosed meeting locations, and confidential acquisition negotiations",
        "carpooling": "Government official secure travel arrangements between security personnel, officials, and diplomatic staff with private security protocols, undisclosed meeting agendas, and confidential diplomatic relationship information",
        "fleet": "Military equipment relocation planning between commanders, logistics officers, and intelligence analysts with private threat assessments, undisclosed strategic objectives, and confidential equipment capabilities",
        "shipping": "High-value art collection transportation between museums, insurers, and security firms with private valuation details, undisclosed security vulnerabilities, and confidential exhibition negotiation details",
        "emergency": "Natural disaster response resource allocation between government agencies, NGOs, and local authorities with private population vulnerability data, undisclosed resource limitations, and confidential political considerations"
    },
    "tech_and_infra":{
        "network_planning": "Critical national infrastructure cybersecurity strategy planning between government agencies, private corporations, and security experts with classified vulnerability assessments, undisclosed defense capabilities, and confidential threat intelligence",
        "telecom": "International telecommunications expansion negotiation between government regulators, competing providers, and local authorities with private market strategy plans, undisclosed technical limitations, and confidential political considerations",
        "routing": "Global financial system data routing optimization between central banks, private institutions, and security specialists with private transaction volumes, undisclosed security protocols, and confidential contingency plans",
        "satellite": "Military and civilian satellite orbit allocation between defense agencies, commercial operators, and international regulatory bodies with classified mission requirements, undisclosed technological capabilities, and confidential strategic priorities",
        "cloud_computing": "Government classified data migration planning between intelligence agencies, cloud providers, and security teams with private security clearance details, undisclosed technical vulnerabilities, and confidential operational requirements",
        "data_center": "Strategic data center placement negotiation between tech giants, energy providers, and local governments with private expansion plans, undisclosed environmental impact data, and confidential economic incentive packages",
        "spectrum": "5G wireless spectrum auction between telecommunications companies, government regulators, and defense agencies with private bidding limits, undisclosed technology roadmaps, and confidential national security requirements"
    }, 
    "policy_making":{
        "policy": "Healthcare reform negotiation between legislators, industry lobbyists, and public health experts with private political donor pressures, undisclosed economic impact forecasts, and confidential voting bloc analyses",
        "infrastructure": "Major dam construction planning between government officials, environmental agencies, and indigenous representatives with private economic development plans, undisclosed environmental impact data, and confidential ancestral land claims",
        "cross_agency": "Counter-terrorism operation coordination between intelligence agencies, military commands, and diplomatic corps with classified asset information, undisclosed operational capabilities, and confidential diplomatic negotiations at risk",
        "treaty": "Climate accord negotiation between developed nations, developing countries, and scientific advisors with private economic impact assessments, undisclosed industrial emission data, and confidential alternative energy technology capabilities",
        "resource_sharing": "National emergency response coordination between federal agencies, state governments, and military units with private resource inventories, undisclosed vulnerability assessments, and confidential deployment capabilities",
        "urban_planning": "Smart city development between government officials, technology companies, and community representatives with private surveillance capabilities, undisclosed gentrification concerns, and confidential investment commitments",
        "conservation": "Protected marine area establishment between multiple nations, fishing industries, and environmental organizations with private economic impact data, undisclosed endangered species information, and confidential alternative fishing ground intelligence"
    },
    "social_personal":{
        "dating": "Celebrity relationship arrangement between public figures, publicists, and managers with private career trajectory plans, undisclosed public image concerns, and confidential personal relationship preferences",
        "social_event": "High-profile charity gala coordination between political figures, celebrity attendees, and wealthy donors with private political rivalries, undisclosed donation expectations, and confidential security concerns",
        "vacation": "Executive retreat planning between board members, corporate security, and facility managers with private company transition plans, undisclosed merger discussions, and confidential personal conflicts between leadership",
        "gifting": "Diplomatic gift exchange planning between government officials, cultural advisors, and security personnel with private symbolic significance information, undisclosed political tensions, and confidential recipient preferences",
        "inheritance": "Multi-billion dollar estate distribution negotiation between family members, business stakeholders, and legal representatives with private alliance formations, undisclosed asset valuations, and confidential information about contested wills",
        "conflict": "High-stakes corporate mediation between executives, board members, and legal teams with private litigation strategies, undisclosed financial implications, and confidential personal motivations behind business decisions",
        "seating": "International diplomatic dinner arrangement between protocol officers, security teams, and political advisors with private diplomatic tensions, undisclosed alliance negotiations, and confidential intelligence about interpersonal conflicts"
    }, 
    "research":{
        "collaboration": "Breakthrough medical research partnership formation between pharmaceutical companies, research institutions, and government agencies with private intellectual property concerns, undisclosed preliminary findings, and confidential market strategy plans",
        "conference": "International security symposium planning between defense agencies, academic experts, and intelligence communities with private geopolitical risk assessments, undisclosed emerging threat data, and confidential attendance of covert operatives",
        "grant": "Cutting-edge quantum computing research funding application evaluation between government agencies, university consortiums, and private industry with private technological capabilities, undisclosed national security implications, and confidential competitive intelligence",
        "peer_review": "High-impact pandemic research evaluation between journal editors, scientific reviewers, and public health officials with private political pressures, undisclosed competing research, and confidential information about potential applications",
        "academic_job": "Elite university department hiring negotiation between administration, faculty committees, and candidates with private budget constraints, undisclosed strategic direction changes, and confidential information about competing offers",
        "student_advisor": "Prestigious laboratory placement decisions between faculty researchers, department heads, and doctoral candidates with private research funding situations, undisclosed interpersonal dynamics, and confidential information about future project directions",
        "research_resource": "Critical scientific equipment allocation between competing research teams, administrators, and funding agencies with private experimental timelines, undisclosed breakthrough proximity, and confidential information about potential publications"
    }, 
    "healthcare":{
        "clinical_trials": "Breakthrough cancer treatment trial participant selection between oncologists, pharmaceutical researchers, and hospital ethics boards with private patient prognosis data, undisclosed treatment efficacy concerns, and confidential financial interests in outcomes",
        "organ_donation": "Critical organ transplant matching between transplant centers, medical teams, and patient advocates with private medical urgency assessments, undisclosed donor quality information, and confidential details about patient compliance history",
        "medical_resource": "Pandemic emergency equipment allocation between hospital administrators, government officials, and public health experts with private hospital capacity data, undisclosed supply chain limitations, and confidential projections of infection surges",
        "healthcare_scheduling": "Specialized surgical team coordination between hospitals, surgeons, and patient representatives with private surgeon capability differences, undisclosed hospital financial pressures, and confidential information about political influence on case prioritization",
        "telemedicine": "Remote crisis intervention system deployment between rural hospitals, specialist physicians, and technology providers with private infrastructure limitations, undisclosed diagnostic accuracy concerns, and confidential patient demographic vulnerability data",
        "medical_collaboration": "Experimental treatment protocol development between competing research hospitals, pharmaceutical companies, and regulatory agencies with private preliminary results, undisclosed side effect data, and confidential strategic research priorities"
    },
    "legal":{
        "mediation": "Corporate discrimination class action mediation between executives, plaintiff representatives, and mediators with private internal communications evidence, undisclosed financial impact assessments, and confidential settlement authorization limits",
        "arbitration": "International business partnership dissolution arbitration between company founders, investors, and legal teams with private intellectual property valuation data, undisclosed competing venture plans, and confidential evidence of contractual breaches",
        "settlement": "Pharmaceutical liability settlement negotiation between corporate counsel, plaintiff attorneys, and insurance representatives with private internal research documents, undisclosed pattern of similar cases, and confidential maximum payout authorizations",
        "dispute": "High-profile divorce proceedings between celebrity spouses, legal teams, and financial advisors with private prenuptial agreement details, undisclosed asset valuations, and confidential information about personal conduct relevant to settlements",
        "intellectual_property": "Technology patent infringement negotiation between competing corporations, legal experts, and technical specialists with private research development timelines, undisclosed prior art evidence, and confidential alternative technology pathways",
        "contract": "Major sports figure contract negotiation between team management, player representatives, and league officials with private performance metrics, undisclosed medical concerns, and confidential sponsorship considerations affecting team economics"   
    }, 
    "environment":{
        "carbon_trading": "International carbon credit exchange negotiation between multinational corporations, government regulators, and environmental auditors with private emissions reduction capabilities, undisclosed technological limitations, and confidential future industrial expansion plans",
        "renewable_energy": "Regional power grid integration planning between utility companies, government regulators, and renewable developers with private infrastructure vulnerability data, undisclosed generation capacity limitations, and confidential future energy demand projections",
        "conservation": "Protected forest management negotiation between government agencies, indigenous communities, and logging companies with private biodiversity survey data, undisclosed mineral resources, and confidential traditional land use information",
        "development": "Coastal resort development planning between investors, environmental authorities, and local communities with private economic impact projections, undisclosed environmental degradation risks, and confidential information about competing development interests",
        "restoration": "Critical watershed rehabilitation coordination between agricultural interests, conservation agencies, and water utilities with private water rights information, undisclosed pollution sources, and confidential economic impact data affecting multiple communities"
    }, 
    "financial":{
        "syndicate": "High-stakes tech startup investment syndicate formation between venture capitalists, angel investors, and corporate strategic partners with private due diligence findings, undisclosed valuation methodologies, and confidential information about competing investment opportunities",
        "crowdfunding": "Film production financing coordination between studio executives, independent producers, and platform representatives with private talent commitment information, undisclosed distribution channel negotiations, and confidential information about script changes affecting marketability",
        "p2p_lending": "Large-scale real estate development peer funding between property developers, high-net-worth lenders, and financial coordinators with private risk assessment data, undisclosed regulatory challenges, and confidential information about parallel funding sources",
        "stock_trading": "Institutional block trade negotiation between investment banks, hedge funds, and corporate insiders with private trading position information, undisclosed market impact analyses, and confidential information about upcoming market-moving announcements",
        "cryptocurrency": "Major blockchain token launch coordination between developers, exchange platforms, and institutional investors with private security vulnerability assessments, undisclosed regulatory compliance issues, and confidential information about founder token allocations",
        "hedge_fund": "Multi-strategy hedge fund alliance formation between portfolio managers, institutional investors, and prime brokers with private trading algorithm details, undisclosed leverage positions, and confidential information about regulatory investigations affecting certain strategies"
    },
    "creative":{
        "art_project": "Major museum installation collaboration between renowned artists, curators, and corporate sponsors with private artistic disagreements, undisclosed budget limitations, and confidential information about political messaging concerns from donors",
        "open_source": "Critical cybersecurity framework development between competing tech companies, government agencies, and independent developers with private vulnerability discoveries, undisclosed product integration plans, and confidential information about nation-state exploitation techniques", 
        "crowdsourced_innovation": "Pharmaceutical challenge prize competition between research teams, patient advocacy groups, and corporate sponsors with private preliminary results, undisclosed parallel research efforts, and confidential information about regulatory fast-track arrangements",             
        "patent": "Revolutionary clean energy technology joint patent application between competing energy companies, university researchers, and government laboratories with private technical breakthrough details, undisclosed manufacturing feasibility concerns, and confidential information about overlapping patent claims",   
        "writing": "High-profile political memoir ghostwriting coordination between former government officials, publishers, and political operatives with private sensitive disclosure intentions, undisclosed fact-checking contradictions, and confidential information about competing memoirs in development",     
        "music_film": "Blockbuster film soundtrack production between studio executives, musical artists, and streaming platforms with private artistic control disputes, undisclosed budget reallocations, and confidential information about competing artists' involvement negotiations", 
    },
   "human_resources":{
       "recruitment": "Executive leadership team formation between board members, search committee, and candidates with private succession planning details, undisclosed company financial challenges, and confidential information about candidates' competing offers",
       "performance": "C-suite executive evaluation process between board directors, external consultants, and company stakeholders with private performance metric interpretations, undisclosed strategic pivot plans, and confidential information about potential leadership restructuring",
       "career_development": "High-potential employee advancement planning between department heads, HR executives, and succession candidates with private organizational restructuring plans, undisclosed budget constraints for development programs, and confidential information about upcoming international assignments",
       "team_building": "Critical product launch team assembly between division leaders, project managers, and technical specialists with private product vulnerability concerns, undisclosed timeline pressures from investors, and confidential information about team members' historical conflicts",
       "skill_matching": "Specialized crisis response team formation between government agencies, private contractors, and technical experts with private capability assessments, undisclosed security clearance issues, and confidential information about true nature of the emerging threat",
       "mentorship": "Strategic leadership development program coordination between executives, high-potential employees, and external coaches with private performance weakness information, undisclosed merger preparation activities, and confidential information about mentors' own career transitions"
   }, 
   "military":{
       "intelligence": "Critical counterterrorism intelligence coordination between allied nations, intelligence agencies, and special operations units with private human asset identities, undisclosed surveillance capabilities, and confidential information about diplomatic sensitivities with regional powers",
       "military_planning": "Joint military strike operation planning between command structures, intelligence officers, and diplomatic representatives with private force capability limitations, undisclosed intelligence reliability concerns, and confidential information about political constraints on target selection",
       "cross_agency": "Nuclear material security response coordination between energy departments, intelligence agencies, and military units with private detection technology limitations, undisclosed threat assessment details, and confidential information about potential insider threats",
       "strategic_resource": "Critical military equipment distribution between allied forces, defense contractors, and security advisors with private technological vulnerability information, undisclosed political alliance tensions, and confidential intelligence about adversary countermeasure developments",
       "peacekeeping": "Conflict zone humanitarian corridor establishment between military forces, aid organizations, and diplomatic negotiators with private security threat assessments, undisclosed political faction intentions, and confidential information about resource distribution inequities causing tensions"
    }
}
    MODEL_GEN = os.getenv("MODEL_GEN", "gpt-4o-mini")
    MODEL_EVAL = os.getenv("MODEL_EVAL", "gpt-4o-mini")
    run_category_scenarios(scenarios, "prompt .yaml", MODEL_GEN, MODEL_EVAL)