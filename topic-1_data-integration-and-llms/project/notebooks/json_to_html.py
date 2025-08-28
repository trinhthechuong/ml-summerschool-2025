import json
import os
import re
from IPython.display import display, HTML

# --------- Formatting helpers ---------

def _markdown_bold(s: str) -> str:
    # **bold** -> <b>bold</b>
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", s)

def _bullets_to_html_blocks(text: str) -> str:
    """
    Convert lines:
      * item -> <ul><li>item</li>...</ul>
    Keep non-bullet lines as <p>.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    out, in_list = [], False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("* "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{stripped[2:].strip()}</li>")
        elif stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<p>{stripped}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)

def format_clinical_text(text: str) -> str:
    """Structured 'Clinical Considerations':
       - bold markers
       - bullet lines
    """
    if not text:
        return "<p>No clinical considerations provided.</p>"
    text = _markdown_bold(text)
    return _bullets_to_html_blocks(text)

# --------- Decision rationale parsing ---------

_FEATURE_PREFIX = re.compile(r"^(Prot_|Cyto_|Clinical_)", re.I)

def _clean_feature_token(tok: str) -> tuple[str, str]:
    """
    Given a token like:
      'Prot_YY1 (Yin Yang 1) implicated in ...'
    return:
      ('YY1', '(Yin Yang 1) implicated in ...')
    """
    tok = tok.strip().strip(".").strip(";")
    if not tok:
        return "", ""
    tok = _FEATURE_PREFIX.sub("", tok)       # drop Prot_/Cyto_/Clinical_
    # split: first token = name, rest = description
    m = re.match(r"([^\s]+)\s*(.*)", tok)
    if not m:
        return tok, ""
    name, rest = m.group(1), m.group(2).strip(" -—:").strip()
    return name, rest

def _tokens_to_ul(tokens: list[str]) -> str:
    items = []
    for t in tokens:
        name, desc = _clean_feature_token(t)
        if not name and not desc:
            continue
        if desc:
            items.append(f"<li><b>{name}</b> — {desc}</li>")
        else:
            items.append(f"<li><b>{name}</b></li>")
    if not items:
        return ""
    return "<ul>\n" + "\n".join(items) + "\n</ul>"

def _extract_section_lines(lines: list[str], start_idx: int) -> tuple[str, int]:
    """
    From a heading line at start_idx, collect the text for this section until
    the next recognized heading or end. Return (section_text, next_index).
    """
    buf = []
    i = start_idx
    # take text on same line after ':' first
    line = lines[i]
    colon_split = line.split(":", 1)
    if len(colon_split) == 2:
        buf.append(colon_split[1].strip())
    i += 1
    # keep consuming until next heading or end/blank
    while i < len(lines):
        nxt = lines[i].strip()
        if re.match(r"^(Top Positive SHAP Features|Top Negative SHAP Features|Summary of Feature Relevance)", nxt, re.I):
            break
        buf.append(nxt)
        i += 1
    section_text = " ".join([b for b in buf if b])
    return section_text, i

def format_decision_text(text: str) -> str:
    """
    Structured 'Decision Rationale':
      - bold markers
      - bullets (* ) respected
      - Recognize and parse:
          'Top Positive SHAP Features ...'
          'Top Negative SHAP Features ...'
          'Summary of Feature Relevance ...'
        into clean lists.
    Fallback: paragraph + bullets formatting.
    """
    if not text:
        return "<p>No decision rationale provided.</p>"

    # Normalize and bold
    text = _markdown_bold(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    i = 0
    html_parts = []
    saw_any_structured = False

    while i < len(lines):
        ln = lines[i]

        # Top Positive
        if re.match(r"^Top Positive SHAP Features", ln, re.I):
            sec, i = _extract_section_lines(lines, i)
            # Split tokens by ';'
            tokens = [t.strip() for t in sec.split(";") if t.strip()]
            html_parts.append("<h3>Top Positive SHAP Features</h3>")
            html_parts.append(_tokens_to_ul(tokens))
            saw_any_structured = True
            continue

        # Top Negative
        if re.match(r"^Top Negative SHAP Features", ln, re.I):
            sec, i = _extract_section_lines(lines, i)
            tokens = [t.strip() for t in sec.split(";") if t.strip()]
            html_parts.append("<h3>Top Negative SHAP Features</h3>")
            html_parts.append(_tokens_to_ul(tokens))
            saw_any_structured = True
            continue

        # Summary of Feature Relevance
        if re.match(r"^Summary of Feature Relevance", ln, re.I):
            sec, i = _extract_section_lines(lines, i)
            # split by ';' to make compact bullets
            bullets = [b.strip() for b in sec.split(";") if b.strip()]
            if bullets:
                html_parts.append("<h3>Summary of Feature Relevance</h3>")
                html_parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
                saw_any_structured = True
            continue

        # Generic bullets/paragraphs
        # consume consecutive non-heading lines until next heading
        block = [ln]
        i += 1
        while i < len(lines) and not re.match(
            r"^(Top Positive SHAP Features|Top Negative SHAP Features|Summary of Feature Relevance)",
            lines[i], re.I
        ):
            block.append(lines[i])
            i += 1
        block_text = "\n".join(block)
        html_parts.append(_bullets_to_html_blocks(block_text))

    if not saw_any_structured:
        # Fallback to simple bullets formatting
        return _bullets_to_html_blocks(text)

    return "\n".join(html_parts)

# --------- Report builders ---------

def json_to_html(data: dict) -> str:
    """Convert summary JSON to a styled hospital-ready HTML string."""
    patient_id = data.get("patient_ID", "Unknown")
    disease = data.get("disease_type", "Unknown")
    drug = data.get("recomended_drug_name", "N/A")
    drug_info = data.get("info_on_recommended_drug", "")
    decision = data.get("decision_making_process", "")

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 40px; background: #f9f9f9; color: #2c3e50; }",
        "header { background: #004080; color: white; padding: 20px; border-radius: 8px; }",
        "header h1 { margin: 0; font-size: 26px; }",
        "header p { margin: 5px 0; font-size: 16px; }",
        "section { background: white; padding: 20px; margin-top: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }",
        "h2 { color: #004080; margin-top: 0; border-bottom: 2px solid #004080; padding-bottom: 6px; }",
        "h3 { color: #16a085; margin-top: 18px; }",
        "p, li { font-size: 15px; line-height: 1.6; }",
        "ul { margin: 0.25rem 0 0.75rem 1.25rem; }",
        "pre { background: #f4f6f7; padding: 12px; border-left: 4px solid #004080; border-radius: 4px; white-space: pre-wrap; font-size: 14px; }",
        "</style></head><body>",

        "<header>",
        "<h1>Patient Drug Response Prediction Report</h1>",
        f"<p><strong>Patient ID:</strong> {patient_id}</p>",
        f"<p><strong>Disease:</strong> {disease}</p>",
        "</header>",

        "<section>",
        "<h2>Recommended Treatment</h2>",
        f"<p><strong>Drug Regimen:</strong> {drug}</p>",
        "</section>",

        "<section>",
        "<h2>Clinical Considerations</h2>",
        format_clinical_text(drug_info),
        "</section>",

        "<section>",
        "<h2>Decision Rationale</h2>",
        format_decision_text(decision),
        "</section>",

        "</body></html>"
    ]
    return "\n".join(html_parts)

def save_html_report(data: dict, output_file: str):
    """Save the HTML report to disk."""
    html = json_to_html(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Report saved to {output_file}")

def display_html_report(data: dict):
    """Render the report directly inside a Jupyter Notebook."""
    html = json_to_html(data)
    display(HTML(html))
