import json
from pathlib import Path
from typing import Dict
from IPython.display import IFrame, display 


def json_to_html(data: dict) -> str:
    """Convert summary JSON (various schema versions) into styled HTML."""

    # --- Extract fields robustly ---
    # Patient ID
    patient_id = (
        data.get("patient_ID")
        or data.get("patient_id")
        or data.get("assistant_structured_report", {}).get("patient_ID")
        or "Unknown"
    )

    # Disease type
    disease = (
        data.get("disease_type")
        or data.get("assistant_structured_report", {}).get("disease_type")
        or "Unknown"
    )

    # Recommended drug
    drug = (
        data.get("recomended_drug_name")
        or data.get("recommended_drug_name")  # just in case
        or data.get("assistant_structured_report", {}).get("recomended_drug_name")
        or "N/A"
    )

    # Drug info
    drug_info = (
        data.get("info_on_recommended_drug")
        or data.get("assistant_structured_report", {}).get("info_on_recommended_drug")
        or data.get("assistant_free_text_report")  # fallback: use the whole free text
        or ""
    )

    # Decision rationale
    decision = (
        data.get("decision_making_process")
        or data.get("assistant_structured_report", {}).get("decision_making_process")
        or data.get("assistant_free_text_report")  # fallback
        or ""
    )

    # --- Build HTML ---
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
