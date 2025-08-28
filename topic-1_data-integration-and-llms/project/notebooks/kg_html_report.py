import json
from pyvis.network import Network

def _apply_options(net: Network):
    options = {
        "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 110,
                "springConstant": 0.06,
                "avoidOverlap": 0.5
            },
            "stabilization": {"iterations": 200, "fit": True}
        },
        "nodes": {"shape": "dot"},
        "edges": {"smooth": {"enabled": True, "type": "dynamic"}}
    }
    net.set_options(json.dumps(options))


import json, ast
from typing import Any, Dict

def load_kg_json(path: str) -> Dict[str, Any]:
    """Load a KG from JSON. Also tolerates:
       - stringified JSON (double-encoded)
       - Python dict literal (single quotes) via ast.literal_eval
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Try pure JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # If it looks like a quoted JSON string, unquote once and try again
        if raw.startswith('"') and raw.endswith('"'):
            try:
                data = json.loads(json.loads(raw))
            except Exception:
                # Fall back to Python literal (single quotes etc.)
                data = ast.literal_eval(json.loads(raw))
        else:
            # Try Python literal (single quotes / trailing commas)
            data = ast.literal_eval(raw)

    if not isinstance(data, dict):
        raise ValueError("KG file must decode to a dict with 'nodes' and 'links' keys.")
    return data


from pyvis.network import Network
from pathlib import Path
from IPython.display import IFrame, display

def build_pyvis(data, height="720px", width="100%"):
    net = Network(height=height, width=width, bgcolor="#ffffff", font_color="#111827", directed=False)
    _apply_options(net)

    nodes = data.get("nodes", [])
    links = data.get("links", data.get("edges", []))

    # Colors
    def node_color(role):
        return {"background": "#DAA520" if (role or "").lower()=="test" else "#4682B4", "border": "#3b82f6"}

    # Nodes
    for n in nodes:
        nid = str(n.get("id", ""))
        if not nid:
            continue
        deceased = str(n.get("Clinical_Deceased", "0")) == "1"
        color = node_color(n.get("role"))
        if deceased:
            color["border"] = "#e11d48"
        title = "<br>".join([
            f"<b>Patient:</b> {nid}",
            f"<b>Role:</b> {n.get('role','')}",
            f"<b>Diagnosis:</b> {n.get('Clinical_Diagnosis','N/A')}",
            f"<b>Age:</b> {n.get('Clinical_Age_At_Sampling','N/A')} &nbsp; <b>Sex:</b> {n.get('Clinical_Gender','N/A')}",
            f"<b>Stage:</b> {n.get('Clinical_Treatment_stage','N/A')}",
            f"<b>Pheno group:</b> {n.get('Clinical_PhenoGroups','N/A')}",
        ])
        net.add_node(nid, label=nid, title=title, color=color, borderWidth=3 if deceased else 1.5)

    # Edges
    def edge_color(true_cls):
        m = {"positive_effect": "#2ca02c", "no_effect": "#1f77b4", "adverse_effect": "#d62728"}
        return m.get((true_cls or "").lower(), "#888888")

    def to_float(x, default=0.0):
        try: return float(x)
        except: return default

    for e in links:
        src, dst = str(e.get("source","")), str(e.get("target",""))
        if not src or not dst:
            continue
        title = "<br>".join([
            f"<b>Drug:</b> {e.get('drug','')}",
            f"<b>Predicted class (test):</b> {e.get('Predicted_Class','')}",
            f"<b>Probability:</b> {to_float(e.get('Predicted_Prob'), 0.0):.3f}",
            f"<b>Training true class:</b> {e.get('Training_True_Class','')}",
            f"<b>Train patient:</b> {dst}",
        ])
        net.add_edge(src, dst, color=edge_color(e.get("Training_True_Class")), width=1+4*to_float(e.get("Predicted_Prob"),0.0), title=title)

    return net

def save_kg_html(data, out_html="KG.html"):
    net = build_pyvis(data)
    out_html = str(Path(out_html))
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    net.show(out_html)
    return out_html


