import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from collections import Counter
import pandas as pd
import shap
import numpy as np
import os
from collections import defaultdict
import json
from pathlib import Path

import os
import json
from collections import Counter, defaultdict
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

def run_drug_classification(
    proteomics, cytokines, clinical_numeric, drug_classes, 
    output_dir="results/patient_res_jsons", top_k=10
):
    """
    Train XGBoost classifiers for each drug, evaluate performance, 
    and generate patient-level SHAP explanations in structured JSON.

    Parameters
    ----------
    proteomics : pd.DataFrame
        Proteomics features (patients x features).
    cytokines : pd.DataFrame
        Cytokine features (patients x features).
    clinical_numeric : pd.DataFrame
        Clinical numeric features (patients x features).
    drug_classes : pd.DataFrame
        Target labels for each drug (patients x drugs).
    output_dir : str, default="patient_res_jsons"
        Directory to save per-patient JSON SHAP explanations.
    top_k : int, default=10
        Number of top positive/negative SHAP features to keep.

    Returns
    -------
    results : dict
        Per-drug dictionary with model, encoder, report, and class counts.
    patient_explanations : dict
        Structured patient-level predictions + SHAP values.
    """

    # --- Prepare features ---
    X = pd.concat(
        [
            proteomics.add_prefix("Prot_"),
            cytokines.add_prefix("Cyto_"),
            clinical_numeric.astype(int).add_prefix("Clinical_"),
        ],
        axis=1,
    )

    imputer = SimpleImputer(strategy="mean")
    results = {}
    patient_explanations = defaultdict(lambda: {"Disease": "Multiple Myeloma", "Drugs": {}})

    for drug in drug_classes.columns:
        y_raw = drug_classes[drug].dropna()
        patients = y_raw.index
        X_drug, y_drug = X.loc[patients], y_raw

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y_drug)
        counts = Counter(y)

        # Require 3 classes with at least 2 samples each
        if len(counts) < 3 or min(counts.values()) < 2:
            print(f"Skipping {drug} (class counts: {counts})")
            continue

        # Split
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_drug, y, patients, test_size=0.2, random_state=42, stratify=y
        )

        # Impute
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # Train model
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        model.fit(X_train_imp, y_train)

        # Evaluate
        y_pred = model.predict(X_test_imp)
        y_proba = model.predict_proba(X_test_imp)
        report = classification_report(
            y_test,
            y_pred,
            labels=list(range(len(le.classes_))),
            target_names=le.classes_,
            zero_division=0,
        )

        results[drug] = {
            "label_encoder": le,
            "model": model,
            "report": report,
            "class_counts": counts,
            "train_labels": dict(zip(idx_train, y_train)),
        }

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_imp, check_additivity=False)
        feature_names = X.columns.tolist()

        for i, pid in enumerate(idx_test):
            pred_idx = y_pred[i]
            pred_label = le.inverse_transform([pred_idx])[0]
            pred_prob = float(y_proba[i, pred_idx])

            patient_shap = shap_values[i, :, pred_idx]

            # Sort features
            sorted_idx = patient_shap.argsort()
            neg_idx = sorted_idx[:top_k]       # lowest
            pos_idx = sorted_idx[::-1][:top_k] # highest

            top_pos_features = [
                {"Feature": feature_names[j], "Value": float(patient_shap[j])}
                for j in pos_idx if patient_shap[j] > 0
            ]
            top_neg_features = [
                {"Feature": feature_names[j], "Value": float(patient_shap[j])}
                for j in neg_idx if patient_shap[j] < 0
            ]

            # Store per-drug under patient
            patient_explanations[pid]["Drugs"][drug] = {
                "Predicted_Class": pred_label,
                "Predicted_Prob": pred_prob,
                "SHAP": {
                    "Top_Positive": top_pos_features,
                    "Top_Negative": top_neg_features,
                },
            }

    # Save JSONs
    os.makedirs(output_dir, exist_ok=True)
    for pid, pdata in patient_explanations.items():
        with open(os.path.join(output_dir, f"{pid}.json"), "w") as f:
            json.dump({pid: pdata}, f, indent=2)

    return results, patient_explanations




import networkx as nx

def build_patient_graph(patient_exp, results, clinical_numeric):
    """
    Build a knowledge graph for a single test patient.
    
    Parameters
    ----------
    patient_exp : dict
        One entry from patient_explanations list
    results : dict
        Results dict from run_drug_classification
    clinical_numeric : pd.DataFrame
        Clinical features
    
    Returns
    -------
    G : networkx.Graph
        Patient-level knowledge graph
    """
    pid_test   = patient_exp["Patient_ID"]
    drug       = patient_exp["Drug"]
    pred_class = patient_exp["Predicted_Class"]

    # Graph
    G = nx.Graph()

    # Add test patient node with metadata
    test_meta = {
        f"Clinical_{col}": val
        for col, val in clinical_numeric.loc[pid_test].items()
        if pd.notna(val)
    }
    G.add_node(pid_test, role="test", **test_meta)

    # Add training patients with same class
    train_labels = results[drug]["train_labels"]
    le = results[drug]["label_encoder"]

    for pid_train, label_int in train_labels.items():
        label_str = le.inverse_transform([label_int])[0]
        if label_str == pred_class:
            # add node with metadata
            train_meta = {
                f"Clinical_{col}": val
                for col, val in clinical_numeric.loc[pid_train].items()
                if pd.notna(val)
            }
            G.add_node(pid_train, role="train", **train_meta)
            G.add_edge(pid_test, pid_train, drug=drug, class_label=pred_class)

    return G


import os
import json
import networkx as nx
from networkx.readwrite import json_graph
import pandas as pd

def _safe_clinical_attrs(df: pd.DataFrame, pid: str) -> dict:
    """Return clinical attributes for pid with 'Clinical_' prefix; empty if missing."""
    if (df is None) or (pid not in df.index):
        return {}
    row = df.loc[pid]
    return {f"Clinical_{col}": val for col, val in row.items() if pd.notna(val)}

def build_patient_graph(pid, patient_exp, results, clinical_numeric):
    """
    Build a knowledge graph for a single (unlabeled) test patient across all drugs.

    - Test patient node: role='test', clinical attrs, and per-drug prediction attrs.
    - Training patient nodes: role='train', clinical attrs (no true labels stored on node).
    - One edge per (test, training, drug) with attributes:
        drug, Predicted_Class (test), Predicted_Prob (test), True_Class (training)

    Uses MultiGraph so multiple drugs between the same two patients create parallel edges.

    Parameters
    ----------
    pid : str
        Patient ID (key in patient_explanations).
    patient_exp : dict
        patient_explanations[pid] as produced by run_drug_classification (new JSON style).
    results : dict
        Output from run_drug_classification; must contain per-drug 'train_labels' and 'label_encoder'.
    clinical_numeric : pd.DataFrame
        Clinical features (index = patient IDs).

    Returns
    -------
    G : networkx.MultiGraph
        Patient-level knowledge graph.
    """
    # Allow multiple edges between the same nodes (one per drug)
    G = nx.MultiGraph()

    # --- Add test patient node (no true label) ---
    test_attrs = _safe_clinical_attrs(clinical_numeric, pid)
    test_attrs.update({"role": "test", "Disease": patient_exp.get("Disease", None)})
    G.add_node(pid, **test_attrs)

    # --- For each drug the test patient was scored on ---
    for drug, drug_info in patient_exp.get("Drugs", {}).items():
        pred_class = drug_info.get("Predicted_Class")
        pred_prob  = float(drug_info.get("Predicted_Prob", float("nan")))

        # Store per-drug prediction on the test node (namespacing with drug)
        G.nodes[pid][f"{drug}__Predicted_Class"] = pred_class
        G.nodes[pid][f"{drug}__Predicted_Prob"]  = pred_prob

        # Get training labels for this drug
        train_labels = results[drug]["train_labels"]          # {pid_train: class_int}
        le = results[drug]["label_encoder"]                   # LabelEncoder
        # Connect to ALL training patients for this drug; only they have True_Class
        for pid_train, y_int in train_labels.items():
            true_class = le.inverse_transform([y_int])[0]

            # Add / update training node (no true label baked into node; it's drug-specific)
            if pid_train not in G:
                train_attrs = _safe_clinical_attrs(clinical_numeric, pid_train)
                train_attrs.update({"role": "train"})
                G.add_node(pid_train, **train_attrs)

            # Add an edge specific to this drug
            # Key by drug to avoid overwriting parallel edges between the same nodes
            G.add_edge(
                pid, pid_train,
                key=drug,
                drug=drug,
                Predicted_Class=pred_class,
                Predicted_Prob=pred_prob,
                Training_True_Class=true_class
            )

    return G


def save_patient_graphs_json(patient_explanations, results, clinical_numeric, outdir="patient_graphs_json"):
    """
    Build and save one knowledge graph (JSON) per test patient (unlabeled).

    Parameters
    ----------
    patient_explanations : dict
        { patient_id: { "Disease": ..., "Drugs": { drug: {...} } } }
    results : dict
        From run_drug_classification (per drug: label_encoder, train_labels, ...)
    clinical_numeric : pd.DataFrame
        Clinical features for both test and training patients.
    outdir : str
        Output directory for node-link JSON graphs.
    """
    os.makedirs(outdir, exist_ok=True)

    for pid, pdata in patient_explanations.items():
        G = build_patient_graph(pid, pdata, results, clinical_numeric)
        data = json_graph.node_link_data(G)  # works with MultiGraph too
        fname = os.path.join(outdir, f"{pid}_KG.json")
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {fname}")



