# bioassistant.py

import json
from typing import List, Dict
from collections import defaultdict

from gseapy import enrichr
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


# -------------------------------
# Tool: Enrichment
# -------------------------------
@tool
def enrichr_query(gene_list: List[str]):
    """Run enrichment analysis on a list of genes using gseapy (GO Biological Process)."""
    enr = enrichr(
        gene_list=gene_list,
        gene_sets='GO_Biological_Process_2021',
        organism='Human',
        outdir=None,
        cutoff=0.05
    )
    return enr.results  # DataFrame


# -------------------------------
# LLM setup
# -------------------------------
def get_llm_with_tools(model: str = "gemini-2.5-flash", provider: str = "google_genai"):
    """Initialize the chat model and bind the enrichment tool."""
    llm = init_chat_model(model=model, model_provider=provider, temperature=0.2)
    return llm.bind_tools([enrichr_query])


def get_prompt_chain(llm_with_tools):
    """Return a chain with system+human prompt bound to the LLM with tools."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful bioinformatics assistant. Use tools when needed."),
        ("human", "{question}")
    ])
    return prompt | llm_with_tools


# -------------------------------
# SHAP → Gene Sets → Enrichment → Summarization
# -------------------------------
def analyze_patient(patient_json: Dict, patient_id: str, chain):
    """
    Collect SHAP features per predicted class, run enrichment, and ask LLM to summarize.
    
    Parameters
    ----------
    patient_json : dict
        JSON object with structure { patient_id: { "Drugs": {...}} }
    patient_id : str
        Patient ID key in patient_json
    chain : LangChain runnable (prompt | llm_with_tools)
    
    Returns
    -------
    results_by_class : dict
        { predicted_class: { "positive": enrichment_df, "negative": enrichment_df } }
    """
    drug_keys = list(patient_json[patient_id]["Drugs"].keys())
    class_features = defaultdict(lambda: {"positive": [], "negative": []})

    # Step 1: Collect SHAP features by predicted class
    for drug in drug_keys:
        predicted_class = patient_json[patient_id]["Drugs"][drug]["Predicted_Class"]
        pos_features = patient_json[patient_id]["Drugs"][drug]['SHAP']['Top_Positive']
        neg_features = patient_json[patient_id]["Drugs"][drug]['SHAP']['Top_Negative']

        class_features[predicted_class]["positive"].extend(
            [item["Feature"].split("_", 1)[1] if "_" in item["Feature"] else item["Feature"]
             for item in pos_features]
        )
        class_features[predicted_class]["negative"].extend(
            [item["Feature"].split("_", 1)[1] if "_" in item["Feature"] else item["Feature"]
             for item in neg_features]
        )

    # Step 2: Run enrichment
    results_by_class = {}
    for cls, feats in class_features.items():
        results_by_class[cls] = {}
        results_by_class[cls]["positive"] = (
            enrichr_query({"gene_list": list(set(feats["positive"]))}) if feats["positive"] else None
        )
        results_by_class[cls]["negative"] = (
            enrichr_query({"gene_list": list(set(feats["negative"]))}) if feats["negative"] else None
        )

    # Step 3: Summarize with LLM
    summaries = {}
    for cls, res in results_by_class.items():
        question = f"Predicted class: {cls}\nSummarize functional biology or pathways of SHAP features.\n"

        if res["positive"] is not None and not res["positive"].empty:
            question += f"\nPositive SHAP features (supporting {cls}):\n{res['positive'].head(10).to_string(index=False)}\n"
        if res["negative"] is not None and not res["negative"].empty:
            question += f"\nNegative SHAP features (against {cls}):\n{res['negative'].head(10).to_string(index=False)}\n"

        ai_msg = chain.invoke({"question": question})
        summaries[cls] = ai_msg.content

    return results_by_class, summaries
