from __future__ import annotations

import json
import os
import re

import pandas as pd


ALLOWED_ACTION_TYPES = [
    "inhibitor",
    "agonist",
    "antagonist",
    "degrader",
    "DNA damaging agent",
    "antimetabolite",
    "topoisomerase inhibitor",
    "microtubule stabilizer",
    "microtubule destabilizer",
    "epigenetic inhibitor",
    "proteasome inhibitor",
    "PARP inhibitor",
    "kinase inhibitor",
    "Unknown",
]


def normalize_scalar(value, default: str = "Unknown") -> str:
    text = str(value).replace("\n", " ").strip() if value is not None else ""
    return text if text else default


def normalize_list(value, max_items: int = 5) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    elif value is None:
        raw_items = []
    else:
        raw_items = re.split(r"[;,|]", str(value))

    cleaned = []
    for item in raw_items:
        text = normalize_scalar(item, default="")
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break

    return cleaned if cleaned else ["Unknown"]


def normalize_confidence(value) -> str:
    mapping = {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "unknown": "Unknown",
    }
    key = normalize_scalar(value, default="Unknown").lower()
    return mapping.get(key, "Unknown")


def extract_json_object(text: str) -> dict:
    content = (text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {content[:200]}")

    return json.loads(content[start:end + 1])


def sanitize_profile(profile: dict, drug_name: str, smiles: str) -> dict:
    return {
        "drug_name": normalize_scalar(drug_name),
        "smiles": normalize_scalar(smiles),
        "primary_targets": normalize_list(profile.get("primary_targets"), max_items=5),
        "action_type": normalize_scalar(profile.get("action_type")),
        "pathways": normalize_list(profile.get("pathways"), max_items=5),
        "biological_processes": normalize_list(profile.get("biological_processes"), max_items=5),
        "mechanism_summary": normalize_scalar(profile.get("mechanism_summary")),
        "confidence": normalize_confidence(profile.get("confidence")),
        "notes": normalize_scalar(profile.get("notes"), default="None"),
    }


def build_profile_text(profile: dict) -> str:
    return " | ".join(
        [
            f"Drug={profile['drug_name']}",
            f"PrimaryTargets={'; '.join(profile['primary_targets'])}",
            f"ActionType={profile['action_type']}",
            f"Pathways={'; '.join(profile['pathways'])}",
            f"BiologicalProcesses={'; '.join(profile['biological_processes'])}",
            f"MechanismSummary={profile['mechanism_summary']}",
            f"Confidence={profile['confidence']}",
            f"Notes={profile['notes']}",
        ]
    )


def load_reference_table(reference_csv: str):
    if not reference_csv or not os.path.exists(reference_csv):
        return None

    reference = pd.read_csv(reference_csv).copy()
    if "smiles" in reference.columns:
        reference["smiles"] = reference["smiles"].astype(str).str.strip()
    if "drug_name" in reference.columns:
        reference["drug_name"] = reference["drug_name"].astype(str).str.strip()
    return reference


def load_done_smiles(out_csv: str) -> set[str]:
    if not out_csv or not os.path.exists(out_csv):
        return set()

    done = pd.read_csv(out_csv, usecols=["smiles"])
    done["smiles"] = done["smiles"].astype(str).str.strip()
    return set(done["smiles"])


def build_reference_context(reference_df, drug_name: str, smiles: str) -> str:
    if reference_df is None or reference_df.empty:
        return ""

    matched = pd.DataFrame()
    if "smiles" in reference_df.columns:
        matched = reference_df[reference_df["smiles"] == smiles]
    if matched.empty and "drug_name" in reference_df.columns:
        matched = reference_df[reference_df["drug_name"].str.lower() == drug_name.lower()]
    if matched.empty:
        return ""

    row = matched.iloc[0]
    field_map = [
        ("primary_targets", "Primary targets"),
        ("action_type", "Action type"),
        ("pathways", "Pathways"),
        ("biological_processes", "Biological processes"),
        ("mechanism_summary", "Mechanism summary"),
        ("notes", "Notes"),
    ]

    facts = []
    for column, label in field_map:
        if column in matched.columns:
            value = normalize_scalar(row[column], default="")
            if value:
                facts.append(f"{label}: {value}")

    return "\n".join(facts)


def build_system_prompt() -> str:
    return (
        "You are extracting mechanism priors for drug synergy prediction. "
        "Return only valid JSON. "
        "The goal is not to write a pharmacology summary, but to produce a compact, canonical, "
        "mechanism-focused profile for one drug. "
        "Prioritize targets, action type, pathway modules, and biological processes that are likely "
        "to influence combination response in cancer cells. "
        "Prefer HGNC gene symbols for human protein targets. "
        "Prefer canonical pathway names aligned with Reactome or KEGG conventions. "
        "Do not infer targets from disease indications alone. "
        "If uncertain, use Unknown rather than guessing."
    )


def build_user_prompt(drug_name: str, smiles: str, reference_context: str = "") -> str:
    action_type_text = ", ".join(ALLOWED_ACTION_TYPES)
    prompt = (
        f"Drug name: {drug_name}\n"
        f"SMILES: {smiles}\n\n"
        "Task:\n"
        "Generate a machine-learning-oriented mechanism profile for drug synergy prediction.\n\n"
        "Important priorities:\n"
        "- Keep only mechanism-relevant facts.\n"
        "- Prioritize pathways or processes related to RTK signaling, MAPK, PI3K-AKT-mTOR, DNA damage response, "
        "cell cycle, apoptosis, chromatin regulation, microtubules, proteasome, and oxidative stress when relevant.\n"
        "- Do not include broad clinical indications, brand names, trial history, or narrative filler.\n"
        "- If the compound is a salt, hydrate, or ester form and the mechanism is unchanged, describe the active moiety "
        "and mention the normalization briefly in notes.\n"
        "- If the input appears to be a combination or regimen name rather than a single compound, set confidence to Low "
        "and keep ambiguous fields as Unknown.\n"
        "- If a pathway is not directly supported by the known targets or established mechanism, do not include it.\n\n"
        "Return valid JSON only with these keys:\n"
        "primary_targets, action_type, pathways, biological_processes, mechanism_summary, confidence, notes\n\n"
        "Field rules:\n"
        "- primary_targets: array of short strings, max 5; use HGNC gene symbols when possible\n"
        f"- action_type: short normalized phrase, preferably one of: {action_type_text}\n"
        "- pathways: array of short canonical pathway or module names, max 5\n"
        "- biological_processes: array of short strings, max 5\n"
        "- mechanism_summary: one sentence, mechanism-focused, <= 35 words\n"
        "- confidence: one of High, Medium, Low, Unknown\n"
        "- notes: short string; use None if nothing useful remains\n"
        "- If a field is unknown, use 'Unknown' or ['Unknown']\n"
        "- Output JSON only, no markdown\n"
    )

    if reference_context:
        prompt += (
            "\nTrusted reference facts are provided below.\n"
            "Use these facts as the primary evidence source.\n"
            "Do not contradict them.\n"
            "Only add extra information if it is highly consistent with these facts.\n"
            "If the reference facts are sparse, keep unsupported fields as Unknown.\n\n"
            f"{reference_context}\n"
        )
    return prompt


def generate_mechanism_profile(client, text_model: str, drug_name: str, smiles: str, reference_context: str = "") -> dict:
    response = client.chat.completions.create(
        model=text_model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(drug_name, smiles, reference_context=reference_context)},
        ],
        temperature=0.1,
    )
    raw_text = response.choices[0].message.content or ""
    profile = extract_json_object(raw_text)
    return sanitize_profile(profile, drug_name=drug_name, smiles=smiles)


def build_mechanism_text(client, text_model: str, drug_name: str, smiles: str, reference_df=None) -> str:
    reference_context = build_reference_context(reference_df, drug_name=drug_name, smiles=smiles)
    profile = generate_mechanism_profile(
        client=client,
        text_model=text_model,
        drug_name=drug_name,
        smiles=smiles,
        reference_context=reference_context,
    )
    return build_profile_text(profile)
