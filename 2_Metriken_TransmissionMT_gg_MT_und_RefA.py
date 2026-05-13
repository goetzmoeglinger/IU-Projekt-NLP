from sacrebleu.metrics import BLEU, CHRF
import pandas as pd
import numpy as np
import requests

# ========================= BASIS-URLS =========================

base_url = "https://raw.githubusercontent.com/wmt-conference/wmt24-news-systems/main/txt"

systems_api_url = "https://api.github.com/repos/wmt-conference/wmt24-news-systems/contents/txt/system-outputs/en-de"

human_ref_url = f"{base_url}/references/en-de.refA.txt"

machine_ref_file = "TranssionMT.txt"
machine_ref_url = f"{base_url}/system-outputs/en-de/{machine_ref_file}"

# ========================= HILFSFUNKTION =========================

def load_lines_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text.splitlines()

# ========================= REFERENZEN LADEN =========================

machine_reference = load_lines_from_url(machine_ref_url)
human_reference = load_lines_from_url(human_ref_url)

# ========================= METRIKEN =========================

bleu = BLEU(effective_order=True)
chrf = CHRF()

# ========================= SYSTEMDATEIEN AUS GITHUB =========================

response = requests.get(systems_api_url)
response.raise_for_status()

system_files = [
    item["name"]
    for item in response.json()
    if item["name"].endswith(".txt")
]

results = []

# ========================= ALLE SYSTEME GEGEN TRANSSIONMT VERGLEICHEN =========================

for system_file in system_files:

    system_url = f"{base_url}/system-outputs/en-de/{system_file}"

    hypothesis = load_lines_from_url(system_url)

    if len(hypothesis) != len(machine_reference):
        print(f"Übersprungen: {system_file}")
        continue

    bleu_scores = [
        bleu.sentence_score(hyp, [ref]).score
        for hyp, ref in zip(hypothesis, machine_reference)
    ]

    chrf_scores = [
        chrf.sentence_score(hyp, [ref]).score
        for hyp, ref in zip(hypothesis, machine_reference)
    ]

    results.append({
        "System": system_file.replace(".txt", ""),
        "BLEU_avg_vs_TranssionMT": round(np.mean(bleu_scores), 2),
        "chrF_avg_vs_TranssionMT": round(np.mean(chrf_scores), 2)
    })

# ========================= MENSCHLICHE REFERENZ GEGEN TRANSSIONMT =========================

human_bleu_scores = [
    bleu.sentence_score(hyp, [ref]).score
    for hyp, ref in zip(human_reference, machine_reference)
]

human_chrf_scores = [
    chrf.sentence_score(hyp, [ref]).score
    for hyp, ref in zip(human_reference, machine_reference)
]

results.append({
    "System": "Human_refA",
    "BLEU_avg_vs_TranssionMT": round(np.mean(human_bleu_scores), 2),
    "chrF_avg_vs_TranssionMT": round(np.mean(human_chrf_scores), 2)
})

# ========================= DATAFRAME =========================

df_transsion_ref = pd.DataFrame(results)

df_transsion_ref = df_transsion_ref.sort_values(
    by="BLEU_avg_vs_TranssionMT",
    ascending=False
)

display(df_transsion_ref)