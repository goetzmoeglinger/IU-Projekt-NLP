# ========================= Berechnung BLEU, chrF und chrF++ =========================

from sacrebleu.metrics import BLEU, CHRF
import pandas as pd
import numpy as np
import requests

# ========================= GITHUB RAW BASIS-URL =========================

base_url = "https://raw.githubusercontent.com/wmt-conference/wmt24-news-systems/main/txt"

ref_url = f"{base_url}/references/en-de.refA.txt" # Referenzübersetzung

systems_api_url = "https://api.github.com/repos/wmt-conference/wmt24-news-systems/contents/txt/system-outputs/en-de" # System Übersetzungen

# ========================= HILFSFUNKTIONEN =========================

def load_lines_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text.splitlines()

# ========================= REFERENZ LADEN =========================

reference = load_lines_from_url(ref_url)

# ========================= SYSTEMDATEIEN AUS GITHUB-ORDNER LADEN =========================

response = requests.get(systems_api_url)
response.raise_for_status()

system_files = [
    item["name"]
    for item in response.json()
    if item["name"].endswith(".txt")
]

# ========================= METRIKEN =========================

bleu = BLEU(effective_order=True) # SacreBLEU passt maximale n-Gramm-Länge automatisch an.
chrf = CHRF()
chrfpp = CHRF(word_order=2) # chrF + Bigramme

results = []

# ========================= ALLE 26 SYSTEME AUSWERTEN =========================

for system_file in system_files:

    sys_url = f"{base_url}/system-outputs/en-de/{system_file}"
    hypothesis = load_lines_from_url(sys_url)

    if len(hypothesis) != len(reference):
        print(f"Übersprungen wegen unterschiedlicher Zeilenzahl: {system_file}")
        continue

    corpus_bleu = bleu.corpus_score(hypothesis, [reference]).score
    corpus_chrf = chrf.corpus_score(hypothesis, [reference]).score
    corpus_chrfpp = chrfpp.corpus_score(hypothesis, [reference]).score

    sentence_bleu_scores = []
    sentence_chrf_scores = []
    sentence_chrfpp_scores = []

    for hyp, ref in zip(hypothesis, reference):
        sentence_bleu_scores.append(bleu.sentence_score(hyp, [ref]).score)
        sentence_chrf_scores.append(chrf.sentence_score(hyp, [ref]).score)
        sentence_chrfpp_scores.append(chrfpp.sentence_score(hyp, [ref]).score)

    results.append({
        "System": system_file.replace(".txt", ""),
        "BLEU_corpus": round(corpus_bleu, 2),
        "BLEU_sentence_avg": round(np.mean(sentence_bleu_scores), 2),
        "chrF_corpus": round(corpus_chrf, 2),
        "chrF_sentence_avg": round(np.mean(sentence_chrf_scores), 2),
        "chrF++_corpus": round(corpus_chrfpp, 2),
        "chrF++_sentence_avg": round(np.mean(sentence_chrfpp_scores), 2)
    })

# ========================= DATAFRAME =========================

df = pd.DataFrame(results)
df = df.sort_values(by="BLEU_corpus", ascending=False)

display(df)