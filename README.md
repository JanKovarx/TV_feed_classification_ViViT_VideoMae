TV Feed Classification – ViViT & VideoMAE
Tento projekt implementuje a porovnává dvě moderní transformerové architektury pro klasifikaci scén v televizním vysílání: Video Vision Transformer (ViViT) a Video Masked Autoencoder (VideoMAE).

Popis projektu
Cílem je automaticky klasifikovat krátké úseky televizních záznamů do předdefinovaných tříd na základě vizuálních dat.
Projekt obsahuje:

Kompletní datové zpracování a přípravu datasetu

Implementaci modelů ViViT a VideoMAE

Trénování a vyhodnocení modelů

Analýzu výsledků a porovnání výkonu

Struktura repozitáře

- **data/** – složka pro dataset (není součástí repozitáře)  
- **models/** – implementace architektur ViViT a VideoMAE  
- **notebooks/** – Jupyter notebooky pro testování a analýzu  
- **utils/** – pomocné funkce pro načítání a zpracování dat  
- **requirements.txt** – seznam Python závislostí  
- **train.py** – hlavní skript pro trénování modelů  
- **evaluate.py** – skript pro vyhodnocení výsledků  

Instalace
Naklonujte repozitář:
git clone https://github.com/JanKovarx/TV_feed_classification_ViViT_VideoMae.git
cd TV_feed_classification_ViViT_VideoMae

Vytvořte a aktivujte virtuální prostředí:
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows

Nainstalujte závislosti:
pip install -r requirements.txt

Dataset
181 videí z různých televizních stanic

Anotace do 8 tříd (studio, indoor, outdoor, předěl, reklama, upoutávka, grafika, zábava)

Přísné rozdělení datasetu na trénovací, validační a testovací část bez sdílených videí

Modely
ViViT
Architektura factorised encoder

Inicializováno z vit_b_16 předtrénovaného na ImageNet-1K

Trénováno v plně supervidovaném režimu

VideoMAE
Předtrénováno na Kinetics-400 v samo-supervidovaném režimu

Použita Base varianta (videomae-base)

Výsledky
|Metric | ViViT | VideoMAE |
|-------|-------|----------|
|F1 score | 80.8% | 79.6% |
|Accuracy | 81.1% | 79.9% |
|Precision | 81.2% | 80.3% |
|Recall | 81.1% | 78.9% |
|Loss | 1.07 | 1.11 |
|Overall success rate | 81.1% | 80.9% |

Nejlepší výsledky pro vizuálně výrazné třídy (entertainment, advertisement)

Slabší pro vizuálně podobné kategorie (indoor vs. studio)

Finetuning na doménově specifických datech
