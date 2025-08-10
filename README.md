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
.
├── data/ # Složka pro dataset (není součástí repozitáře)
├── models/ # Implementace architektur ViViT a VideoMAE
├── notebooks/ # Jupyter notebooky pro testování a analýzu
├── utils/ # Pomocné funkce pro načítání a zpracování dat
├── requirements.txt # Seznam Python závislostí
├── train.py # Hlavní skript pro trénování modelů
├── evaluate.py # Skript pro vyhodnocení výsledků
└── README.md # Tento soubor

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

Ruční anotace do 8 tříd (mix „scene environment“ a „content type“)

Přísné rozdělení datasetu na trénovací, validační a testovací část bez sdílených videí

Modely
ViViT
Architektura factorised encoder (nejprve prostorová, pak časová pozornost)

Inicializováno z vit_b_16 předtrénovaného na ImageNet-1K

Trénováno v plně supervidovaném režimu

VideoMAE
Předtrénováno na Kinetics-400 v samo-supervidovaném režimu

Použita Base varianta (videomae-base)

Finetuning na doménově specifických datech
