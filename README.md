---
title: CycleSync
emoji: 🌸
colorFrom: pink
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Zyklusbasierter Trainings- & Recovery-Coach mit ML + RAG
---

# CycleSync — Zyklusbasierter Trainings- & Recovery-Coach

Semesterprojekt im Modul **KI-Anwendungen** (ZHAW, FS 2026)
Kombination der Blöcke **ML Numeric Data** + **NLP (RAG)**

## Projektidee

CycleSync generiert Trainingsempfehlungen, die sich an der aktuellen Zyklusphase einer Nutzerin orientieren, und erklärt diese mit wissenschaftlicher Literatur.

- **ML**: Klassifiziert aus Zyklustag, Symptomen und Schlafdaten die empfohlene Trainingsintensität, Recovery-Zeit und Belastungsrisiko.
- **NLP (RAG)**: Sucht passende PubMed-Abstracts und ein LLM generiert eine personalisierte Erklärung mit Quellenangaben.

## Live-Demo

Diese App läuft auf HuggingFace Spaces. Für lokales Setup siehe [Repository](https://github.com/carvacla/cyclesync_KI_project).

## Disclaimer

Forschungsprojekt — keine medizinische Beratung. Empfehlungen ersetzen keinen Arztbesuch.
