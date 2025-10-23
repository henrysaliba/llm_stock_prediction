# Topics in Computer Science – Stock Prediction with LLMs

This repo is for my university project on using a Large Language Model to help with stock prediction.

## Setup

Create and activate a virtual environment:
    python3 -m venv .venv
    source .venv/bin/activate

Install packages if needed:
    pip install -r requirements.txt

## Run Steps

Collect data and news sentiment:
    python3 scripts/collect_data.py


Process and visualize:
    python3 scripts/process_data.py


Build and train models:
    python3 scripts/build_dataset_baseline.py
    python3 scripts/train_baseline.py
    python3 scripts/build_dataset_hybrid.py
    python3 scripts/train_hybrid.py


Compare results:
    python3 scripts/compare_models.py

## Notes
Baseline = traditional model  
Hybrid = traditional model + LLM sentiment  
Results are compared at the end
