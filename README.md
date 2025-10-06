This is my repository for my LLM integration with stock prediction project.

virtual environment activation, necessary as ubuntu's system python may lack few packages 
    python3 -m venv .venv
    source .venv/bin/activate

to collect data and gather news sentiment
    python3 scripts/collect_data.py

to illustrate & further process
    python3 scripts/process_data.py



build baseline & hybrid
    python3 scripts/build_dataset_baseline.py
    python3 scripts/train_baseline.py
    python3 scripts/build_dataset_hybrid.py
    python3 scripts/train_hybrid.py

