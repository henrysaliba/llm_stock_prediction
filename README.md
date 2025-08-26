This is my repository for my LLM integration with stock prediction project.


virtual environment activation, necessary as ubuntu's system python may lack few packages 
source .venv/bin/activate

to collect data + news sentiment
python3 scripts/collect_data.py

to illustrate graphs
python3 scripts/make_figures.py

current errors consist of:
    News headlines lacks extensive history
    Must run collect_data.py once a day in order to keep updating the csv properly. 


    Need to find a way to backfill data on every pull
