"""  Runs: prices -> news -> sentiment
Outputs:
    data/raw/prices.csv """
 

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

    
from src.utils.config import TICKERS
from src.data.fetch_price import download_prices, save_prices

def main():
    prices = download_prices(TICKERS)
    save_prices(prices, "data/raw/prices.csv")

    print("\nData collection system finished.")

if __name__ == "__main__":
    main()

