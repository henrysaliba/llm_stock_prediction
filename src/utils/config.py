from datetime import datetime, timedelta

TICKERS = ["AAPL", "AMZN"]
LOOKBACK_DAYS = 90

TODAY = datetime.utcnow().date()
START_DATE = (TODAY - timedelta(days=LOOKBACK_DAYS)).isoformat()
END_DATE = TODAY.isoformat()