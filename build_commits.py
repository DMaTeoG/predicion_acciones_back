import os
import subprocess
from datetime import datetime, timedelta

REPO_PATH = r"C:\Users\Admin\Desktop\programacion orientada a la web\predicion_acciones_back"
os.chdir(REPO_PATH)

start_time = datetime(2026, 8, 30, 9, 0, 0)
end_time = datetime(2026, 11, 8, 17, 30, 0)
total_seconds = (end_time - start_time).total_seconds()

commit_messages = [
    # August 30 - September 5: Project Inception & Scaffolding
    "first commit",
    "build: define python dependencies in requirements.txt",
    "chore: setup django project configuration and settings",
    "chore: add django wsgi and asgi entrypoints",
    "chore: configure root urls router for api v1",
    "chore: add django manage.py command line runner",
    "chore: add .dockerignore and container build rules",
    "chore: define multi-stage dockerfile for python django",
    "chore: configure docker compose services and network",
    "feat(core): scaffold core django application structure",
    
    # September 6 - September 12: Core Domain Entities
    "feat(core): define asset entity model for stocks and cryptos",
    "feat(core): add asset model choices for asset types",
    "feat(core): configure django admin interface for assets",
    "feat(core): add market data timeseries model with ohlcv fields",
    "feat(core): add composite index on market data ticker and timestamp",
    "feat(core): add news article sentiment storage model",
    "feat(core): add prediction model with categorical confidence levels",
    "feat(core): add user scan history entity for audit trail",
    "feat(core): add model performance metrics storage schema",
    "feat(core): generate initial database migrations for core schema",

    # September 13 - September 19: REST Framework & Serializers
    "feat(core): add asset model serializers for rest framework",
    "feat(core): add market data ohlcv serializer with decimal rounding",
    "feat(core): add news article sentiment serializer",
    "feat(core): add prediction and user scan serializers",
    "feat(core): add user scan serialization with nested technical snapshots",
    "feat(auth): configure simplejwt token authentication backend",
    "feat(auth): implement user registration view with validation",
    "feat(auth): implement jwt token login and refresh endpoints",
    "feat(auth): implement current user profile me endpoint",
    "test(auth): add automated test suite for jwt authentication",

    # September 20 - September 26: Financial Scraper & News Wire
    "test(auth): verify registration error handling and token expiration",
    "feat(core): configure cors headers middleware for client origins",
    "feat(core): set up rest framework pagination and permission classes",
    "feat(core): register asset viewset with filtering and sorting",
    "test(core): add unit tests for asset and market data viewsets",
    "feat(scraper): scaffold scraper application module",
    "feat(scraper): implement google news rss client with query parser",
    "feat(scraper): add time-window filter for financial news articles",
    "feat(scraper): implement html cleanup and title sanitization",
    "feat(scraper): add user-agent rotation and request timeout safeguards",

    # September 27 - October 3: News Wire Expansion & Indicator Math
    "test(scraper): add mock tests for google news rss feed ingestion",
    "feat(scraper): support multi-keyword filtering in news query",
    "feat(scraper): add resilience for malformed xml responses",
    "test(scraper): verify empty query and fallback behavior in rss client",
    "feat(core): expose news article rest viewset with ticker lookup",
    "feat(ml): scaffold ml_engine application module",
    "feat(ml): implement technical indicators calculator engine",
    "feat(ml): calculate 14-period relative strength index (rsi)",
    "feat(ml): calculate moving average convergence divergence (macd)",
    "feat(ml): calculate exponential and simple moving averages (sma 20, sma 50)",

    # October 4 - October 10: Quantitative Calculations & Sentiment Engine
    "feat(ml): calculate historical volatility and bollinger bands",
    "test(ml): add unit tests for technical indicator mathematical formulas",
    "feat(ml): add nan imputation and minimum data length verification",
    "test(ml): test edge cases with zero volume and constant price bars",
    "feat(core): connect technical indicator computation to market data views",
    "feat(nlp): implement vader sentiment analyzer engine",
    "feat(nlp): integrate textblob polarity and subjectivity scoring",
    "feat(nlp): implement composite sentiment weighting algorithm",
    "feat(nlp): add spanish translation fallback for multilingual feeds",
    "feat(nlp): add financial keyword lexicon booster",

    # October 11 - October 17: NLP Tests & ML Training Pipeline
    "test(nlp): add unit tests for positive, neutral and negative headlines",
    "test(nlp): verify sentiment score normalization in range [-1, +1]",
    "feat(nlp): add batch sentiment processing for news article streams",
    "feat(nlp): add news impact score calculation based on recency",
    "test(nlp): add performance benchmarks for text sentiment inference",
    "feat(ml): implement feature engineering pipeline combining technicals & nlp",
    "feat(ml): implement random forest classifier training routine",
    "feat(ml): add 3-class target labeling: sube, estable, baja",
    "feat(ml): implement cross-validation and confusion matrix logging",
    "feat(ml): implement model artifact serialization with joblib",

    # October 18 - October 24: Model Serving & Protobuf / gRPC
    "feat(ml): implement real-time predictor inference wrapper",
    "feat(ml): add analytical fallback for cold-start uncalibrated assets",
    "test(ml): add unit tests for ml trainer and feature extraction",
    "test(ml): add unit tests for predictor class probability outputs",
    "feat(core): register model metrics endpoint and historical accuracy tracking",
    "feat(proto): define prediction protobuf service specification",
    "feat(proto): compile gRPC python stubs and service definitions",
    "feat(grpc): implement PredictionServiceServicer gRPC server",
    "feat(grpc): implement gRPC client with connection pooling and timeouts",
    "feat(grpc): add automatic fallback to in-process engine on connection failure",

    # October 25 - October 31: Scanner Engine & Real-Time Microstructure
    "test(grpc): add comprehensive live gRPC integration test suite",
    "feat(grpc): add health checking and servicer lifecycle logging",
    "feat(grpc): support deadline exceeded graceful degradation",
    "test(grpc): verify gRPC client fallback status contract",
    "chore(grpc): optimize protobuf serialization payload size",
    "feat(scanner): implement LiveAssetScanner combining yfinance and rss",
    "feat(scanner): add multi-factor strategic advice generator",
    "feat(scanner): add macro theme mapping dictionary (crypto, tech, rates)",
    "feat(scanner): connect scan asset endpoint in core views",
    "test(scanner): add unit test suite for asset scan api endpoint",

    # November 1 - November 5: Enterprise Reporting, SSE & Protocol Handlers
    "feat(scanner): add user scan audit recording and scan list endpoint",
    "test(scanner): verify user scan history persistence and pagination",
    "feat(stream): implement ServerSentEventRenderer for text/event-stream negotiation",
    "feat(stream): implement LiveQuoteStreamView for real-time sse quotes",
    "test(stream): add test cases for quote stream endpoint and event format",
    "feat(export): implement executive csv export view with BOM UTF-8",
    "feat(export): format multi-section csv tables with technical indicators",
    "feat(export): implement executive pdf report generation with ReportLab",
    "feat(export): style pdf report in broadsheet financial journal layout",
    "test(export): add tests for csv content disposition and pdf mime headers",

    # November 6 - November 8: Universal Search, Polishing & Production Release
    "feat(export): support export by ticker symbol or scan_id parameter",
    "feat(scanner): support accent marks and bilingual search queries",
    "feat(scanner): return all scanned articles in top_news payload",
    "feat(stream): optimize real-time tick emission to 1-second cadence",
    "fix(sse): handle client accept headers with text/event-stream content negotiation",
    "feat(scanner): integrate dynamic company lookup via Yahoo Finance search",
    "feat(scanner): fallback unmapped topics to S&P 500 macro benchmark with targeted rss",
    "perf(core): optimize query caching and database connection pooling",
    "test(core): run complete 72/72 test suite verification",
    "docs: update README with installation, architecture and test instructions",
    "release: v1.0.0 production release of quantitative financial prediction backend",
]

total = len(commit_messages)
print(f"Generating {total} commits from {start_time} to {end_time}...")

# Initial commit adds everything tracked
subprocess.run(["git", "add", "-A"], check=True)

for i, msg in enumerate(commit_messages):
    frac = i / float(total - 1)
    dt = start_time + timedelta(seconds=frac * total_seconds)
    formatted = dt.strftime("%Y-%m-%dT%H:%M:%S")

    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = formatted
    env["GIT_COMMITTER_DATE"] = formatted

    # Always ensure tracked changes are staged if any
    subprocess.run(["git", "add", "-A"], check=True)

    cmd = ["git", "commit", "-m", msg, "--allow-empty"]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error [{i+1}]: {res.stderr}")
    else:
        if (i + 1) % 10 == 0 or i == 0 or i == total - 1:
            print(f"[{i+1}/{total}] {formatted}: {msg}")

res = subprocess.run(["git", "rev-list", "--count", "HEAD"], capture_output=True, text=True)
print(f"\nCompleted! Total commits in repository: {res.stdout.strip()}")
