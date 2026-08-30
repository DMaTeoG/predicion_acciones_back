"""gRPC Client for ML Engine with smart local fallback."""
import logging
import os
import grpc
from django.conf import settings

from apps.ml_engine.proto import prediction_pb2, prediction_pb2_grpc
from apps.ml_engine.live_scanner import LiveAssetScanner
from apps.nlp.sentiment_engine import SentimentEngine

logger = logging.getLogger("ml_engine.grpc_client")


class GrpcMLClient:
    """Client that communicates with the ML Engine gRPC microservice,

    with automatic fallback to in-process execution when gRPC is unavailable.
    """

    def __init__(self, host: str = None, timeout: float = 8.0):
        self.host = host or os.environ.get("GRPC_ML_HOST") or getattr(settings, "GRPC_ML_HOST", None)
        self.timeout = timeout
        self._channel = None
        self._stub = None
        self._fallback_scanner = None
        self._fallback_sentiment = None

    def _get_stub(self):
        if not self.host:
            return None
        if self._stub is None:
            self._channel = grpc.insecure_channel(self.host)
            self._stub = prediction_pb2_grpc.PredictionServiceStub(self._channel)
        return self._stub

    def _get_fallback_scanner(self) -> LiveAssetScanner:
        if self._fallback_scanner is None:
            self._fallback_scanner = LiveAssetScanner()
        return self._fallback_scanner

    def _get_fallback_sentiment(self) -> SentimentEngine:
        if self._fallback_sentiment is None:
            self._fallback_sentiment = SentimentEngine()
        return self._fallback_sentiment

    def scan_asset(self, query: str) -> dict:
        """Scan asset using gRPC microservice if available, falling back to local engine."""
        stub = self._get_stub()
        if stub:
            try:
                logger.info("Attempting gRPC LiveScan to %s for query: %s", self.host, query)
                req = prediction_pb2.LiveScanRequest(query=query)
                resp = stub.LiveScan(req, timeout=self.timeout)

                if resp.status == "ERROR":
                    logger.warning("gRPC LiveScan returned ERROR status: %s", resp.error_message)
                    return {"status": "error", "error_message": resp.error_message}

                top_news = []
                for n in resp.top_news:
                    top_news.append({
                        "title": n.title,
                        "source": n.source,
                        "link": n.link,
                        "published": n.published,
                        "sentiment_score": round(n.sentiment_score, 4),
                        "sentiment_label": n.sentiment_label,
                        "impact_score": round(n.impact_score, 2),
                    })

                return {
                    "status": "success",
                    "source": "grpc",
                    "query": resp.query,
                    "ticker": resp.ticker,
                    "current_price": round(resp.current_price, 2),
                    "change_percent": round(resp.change_percent, 2),
                    "currency": resp.currency,
                    "direction_signal": resp.direction_signal,
                    "probability_up": round(resp.probability_up, 4),
                    "probability_down": round(resp.probability_down, 4),
                    "advice": {
                        "action": resp.advice_action,
                        "summary": resp.advice_summary,
                        "rationale": resp.advice_rationale,
                        "risk_level": resp.risk_level,
                    },
                    "average_sentiment": round(resp.average_sentiment, 4),
                    "sentiment_label": resp.sentiment_label,
                    "news_count": resp.news_count,
                    "top_news": top_news,
                    "market_snapshot": {
                        "sma_20": round(resp.market_snapshot.sma_20, 2),
                        "sma_50": round(resp.market_snapshot.sma_50, 2),
                        "rsi_14": round(resp.market_snapshot.rsi_14, 2),
                        "macd": round(resp.market_snapshot.macd, 4),
                        "macd_signal": round(resp.market_snapshot.macd_signal, 4),
                        "volume": resp.market_snapshot.volume,
                    },
                }
            except Exception as exc:
                logger.warning(
                    "gRPC communication to %s failed (%s). Falling back to local LiveAssetScanner.",
                    self.host,
                    exc,
                )

        # Fallback to local execution
        result = self._get_fallback_scanner().scan_asset(query)
        result["source"] = "local_fallback"
        return result

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment via gRPC or local fallback."""
        stub = self._get_stub()
        if stub:
            try:
                req = prediction_pb2.SentimentRequest(text=text)
                resp = stub.AnalyzeSentiment(req, timeout=self.timeout)
                return {
                    "status": "success",
                    "source": "grpc",
                    "sentiment_label": resp.sentiment_label,
                    "sentiment_score": round(resp.sentiment_score, 4),
                    "impact_score": round(resp.impact_score, 2),
                }
            except Exception as exc:
                logger.warning("gRPC AnalyzeSentiment failed (%s). Falling back to local engine.", exc)

        res = self._get_fallback_sentiment().analyze_text(text)
        res["status"] = "success"
        res["source"] = "local_fallback"
        return res

    def close(self):
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
