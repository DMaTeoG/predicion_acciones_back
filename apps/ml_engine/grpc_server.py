"""gRPC Server implementation for ML Engine Prediction and Live Scan."""
import logging
import os
import sys
from concurrent import futures
import grpc

from apps.ml_engine.proto import prediction_pb2, prediction_pb2_grpc
from apps.ml_engine.live_scanner import LiveAssetScanner
from apps.nlp.sentiment_engine import SentimentEngine
from apps.ml_engine.predictor import MLPredictor

logger = logging.getLogger("ml_engine.grpc_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class PredictionServiceServicer(prediction_pb2_grpc.PredictionServiceServicer):
    """Servicer implementing PredictionService RPCs."""

    def __init__(self):
        logger.info("Initializing PredictionServiceServicer engines...")
        self.live_scanner = LiveAssetScanner()
        self.sentiment_engine = SentimentEngine()
        try:
            self.predictor = MLPredictor()
        except Exception as exc:
            logger.warning("MLPredictor initialization warning: %s", exc)
            self.predictor = None

    def Predict(self, request, context):
        """Perform quantitative prediction on ticker or feature vector."""
        logger.info("gRPC Predict called for ticker: %s", request.ticker)
        try:
            ticker = request.ticker or "NVDA"
            scan = self.live_scanner.scan_asset(ticker)
            return prediction_pb2.PredictResponse(
                ticker=ticker,
                direction_signal=scan.get("direction_signal", "HOLD"),
                probability_up=float(scan.get("probability_up", 0.5)),
                probability_down=float(scan.get("probability_down", 0.5)),
                model_used="EnsemblePredictor",
                status="SUCCESS",
            )
        except Exception as exc:
            logger.error("Predict RPC error: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return prediction_pb2.PredictResponse(status="ERROR")

    def AnalyzeSentiment(self, request, context):
        """Analyze sentiment of provided text."""
        logger.info("gRPC AnalyzeSentiment called for text length: %d", len(request.text))
        try:
            res = self.sentiment_engine.analyze_text(request.text)
            return prediction_pb2.SentimentResponse(
                sentiment_label=res.get("sentiment_label", "Neutral"),
                sentiment_score=float(res.get("sentiment_score", 0.0)),
                impact_score=float(res.get("impact_score", 0.0)),
                status="SUCCESS",
            )
        except Exception as exc:
            logger.error("AnalyzeSentiment RPC error: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return prediction_pb2.SentimentResponse(status="ERROR")

    def LiveScan(self, request, context):
        """Perform complete multi-factor scan on query/ticker."""
        logger.info("gRPC LiveScan called for query: %s", request.query)
        try:
            scan = self.live_scanner.scan_asset(request.query)
            if scan.get("status") == "error":
                return prediction_pb2.LiveScanResponse(
                    status="ERROR",
                    query=request.query,
                    error_message=scan.get("error_message", "Unknown error in live scan"),
                )

            top_news_protos = []
            for item in scan.get("top_news", []):
                top_news_protos.append(
                    prediction_pb2.NewsItemProto(
                        title=item.get("title", ""),
                        source=item.get("source", ""),
                        link=item.get("link", ""),
                        published=item.get("published", ""),
                        sentiment_score=float(item.get("sentiment_score", 0.0)),
                        sentiment_label=item.get("sentiment_label", "Neutral"),
                        impact_score=float(item.get("impact_score", 0.0)),
                    )
                )

            m_snap = scan.get("market_snapshot", {})
            market_snapshot_proto = prediction_pb2.MarketSnapshotProto(
                sma_20=float(m_snap.get("sma_20", 0.0)),
                sma_50=float(m_snap.get("sma_50", 0.0)),
                rsi_14=float(m_snap.get("rsi_14", 50.0)),
                macd=float(m_snap.get("macd", 0.0)),
                macd_signal=float(m_snap.get("macd_signal", 0.0)),
                volume=int(m_snap.get("volume", 0)),
            )

            advice = scan.get("advice", {})
            return prediction_pb2.LiveScanResponse(
                status="SUCCESS",
                query=scan.get("query", request.query),
                ticker=scan.get("ticker", ""),
                current_price=float(scan.get("current_price", 0.0)),
                change_percent=float(scan.get("change_percent", 0.0)),
                currency=scan.get("currency", "USD"),
                direction_signal=scan.get("direction_signal", "HOLD"),
                probability_up=float(scan.get("probability_up", 0.5)),
                probability_down=float(scan.get("probability_down", 0.5)),
                advice_action=advice.get("action", "MANTENER"),
                advice_summary=advice.get("summary", ""),
                advice_rationale=advice.get("rationale", ""),
                risk_level=advice.get("risk_level", "MEDIO"),
                average_sentiment=float(scan.get("average_sentiment", 0.0)),
                sentiment_label=scan.get("sentiment_label", "Neutral"),
                news_count=int(scan.get("news_count", 0)),
                top_news=top_news_protos,
                market_snapshot=market_snapshot_proto,
            )
        except Exception as exc:
            logger.error("LiveScan RPC error: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return prediction_pb2.LiveScanResponse(
                status="ERROR",
                query=request.query,
                error_message=str(exc),
            )


def serve(port: int = 50051, max_workers: int = 10):
    """Start gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    prediction_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionServiceServicer(), server)
    bind_address = f"0.0.0.0:{port}"
    server.add_insecure_port(bind_address)
    logger.info("Starting gRPC server on %s", bind_address)
    server.start()
    return server


if __name__ == "__main__":
    port = int(os.environ.get("GRPC_PORT", 50051))
    server = serve(port=port)
    logger.info("gRPC Server running successfully on port %d. Waiting for termination...", port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Stopping gRPC server...")
        server.stop(0)
