"""Unit tests for gRPC Protobuf stubs, PredictionServiceServicer, and GrpcMLClient."""
from concurrent import futures
import socket
from unittest.mock import MagicMock, patch
from django.test import TestCase
from rest_framework.test import APIClient
import grpc

from apps.ml_engine.proto import prediction_pb2, prediction_pb2_grpc
from apps.ml_engine.grpc_server import PredictionServiceServicer, serve
from apps.ml_engine.grpc_client import GrpcMLClient


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class GrpcProtoAndStubsTestCase(TestCase):
    def test_proto_classes_exist(self):
        req = prediction_pb2.LiveScanRequest(query="AAPL")
        self.assertEqual(req.query, "AAPL")

        sent_req = prediction_pb2.SentimentRequest(text="Record profits reported!")
        self.assertEqual(sent_req.text, "Record profits reported!")

        pred_req = prediction_pb2.PredictRequest(ticker="NVDA")
        self.assertEqual(pred_req.ticker, "NVDA")


class GrpcServicerDirectTestCase(TestCase):
    def setUp(self):
        self.servicer = PredictionServiceServicer()

    def test_analyze_sentiment_servicer(self):
        req = prediction_pb2.SentimentRequest(text="Record surge in revenue and high margins")
        context = MagicMock()
        resp = self.servicer.AnalyzeSentiment(req, context)
        self.assertEqual(resp.status, "SUCCESS")
        self.assertIn(resp.sentiment_label.upper(), ["POSITIVE", "BULLISH", "NEUTRAL", "POSITIVO"])
        self.assertGreaterEqual(resp.sentiment_score, 0.0)

    @patch("apps.ml_engine.grpc_server.LiveAssetScanner.scan_asset")
    def test_live_scan_servicer_mocked(self, mock_scan):
        mock_scan.return_value = {
            "status": "success",
            "query": "TEST",
            "ticker": "TEST",
            "current_price": 150.0,
            "change_percent": 2.5,
            "currency": "USD",
            "direction_signal": "BUY",
            "probability_up": 0.65,
            "probability_down": 0.35,
            "average_sentiment": 0.45,
            "sentiment_label": "Positivo",
            "news_count": 1,
            "top_news": [{
                "title": "Positive Earnings",
                "source": "Reuters",
                "link": "https://example.com",
                "published": "2026-10-19",
                "sentiment_score": 0.8,
                "sentiment_label": "Positivo",
                "impact_score": 0.9,
            }],
            "market_snapshot": {
                "sma_20": 145.0,
                "sma_50": 140.0,
                "rsi_14": 62.0,
                "macd": 1.2,
                "macd_signal": 0.8,
                "volume": 1000000,
            },
            "advice": {
                "action": "COMPRA",
                "summary": "Señal alcista",
                "rationale": "Buenas métricas",
                "risk_level": "MEDIO",
            },
        }

        req = prediction_pb2.LiveScanRequest(query="TEST")
        context = MagicMock()
        resp = self.servicer.LiveScan(req, context)

        self.assertEqual(resp.status, "SUCCESS")
        self.assertEqual(resp.ticker, "TEST")
        self.assertEqual(resp.current_price, 150.0)
        self.assertEqual(resp.direction_signal, "BUY")
        self.assertEqual(len(resp.top_news), 1)
        self.assertEqual(resp.top_news[0].title, "Positive Earnings")


class GrpcClientTestCase(TestCase):
    def test_client_fallback_mode_no_host(self):
        client = GrpcMLClient(host=None)
        res = client.analyze_sentiment("Good quarterly revenue growth")
        self.assertIn("sentiment_score", res)
        self.assertEqual(res.get("source"), "local_fallback")

    def test_client_fallback_on_unreachable_host(self):
        # Point client to a port that is definitely not running a gRPC server
        dead_port = get_free_port()
        client = GrpcMLClient(host=f"127.0.0.1:{dead_port}", timeout=0.5)
        res = client.analyze_sentiment("Neutral market conditions")
        self.assertIn("sentiment_score", res)
        self.assertEqual(res.get("source"), "local_fallback")
        client.close()

    def test_live_grpc_client_server_integration(self):
        port = get_free_port()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        prediction_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionServiceServicer(), server)
        server.add_insecure_port(f"127.0.0.1:{port}")
        server.start()

        try:
            client = GrpcMLClient(host=f"127.0.0.1:{port}", timeout=5.0)
            res = client.analyze_sentiment("Nvidia unveils breakthrough next-generation chips")
            self.assertEqual(res.get("status"), "success")
            self.assertEqual(res.get("source"), "grpc")
            self.assertIn("sentiment_score", res)
            client.close()
        finally:
            server.stop(0)
