"""
Dynamic Reputation System

Tracks source performance over time and adjusts reputation scores
based on classification accuracy and other metrics.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque


@dataclass
class SourceMetrics:
    """Metrics tracked for each source"""
    source_name: str
    total_urls: int = 0
    phishing_confirmed: int = 0
    safe_confirmed: int = 0
    suspicious_confirmed: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    # Rolling window of recent classifications
    recent_classifications: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy"""
        total = self.phishing_confirmed + self.safe_confirmed + self.false_positives + self.false_negatives
        if total == 0:
            return 0.5  # Default to neutral

        correct = self.phishing_confirmed + self.safe_confirmed
        return correct / total

    @property
    def precision(self) -> float:
        """Calculate precision (of predicted phishing, how many were correct)"""
        total_predicted_phishing = self.phishing_confirmed + self.false_positives
        if total_predicted_phishing == 0:
            return 0.5
        return self.phishing_confirmed / total_predicted_phishing

    @property
    def recall(self) -> float:
        """Calculate recall (of actual phishing, how many did we catch)"""
        total_actual_phishing = self.phishing_confirmed + self.false_negatives
        if total_actual_phishing == 0:
            return 0.5
        return self.phishing_confirmed / total_actual_phishing

    @property
    def f1_score(self) -> float:
        """Calculate F1 score"""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "source_name": self.source_name,
            "total_urls": self.total_urls,
            "phishing_confirmed": self.phishing_confirmed,
            "safe_confirmed": self.safe_confirmed,
            "suspicious_confirmed": self.suspicious_confirmed,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class ClassificationEvent:
    """A single classification event for tracking"""
    url: str
    source_name: str
    predicted: str
    actual: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0


class ReputationManager:
    """
    Manages dynamic reputation scores for URL sources.

    Blends base reputation with observed performance metrics to
    produce adaptive reputation scores.
    """

    # Minimum samples before we start adjusting reputation
    MIN_SAMPLES = 50

    # How much to weight historical data vs recent data (0-1)
    # Higher = more weight on recent
    DECAY_FACTOR = 0.95

    # Base reputations for known sources
    BASE_REPUTATIONS = {
        "synthetic": 0.50,
        "phishtank": 0.90,
        "openphish": 0.85,
        "urlhaus": 0.80,
        "crtsh": 0.85,
        "twitter": 0.88,
        "pastebin": 0.70,
    }

    # Maximum reputation adjustment from base
    MAX_ADJUSTMENT = 0.2

    def __init__(self):
        self._metrics: Dict[str, SourceMetrics] = {}
        self._lock = asyncio.Lock()

    def get_base_reputation(self, source_name: str) -> float:
        """Get the base reputation for a source"""
        return self.BASE_REPUTATIONS.get(source_name.lower(), 0.5)

    def get_reputation(self, source_name: str) -> float:
        """
        Get the dynamic reputation score for a source.

        Blends base reputation with observed metrics if we have enough data.
        """
        base_rep = self.get_base_reputation(source_name)

        metrics = self._metrics.get(source_name.lower())
        if not metrics or metrics.total_urls < self.MIN_SAMPLES:
            return base_rep

        # Calculate performance-based adjustment
        # Use F1 score as the primary metric
        performance = metrics.f1_score

        # Blend with base reputation
        # More weight to observed performance as sample size increases
        sample_weight = min(1.0, metrics.total_urls / 500)  # Max weight at 500 samples

        # Calculate adjustment from base
        adjustment = (performance - 0.5) * self.MAX_ADJUSTMENT * sample_weight

        # Apply decay for recency
        time_since_update = (datetime.utcnow() - metrics.last_updated).total_seconds()
        decay = self.DECAY_FACTOR ** (time_since_update / 3600)  # Decay per hour

        final_rep = base_rep + (adjustment * decay)

        # Clamp to valid range
        return max(0.1, min(0.99, final_rep))

    async def record_classification(
        self,
        source_name: str,
        url: str,
        predicted: str,
        confidence: float = 0.0,
        actual: Optional[str] = None,
    ):
        """
        Record a classification for a source.

        Args:
            source_name: Name of the URL source
            url: The URL that was classified
            predicted: Predicted classification (PHISHING, SAFE, SUSPICIOUS)
            confidence: Classification confidence
            actual: Actual classification if known (for ground truth)
        """
        async with self._lock:
            source_key = source_name.lower()

            if source_key not in self._metrics:
                self._metrics[source_key] = SourceMetrics(source_name=source_key)

            metrics = self._metrics[source_key]
            metrics.total_urls += 1
            metrics.last_updated = datetime.utcnow()

            # Create event
            event = ClassificationEvent(
                url=url,
                source_name=source_key,
                predicted=predicted,
                actual=actual,
                confidence=confidence,
            )
            metrics.recent_classifications.append(event)

            # If we have ground truth, update counts
            if actual:
                self._update_metrics_with_ground_truth(metrics, predicted, actual)

    def _update_metrics_with_ground_truth(
        self,
        metrics: SourceMetrics,
        predicted: str,
        actual: str,
    ):
        """Update metrics when we have ground truth"""
        predicted = predicted.upper()
        actual = actual.upper()

        if predicted == actual:
            # Correct prediction
            if actual == "PHISHING":
                metrics.phishing_confirmed += 1
            elif actual == "SAFE":
                metrics.safe_confirmed += 1
            elif actual == "SUSPICIOUS":
                metrics.suspicious_confirmed += 1
        else:
            # Incorrect prediction
            if predicted == "PHISHING" and actual in ("SAFE", "SUSPICIOUS"):
                # False positive (said phishing but wasn't)
                metrics.false_positives += 1
            elif predicted in ("SAFE", "SUSPICIOUS") and actual == "PHISHING":
                # False negative (said safe but was phishing)
                metrics.false_negatives += 1

    async def update_ground_truth(
        self,
        source_name: str,
        url: str,
        actual: str,
    ):
        """
        Update metrics with ground truth for a previously classified URL.

        Args:
            source_name: Name of the URL source
            url: The URL
            actual: The actual classification
        """
        async with self._lock:
            source_key = source_name.lower()

            if source_key not in self._metrics:
                return

            metrics = self._metrics[source_key]

            # Find the classification event
            for event in metrics.recent_classifications:
                if event.url == url and event.actual is None:
                    event.actual = actual
                    self._update_metrics_with_ground_truth(
                        metrics, event.predicted, actual
                    )
                    break

    def get_metrics(self, source_name: str) -> Optional[SourceMetrics]:
        """Get metrics for a source"""
        return self._metrics.get(source_name.lower())

    def get_all_metrics(self) -> Dict[str, SourceMetrics]:
        """Get metrics for all sources"""
        return dict(self._metrics)

    def get_all_reputations(self) -> Dict[str, float]:
        """Get reputation scores for all sources"""
        reputations = {}

        # Include all known base sources
        for source in self.BASE_REPUTATIONS:
            reputations[source] = self.get_reputation(source)

        # Include any additional sources we've tracked
        for source in self._metrics:
            if source not in reputations:
                reputations[source] = self.get_reputation(source)

        return reputations

    def get_statistics(self) -> Dict:
        """Get overall reputation system statistics"""
        total_sources = len(self._metrics)
        total_urls = sum(m.total_urls for m in self._metrics.values())
        avg_accuracy = 0.0

        if total_sources > 0:
            accuracies = [m.accuracy for m in self._metrics.values() if m.total_urls >= self.MIN_SAMPLES]
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)

        return {
            "total_sources_tracked": total_sources,
            "total_urls_tracked": total_urls,
            "sources_with_enough_data": len([m for m in self._metrics.values() if m.total_urls >= self.MIN_SAMPLES]),
            "average_accuracy": avg_accuracy,
            "reputations": self.get_all_reputations(),
        }

    def reset_source(self, source_name: str):
        """Reset metrics for a source"""
        source_key = source_name.lower()
        if source_key in self._metrics:
            del self._metrics[source_key]

    def reset_all(self):
        """Reset all metrics"""
        self._metrics.clear()


# Global reputation manager instance
reputation_manager = ReputationManager()
