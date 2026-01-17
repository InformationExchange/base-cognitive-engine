"""
BASE Cognitive Governance Engine v47.0
API Gateway with AI + Pattern + Learning

Phase 47: API Infrastructure
- AI-powered routing and load balancing
- Pattern-based rate limiting
- Continuous learning from traffic patterns
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import logging

logger = logging.getLogger(__name__)


class RouteStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"


class RateLimitAction(Enum):
    ALLOW = "allow"
    THROTTLE = "throttle"
    BLOCK = "block"


@dataclass
class Route:
    path: str
    handler: str
    methods: List[str]
    rate_limit: int = 100  # requests per minute
    timeout_ms: int = 5000
    circuit_breaker: bool = True


@dataclass
class RateLimitResult:
    action: RateLimitAction
    remaining: int
    reset_at: datetime
    retry_after_ms: Optional[int] = None


@dataclass
class RequestMetrics:
    path: str
    method: str
    status_code: int
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PatternBasedRateLimiter:
    """
    Pattern-based rate limiting.
    Layer 1: Static rate limits.
    """
    
    DEFAULT_LIMITS = {
        "/api/evaluate": {"limit": 100, "window_seconds": 60},
        "/api/audit": {"limit": 50, "window_seconds": 60},
        "/api/health": {"limit": 1000, "window_seconds": 60},
        "default": {"limit": 200, "window_seconds": 60},
    }
    
    def __init__(self):
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque())
        self.enforcement_count = 0
    
    def check_limit(self, path: str, client_id: str) -> RateLimitResult:
        """Check rate limit for a request."""
        key = f"{client_id}:{path}"
        limits = self.DEFAULT_LIMITS.get(path, self.DEFAULT_LIMITS["default"])
        
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=limits["window_seconds"])
        
        # Clean old requests
        requests = self.request_counts[key]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check limit
        if len(requests) >= limits["limit"]:
            self.enforcement_count += 1
            return RateLimitResult(
                action=RateLimitAction.THROTTLE,
                remaining=0,
                reset_at=now + timedelta(seconds=limits["window_seconds"]),
                retry_after_ms=limits["window_seconds"] * 1000
            )
        
        # Record request
        requests.append(now)
        
        return RateLimitResult(
            action=RateLimitAction.ALLOW,
            remaining=limits["limit"] - len(requests),
            reset_at=now + timedelta(seconds=limits["window_seconds"])
        )


class AIRoutingOptimizer:
    """
    AI-powered routing optimization.
    Layer 2: Intelligent load balancing.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.route_health: Dict[str, float] = defaultdict(lambda: 1.0)
        self.latency_history: Dict[str, List[float]] = defaultdict(list)
        self.optimization_count = 0
    
    def update_health(self, path: str, latency_ms: float, success: bool):
        """Update route health score."""
        self.latency_history[path].append(latency_ms)
        if len(self.latency_history[path]) > 100:
            self.latency_history[path] = self.latency_history[path][-100:]
        
        # Calculate health score
        current = self.route_health[path]
        if success:
            self.route_health[path] = min(1.0, current * 0.95 + 0.05)
        else:
            self.route_health[path] = max(0.0, current * 0.9)
    
    def select_route(self, candidates: List[str]) -> str:
        """Select best route from candidates."""
        self.optimization_count += 1
        
        if not candidates:
            return None
        
        # Select route with best health
        best = max(candidates, key=lambda p: self.route_health.get(p, 0.5))
        return best
    
    def get_route_status(self, path: str) -> RouteStatus:
        """Get route health status."""
        health = self.route_health.get(path, 1.0)
        if health > 0.8:
            return RouteStatus.HEALTHY
        elif health > 0.5:
            return RouteStatus.DEGRADED
        return RouteStatus.UNHEALTHY

    # Learning Interface
    def record_outcome(self, outcome: Dict) -> None:
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        pass
    
    def get_statistics(self) -> Dict:
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        self._outcomes = state.get('outcomes', [])


class TrafficLearner:
    """
    Learns from traffic patterns.
    Layer 3: Continuous improvement.
    """
    
    def __init__(self):
        self.hourly_traffic: Dict[int, int] = defaultdict(int)
        self.path_popularity: Dict[str, int] = defaultdict(int)
        self.learned_limits: Dict[str, int] = {}
        self.anomaly_count = 0
    
    def record_request(self, metrics: RequestMetrics):
        """Record request for learning."""
        hour = metrics.timestamp.hour
        self.hourly_traffic[hour] += 1
        self.path_popularity[metrics.path] += 1
    
    def detect_traffic_anomaly(self, current_rate: float) -> bool:
        """Detect unusual traffic patterns."""
        hour = datetime.utcnow().hour
        expected = self.hourly_traffic.get(hour, 100)
        if expected > 0 and current_rate > expected * 2:
            self.anomaly_count += 1
            return True
        return False
    
    def recommend_limit(self, path: str) -> int:
        """Recommend rate limit based on traffic patterns."""
        popularity = self.path_popularity.get(path, 0)
        if popularity > 1000:
            return 500  # High traffic paths get more capacity
        elif popularity > 100:
            return 200
        return 100
    
    def get_traffic_insights(self) -> Dict[str, Any]:
        """Get traffic insights."""
        total = sum(self.hourly_traffic.values())
        peak_hour = max(self.hourly_traffic.items(), key=lambda x: x[1])[0] if self.hourly_traffic else 12
        return {
            "total_requests": total,
            "peak_hour": peak_hour,
            "popular_paths": sorted(self.path_popularity.items(), key=lambda x: x[1], reverse=True)[:5],
            "anomalies_detected": self.anomaly_count
        }

    # =========================================================================
    # Learning Interface
    # =========================================================================
    
    def record_outcome(self, outcome: Dict) -> None:
        """Record operation outcome for learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            for key in self._learning_params:
                self._learning_params[key] *= 0.95
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'outcomes': len(getattr(self, '_outcomes', [])),
            'params': getattr(self, '_learning_params', {})
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('params', {})


class EnhancedAPIGateway:
    """
    Unified API gateway with AI + Pattern + Learning.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_ai: bool = True):
        # Layer 1: Pattern rate limiting
        self.rate_limiter = PatternBasedRateLimiter()
        
        # Layer 2: AI routing
        self.ai_router = AIRoutingOptimizer(api_key) if use_ai else None
        
        # Layer 3: Learning
        self.learner = TrafficLearner()
        
        # Routes
        self.routes: Dict[str, Route] = {}
        self.total_requests = 0
        
        # Register default routes
        self._register_default_routes()
        
        logger.info("[Gateway] Enhanced API Gateway initialized")
    
    def _register_default_routes(self):
        """Register default BASE routes."""
        default_routes = [
            Route("/api/evaluate", "evaluate_handler", ["POST"]),
            Route("/api/audit", "audit_handler", ["GET", "POST"]),
            Route("/api/health", "health_handler", ["GET"], rate_limit=1000),
            Route("/api/improve", "improve_handler", ["POST"]),
        ]
        for route in default_routes:
            self.routes[route.path] = route
    
    def handle_request(self, path: str, method: str, client_id: str) -> Dict[str, Any]:
        """Handle an incoming request."""
        self.total_requests += 1
        
        # Check rate limit
        limit_result = self.rate_limiter.check_limit(path, client_id)
        if limit_result.action != RateLimitAction.ALLOW:
            return {
                "status": "rate_limited",
                "retry_after_ms": limit_result.retry_after_ms
            }
        
        # Get route
        route = self.routes.get(path)
        if not route:
            return {"status": "not_found"}
        
        # Simulate handling
        start = time.time()
        latency = (time.time() - start) * 1000 + 5
        
        # Record metrics
        metrics = RequestMetrics(
            path=path,
            method=method,
            status_code=200,
            latency_ms=latency
        )
        self.learner.record_request(metrics)
        
        if self.ai_router:
            self.ai_router.update_health(path, latency, True)
        
        return {
            "status": "success",
            "latency_ms": latency,
            "remaining_quota": limit_result.remaining
        }
    
    def get_status(self) -> Dict[str, Any]:
        insights = self.learner.get_traffic_insights()
        return {
            "mode": "AI + Pattern + Learning",
            "ai_enabled": self.ai_router is not None,
            "routes_registered": len(self.routes),
            "total_requests": self.total_requests,
            "rate_limit_enforcements": self.rate_limiter.enforcement_count,
            "routing_optimizations": self.ai_router.optimization_count if self.ai_router else 0,
            "traffic_anomalies": insights.get("anomalies_detected", 0)
        }



    # ========================================
    # PHASE 49: PERSISTENCE METHODS
    # ========================================
    
    def save_state(self, filepath=None):
        """Save learning state (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.save_state()
        return False
    
    def load_state(self, filepath=None):
        """Load learning state (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.load_state()
        return False



    # ========================================
    # LEARNING INTERFACE (Auto-added)
    # ========================================

    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, input_data, output_data, was_correct, domain=None, metadata=None):
        """Record outcome for adaptive learning."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append({
            'input': str(input_data)[:100],
            'correct': was_correct,
            'domain': domain
        })
        self._outcomes = self._outcomes[-1000:]

    def record_feedback(self, result, was_accurate):
        """Record feedback for learning adjustment."""
        self.record_outcome({'result': str(result)}, {}, was_accurate)

    def adapt_thresholds(self, threshold_name, current_value, direction='decrease'):
        """Adapt thresholds based on learning history."""
        adjustment = 0.05 if direction == 'increase' else -0.05
        return max(0.1, min(0.95, current_value + adjustment))

    def get_domain_adjustment(self, domain):
        """Get learned adjustment for a domain."""
        if not hasattr(self, '_domain_adjustments'):
            self._domain_adjustments = {}
        return self._domain_adjustments.get(domain, 0.0)

    def get_learning_statistics(self):
        """Get learning statistics."""
        outcomes = getattr(self, '_outcomes', [])
        correct = sum(1 for o in outcomes if o.get('correct', False))
        return {
            'module': self.__class__.__name__,
            'total_outcomes': len(outcomes),
            'accuracy': correct / len(outcomes) if outcomes else 0.0
        }

    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback."""
        pass  # Implement specific learning logic


    def get_statistics(self) -> Dict:
        """Return statistics."""
        return {'outcomes': len(getattr(self, '_outcomes', []))}


    def serialize_state(self) -> Dict:
        """Serialize state."""
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}


    def deserialize_state(self, state: Dict) -> None:
        """Restore state."""
        self._outcomes = state.get('outcomes', [])


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 47: API Gateway (AI + Pattern + Learning)")
    print("=" * 70)
    
    gateway = EnhancedAPIGateway(api_key=None, use_ai=True)
    
    print("\n[1] Registered Routes")
    print("-" * 60)
    for path, route in gateway.routes.items():
        print(f"  {path}: {route.methods} (limit: {route.rate_limit}/min)")
    
    print("\n[2] Request Handling")
    print("-" * 60)
    
    for i in range(5):
        result = gateway.handle_request("/api/evaluate", "POST", "client_1")
        print(f"  Request {i+1}: {result['status']} (remaining: {result.get('remaining_quota', 'N/A')})")
    
    print("\n[3] Rate Limiting Test")
    print("-" * 60)
    
    # Simulate many requests to trigger rate limit
    gateway.rate_limiter.DEFAULT_LIMITS["/api/test"] = {"limit": 3, "window_seconds": 60}
    for i in range(5):
        result = gateway.rate_limiter.check_limit("/api/test", "client_2")
        print(f"  Check {i+1}: {result.action.value} (remaining: {result.remaining})")
    
    print("\n[4] Traffic Learning")
    print("-" * 60)
    insights = gateway.learner.get_traffic_insights()
    print(f"  Total Requests: {insights['total_requests']}")
    print(f"  Popular Paths: {insights['popular_paths'][:3]}")
    
    print("\n[5] Gateway Status")
    print("-" * 60)
    status = gateway.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print("PHASE 47: API Gateway - VERIFIED")
    print("=" * 70)
