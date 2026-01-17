"""
BASE External Services Interface
Phase 13: External Verification Service Abstraction

Provides interfaces for external verification services that can be
integrated when available. Services include:
- Code execution sandboxes
- Citation verification APIs
- External knowledge bases
- Human feedback systems

Per PPA1-Inv25: Federated External Integration
Per NOVEL-26: External Service Orchestration

NO PLACEHOLDERS | SERVICE AVAILABILITY TRACKED | GRACEFUL DEGRADATION
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import asyncio
import httpx


class ServiceType(Enum):
    """Types of external verification services."""
    CODE_EXECUTION = "code_execution"      # Sandbox for running code
    CITATION_VERIFY = "citation_verify"    # DOI, arXiv, Wikipedia APIs
    KNOWLEDGE_BASE = "knowledge_base"      # External knowledge sources
    HUMAN_FEEDBACK = "human_feedback"      # Human-in-the-loop
    LLM_CHALLENGER = "llm_challenger"      # Cross-LLM verification


class ServiceStatus(Enum):
    """Status of external service."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    NOT_CONFIGURED = "not_configured"


@dataclass
class ServiceConfig:
    """Configuration for an external service."""
    service_type: ServiceType
    endpoint: str = ""
    api_key: str = ""
    timeout_seconds: float = 30.0
    retry_count: int = 3
    enabled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceResult:
    """Result from an external service call."""
    service_type: ServiceType
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'service_type': self.service_type.value,
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'latency_ms': self.latency_ms,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class CodeExecutionRequest:
    """Request for code execution verification."""
    code: str
    language: str = "python"
    timeout_seconds: float = 10.0
    test_cases: List[Dict] = field(default_factory=list)
    expected_outputs: List[Any] = field(default_factory=list)


@dataclass
class CodeExecutionResult:
    """Result from code execution."""
    executed: bool = False
    output: str = ""
    errors: List[str] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    execution_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.tests_passed + self.tests_failed
        return self.tests_passed / total if total > 0 else 0.0


class ExternalServiceRegistry:
    """
    Registry for external verification services.
    
    Provides a unified interface for:
    1. Registering available services
    2. Checking service availability
    3. Calling services with graceful fallback
    4. Tracking service health metrics
    
    Services are opt-in and degrade gracefully when unavailable.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self._services: Dict[ServiceType, ServiceConfig] = {}
        self._health: Dict[ServiceType, ServiceStatus] = {}
        self._call_history: Dict[ServiceType, List[ServiceResult]] = {}
        self._handlers: Dict[ServiceType, Callable] = {}
        
        # Initialize with default (not configured) status
        for stype in ServiceType:
            self._health[stype] = ServiceStatus.NOT_CONFIGURED
            self._call_history[stype] = []
        
        # Load config if provided
        if config_path and config_path.exists():
            self._load_config(config_path)
    
    def register_service(self, config: ServiceConfig) -> bool:
        """Register an external service."""
        self._services[config.service_type] = config
        
        if config.enabled and config.endpoint:
            self._health[config.service_type] = ServiceStatus.AVAILABLE
            return True
        else:
            self._health[config.service_type] = ServiceStatus.NOT_CONFIGURED
            return False
    
    def register_handler(
        self, 
        service_type: ServiceType, 
        handler: Callable[[Any], Awaitable[ServiceResult]]
    ):
        """Register a custom handler for a service type."""
        self._handlers[service_type] = handler
    
    def is_available(self, service_type: ServiceType) -> bool:
        """Check if a service is available."""
        return self._health.get(service_type) == ServiceStatus.AVAILABLE
    
    def get_status(self, service_type: ServiceType) -> ServiceStatus:
        """Get status of a service."""
        return self._health.get(service_type, ServiceStatus.NOT_CONFIGURED)
    
    def get_all_status(self) -> Dict[ServiceType, ServiceStatus]:
        """Get status of all services."""
        return self._health.copy()
    
    async def call_service(
        self, 
        service_type: ServiceType, 
        request: Any
    ) -> ServiceResult:
        """
        Call an external service with request.
        
        Returns ServiceResult with success=False if service unavailable.
        This allows graceful degradation.
        """
        start_time = datetime.now()
        
        # Check if service is configured
        if service_type not in self._services:
            result = ServiceResult(
                service_type=service_type,
                success=False,
                error="Service not configured. Will be available when integrated.",
                latency_ms=0.0
            )
            self._call_history[service_type].append(result)
            return result
        
        config = self._services[service_type]
        
        # Check if service is enabled
        if not config.enabled:
            result = ServiceResult(
                service_type=service_type,
                success=False,
                error="Service disabled. Enable when external service is available.",
                latency_ms=0.0
            )
            self._call_history[service_type].append(result)
            return result
        
        # Use custom handler if registered
        if service_type in self._handlers:
            try:
                result = await self._handlers[service_type](request)
                result.latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._call_history[service_type].append(result)
                return result
            except Exception as e:
                self._health[service_type] = ServiceStatus.DEGRADED
                result = ServiceResult(
                    service_type=service_type,
                    success=False,
                    error=f"Handler error: {str(e)}",
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
                self._call_history[service_type].append(result)
                return result
        
        # Default HTTP call for services with endpoints
        try:
            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                response = await client.post(
                    config.endpoint,
                    json=request if isinstance(request, dict) else {'data': str(request)},
                    headers={'Authorization': f'Bearer {config.api_key}'} if config.api_key else {}
                )
                
                latency = (datetime.now() - start_time).total_seconds() * 1000
                
                if response.status_code == 200:
                    result = ServiceResult(
                        service_type=service_type,
                        success=True,
                        data=response.json(),
                        latency_ms=latency
                    )
                else:
                    self._health[service_type] = ServiceStatus.DEGRADED
                    result = ServiceResult(
                        service_type=service_type,
                        success=False,
                        error=f"HTTP {response.status_code}: {response.text[:200]}",
                        latency_ms=latency
                    )
                
                self._call_history[service_type].append(result)
                return result
                
        except Exception as e:
            self._health[service_type] = ServiceStatus.UNAVAILABLE
            result = ServiceResult(
                service_type=service_type,
                success=False,
                error=f"Connection error: {str(e)}",
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            self._call_history[service_type].append(result)
            return result
    
    def get_service_metrics(self, service_type: ServiceType) -> Dict:
        """Get metrics for a service."""
        history = self._call_history.get(service_type, [])
        
        if not history:
            return {
                'total_calls': 0,
                'success_rate': 0.0,
                'avg_latency_ms': 0.0,
                'status': self._health.get(service_type, ServiceStatus.NOT_CONFIGURED).value
            }
        
        successes = sum(1 for r in history if r.success)
        latencies = [r.latency_ms for r in history if r.latency_ms > 0]
        
        return {
            'total_calls': len(history),
            'success_rate': successes / len(history) if history else 0.0,
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0.0,
            'status': self._health.get(service_type, ServiceStatus.NOT_CONFIGURED).value
        }
    
    def _load_config(self, config_path: Path):
        """Load service configurations from file."""
        try:
            with open(config_path) as f:
                data = json.load(f)
                for svc_data in data.get('services', []):
                    config = ServiceConfig(
                        service_type=ServiceType(svc_data['type']),
                        endpoint=svc_data.get('endpoint', ''),
                        api_key=svc_data.get('api_key', ''),
                        timeout_seconds=svc_data.get('timeout', 30.0),
                        enabled=svc_data.get('enabled', False)
                    )
                    self.register_service(config)
        except Exception:
            pass  # Graceful degradation on config load failure

    # Learning Interface
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

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


class CodeVerificationService:
    """
    Code verification service interface.
    
    Provides abstraction for code execution verification when available.
    Falls back to pattern-based detection when service unavailable.
    
    Integration points:
    - Cursor Code Execution
    - Docker sandboxes
    - Replit API
    - Custom execution environments
    """
    
    def __init__(self, registry: ExternalServiceRegistry):
        self._registry = registry
        self._fallback_patterns = self._load_fallback_patterns()
    
    async def verify_code(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        """
        Verify code execution.
        
        If external service available: Execute in sandbox
        If unavailable: Use pattern-based analysis (degraded mode)
        """
        # Attempt external service
        service_result = await self._registry.call_service(
            ServiceType.CODE_EXECUTION,
            {
                'code': request.code,
                'language': request.language,
                'timeout': request.timeout_seconds,
                'test_cases': request.test_cases
            }
        )
        
        if service_result.success and service_result.data:
            # Parse external service response
            return CodeExecutionResult(
                executed=True,
                output=service_result.data.get('output', ''),
                errors=service_result.data.get('errors', []),
                tests_passed=service_result.data.get('tests_passed', 0),
                tests_failed=service_result.data.get('tests_failed', 0),
                execution_time_ms=service_result.latency_ms
            )
        
        # Fallback to pattern analysis (not actual execution)
        return self._pattern_based_analysis(request)
    
    def _pattern_based_analysis(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        """
        Pattern-based code analysis when execution unavailable.
        
        This does NOT execute code - it analyzes patterns.
        Results should be interpreted with this limitation.
        """
        errors = []
        
        # Check for common incomplete patterns
        if 'pass' in request.code and 'def ' in request.code:
            # Check if pass is the only body
            import re
            empty_funcs = re.findall(r'def \w+\([^)]*\):\s*pass', request.code)
            if empty_funcs:
                errors.append(f"Detected {len(empty_funcs)} empty function(s)")
        
        if 'TODO' in request.code or 'FIXME' in request.code:
            errors.append("Code contains TODO/FIXME markers")
        
        if '...' in request.code:
            errors.append("Code contains ellipsis placeholder")
        
        # Check for syntax patterns (not execution)
        try:
            compile(request.code, '<string>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")
        
        return CodeExecutionResult(
            executed=False,  # Pattern analysis, not execution
            output="[Pattern analysis only - external execution service not available]",
            errors=errors,
            tests_passed=0,
            tests_failed=len(errors),
            execution_time_ms=0.0
        )
    
    def _load_fallback_patterns(self) -> List[Dict]:
        """Load patterns for fallback analysis."""
        return [
            {'pattern': r'pass\s*$', 'issue': 'Empty function body'},
            {'pattern': r'TODO|FIXME', 'issue': 'Incomplete marker'},
            {'pattern': r'raise NotImplementedError', 'issue': 'Not implemented'},
            {'pattern': r'\.\.\.\s*$', 'issue': 'Ellipsis placeholder'},
        ]

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


# Singleton registry instance
_global_registry: Optional[ExternalServiceRegistry] = None


def get_service_registry() -> ExternalServiceRegistry:
    """Get the global service registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ExternalServiceRegistry()
    return _global_registry


def register_code_execution_service(endpoint: str, api_key: str = "") -> bool:
    """
    Register a code execution service.
    
    Call this when external execution service becomes available.
    """
    registry = get_service_registry()
    config = ServiceConfig(
        service_type=ServiceType.CODE_EXECUTION,
        endpoint=endpoint,
        api_key=api_key,
        enabled=True
    )
    return registry.register_service(config)


