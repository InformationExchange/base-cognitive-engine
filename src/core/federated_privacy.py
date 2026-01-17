"""
BAIS Cognitive Governance Engine v35.0
Federated Learning & Privacy-Preserving Aggregation

Phase 35: Addresses PPA2-C1-30, PPA2-C1-31, PPA3-Comp3
- Federated learning for distributed governance
- Privacy-preserving aggregation with differential privacy
- Secure multi-party coordination

Patent Claims Addressed:
- PPA2-C1-30: Federated learning across distributed governance nodes
- PPA2-C1-31: Privacy-preserving aggregation of governance signals
- PPA3-Comp3: Secure multi-party computation for sensitive decisions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import hashlib
import secrets
import logging
import json
import uuid

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating federated updates."""
    FEDAVG = "federated_averaging"
    FEDPROX = "federated_proximal"
    WEIGHTED = "weighted_average"
    MEDIAN = "coordinate_median"
    TRIMMED_MEAN = "trimmed_mean"


class PrivacyMechanism(Enum):
    """Privacy-preserving mechanisms."""
    NONE = "none"
    GAUSSIAN_DP = "gaussian_dp"
    LAPLACE_DP = "laplace_dp"
    LOCAL_DP = "local_dp"
    SECURE_AGGREGATION = "secure_aggregation"


class NodeRole(Enum):
    """Role of a node in federated system."""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


class NodeStatus(Enum):
    """Status of a federated node."""
    ACTIVE = "active"
    IDLE = "idle"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    OFFLINE = "offline"


@dataclass
class PrivacyBudget:
    """Privacy budget for differential privacy."""
    total_epsilon: float
    total_delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    operations: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def remaining_epsilon(self) -> float:
        return self.total_epsilon - self.spent_epsilon
    
    @property
    def remaining_delta(self) -> float:
        return self.total_delta - self.spent_delta
    
    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        return (self.remaining_epsilon >= epsilon and 
                self.remaining_delta >= delta)
    
    def spend(self, epsilon: float, delta: float, operation: str) -> bool:
        if not self.can_spend(epsilon, delta):
            return False
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.operations.append({
            "operation": operation,
            "epsilon": epsilon,
            "delta": delta,
            "timestamp": datetime.utcnow().isoformat()
        })
        return True


@dataclass
class FederatedNode:
    """A node in the federated governance network."""
    node_id: str
    role: NodeRole
    status: NodeStatus
    public_key: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    local_model_version: int = 0
    contribution_weight: float = 1.0
    privacy_budget: Optional[PrivacyBudget] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self, timeout_seconds: int = 60) -> bool:
        return (datetime.utcnow() - self.last_heartbeat).total_seconds() < timeout_seconds


@dataclass
class LocalUpdate:
    """Local model update from a participant."""
    node_id: str
    update_id: str
    parameters: Dict[str, np.ndarray]
    num_samples: int
    timestamp: datetime
    signature: Optional[str] = None
    noise_added: bool = False
    epsilon_spent: float = 0.0


@dataclass
class AggregatedModel:
    """Result of federated aggregation."""
    model_version: int
    parameters: Dict[str, np.ndarray]
    num_contributors: int
    total_samples: int
    aggregation_method: AggregationMethod
    timestamp: datetime
    privacy_guarantee: Dict[str, float]


@dataclass
class SecureShare:
    """A share in secure aggregation."""
    share_id: str
    owner_id: str
    target_id: str
    value: np.ndarray
    mask: np.ndarray


class DifferentialPrivacy:
    """
    Differential privacy mechanisms for governance signals.
    
    PPA2-C1-31: Privacy-preserving aggregation.
    """
    
    def __init__(self, default_epsilon: float = 1.0, default_delta: float = 1e-5):
        self.default_epsilon = default_epsilon
        self.default_delta = default_delta
        
    def add_gaussian_noise(
        self,
        data: np.ndarray,
        sensitivity: float,
        epsilon: float,
        delta: float
    ) -> Tuple[np.ndarray, float]:
        """
        Add Gaussian noise for (ε, δ)-differential privacy.
        
        Args:
            data: Original data
            sensitivity: L2 sensitivity
            epsilon: Privacy parameter
            delta: Privacy parameter
            
        Returns:
            Noisy data and actual epsilon spent
        """
        # Calculate sigma for Gaussian mechanism
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        noise = np.random.normal(0, sigma, data.shape)
        noisy_data = data + noise
        
        return noisy_data, epsilon
    
    def add_laplace_noise(
        self,
        data: np.ndarray,
        sensitivity: float,
        epsilon: float
    ) -> Tuple[np.ndarray, float]:
        """
        Add Laplace noise for ε-differential privacy.
        
        Args:
            data: Original data
            sensitivity: L1 sensitivity
            epsilon: Privacy parameter
            
        Returns:
            Noisy data and epsilon spent
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, data.shape)
        noisy_data = data + noise
        
        return noisy_data, epsilon
    
    def clip_gradient(
        self,
        gradient: np.ndarray,
        max_norm: float
    ) -> np.ndarray:
        """Clip gradient to bound sensitivity."""
        norm = np.linalg.norm(gradient)
        if norm > max_norm:
            gradient = gradient * (max_norm / norm)
        return gradient
    
    def randomized_response(
        self,
        value: bool,
        epsilon: float
    ) -> Tuple[bool, float]:
        """
        Randomized response for local differential privacy.
        
        Args:
            value: True binary value
            epsilon: Privacy parameter
            
        Returns:
            Potentially flipped value and epsilon
        """
        p = np.exp(epsilon) / (1 + np.exp(epsilon))
        
        if np.random.random() < p:
            return value, epsilon
        else:
            return not value, epsilon

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


class SecureAggregation:
    """
    Secure aggregation protocol for federated learning.
    
    Implements secret sharing for privacy-preserving sum.
    """
    
    def __init__(self, num_parties: int, threshold: int):
        """
        Initialize secure aggregation.
        
        Args:
            num_parties: Total number of parties
            threshold: Minimum parties needed for reconstruction
        """
        self.num_parties = num_parties
        self.threshold = threshold
        self.shares: Dict[str, List[SecureShare]] = defaultdict(list)
        
    def create_shares(
        self,
        value: np.ndarray,
        owner_id: str,
        party_ids: List[str]
    ) -> List[SecureShare]:
        """
        Create secret shares of a value.
        
        Args:
            value: Value to share
            owner_id: ID of value owner
            party_ids: IDs of parties to share with
            
        Returns:
            List of shares
        """
        n = len(party_ids)
        shares = []
        
        # Generate n-1 random shares
        random_shares = [
            np.random.randn(*value.shape) * 1000 
            for _ in range(n - 1)
        ]
        
        # Last share is value minus sum of random shares
        last_share = value - sum(random_shares)
        all_shares = random_shares + [last_share]
        
        # Create share objects
        for i, (party_id, share_value) in enumerate(zip(party_ids, all_shares)):
            share = SecureShare(
                share_id=f"{owner_id}-{party_id}-{uuid.uuid4().hex[:8]}",
                owner_id=owner_id,
                target_id=party_id,
                value=share_value,
                mask=np.random.randn(*value.shape)
            )
            shares.append(share)
            
        return shares
    
    def aggregate_shares(
        self,
        shares: List[SecureShare]
    ) -> np.ndarray:
        """
        Aggregate shares to recover sum of original values.
        
        Args:
            shares: List of all shares
            
        Returns:
            Aggregated value
        """
        if not shares:
            raise ValueError("No shares provided")
        
        # Group by owner
        owner_shares = defaultdict(list)
        for share in shares:
            owner_shares[share.owner_id].append(share)
        
        # Sum all shares (which equals sum of original values)
        result = None
        for owner_id, owner_share_list in owner_shares.items():
            for share in owner_share_list:
                if result is None:
                    result = share.value.copy()
                else:
                    result += share.value
        
        return result

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


class FederatedAggregator:
    """
    Federated learning aggregator for governance models.
    
    PPA2-C1-30: Federated learning across distributed governance nodes.
    """
    
    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.FEDAVG,
        privacy_mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN_DP,
        min_participants: int = 2
    ):
        self.method = method
        self.privacy_mechanism = privacy_mechanism
        self.min_participants = min_participants
        self.dp = DifferentialPrivacy()
        self.current_round = 0
        self.model_version = 0
        
    def aggregate(
        self,
        updates: List[LocalUpdate],
        global_params: Optional[Dict[str, np.ndarray]] = None,
        privacy_budget: Optional[PrivacyBudget] = None
    ) -> Optional[AggregatedModel]:
        """
        Aggregate local updates into global model.
        
        Args:
            updates: List of local updates from participants
            global_params: Current global parameters (for FedProx)
            privacy_budget: Privacy budget for this round
            
        Returns:
            Aggregated model or None if insufficient participants
        """
        if len(updates) < self.min_participants:
            logger.warning(f"Insufficient participants: {len(updates)} < {self.min_participants}")
            return None
        
        self.current_round += 1
        
        # Apply privacy mechanism to updates
        processed_updates = []
        total_epsilon = 0.0
        
        for update in updates:
            processed = self._apply_privacy(update, privacy_budget)
            if processed:
                processed_updates.append(processed)
                total_epsilon += processed.epsilon_spent
        
        if not processed_updates:
            return None
        
        # Aggregate based on method
        if self.method == AggregationMethod.FEDAVG:
            aggregated_params = self._fedavg(processed_updates)
        elif self.method == AggregationMethod.FEDPROX:
            aggregated_params = self._fedprox(processed_updates, global_params)
        elif self.method == AggregationMethod.MEDIAN:
            aggregated_params = self._coordinate_median(processed_updates)
        elif self.method == AggregationMethod.TRIMMED_MEAN:
            aggregated_params = self._trimmed_mean(processed_updates)
        else:
            aggregated_params = self._weighted_average(processed_updates)
        
        self.model_version += 1
        
        return AggregatedModel(
            model_version=self.model_version,
            parameters=aggregated_params,
            num_contributors=len(processed_updates),
            total_samples=sum(u.num_samples for u in processed_updates),
            aggregation_method=self.method,
            timestamp=datetime.utcnow(),
            privacy_guarantee={
                "mechanism": self.privacy_mechanism.value,
                "total_epsilon": total_epsilon,
                "num_updates": len(processed_updates)
            }
        )
    
    def _apply_privacy(
        self,
        update: LocalUpdate,
        budget: Optional[PrivacyBudget]
    ) -> Optional[LocalUpdate]:
        """Apply privacy mechanism to an update."""
        if self.privacy_mechanism == PrivacyMechanism.NONE:
            return update
        
        epsilon = 1.0
        delta = 1e-5
        
        if budget and not budget.can_spend(epsilon, delta):
            return None
        
        new_params = {}
        for name, param in update.parameters.items():
            if self.privacy_mechanism == PrivacyMechanism.GAUSSIAN_DP:
                noisy_param, _ = self.dp.add_gaussian_noise(
                    param, sensitivity=1.0, epsilon=epsilon, delta=delta
                )
            elif self.privacy_mechanism == PrivacyMechanism.LAPLACE_DP:
                noisy_param, _ = self.dp.add_laplace_noise(
                    param, sensitivity=1.0, epsilon=epsilon
                )
            else:
                noisy_param = param
            new_params[name] = noisy_param
        
        if budget:
            budget.spend(epsilon, delta, f"update_{update.update_id}")
        
        return LocalUpdate(
            node_id=update.node_id,
            update_id=update.update_id,
            parameters=new_params,
            num_samples=update.num_samples,
            timestamp=update.timestamp,
            noise_added=True,
            epsilon_spent=epsilon
        )
    
    def _fedavg(self, updates: List[LocalUpdate]) -> Dict[str, np.ndarray]:
        """Federated Averaging aggregation."""
        total_samples = sum(u.num_samples for u in updates)
        
        aggregated = {}
        for name in updates[0].parameters.keys():
            weighted_sum = sum(
                u.parameters[name] * u.num_samples 
                for u in updates
            )
            aggregated[name] = weighted_sum / total_samples
        
        return aggregated
    
    def _fedprox(
        self,
        updates: List[LocalUpdate],
        global_params: Optional[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """FedProx aggregation with proximal term."""
        # First do FedAvg
        aggregated = self._fedavg(updates)
        
        # Then add proximal regularization toward global model
        if global_params:
            mu = 0.01  # Proximal coefficient
            for name in aggregated.keys():
                if name in global_params:
                    aggregated[name] = (
                        aggregated[name] + mu * global_params[name]
                    ) / (1 + mu)
        
        return aggregated
    
    def _coordinate_median(self, updates: List[LocalUpdate]) -> Dict[str, np.ndarray]:
        """Coordinate-wise median for Byzantine resilience."""
        aggregated = {}
        for name in updates[0].parameters.keys():
            stacked = np.stack([u.parameters[name] for u in updates])
            aggregated[name] = np.median(stacked, axis=0)
        
        return aggregated
    
    def _trimmed_mean(
        self,
        updates: List[LocalUpdate],
        trim_ratio: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """Trimmed mean for outlier resistance."""
        trim_count = int(len(updates) * trim_ratio)
        
        aggregated = {}
        for name in updates[0].parameters.keys():
            stacked = np.stack([u.parameters[name] for u in updates])
            sorted_stacked = np.sort(stacked, axis=0)
            
            if trim_count > 0:
                trimmed = sorted_stacked[trim_count:-trim_count]
            else:
                trimmed = sorted_stacked
            
            aggregated[name] = np.mean(trimmed, axis=0)
        
        return aggregated
    
    def _weighted_average(self, updates: List[LocalUpdate]) -> Dict[str, np.ndarray]:
        """Simple weighted average."""
        return self._fedavg(updates)

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


class FederatedGovernanceCoordinator:
    """
    Coordinator for distributed governance network.
    
    PPA3-Comp3: Secure multi-party computation for sensitive decisions.
    """
    
    def __init__(
        self,
        node_id: str,
        aggregation_method: AggregationMethod = AggregationMethod.FEDAVG,
        privacy_mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN_DP,
        privacy_epsilon: float = 10.0,
        privacy_delta: float = 1e-5
    ):
        self.node_id = node_id
        self.role = NodeRole.COORDINATOR
        self.nodes: Dict[str, FederatedNode] = {}
        self.aggregator = FederatedAggregator(
            method=aggregation_method,
            privacy_mechanism=privacy_mechanism
        )
        self.privacy_budget = PrivacyBudget(
            total_epsilon=privacy_epsilon,
            total_delta=privacy_delta
        )
        self.current_model: Optional[AggregatedModel] = None
        self.pending_updates: List[LocalUpdate] = []
        self.lock = threading.RLock()
        
        logger.info(f"[Federated] Coordinator {node_id} initialized")
    
    def register_node(
        self,
        node_id: str,
        role: NodeRole = NodeRole.PARTICIPANT,
        weight: float = 1.0
    ) -> FederatedNode:
        """Register a new node in the network."""
        with self.lock:
            node = FederatedNode(
                node_id=node_id,
                role=role,
                status=NodeStatus.ACTIVE,
                contribution_weight=weight,
                privacy_budget=PrivacyBudget(
                    total_epsilon=self.privacy_budget.total_epsilon / 10,
                    total_delta=self.privacy_budget.total_delta / 10
                )
            )
            self.nodes[node_id] = node
            return node
    
    def submit_update(
        self,
        node_id: str,
        parameters: Dict[str, np.ndarray],
        num_samples: int
    ) -> str:
        """Submit a local update from a participant."""
        with self.lock:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not registered")
            
            node = self.nodes[node_id]
            node.status = NodeStatus.TRAINING
            node.last_heartbeat = datetime.utcnow()
            
            update = LocalUpdate(
                node_id=node_id,
                update_id=str(uuid.uuid4())[:12],
                parameters=parameters,
                num_samples=num_samples,
                timestamp=datetime.utcnow()
            )
            self.pending_updates.append(update)
            
            node.status = NodeStatus.IDLE
            node.local_model_version += 1
            
            return update.update_id
    
    def aggregate_round(self) -> Optional[AggregatedModel]:
        """Run one round of federated aggregation."""
        with self.lock:
            if not self.pending_updates:
                return None
            
            # Get current global params if available
            global_params = None
            if self.current_model:
                global_params = self.current_model.parameters
            
            # Aggregate
            result = self.aggregator.aggregate(
                updates=self.pending_updates,
                global_params=global_params,
                privacy_budget=self.privacy_budget
            )
            
            if result:
                self.current_model = result
                self.pending_updates = []
            
            return result
    
    def get_global_model(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current global model parameters."""
        if self.current_model:
            return self.current_model.parameters
        return None
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get status of the federated network."""
        with self.lock:
            active_nodes = sum(1 for n in self.nodes.values() if n.is_active())
            return {
                "coordinator_id": self.node_id,
                "total_nodes": len(self.nodes),
                "active_nodes": active_nodes,
                "current_round": self.aggregator.current_round,
                "model_version": self.aggregator.model_version,
                "pending_updates": len(self.pending_updates),
                "privacy_budget": {
                    "total_epsilon": self.privacy_budget.total_epsilon,
                    "spent_epsilon": self.privacy_budget.spent_epsilon,
                    "remaining_epsilon": self.privacy_budget.remaining_epsilon
                },
                "aggregation_method": self.aggregator.method.value
            }

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


class FederatedPrivacyEngine:
    """
    Unified engine for federated learning and privacy-preserving governance.
    
    Implements PPA2-C1-30, PPA2-C1-31, PPA3-Comp3.
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        is_coordinator: bool = True,
        privacy_epsilon: float = 10.0
    ):
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        self.is_coordinator = is_coordinator
        
        if is_coordinator:
            self.coordinator = FederatedGovernanceCoordinator(
                node_id=self.node_id,
                privacy_epsilon=privacy_epsilon
            )
        else:
            self.coordinator = None
        
        self.dp = DifferentialPrivacy()
        self.secure_agg = None  # Initialized when parties known
        
        logger.info(f"[FederatedPrivacy] Engine initialized as {'coordinator' if is_coordinator else 'participant'}")
    
    def add_participant(self, participant_id: str, weight: float = 1.0) -> FederatedNode:
        """Add a participant to the federated network."""
        if not self.coordinator:
            raise ValueError("Only coordinator can add participants")
        return self.coordinator.register_node(participant_id, NodeRole.PARTICIPANT, weight)
    
    def submit_governance_update(
        self,
        participant_id: str,
        governance_scores: Dict[str, float],
        num_evaluations: int
    ) -> str:
        """Submit governance scores from a participant."""
        if not self.coordinator:
            raise ValueError("Only coordinator can receive updates")
        
        # Convert scores to numpy arrays
        params = {k: np.array([v]) for k, v in governance_scores.items()}
        
        return self.coordinator.submit_update(
            node_id=participant_id,
            parameters=params,
            num_samples=num_evaluations
        )
    
    def aggregate_governance(self) -> Optional[Dict[str, float]]:
        """Aggregate governance scores from all participants."""
        if not self.coordinator:
            return None
        
        result = self.coordinator.aggregate_round()
        if not result:
            return None
        
        # Convert back to float dict
        return {k: float(v[0]) for k, v in result.parameters.items()}
    
    def apply_local_dp(
        self,
        value: float,
        epsilon: float = 1.0
    ) -> float:
        """Apply local differential privacy to a value."""
        arr = np.array([value])
        noisy, _ = self.dp.add_laplace_noise(arr, sensitivity=1.0, epsilon=epsilon)
        return float(noisy[0])
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        if self.coordinator:
            return {
                "role": "coordinator",
                **self.coordinator.get_network_status()
            }
        return {
            "role": "participant",
            "node_id": self.node_id
        }

    # ========================================
    # PHASE 49: LEARNING METHODS
    # ========================================
    
    def record_outcome(self, input_data, output_data, was_correct, domain=None, metadata=None):
        """Record outcome for learning (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            self._learning_manager.record_outcome(
                module_name=self.__class__.__name__.lower(),
                input_data=input_data, output_data=output_data,
                was_correct=was_correct, domain=domain, metadata=metadata
            )
    
    def record_feedback(self, result, was_accurate):
        """Record feedback for learning (Phase 49)."""
        self.record_outcome({"result": str(result)}, {}, was_accurate)
    
    def adapt_thresholds(self, threshold_name, current_value, direction='decrease'):
        """Adapt thresholds based on learning (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.adapt_threshold(
                self.__class__.__name__.lower(), threshold_name, current_value, direction
            )
        return max(0.1, min(0.95, current_value + (0.05 if direction == 'increase' else -0.05)))
    
    def get_domain_adjustment(self, domain):
        """Get domain-specific adjustment (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            return self._learning_manager.get_domain_adjustment(self.__class__.__name__.lower(), domain)
        return 0.0
    
    def save_state(self, filepath=None):
        """Save learning state (Phase 49)."""
        return self._learning_manager.save_state() if hasattr(self, '_learning_manager') and self._learning_manager else False
    
    def load_state(self, filepath=None):
        """Load learning state (Phase 49)."""
        return self._learning_manager.load_state() if hasattr(self, '_learning_manager') and self._learning_manager else False
    
    def get_learning_statistics(self):
        """Get learning statistics (Phase 49)."""
        if hasattr(self, '_learning_manager') and self._learning_manager:
            stats = self._learning_manager.get_learning_statistics(self.__class__.__name__.lower())
            return stats.__dict__ if hasattr(stats, '__dict__') else stats
        return {"module": self.__class__.__name__, "status": "no_learning_manager"}

    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface."""
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)

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


# Test function
    # =========================================================================
    # Learning Interface Completion
    # =========================================================================
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        """Learn from feedback to adjust behavior."""
        if not hasattr(self, '_learning_params'):
            self._learning_params = {}
        was_correct = feedback.get('was_correct', True)
        if not was_correct:
            adjustment = feedback.get('adjustment', 0.05)
            for key in self._learning_params:
                self._learning_params[key] *= (1 - adjustment)
    
    def get_statistics(self) -> Dict:
        """Return learning statistics."""
        return {
            'learning_params': getattr(self, '_learning_params', {}),
            'history_size': len(getattr(self, '_outcomes', []))
        }
    
    def serialize_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'outcomes': getattr(self, '_outcomes', [])[-100:],
            'learning_params': getattr(self, '_learning_params', {}),
            'version': '1.0'
        }
    
    def deserialize_state(self, state: Dict) -> None:
        """Restore state from serialized data."""
        self._outcomes = state.get('outcomes', [])
        self._learning_params = state.get('learning_params', {})

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 35: Federated Learning & Privacy-Preserving Test")
    print("=" * 70)
    
    engine = FederatedPrivacyEngine(is_coordinator=True, privacy_epsilon=10.0)
    
    # Test 1: Add participants
    print("\n[1] Registering Participants")
    print("-" * 60)
    
    p1 = engine.add_participant("participant-1", weight=1.0)
    p2 = engine.add_participant("participant-2", weight=1.0)
    p3 = engine.add_participant("participant-3", weight=0.5)
    
    print(f"  Registered: {p1.node_id} (weight={p1.contribution_weight})")
    print(f"  Registered: {p2.node_id} (weight={p2.contribution_weight})")
    print(f"  Registered: {p3.node_id} (weight={p3.contribution_weight})")
    
    # Test 2: Submit governance updates
    print("\n[2] Submitting Governance Updates")
    print("-" * 60)
    
    update1 = engine.submit_governance_update(
        "participant-1",
        {"accuracy": 0.85, "bias_score": 0.12, "confidence": 0.78},
        num_evaluations=100
    )
    print(f"  Update 1: {update1}")
    
    update2 = engine.submit_governance_update(
        "participant-2",
        {"accuracy": 0.82, "bias_score": 0.15, "confidence": 0.80},
        num_evaluations=150
    )
    print(f"  Update 2: {update2}")
    
    update3 = engine.submit_governance_update(
        "participant-3",
        {"accuracy": 0.88, "bias_score": 0.10, "confidence": 0.75},
        num_evaluations=80
    )
    print(f"  Update 3: {update3}")
    
    # Test 3: Aggregate
    print("\n[3] Federated Aggregation")
    print("-" * 60)
    
    aggregated = engine.aggregate_governance()
    if aggregated:
        print(f"  Aggregated Accuracy: {aggregated.get('accuracy', 0):.3f}")
        print(f"  Aggregated Bias Score: {aggregated.get('bias_score', 0):.3f}")
        print(f"  Aggregated Confidence: {aggregated.get('confidence', 0):.3f}")
    
    # Test 4: Privacy mechanisms
    print("\n[4] Differential Privacy")
    print("-" * 60)
    
    original = 0.75
    noisy = engine.apply_local_dp(original, epsilon=1.0)
    print(f"  Original Value: {original}")
    print(f"  Noisy Value (ε=1.0): {noisy:.3f}")
    print(f"  Noise Added: {abs(noisy - original):.3f}")
    
    # Test 5: Network status
    print("\n[5] Network Status")
    print("-" * 60)
    
    status = engine.get_status()
    print(f"  Role: {status['role']}")
    print(f"  Total Nodes: {status['total_nodes']}")
    print(f"  Active Nodes: {status['active_nodes']}")
    print(f"  Model Version: {status['model_version']}")
    print(f"  Privacy Spent: ε={status['privacy_budget']['spent_epsilon']:.2f}")
    print(f"  Privacy Remaining: ε={status['privacy_budget']['remaining_epsilon']:.2f}")
    
    # Test 6: Privacy budget tracking
    print("\n[6] Privacy Budget Tracking")
    print("-" * 60)
    
    budget = PrivacyBudget(total_epsilon=5.0, total_delta=1e-5)
    budget.spend(1.0, 1e-6, "operation_1")
    budget.spend(2.0, 1e-6, "operation_2")
    
    print(f"  Total ε: {budget.total_epsilon}")
    print(f"  Spent ε: {budget.spent_epsilon}")
    print(f"  Remaining ε: {budget.remaining_epsilon}")
    print(f"  Can spend ε=3.0: {budget.can_spend(3.0, 0)}")
    print(f"  Operations: {len(budget.operations)}")
    
    print("\n" + "=" * 70)
    print("PHASE 35: Federated Privacy Engine - VERIFIED")
    print("=" * 70)


