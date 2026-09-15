"""
RecoveryFlow Active Learning Framework

Instead of just using recovered/failed outcomes passively,
strategically select which cases to learn from.

Core question: "Which new payment failure should we learn from to
most improve our recovery models?"
"""

import numpy as np
from typing import List, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum


class UncertaintyType(Enum):
    """Types of uncertainty in ML predictions"""
    ALEATORIC = "data"      # Inherent randomness (can't reduce)
    EPISTEMIC = "model"     # Lack of knowledge (can reduce with data)


@dataclass
class RecoveryCase:
    """A payment failure case"""
    case_id: str
    customer_segment: str
    failure_type: str
    recovery_strategy: str
    
    # Features
    customer_tenure: int
    subscription_value: float
    backup_methods: int
    payment_success_rate: float
    time_of_day: int
    
    # Model predictions (before observing outcome)
    predicted_recovery_prob: float
    model_confidence: float
    strategy_uncertainty: float
    
    # Recovery cost for cost-aware learning
    recovery_strategy_cost: float = 25.0
    
    # Actual outcome (observed later)
    was_recovered: bool = None
    time_to_recovery: int = None
    customer_churned: bool = None


@dataclass
class LearningValue:
    """Value of learning from this case"""
    case_id: str
    strategy: str
    expected_information_gain: float
    model_improvement_value: float
    business_value: float
    combined_score: float


# ============================================================================
# UNCERTAINTY SAMPLING
# ============================================================================

class UncertaintySampling:
    """
    Select cases where model is most uncertain.
    
    If model predicts 50% recovery (uncertain), outcome provides most information.
    If model predicts 95% (confident), outcome confirms what we already know.
    """
    
    @staticmethod
    def least_confident(model_probabilities: np.ndarray) -> np.ndarray:
        """Select cases where model is least confident in top prediction"""
        max_prob = np.max(model_probabilities, axis=1)
        uncertainty = 1 - max_prob
        return uncertainty
    
    @staticmethod
    def margin_sampling(model_predictions: np.ndarray) -> np.ndarray:
        """
        Select cases where top 2 strategies are closest in probability.
        Small margin = most uncertain = highest value to observe
        """
        sorted_probs = np.sort(model_predictions, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        uncertainty = 1 - margin
        return uncertainty
    
    @staticmethod
    def entropy_sampling(model_probabilities: np.ndarray) -> np.ndarray:
        """
        Select cases with highest entropy (most strategies equally likely).
        High entropy = multiple strategies viable = high info gain
        """
        epsilon = 1e-10
        entropy = -np.sum(
            model_probabilities * np.log(model_probabilities + epsilon),
            axis=1
        )
        max_entropy = np.log(model_probabilities.shape[1])
        normalized_entropy = entropy / max_entropy
        return normalized_entropy


# ============================================================================
# QUERY-BY-COMMITTEE
# ============================================================================

class QueryByCommittee:
    """
    Maintain ensemble of models.
    Select cases where ensemble members disagree most.
    """
    
    def __init__(self, models: List[Callable]):
        self.models = models
    
    def vote_entropy(self, cases: List[RecoveryCase]) -> np.ndarray:
        """For each case, calculate entropy of model votes"""
        entropies = []
        
        for case in cases:
            predictions = [model(case) for model in self.models]
            predictions = np.array(predictions)
            votes = np.digitize(predictions, bins=[0.3, 0.7])
            unique, counts = np.unique(votes, return_counts=True)
            vote_probs = counts / len(predictions)
            entropy = -np.sum(vote_probs * np.log(vote_probs + 1e-10))
            entropies.append(entropy)
        
        return np.array(entropies)
    
    def prediction_variance(self, cases: List[RecoveryCase]) -> np.ndarray:
        """Calculate variance of model predictions for each case"""
        variances = []
        for case in cases:
            predictions = [model(case) for model in self.models]
            variances.append(np.var(predictions))
        return np.array(variances)


# ============================================================================
# COST-AWARE ACTIVE LEARNING
# ============================================================================

class CostAwareActiveLearning:
    """
    Balance information gain vs cost to recover.
    Focus on cheap cases where we're uncertain.
    """
    
    @staticmethod
    def information_gain_per_cost(
        uncertainty_score: np.ndarray,
        recovery_cost: np.ndarray,
    ) -> np.ndarray:
        """Prioritize high-information cases that are cheap to recover"""
        ratio = uncertainty_score / (recovery_cost + 1)
        return ratio
    
    @staticmethod
    def business_value_weighted_learning(
        uncertainty_score: np.ndarray,
        customer_ltv: np.ndarray,
        recovery_cost: np.ndarray,
    ) -> np.ndarray:
        """Prioritize learning from high-value customers where we're uncertain"""
        ltv_weight = np.log1p(customer_ltv)
        cost_weight = 1 / (recovery_cost + 1)
        score = uncertainty_score * ltv_weight * cost_weight
        return score


# ============================================================================
# ACTIVE LEARNING ORCHESTRATOR
# ============================================================================

class ActiveLearningOrchestrator:
    """
    Coordinate multiple active learning strategies.
    Select which cases to observe for maximum learning.
    """
    
    def __init__(
        self,
        uncertainty_weight: float = 0.4,
        disagreement_weight: float = 0.3,
        change_weight: float = 0.3,
    ):
        self.uncertainty_weight = uncertainty_weight
        self.disagreement_weight = disagreement_weight
        self.change_weight = change_weight
    
    def select_cases_to_learn_from(
        self,
        candidates: List[RecoveryCase],
        n_select: int = 100,
    ) -> List[LearningValue]:
        """
        Rank all candidate cases by learning value.
        Return top n_select cases to observe outcome for.
        """
        
        n_cases = len(candidates)
        
        # Strategy 1: Uncertainty sampling
        predictions = np.array([[c.predicted_recovery_prob] for c in candidates])
        uncertainty = UncertaintySampling.least_confident(predictions)
        
        # Strategy 2: Expected model change
        predicted_probs = np.array([c.predicted_recovery_prob for c in candidates])
        epsilon = 1e-10
        best_case_loss = -np.log(1 - predicted_probs + epsilon)
        current_loss = -np.log(predicted_probs + epsilon)
        change = best_case_loss - current_loss
        
        # Strategy 3: Cost-aware weighting
        costs = np.array([c.recovery_strategy_cost for c in candidates])
        cost_adjustment = CostAwareActiveLearning.information_gain_per_cost(uncertainty, costs)
        
        # Normalize each score to [0, 1]
        def normalize(arr):
            max_val = np.max(arr + 1e-10)
            return arr / max_val
        
        combined_scores = (
            self.uncertainty_weight * normalize(uncertainty) +
            self.change_weight * normalize(change) +
            0.3 * normalize(cost_adjustment)
        )
        
        # Create learning values
        learning_values = []
        for i, case in enumerate(candidates):
            lv = LearningValue(
                case_id=case.case_id,
                strategy=case.recovery_strategy,
                expected_information_gain=float(uncertainty[i]),
                model_improvement_value=float(change[i]),
                business_value=float(cost_adjustment[i]),
                combined_score=float(combined_scores[i]),
            )
            learning_values.append(lv)
        
        learning_values.sort(key=lambda x: x.combined_score, reverse=True)
        return learning_values[:n_select]


# ============================================================================
# LEARNING IMPACT MEASUREMENT
# ============================================================================

class LearningImpactMeasurement:
    """Measure how much each observed case improved our models."""
    
    @staticmethod
    def model_improvement_metric(
        predictions_before: np.ndarray,
        predictions_after: np.ndarray,
    ) -> float:
        """Measure reduction in log-loss after observing cases"""
        loss_before = -np.mean(np.log(predictions_before + 1e-10))
        loss_after = -np.mean(np.log(predictions_after + 1e-10))
        improvement = (loss_before - loss_after) / loss_before
        return improvement
    
    @staticmethod
    def ensemble_disagreement_reduction(
        ensemble_before: np.ndarray,
        ensemble_after: np.ndarray,
    ) -> float:
        """Did observing cases reduce ensemble disagreement?"""
        variance_before = np.mean(np.var(ensemble_before, axis=0))
        variance_after = np.mean(np.var(ensemble_after, axis=0))
        if variance_before == 0:
            return 0.0
        reduction = (variance_before - variance_after) / variance_before
        return reduction
    
    @staticmethod
    def strategy_ranking_stability(
        ranking_before: List[str],
        ranking_after: List[str],
    ) -> float:
        """Did the ranking of strategies change after learning?"""
        tau = 0
        for i in range(len(ranking_before)):
            for j in range(i + 1, len(ranking_before)):
                before_agree = (ranking_before[i] < ranking_before[j])
                after_agree = (ranking_after[i] < ranking_after[j])
                if before_agree != after_agree:
                    tau += 1
        
        max_tau = len(ranking_before) * (len(ranking_before) - 1) / 2
        stability = 1 - (tau / max_tau) if max_tau > 0 else 1.0
        return stability
