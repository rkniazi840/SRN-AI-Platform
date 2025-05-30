
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import time
from collections import defaultdict
import uuid

# ===========================================
# PART 1: Feedback Aggregation + Fit Score Calibration
# ===========================================

@dataclass
class Signal:
    source: str
    confidence: float
    note: str

@dataclass
class CandidateEvaluation:
    candidate_id: str
    job_id: str
    initial_fit_score: float
    signals: List[Signal]
    outcome: str

@dataclass
class AdjustedScore:
    adjusted_fit_score: float
    flags: List[str]
    top_influencer: str
    retraining_signal: str
    confidence_weighted_avg: float
    signal_breakdown: Dict[str, float]

class ScoreAggregator:
    """Reusable module for aggregating multi-source feedback signals"""
    
    def __init__(self):
        # Weight mapping for different signal sources
        self.source_weights = {
            "recruiter": 0.3,
            "interviewer": 0.4,
            "automated_interview_score": 0.2,
            "peer_review": 0.1
        }
        
        # Keywords for flag detection
        self.flag_keywords = {
            "culture mismatch": ["culture", "alignment", "fit", "team"],
            "low engagement": ["disengaged", "engagement", "participation"],
            "response latency": ["latency", "slow", "response time", "delay"],
            "technical weakness": ["tech", "coding", "technical", "skills"]
        }
    
    def extract_flags(self, signals: List[Signal]) -> List[str]:
        """Extract behavioral/performance flags from signal notes"""
        flags = set()
        
        for signal in signals:
            note_lower = signal.note.lower()
            for flag_name, keywords in self.flag_keywords.items():
                if any(keyword in note_lower for keyword in keywords):
                    flags.add(flag_name)
        
        return list(flags)
    
    def calculate_signal_impact(self, signal: Signal, outcome: str) -> float:
        """Calculate how much a signal should impact the score based on outcome"""
        base_weight = self.source_weights.get(signal.source, 0.15)
        confidence_multiplier = signal.confidence
        
        # Adjust based on outcome - negative outcomes increase impact of negative signals
        outcome_multiplier = 1.2 if outcome in ["Offer declined", "Interview failed"] else 1.0
        
        return base_weight * confidence_multiplier * outcome_multiplier
    
    def aggregate_score(self, evaluation: CandidateEvaluation) -> AdjustedScore:
        """Main aggregation logic with weighted scoring"""
        
        # Extract flags first
        flags = self.extract_flags(evaluation.signals)
        
        # Calculate weighted adjustments
        total_weight = 0
        weighted_adjustment = 0
        signal_impacts = {}
        
        for signal in evaluation.signals:
            impact = self.calculate_signal_impact(signal, evaluation.outcome)
            
            # Determine score adjustment based on signal content
            note_lower = signal.note.lower()
            if any(neg in note_lower for neg in ["weak", "poor", "bad", "failed", "declined"]):
                adjustment = -0.5 * impact
            elif any(pos in note_lower for pos in ["great", "excellent", "strong", "good"]):
                adjustment = 0.2 * impact
            else:
                adjustment = -0.1 * impact  # Neutral/mixed signals slightly negative
            
            weighted_adjustment += adjustment
            total_weight += impact
            signal_impacts[signal.source] = adjustment
        
        # Calculate final adjusted score
        base_adjustment = weighted_adjustment / max(total_weight, 0.1)
        adjusted_score = max(0, evaluation.initial_fit_score + base_adjustment)
        
        # Find top influencer
        top_influencer = max(evaluation.signals, 
                           key=lambda s: abs(signal_impacts.get(s.source, 0))).source
        
        # Generate retraining signal
        retraining_signal = self._generate_retraining_signal(flags, evaluation.outcome)
        
        return AdjustedScore(
            adjusted_fit_score=round(adjusted_score, 1),
            flags=flags,
            top_influencer=top_influencer,
            retraining_signal=retraining_signal,
            confidence_weighted_avg=round(sum(s.confidence for s in evaluation.signals) / len(evaluation.signals), 2),
            signal_breakdown=signal_impacts
        )
    
    def _generate_retraining_signal(self, flags: List[str], outcome: str) -> str:
        """Generate actionable retraining recommendations"""
        if "culture mismatch" in flags:
            return "Improve evaluation of communication/culture alignment"
        elif "technical weakness" in flags:
            return "Enhance technical assessment accuracy"
        elif "low engagement" in flags:
            return "Better predict candidate engagement levels"
        else:
            return f"Improve prediction accuracy for {outcome.lower()} scenarios"

# ===========================================
# PART 2: Self-Healing Retry Loop + Agent Reflection
# ===========================================

class RetryDecision(Enum):
    RETRY_NEW_JOB = "Retry with new job"
    RETRY_SAME_JOB = "Re-engage with same job"
    SUPPRESS = "Suppress candidate"
    HOLD = "Hold for manual review"

class AgentVote(Enum):
    RETRY = "Retry"
    SUPPRESS = "Suppress"
    HOLD = "Hold"

@dataclass
class CandidateHistory:
    candidate_id: str
    job_history: List[str]
    last_outcome: str
    engagement_logs: List[str]
    fit_scores: List[float]
    tags: List[str]

@dataclass
class RetryDecisionOutput:
    retry_decision: str
    reasoning: str
    agent_votes: Dict[str, str]
    final_score_adjustment: float
    confidence: float

class Agent(ABC):
    """Base class for all decision agents"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.memory = {}
    
    @abstractmethod
    def vote(self, candidate_history: CandidateHistory) -> Tuple[AgentVote, str, float]:
        """Return vote, reasoning, and confidence"""
        pass

class OutcomeValidator(Agent):
    def vote(self, candidate_history: CandidateHistory) -> Tuple[AgentVote, str, float]:
        # Analyze patterns in outcomes
        declining_scores = len(candidate_history.fit_scores) > 1 and \
                          candidate_history.fit_scores[-1] < candidate_history.fit_scores[0]
        
        if candidate_history.last_outcome == "Offer declined" and not declining_scores:
            return AgentVote.RETRY, "Good fit scores despite decline - worth retry", 0.8
        elif "failed" in candidate_history.last_outcome.lower():
            return AgentVote.SUPPRESS, "Multiple failures indicate poor fit", 0.9
        else:
            return AgentVote.RETRY, "Outcome salvageable with adjustments", 0.6

class RetryStrategist(Agent):
    def vote(self, candidate_history: CandidateHistory) -> Tuple[AgentVote, str, float]:
        # Analyze engagement patterns
        positive_engagement = any("clicked" in log.lower() or "replied" in log.lower() 
                                for log in candidate_history.engagement_logs)
        
        job_count = len(candidate_history.job_history)
        
        if job_count >= 3:
            return AgentVote.SUPPRESS, "Too many job attempts", 0.9
        elif positive_engagement and "culture mismatch" in candidate_history.tags:
            return AgentVote.RETRY, "Strong engagement, try different team culture", 0.8
        elif positive_engagement:
            return AgentVote.RETRY, "Good engagement signals", 0.7
        else:
            return AgentVote.HOLD, "Mixed signals require review", 0.5

class SuppressionMonitor(Agent):
    def vote(self, candidate_history: CandidateHistory) -> Tuple[AgentVote, str, float]:
        # Conservative approach - look for clear suppress signals
        suppress_tags = ["unresponsive", "rude", "unprofessional"]
        
        if any(tag in candidate_history.tags for tag in suppress_tags):
            return AgentVote.SUPPRESS, "Behavioral red flags detected", 0.9
        elif len(candidate_history.job_history) > 2:
            return AgentVote.SUPPRESS, "Pattern of multiple rejections", 0.7
        else:
            return AgentVote.HOLD, "No clear suppress signals", 0.4

class MemoryScorer(Agent):
    def vote(self, candidate_history: CandidateHistory) -> Tuple[AgentVote, str, float]:
        # Score based on historical patterns and learning
        avg_score = sum(candidate_history.fit_scores) / len(candidate_history.fit_scores)
        score_trend = candidate_history.fit_scores[-1] - candidate_history.fit_scores[0] if len(candidate_history.fit_scores) > 1 else 0
        
        if avg_score > 7.5 and score_trend >= -0.5:
            return AgentVote.RETRY, "Historical scores justify retry", 0.8
        elif avg_score < 6.0:
            return AgentVote.SUPPRESS, "Consistently low fit scores", 0.7
        else:
            return AgentVote.RETRY, "Moderate scores warrant another chance", 0.6

class MultiAgentRetryOrchestrator:
    """Orchestrates multi-agent decision making with reflection"""
    
    def __init__(self):
        self.agents = [
            OutcomeValidator("Validator", weight=1.2),
            RetryStrategist("Strategist", weight=1.0),
            SuppressionMonitor("Suppressor", weight=0.8),
            MemoryScorer("Memory", weight=1.1)
        ]
        self.decision_history = []
    
    def make_retry_decision(self, candidate_history: CandidateHistory) -> RetryDecisionOutput:
        """Coordinate agents to make retry decision"""
        
        # Collect votes from all agents
        agent_votes = {}
        agent_reasoning = {}
        weighted_scores = defaultdict(float)
        
        for agent in self.agents:
            vote, reasoning, confidence = agent.vote(candidate_history)
            agent_votes[agent.name] = vote.value
            agent_reasoning[agent.name] = reasoning
            
            # Weight the vote by agent weight and confidence
            weighted_scores[vote] += agent.weight * confidence
        
        # Determine final decision based on weighted voting
        final_vote = max(weighted_scores.items(), key=lambda x: x[1])[0]
        
        # Generate reasoning and decision
        if final_vote == AgentVote.RETRY:
            if "culture mismatch" in candidate_history.tags:
                decision = RetryDecision.RETRY_NEW_JOB.value
                reasoning = "Strong tech fit, weak culture fit -> test against different team"
                score_adjustment = -0.3
            else:
                decision = RetryDecision.RETRY_SAME_JOB.value
                reasoning = "Good overall signals warrant re-engagement"
                score_adjustment = -0.1
        elif final_vote == AgentVote.SUPPRESS:
            decision = RetryDecision.SUPPRESS.value
            reasoning = "Multiple negative signals indicate poor long-term fit"
            score_adjustment = -0.8
        else:
            decision = RetryDecision.HOLD.value
            reasoning = "Mixed signals require human review"
            score_adjustment = 0.0
        
        # Calculate overall confidence
        total_weight = sum(agent.weight for agent in self.agents)
        confidence = weighted_scores[final_vote] / total_weight
        
        result = RetryDecisionOutput(
            retry_decision=decision,
            reasoning=reasoning,
            agent_votes={name: vote for name, vote in agent_votes.items()},
            final_score_adjustment=score_adjustment,
            confidence=round(confidence, 2)
        )
        
        # Store for reflection
        self.decision_history.append({
            'candidate_id': candidate_history.candidate_id,
            'decision': result,
            'input_data': candidate_history
        })
        
        return result
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def simulate_embeddings(self, text: str) -> List[float]:
        """Mock embedding generation for similarity comparison"""
        # Simple hash-based mock embedding
        hash_val = hash(text)
        return [(hash_val >> i) % 100 / 50.0 - 1.0 for i in range(10)]
    
    def compare_failure_patterns(self, candidate1: CandidateHistory, candidate2: CandidateHistory) -> float:
        """Compare failure patterns using mock embeddings"""
        failure_summary1 = f"{candidate1.last_outcome} {' '.join(candidate1.tags)}"
        failure_summary2 = f"{candidate2.last_outcome} {' '.join(candidate2.tags)}"
        
        emb1 = self.simulate_embeddings(failure_summary1)
        emb2 = self.simulate_embeddings(failure_summary2)
        
        return self.cosine_similarity(emb1, emb2)