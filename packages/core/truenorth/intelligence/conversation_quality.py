"""Score conversation quality and predict abandonment risk."""
from __future__ import annotations
import statistics
from dataclasses import dataclass
from truenorth.core.graph_state import GraphState


@dataclass
class QualityResult:
    score: float
    abandonment_risk: float
    signals: list


class ConversationQualityScorer:
    def score(self, state: GraphState) -> QualityResult:
        if len(state.conversation) < 2:
            return QualityResult(1.0, 0.1, [])
        signals = []
        score = 1.0
        user_turns = [t for t in state.conversation if t.role == "user"]
        if not user_turns:
            return QualityResult(1.0, 0.1, [])
        lengths = [len(t.content.split()) for t in user_turns]
        if len(lengths) >= 3:
            recent_avg = statistics.mean(lengths[-3:])
            early_avg = statistics.mean(lengths[:3])
            if early_avg > 0 and recent_avg / early_avg < 0.5:
                score -= 0.2; signals.append("responses_getting_shorter")
        if sum(1 for l in lengths[-3:] if l <= 3) >= 2:
            score -= 0.15; signals.append("many_very_short_responses")
        if state.consecutive_confusion_turns >= 2:
            score -= 0.2 * state.consecutive_confusion_turns
            signals.append(f"confused_{state.consecutive_confusion_turns}_turns")
        penalties = {"frustrated": 0.25, "distressed": 0.40, "anxious": 0.10, "rushed": 0.10}
        p = penalties.get(state.emotion_state, 0)
        if p:
            score -= p; signals.append(f"emotion_{state.emotion_state}")
        if len(user_turns) > 5 and len(state.profile) == 0:
            score -= 0.3; signals.append("no_fields_after_5_turns")
        score = max(0.0, min(1.0, score))
        return QualityResult(score, max(0.0, min(1.0, 1.0 - score)), signals)
