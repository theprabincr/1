"""
Unified BetPredictor - Combines V5 and V6 into Single Prediction System
Leverages both line movement analysis (V5) and ML ensemble (V6) with V6 weighted heavier
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from betpredictor_v5 import generate_v5_prediction
from betpredictor_v6 import generate_v6_prediction

logger = logging.getLogger(__name__)


class UnifiedBetPredictor:
    """
    Unified predictor combining V5 (line movement) and V6 (ML ensemble).
    
    Weighting:
    - V6 (ML Ensemble): 70% weight - Primary decision maker
    - V5 (Line Movement): 30% weight - Validation signal
    
    Philosophy:
    - Use V6's comprehensive analysis as foundation
    - Use V5's sharp money signals for confirmation
    - Only recommend when both algorithms align or V6 has strong conviction
    """
    
    def __init__(self):
        # Weighting for final decision
        self.v6_weight = 0.70  # V6 is primary (ML ensemble)
        self.v5_weight = 0.30  # V5 is secondary (line movement validation)
        
        # Minimum thresholds
        self.min_unified_confidence = 0.60  # 60% combined confidence
        self.min_edge = 0.04  # 4% edge minimum
        
        # Agreement bonus
        self.agreement_bonus = 0.10  # 10% boost when both agree
    
    async def generate_unified_prediction(
        self,
        event: Dict,
        sport_key: str,
        squad_data: Dict,
        matchup_data: Dict,
        line_movement_history: List[Dict],
        opening_odds: Dict,
        current_odds: Dict
    ) -> Dict:
        """
        Generate unified prediction combining V5 and V6 analysis.
        
        Returns single pick with combined reasoning from both algorithms.
        """
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        
        logger.info(f"🔄 UNIFIED PREDICTOR: {home_team} vs {away_team}")
        
        # Run both predictors in parallel
        try:
            # V6 Analysis (ML Ensemble - Primary)
            logger.info("  📊 Running V6 (ML Ensemble)...")
            v6_result = await generate_v6_prediction(
                event, sport_key, squad_data, matchup_data,
                line_movement_history, opening_odds, current_odds
            )
            
            # V5 Analysis (Line Movement - Secondary)
            logger.info("  📈 Running V5 (Line Movement)...")
            v5_result = await generate_v5_prediction(
                event, sport_key, squad_data, matchup_data,
                line_movement_history, opening_odds, current_odds
            )
            
        except Exception as e:
            logger.error(f"Error running predictors: {e}")
            return {
                "has_pick": False,
                "reasoning": f"Analysis failed: {str(e)}",
                "algorithm": "unified"
            }
        
        # Analyze results and combine
        unified_prediction = self._combine_predictions(
            v5_result, v6_result, home_team, away_team, event
        )
        
        return unified_prediction
    
    def _combine_predictions(
        self,
        v5_result: Dict,
        v6_result: Dict,
        home_team: str,
        away_team: str,
        event: Dict
    ) -> Dict:
        """
        Combine V5 and V6 predictions into single unified recommendation.
        """
        v5_has_pick = v5_result.get("has_pick", False)
        v6_has_pick = v6_result.get("has_pick", False)
        
        # SCENARIO 1: Both have picks
        if v5_has_pick and v6_has_pick:
            return self._handle_both_pick(v5_result, v6_result, home_team, away_team, event)
        
        # SCENARIO 2: Only V6 has pick (strong enough on its own)
        elif v6_has_pick and not v5_has_pick:
            return self._handle_v6_only(v6_result, v5_result, event)
        
        # SCENARIO 3: Only V5 has pick (needs V6 confirmation)
        elif v5_has_pick and not v6_has_pick:
            return self._handle_v5_only(v5_result, v6_result, event)
        
        # SCENARIO 4: Neither has pick
        else:
            return self._handle_no_picks(v5_result, v6_result)
    
    def _handle_both_pick(
        self,
        v5_result: Dict,
        v6_result: Dict,
        home_team: str,
        away_team: str,
        event: Dict
    ) -> Dict:
        """
        Both V5 and V6 generated picks - check if they agree.
        """
        v5_pick = v5_result.get("pick", "")
        v6_pick = v6_result.get("pick", "")
        
        # Check if both pick the same team (ignore pick_type for now)
        v5_team = v5_pick.replace(" ML", "").replace(" Spread", "").replace(" Over", "").replace(" Under", "").strip()
        v6_team = v6_pick.replace(" ML", "").replace(" Spread", "").replace(" Over", "").replace(" Under", "").strip()
        
        both_agree = (v5_team == v6_team) or (v5_pick == v6_pick)
        
        if both_agree:
            # STRONG CONSENSUS - Both agree
            logger.info(f"  ✅ STRONG CONSENSUS: Both V5 and V6 pick {v6_pick}")
            
            # Calculate combined confidence (weighted + agreement bonus)
            v5_conf = v5_result.get("confidence", 0) / 100
            v6_conf = v6_result.get("confidence", 0) / 100
            
            combined_confidence = (
                v6_conf * self.v6_weight +
                v5_conf * self.v5_weight +
                self.agreement_bonus  # Bonus for agreement
            )
            
            # Calculate combined edge
            v5_edge = v5_result.get("edge", 0) / 100
            v6_edge = v6_result.get("edge", 0) / 100
            combined_edge = (v6_edge * self.v6_weight + v5_edge * self.v5_weight)
            
            return {
                "has_pick": True,
                "pick": v6_result.get("pick"),  # Use V6's pick (might have better market)
                "pick_type": v6_result.get("pick_type", "moneyline"),
                "pick_display": v6_result.get("pick_display", v6_pick),
                "confidence": round(combined_confidence * 100, 1),
                "edge": round(combined_edge * 100, 1),
                "odds": v6_result.get("odds", 1.91),
                "algorithm": "unified",
                "consensus_level": "strong",
                "v5_agrees": True,
                "v6_agrees": True,
                "reasoning": self._build_unified_reasoning(
                    "STRONG CONSENSUS",
                    v6_result,
                    v5_result,
                    combined_confidence * 100,
                    combined_edge * 100,
                    agreement=True
                ),
                "v5_analysis": v5_result,
                "v6_analysis": v6_result
            }
        
        else:
            # CONFLICT - Different picks
            logger.info(f"  ⚠️ CONFLICT: V5 picks {v5_pick}, V6 picks {v6_pick}")
            
            # Use V6 (higher weight) but reduce confidence due to conflict
            v6_conf = v6_result.get("confidence", 0) / 100
            conflict_penalty = 0.15  # 15% penalty for disagreement
            
            adjusted_confidence = v6_conf * self.v6_weight - conflict_penalty
            
            # Only recommend if V6 confidence is very high
            if adjusted_confidence >= self.min_unified_confidence:
                logger.info(f"  📊 Using V6 pick despite conflict (high confidence)")
                
                return {
                    "has_pick": True,
                    "pick": v6_result.get("pick"),
                    "pick_type": v6_result.get("pick_type", "moneyline"),
                    "pick_display": v6_result.get("pick_display", v6_pick),
                    "confidence": round(adjusted_confidence * 100, 1),
                    "edge": round(v6_result.get("edge", 0) * 0.7, 1),  # Reduce edge due to conflict
                    "odds": v6_result.get("odds", 1.91),
                    "algorithm": "unified",
                    "consensus_level": "weak",
                    "v5_agrees": False,
                    "v6_agrees": True,
                    "conflict_warning": f"V5 picked {v5_pick} (conflicting signal)",
                    "reasoning": self._build_unified_reasoning(
                        "V6 DOMINANT",
                        v6_result,
                        v5_result,
                        adjusted_confidence * 100,
                        v6_result.get("edge", 0) * 0.7,
                        agreement=False,
                        conflict=f"V5 disagreed: {v5_pick}"
                    ),
                    "v5_analysis": v5_result,
                    "v6_analysis": v6_result
                }
            else:
                # Confidence too low with conflict - decline pick
                return {
                    "has_pick": False,
                    "reasoning": f"V5 and V6 conflict: V5 picks {v5_pick}, V6 picks {v6_pick}. Adjusted confidence {adjusted_confidence*100:.1f}% < 60% threshold.",
                    "algorithm": "unified",
                    "conflict": True,
                    "v5_pick": v5_pick,
                    "v6_pick": v6_pick,
                    "v5_analysis": v5_result,
                    "v6_analysis": v6_result
                }
    
    def _handle_v6_only(self, v6_result: Dict, v5_result: Dict, event: Dict) -> Dict:
        """
        Only V6 has a pick. Check if confidence is high enough.
        """
        v6_conf = v6_result.get("confidence", 0) / 100
        v6_edge = v6_result.get("edge", 0) / 100
        
        # V6 is strong enough on its own (70% weight)
        weighted_confidence = v6_conf * self.v6_weight
        
        logger.info(f"  📊 V6 ONLY: Confidence {weighted_confidence*100:.1f}%")
        
        if weighted_confidence >= self.min_unified_confidence and v6_edge >= self.min_edge:
            # V6 strong enough standalone
            return {
                "has_pick": True,
                "pick": v6_result.get("pick"),
                "pick_type": v6_result.get("pick_type", "moneyline"),
                "pick_display": v6_result.get("pick_display"),
                "confidence": round(weighted_confidence * 100, 1),
                "edge": round(v6_edge * 100, 1),
                "odds": v6_result.get("odds", 1.91),
                "algorithm": "unified",
                "consensus_level": "v6_only",
                "v5_agrees": False,
                "v6_agrees": True,
                "reasoning": self._build_unified_reasoning(
                    "V6 STRONG SIGNAL",
                    v6_result,
                    v5_result,
                    weighted_confidence * 100,
                    v6_edge * 100,
                    agreement=None,
                    v5_declined=True
                ),
                "v5_analysis": v5_result,
                "v6_analysis": v6_result
            }
        else:
            return {
                "has_pick": False,
                "reasoning": f"Only V6 picked, but weighted confidence {weighted_confidence*100:.1f}% < 60% threshold. V5 did not pick (no strong line movement signal).",
                "algorithm": "unified",
                "v6_picked": True,
                "v5_picked": False,
                "v5_analysis": v5_result,
                "v6_analysis": v6_result
            }
    
    def _handle_v5_only(self, v5_result: Dict, v6_result: Dict, event: Dict) -> Dict:
        """
        Only V5 has a pick. V5 alone is NOT enough (needs V6 confirmation).
        """
        logger.info(f"  ⚠️ V5 ONLY (insufficient): V5 picked but V6 declined")
        
        return {
            "has_pick": False,
            "reasoning": f"V5 picked {v5_result.get('pick')}, but V6 declined (ML models did not reach consensus). V6 required for confirmation.",
            "algorithm": "unified",
            "v5_picked": True,
            "v6_picked": False,
            "v5_pick": v5_result.get("pick"),
            "v5_confidence": v5_result.get("confidence"),
            "v6_reason": v6_result.get("reasoning", ""),
            "v5_analysis": v5_result,
            "v6_analysis": v6_result
        }
    
    def _handle_no_picks(self, v5_result: Dict, v6_result: Dict) -> Dict:
        """
        Neither V5 nor V6 generated picks.
        """
        logger.info(f"  ❌ NO PICKS: Both V5 and V6 declined")
        
        return {
            "has_pick": False,
            "reasoning": "Neither V5 (line movement) nor V6 (ML ensemble) found sufficient edge. No recommendation.",
            "algorithm": "unified",
            "v5_reason": v5_result.get("reasoning", ""),
            "v6_reason": v6_result.get("reasoning", ""),
            "v5_analysis": v5_result,
            "v6_analysis": v6_result
        }
    
    def _build_unified_reasoning(
        self,
        consensus_type: str,
        v6_result: Dict,
        v5_result: Dict,
        combined_confidence: float,
        combined_edge: float,
        agreement: Optional[bool] = None,
        conflict: Optional[str] = None,
        v5_declined: bool = False
    ) -> str:
        """
        Build comprehensive reasoning combining both algorithms.
        """
        reasoning_parts = []
        
        # Header
        reasoning_parts.append(f"🎯 {consensus_type}")
        reasoning_parts.append(f"📊 Combined Confidence: {combined_confidence:.1f}% (V6: 70%, V5: 30%)")
        reasoning_parts.append(f"💰 Combined Edge: {combined_edge:+.1f}%")
        
        # Agreement status
        if agreement:
            reasoning_parts.append(f"✅ BOTH ALGORITHMS AGREE (+10% consensus bonus)")
        elif conflict:
            reasoning_parts.append(f"⚠️ {conflict}")
        elif v5_declined:
            reasoning_parts.append(f"⚠️ V5 declined (no strong line movement), relying on V6")
        
        # V6 Key Factors
        reasoning_parts.append(f"\n📊 V6 (ML ENSEMBLE) ANALYSIS:")
        v6_reasoning = v6_result.get("reasoning", "")
        if v6_reasoning:
            for line in v6_reasoning.split("\n")[:6]:  # First 6 lines
                if line.strip():
                    reasoning_parts.append(f"  {line.strip()}")
        
        # V5 Key Factors
        if v5_result.get("has_pick"):
            reasoning_parts.append(f"\n📈 V5 (LINE MOVEMENT) SIGNALS:")
            line_analysis = v5_result.get("line_movement_analysis", {})
            if line_analysis.get("sharp_money_detected"):
                reasoning_parts.append(f"  ✓ Sharp money detected")
            if line_analysis.get("reverse_line_movement"):
                reasoning_parts.append(f"  ✓ Reverse line movement (RLM)")
            
            key_insights = line_analysis.get("key_insights", [])
            for insight in key_insights[:3]:
                reasoning_parts.append(f"  ✓ {insight}")
        
        return "\n".join(reasoning_parts)


# Main entry point
async def generate_unified_prediction(
    event: Dict,
    sport_key: str,
    squad_data: Dict,
    matchup_data: Dict,
    line_movement_history: List[Dict],
    opening_odds: Dict,
    current_odds: Dict
) -> Dict:
    """Generate unified prediction combining V5 and V6."""
    predictor = UnifiedBetPredictor()
    return await predictor.generate_unified_prediction(
        event, sport_key, squad_data, matchup_data,
        line_movement_history, opening_odds, current_odds
    )
