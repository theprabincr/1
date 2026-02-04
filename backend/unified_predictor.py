"""
Unified BetPredictor - Combines V5, V6, and XGBoost ML into Single Prediction System
Leverages line movement analysis (V5), rule-based ensemble (V6), and real XGBoost ML
Now with proper machine learning using trained XGBoost models
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from betpredictor_v5 import generate_v5_prediction
from betpredictor_v6 import generate_v6_prediction

# Import XGBoost ML (optional - graceful fallback if not available)
try:
    from ml_xgboost import get_predictor, FeatureEngineering
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class UnifiedBetPredictor:
    """
    Unified predictor combining V5 (line movement), V6 (rule-based), and XGBoost ML.
    
    Weighting (when XGBoost is trained):
    - XGBoost ML: 40% weight - Real trained machine learning
    - V6 (Rule-based): 35% weight - Comprehensive analysis
    - V5 (Line Movement): 25% weight - Sharp money validation
    
    Weighting (without XGBoost):
    - V6 (Rule-based): 70% weight - Primary decision maker
    - V5 (Line Movement): 30% weight - Validation signal
    
    Philosophy:
    - Use XGBoost when trained for real ML predictions
    - Use V6's comprehensive analysis as backup/validation
    - Use V5's sharp money signals for confirmation
    - Only recommend when multiple signals align
    """
    
    def __init__(self):
        # Weighting when XGBoost is available
        self.xgb_weight = 0.40  # XGBoost ML
        self.v6_weight_with_xgb = 0.35  # V6 when XGBoost available
        self.v5_weight_with_xgb = 0.25  # V5 when XGBoost available
        
        # Weighting without XGBoost (fallback)
        self.v6_weight = 0.70  # V6 is primary (ML ensemble)
        self.v5_weight = 0.30  # V5 is secondary (line movement validation)
        
        # Minimum thresholds
        self.min_unified_confidence = 0.60  # 60% combined confidence
        self.min_edge = 0.04  # 4% edge minimum
        
        # Agreement bonus
        self.agreement_bonus = 0.10  # 10% boost when all models agree
        self.two_agree_bonus = 0.05  # 5% boost when 2 models agree
    
    async def generate_unified_prediction(
        self,
        event: Dict,
        sport_key: str,
        squad_data: Dict,
        matchup_data: Dict,
        line_movement_history: List[Dict],
        opening_odds: Dict,
        current_odds: Dict,
        player_stats_comparison: Dict = None  # Player stats analysis
    ) -> Dict:
        """
        Generate unified prediction combining V5, V6, and XGBoost ML analysis.
        
        Args:
            player_stats_comparison: Dict with home/away lineup impact and advantages
        
        Returns single pick with combined reasoning from all algorithms.
        """
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        
        logger.info(f"🔄 UNIFIED PREDICTOR: {home_team} vs {away_team}")
        
        # Log player stats if available
        if player_stats_comparison:
            impact_adv = player_stats_comparison.get("impact_advantage", "even")
            impact_diff = player_stats_comparison.get("impact_diff", 0)
            logger.info(f"  📊 Player Stats: {impact_adv.upper()} advantage (impact diff: {impact_diff})")
            for adv in player_stats_comparison.get("key_advantages", [])[:2]:
                logger.info(f"     - {adv}")
        
        # Initialize XGBoost result
        xgb_result = None
        xgb_available = False
        
        # Run XGBoost ML if available
        if XGBOOST_AVAILABLE:
            try:
                predictor = get_predictor(sport_key)
                if predictor.is_loaded:
                    logger.info("  🤖 Running XGBoost ML (All Markets)...")
                    
                    # Build team data for XGBoost
                    home_team_data = {
                        "elo_rating": matchup_data.get("home_team", {}).get("elo_rating", 1500),
                        "form": matchup_data.get("home_team", {}).get("form", {}),
                        "stats": matchup_data.get("home_team", {}).get("stats", {})
                    }
                    away_team_data = {
                        "elo_rating": matchup_data.get("away_team", {}).get("elo_rating", 1500),
                        "form": matchup_data.get("away_team", {}).get("form", {}),
                        "stats": matchup_data.get("away_team", {}).get("stats", {})
                    }
                    odds_data = event.get("odds", {})
                    
                    # Get predictions for ALL markets - NOW WITH TEAM NAMES
                    xgb_prediction = predictor.predict(
                        home_team_data, 
                        away_team_data, 
                        odds_data,
                        home_team_name=home_team,
                        away_team_name=away_team
                    )
                    
                    # ===== FAVORED OUTCOMES (NEW) =====
                    # Moneyline - show favored team
                    ml_favored_team = xgb_prediction.get("ml_favored_team", home_team)
                    ml_favored_prob = xgb_prediction.get("ml_favored_prob", 0.5)
                    ml_underdog_team = xgb_prediction.get("ml_underdog_team", away_team)
                    ml_underdog_prob = xgb_prediction.get("ml_underdog_prob", 0.5)
                    ml_conf = xgb_prediction.get("ml_confidence", 50)
                    
                    # Spread - show favored team to cover
                    spread_favored_team = xgb_prediction.get("spread_favored_team", home_team)
                    spread_favored_prob = xgb_prediction.get("spread_favored_prob", 0.5)
                    spread_favored_line = xgb_prediction.get("spread_favored_line", 0)
                    spread_conf = xgb_prediction.get("spread_confidence", 50)
                    
                    # Totals - show favored direction
                    totals_favored = xgb_prediction.get("totals_favored", "OVER")
                    totals_favored_prob = xgb_prediction.get("totals_favored_prob", 0.5)
                    totals_line = xgb_prediction.get("totals_line", 220)
                    totals_conf = xgb_prediction.get("totals_confidence", 50)
                    predicted_total = xgb_prediction.get("predicted_total", 220)
                    
                    # Raw probabilities (for backward compat)
                    home_win_prob = xgb_prediction.get("home_win_prob", 0.5)
                    home_cover_prob = xgb_prediction.get("home_cover_prob", 0.5)
                    over_prob = xgb_prediction.get("over_prob", 0.5)
                    
                    # Determine best market and pick
                    best_market = xgb_prediction.get("best_market", "moneyline")
                    best_conf = xgb_prediction.get("best_confidence", 50)
                    best_pick = xgb_prediction.get("best_pick", ml_favored_team)
                    
                    # Get spread and total line from odds
                    spread_line = odds_data.get("spread", 0)
                    total_line = odds_data.get("total", 220)
                    
                    # Determine pick based on best market - USE FAVORED OUTCOMES
                    xgb_pick = None
                    xgb_pick_type = best_market
                    xgb_pick_display = None
                    xgb_conf = best_conf
                    
                    if best_market == "moneyline":
                        # Pick favored team if probability is strong enough
                        if ml_favored_prob >= 0.55:
                            xgb_pick = ml_favored_team
                            xgb_pick_display = f"{ml_favored_team} ML"
                            xgb_conf = ml_conf
                    
                    elif best_market == "spread":
                        # Pick favored team to cover if probability is strong enough
                        if spread_favored_prob >= 0.55:
                            xgb_pick = spread_favored_team
                            spread_display = f"{spread_favored_line:+.1f}" if spread_favored_line else ""
                            xgb_pick_display = f"{spread_favored_team} {spread_display}"
                            xgb_conf = spread_conf
                    
                    elif best_market == "totals":
                        # Pick favored direction if probability is strong enough
                        if totals_favored_prob >= 0.55:
                            xgb_pick = totals_favored
                            xgb_pick_display = f"{totals_favored} {totals_line}"
                            xgb_conf = totals_conf
                    
                    xgb_result = {
                        "has_pick": xgb_pick is not None,
                        "pick": xgb_pick,
                        "pick_type": xgb_pick_type,
                        "pick_display": xgb_pick_display,
                        "confidence": xgb_conf,
                        
                        # ===== FAVORED OUTCOMES (NEW - DISPLAY THESE) =====
                        # Moneyline
                        "ml_favored_team": ml_favored_team,
                        "ml_favored_prob": ml_favored_prob,
                        "ml_underdog_team": ml_underdog_team,
                        "ml_underdog_prob": ml_underdog_prob,
                        
                        # Spread
                        "spread_favored_team": spread_favored_team,
                        "spread_favored_prob": spread_favored_prob,
                        "spread_favored_line": spread_favored_line,
                        
                        # Totals
                        "totals_favored": totals_favored,
                        "totals_favored_prob": totals_favored_prob,
                        "totals_line": totals_line,
                        "predicted_total": predicted_total,
                        
                        # Raw probabilities (backward compat)
                        "ml_probability": home_win_prob,
                        "spread_probability": home_cover_prob,
                        "over_probability": over_prob,
                        
                        # Accuracies
                        "ml_accuracy": predictor.ml_accuracy,
                        "spread_accuracy": predictor.spread_accuracy,
                        "totals_accuracy": predictor.totals_accuracy,
                        
                        # Best market info
                        "best_market": best_market,
                        "best_pick": best_pick,
                        "spread_line": spread_line,
                        "total_line": total_line,
                        
                        # Backward compat
                        "probability": home_win_prob,
                        "model_accuracy": predictor.ml_accuracy
                    }
                    xgb_available = True
                    
                    logger.info(f"  🤖 XGBoost Best Market: {best_market.upper()}")
                    logger.info(f"     ML: {ml_favored_team} {ml_favored_prob*100:.1f}% (vs {ml_underdog_team} {ml_underdog_prob*100:.1f}%)")
                    logger.info(f"     Spread: {spread_favored_team} {spread_favored_line:+.1f} @ {spread_favored_prob*100:.1f}%")
                    logger.info(f"     Totals: {totals_favored} {totals_line} @ {totals_favored_prob*100:.1f}%")
                    logger.info(f"     Pick: {xgb_pick_display} ({xgb_conf:.0f}% conf)")
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Run V5 and V6 predictors
        try:
            # V6 Analysis (Rule-based Ensemble)
            logger.info("  📊 Running V6 (Rule-based Ensemble)...")
            v6_result = await generate_v6_prediction(
                event, sport_key, squad_data, matchup_data,
                line_movement_history, opening_odds, current_odds,
                player_stats=player_stats_comparison
            )
            
            # V5 Analysis (Line Movement)
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
        
        # Analyze results and combine (now with XGBoost)
        unified_prediction = self._combine_predictions(
            v5_result, v6_result, home_team, away_team, event,
            xgb_result=xgb_result, xgb_available=xgb_available
        )
        
        return unified_prediction
    
    def _combine_predictions(
        self,
        v5_result: Dict,
        v6_result: Dict,
        home_team: str,
        away_team: str,
        event: Dict,
        xgb_result: Dict = None,
        xgb_available: bool = False
    ) -> Dict:
        """
        Combine V5, V6, and XGBoost predictions into single unified recommendation.
        
        When XGBoost is available:
        - XGBoost: 40% weight
        - V6: 35% weight  
        - V5: 25% weight
        
        Without XGBoost:
        - V6: 70% weight
        - V5: 30% weight
        """
        v5_has_pick = v5_result.get("has_pick", False)
        v6_has_pick = v6_result.get("has_pick", False)
        xgb_has_pick = xgb_result.get("has_pick", False) if xgb_result else False
        
        # Count how many models have picks
        picks_count = sum([v5_has_pick, v6_has_pick, xgb_has_pick])
        
        # If XGBoost is available and has a pick, use enhanced combination
        if xgb_available and xgb_has_pick:
            return self._combine_with_xgboost(
                v5_result, v6_result, xgb_result, home_team, away_team, event
            )
        
        # Fallback to original V5/V6 combination
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
    
    def _combine_with_xgboost(
        self,
        v5_result: Dict,
        v6_result: Dict,
        xgb_result: Dict,
        home_team: str,
        away_team: str,
        event: Dict
    ) -> Dict:
        """
        Combine predictions when XGBoost ML is available.
        Now supports all three markets: Moneyline, Spread, Totals.
        """
        xgb_pick = xgb_result.get("pick")
        xgb_pick_type = xgb_result.get("pick_type", "moneyline")
        xgb_pick_display = xgb_result.get("pick_display", xgb_pick)
        xgb_conf = xgb_result.get("confidence", 50) / 100
        xgb_best_market = xgb_result.get("best_market", "moneyline")
        
        # All market probabilities
        ml_prob = xgb_result.get("ml_probability", 0.5)
        spread_prob = xgb_result.get("spread_probability", 0.5)
        over_prob = xgb_result.get("over_probability", 0.5)
        predicted_total = xgb_result.get("predicted_total", 220)
        
        # Market accuracies
        ml_accuracy = xgb_result.get("ml_accuracy", 0)
        spread_accuracy = xgb_result.get("spread_accuracy", 0)
        totals_accuracy = xgb_result.get("totals_accuracy", 0)
        
        # Lines
        spread_line = xgb_result.get("spread_line", event.get("odds", {}).get("spread", 0))
        total_line = xgb_result.get("total_line", event.get("odds", {}).get("total", 220))
        
        v5_pick = v5_result.get("pick", "").replace(" ML", "").replace(" Spread", "").strip() if v5_result.get("has_pick") else None
        v6_pick = v6_result.get("pick", "").replace(" ML", "").replace(" Spread", "").strip() if v6_result.get("has_pick") else None
        
        v5_conf = v5_result.get("confidence", 0) / 100 if v5_result.get("has_pick") else 0
        v6_conf = v6_result.get("confidence", 0) / 100 if v6_result.get("has_pick") else 0
        
        # For totals, check if V5/V6 agree with over/under direction
        if xgb_pick_type == "totals":
            # Totals picks don't need team agreement
            agrees_with_xgb = 1  # XGBoost always agrees with itself
        else:
            # Count agreements with XGBoost for team-based picks
            agrees_with_xgb = 0
            if v5_pick and v5_pick == xgb_pick:
                agrees_with_xgb += 1
            if v6_pick and v6_pick == xgb_pick:
                agrees_with_xgb += 1
        
        # Calculate weighted confidence
        weighted_conf = (
            xgb_conf * self.xgb_weight +
            v6_conf * self.v6_weight_with_xgb +
            v5_conf * self.v5_weight_with_xgb
        )
        
        # Add agreement bonus
        if agrees_with_xgb == 2:
            weighted_conf += self.agreement_bonus  # All 3 agree
            consensus_level = "strong_consensus"
            logger.info(f"  ✅ STRONG CONSENSUS: All 3 models agree on {xgb_pick_display}")
        elif agrees_with_xgb == 1:
            weighted_conf += self.two_agree_bonus  # 2 agree
            consensus_level = "moderate_consensus"
            logger.info(f"  📊 MODERATE CONSENSUS: 2/3 models agree on {xgb_pick_display}")
        else:
            consensus_level = "xgb_only"
            logger.info(f"  ⚠️ XGB ONLY: Only XGBoost picks {xgb_pick_display}")
        
        # Calculate edge based on market type
        odds_data = event.get("odds", {})
        
        if xgb_pick_type == "moneyline":
            if xgb_pick == home_team:
                pick_odds = odds_data.get("home_ml_decimal", 1.91)
                edge = ml_prob - (1 / pick_odds if pick_odds > 1 else 0.5)
            else:
                pick_odds = odds_data.get("away_ml_decimal", 1.91)
                edge = (1 - ml_prob) - (1 / pick_odds if pick_odds > 1 else 0.5)
        elif xgb_pick_type == "spread":
            pick_odds = 1.91  # Standard -110
            if xgb_pick == home_team:
                edge = spread_prob - 0.524  # Break-even at -110
            else:
                edge = (1 - spread_prob) - 0.524
        elif xgb_pick_type == "totals":
            pick_odds = 1.91
            if xgb_pick == "OVER":
                edge = over_prob - 0.524
            else:
                edge = (1 - over_prob) - 0.524
        else:
            pick_odds = 1.91
            edge = 0
        
        # Determine if we should make a pick
        should_pick = (
            weighted_conf >= self.min_unified_confidence or
            (xgb_conf >= 0.65 and agrees_with_xgb >= 1)
        )
        
        if should_pick and edge >= self.min_edge:
            return {
                "has_pick": True,
                "pick": xgb_pick,
                "pick_type": xgb_pick_type,
                "pick_display": xgb_pick_display,
                "confidence": round(weighted_conf * 100, 1),
                "edge": round(edge * 100, 1),
                "odds": pick_odds,
                "algorithm": "unified_xgboost",
                "consensus_level": consensus_level,
                "xgb_agrees": True,
                "v5_agrees": v5_pick == xgb_pick if v5_pick else False,
                "v6_agrees": v6_pick == xgb_pick if v6_pick else False,
                # All market probabilities
                "xgb_probability": round(ml_prob, 3),
                "xgb_spread_probability": round(spread_prob, 3),
                "xgb_over_probability": round(over_prob, 3),
                "xgb_predicted_total": predicted_total,
                "xgb_best_market": xgb_best_market,
                # Accuracies
                "xgb_ml_accuracy": ml_accuracy,
                "xgb_spread_accuracy": spread_accuracy,
                "xgb_totals_accuracy": totals_accuracy,
                "xgb_model_accuracy": ml_accuracy,  # Backward compat
                # Lines
                "spread_line": spread_line,
                "total_line": total_line,
                "reasoning": self._build_xgb_reasoning(
                    xgb_result, v5_result, v6_result,
                    weighted_conf * 100, edge * 100,
                    consensus_level, agrees_with_xgb
                ),
                "v5_analysis": v5_result,
                "v6_analysis": v6_result,
                "xgb_analysis": xgb_result
            }
        else:
            reason = []
            if weighted_conf < self.min_unified_confidence:
                reason.append(f"Low confidence: {weighted_conf*100:.1f}%")
            if edge < self.min_edge:
                reason.append(f"Insufficient edge: {edge*100:.1f}%")
            
            return {
                "has_pick": False,
                "reasoning": f"XGBoost picked {xgb_pick} but: {'; '.join(reason)}",
                "algorithm": "unified_xgboost",
                "xgb_pick": xgb_pick,
                "xgb_probability": xgb_prob,
                "weighted_confidence": round(weighted_conf * 100, 1),
                "edge": round(edge * 100, 1),
                "v5_analysis": v5_result,
                "v6_analysis": v6_result,
                "xgb_analysis": xgb_result
            }
    
    def _build_xgb_reasoning(
        self,
        xgb_result: Dict,
        v5_result: Dict,
        v6_result: Dict,
        confidence: float,
        edge: float,
        consensus_level: str,
        agrees_count: int
    ) -> str:
        """Build reasoning text when XGBoost is the primary model."""
        parts = []
        
        parts.append("=" * 60)
        parts.append(f"🤖 XGBOOST ML PREDICTION")
        parts.append("=" * 60)
        parts.append("")
        
        xgb_prob = xgb_result.get("probability", 0.5)
        xgb_accuracy = xgb_result.get("model_accuracy", 0)
        
        parts.append(f"📊 XGBoost Probability: {xgb_prob*100:.1f}% home win")
        parts.append(f"📈 Model Training Accuracy: {xgb_accuracy:.1%}")
        parts.append(f"💰 Combined Confidence: {confidence:.1f}%")
        parts.append(f"🎯 Edge: {edge:+.1f}%")
        parts.append("")
        
        # Model agreement
        parts.append("MODEL AGREEMENT")
        parts.append("")
        if consensus_level == "strong_consensus":
            parts.append("✅ ALL 3 MODELS AGREE (+10% confidence boost)")
        elif consensus_level == "moderate_consensus":
            parts.append("📊 2 OF 3 MODELS AGREE (+5% confidence boost)")
        else:
            parts.append("⚠️ Only XGBoost has a pick")
        
        parts.append("")
        parts.append(f"  • XGBoost ML: {xgb_result.get('pick')} ({xgb_prob*100:.1f}%)")
        
        if v6_result.get("has_pick"):
            v6_pick = v6_result.get("pick", "N/A")
            v6_conf = v6_result.get("confidence", 0)
            parts.append(f"  • V6 Ensemble: {v6_pick} ({v6_conf:.0f}%)")
        else:
            parts.append(f"  • V6 Ensemble: No pick")
        
        if v5_result.get("has_pick"):
            v5_pick = v5_result.get("pick", "N/A")
            v5_conf = v5_result.get("confidence", 0)
            parts.append(f"  • V5 Line Movement: {v5_pick} ({v5_conf:.0f}%)")
        else:
            parts.append(f"  • V5 Line Movement: No pick")
        
        parts.append("")
        parts.append("=" * 60)
        
        # Include V6 detailed reasoning if available
        v6_reasoning = v6_result.get("reasoning", "")
        if v6_reasoning:
            parts.append("")
            parts.append("DETAILED ANALYSIS (V6)")
            parts.append("")
            parts.append(v6_reasoning)
        
        return "\n".join(parts)
    
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
        Shows FULL detailed analysis including all factors considered.
        """
        reasoning_parts = []
        
        # ===== HEADER =====
        reasoning_parts.append(f"{'='*60}")
        reasoning_parts.append(f"🎯 {consensus_type}")
        reasoning_parts.append(f"{'='*60}")
        reasoning_parts.append("")
        reasoning_parts.append(f"📊 Combined Confidence: {combined_confidence:.1f}% (V6: 70%, V5: 30%)")
        reasoning_parts.append(f"💰 Combined Edge: {combined_edge:+.1f}%")
        
        # Agreement status
        if agreement:
            reasoning_parts.append(f"✅ BOTH ALGORITHMS AGREE (+10% consensus bonus)")
        elif conflict:
            reasoning_parts.append(f"⚠️ {conflict}")
        elif v5_declined:
            reasoning_parts.append(f"⚠️ V5 declined (no strong line movement), relying on V6")
        
        reasoning_parts.append("")
        
        # ===== FULL V6 ANALYSIS =====
        reasoning_parts.append(f"{'='*60}")
        reasoning_parts.append(f"📊 V6 ML ENSEMBLE - FULL ANALYSIS")
        reasoning_parts.append(f"{'='*60}")
        reasoning_parts.append("")
        
        v6_reasoning = v6_result.get("reasoning", "")
        if v6_reasoning:
            # Include FULL V6 reasoning - all sections
            reasoning_parts.append(v6_reasoning)
        
        reasoning_parts.append("")
        
        # ===== V5 LINE MOVEMENT ANALYSIS =====
        reasoning_parts.append(f"{'='*60}")
        reasoning_parts.append(f"📈 V5 LINE MOVEMENT ANALYSIS")
        reasoning_parts.append(f"{'='*60}")
        reasoning_parts.append("")
        
        if v5_result.get("has_pick"):
            v5_pick = v5_result.get("pick", "")
            v5_conf = v5_result.get("confidence", 0)
            reasoning_parts.append(f"V5 Pick: {v5_pick} @ {v5_conf:.0f}% confidence")
            reasoning_parts.append("")
            
            line_analysis = v5_result.get("line_movement_analysis", {}) or v5_result.get("market_analysis", {})
            
            if line_analysis.get("sharp_money_detected"):
                reasoning_parts.append(f"✓ SHARP MONEY DETECTED - Professional bettors active")
            if line_analysis.get("reverse_line_movement"):
                reasoning_parts.append(f"✓ REVERSE LINE MOVEMENT - Line moving against public")
            
            key_insights = line_analysis.get("key_insights", []) or v5_result.get("insights", [])
            if key_insights:
                reasoning_parts.append("")
                reasoning_parts.append("Key Insights:")
                for insight in key_insights[:5]:
                    reasoning_parts.append(f"  • {insight}")
            
            # Market breakdown
            markets = line_analysis if isinstance(line_analysis, dict) else {}
            market_analysis = v5_result.get("market_analysis", {})
            
            if market_analysis:
                reasoning_parts.append("")
                reasoning_parts.append("Market Breakdown:")
                for market_name, market_data in market_analysis.items():
                    if isinstance(market_data, dict):
                        opening = market_data.get("opening_value", "N/A")
                        current = market_data.get("current_value", "N/A")
                        movement = market_data.get("total_movement", 0)
                        direction = market_data.get("movement_direction", "neutral")
                        
                        reasoning_parts.append(f"  • {market_name.upper()}: {opening} → {current} ({direction}, {movement:+.1f})")
        else:
            v5_reason = v5_result.get("reasoning", "No strong line movement signal detected")
            reasoning_parts.append(f"V5 did not recommend a pick:")
            reasoning_parts.append(f"  {v5_reason}")
            
            # Still show market status
            market_analysis = v5_result.get("market_analysis", {})
            if market_analysis:
                reasoning_parts.append("")
                reasoning_parts.append("Current Market Status:")
                for market_name, market_data in market_analysis.items():
                    if isinstance(market_data, dict):
                        opening = market_data.get("opening_value", {})
                        current = market_data.get("current_value", {})
                        movement = market_data.get("total_movement", 0)
                        
                        if isinstance(opening, dict):
                            opening_str = f"Home: {opening.get('home', 'N/A')}, Away: {opening.get('away', 'N/A')}"
                            current_str = f"Home: {current.get('home', 'N/A')}, Away: {current.get('away', 'N/A')}"
                        else:
                            opening_str = str(opening)
                            current_str = str(current)
                        
                        reasoning_parts.append(f"  • {market_name.upper()}: {opening_str} (no significant movement)")
        
        reasoning_parts.append("")
        reasoning_parts.append(f"{'='*60}")
        
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
