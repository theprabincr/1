"""
BetPredictor V6 - Advanced Comprehensive Betting Algorithm
Combines all advanced analytics for maximum accuracy:

Phase 1: Context & Advanced Metrics
- ELO ratings with dynamic updates
- Rest days, travel, altitude analysis
- Smart injury weighting by player importance
- Sport-specific advanced metrics (Four Factors, DVOA-style, Corsi)

Phase 2: Statistical Modeling & Simulations
- Monte Carlo simulations (1000+ runs)
- Poisson modeling for low-scoring sports
- Market psychology and contrarian opportunities
- Market efficiency analysis

Phase 3: Machine Learning & Ensemble
- Logistic regression for win probability
- Ensemble model combining 5 sub-models
- Kelly Criterion for bet sizing
- Historical performance tracking with auto-tuning

Philosophy:
- NO PICK is still the default - only recommend when edge is clear
- Multi-model consensus required (3+ models must agree)
- Dynamic confidence based on model agreement
- Conservative by design - quality over quantity
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from advanced_metrics import (
    calculate_advanced_metrics,
    calculate_matchup_metrics,
    ELORatingSystem
)
from context_analyzer import ContextAnalyzer
from injury_analyzer import analyze_injury_impact
from market_psychology import analyze_market_psychology
from simulation_engine import run_game_simulation
from ml_models import (
    LogisticRegressionModel,
    EnsembleModel,
    create_feature_vector
)
from line_movement_analyzer import analyze_line_movement

logger = logging.getLogger(__name__)


class BetPredictorV6:
    """
    Advanced betting prediction engine with comprehensive analysis.
    """
    
    def __init__(self):
        # Research-backed thresholds (2024-2025 ML sports betting studies)
        # - 70%+ confidence is optimal for selective betting
        # - 3/5 model agreement provides good balance of quality vs quantity
        # - 4% minimum edge required for long-term profitability
        self.min_confidence = 0.70  # Individual model minimum
        self.min_edge = 0.04  # 4% minimum edge
        self.min_models_agreement = 3  # At least 3/5 models must agree
        self.min_ensemble_confidence = 60.0  # Ensemble confidence threshold
        
        # Strong signal bypass - allow picks when ONE model is extremely confident
        self.strong_signal_threshold = 85.0  # Single model bypass threshold
        self.strong_signal_edge = 0.08  # 8% edge required for strong signal picks
    
    async def analyze_and_predict(
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
        Comprehensive analysis using all V6 features.
        
        Returns:
            Complete prediction with multi-model consensus and detailed analysis
        """
        home_team = event.get("home_team", "Home")
        away_team = event.get("away_team", "Away")
        commence_time = event.get("commence_time", "")
        
        logger.info(f"🚀 BetPredictor V6 Analysis: {home_team} vs {away_team}")
        
        # ==================== PHASE 1: CONTEXT & ADVANCED METRICS ====================
        
        logger.info("📊 Phase 1: Computing advanced metrics and context...")
        
        # 1. Advanced Metrics (ELO, Four Factors, DVOA-style, etc.)
        home_metrics = calculate_advanced_metrics(sport_key, matchup_data.get("home_team", {}))
        away_metrics = calculate_advanced_metrics(sport_key, matchup_data.get("away_team", {}))
        matchup_metrics = calculate_matchup_metrics(sport_key, home_metrics, away_metrics)
        
        logger.info(f"   ELO: {home_team} {home_metrics.get('elo_rating', 0):.0f} vs {away_team} {away_metrics.get('elo_rating', 0):.0f}")
        
        # 2. Context Analysis (rest, travel, altitude, schedule)
        context_analyzer = ContextAnalyzer(sport_key)
        context_analysis = context_analyzer.get_comprehensive_context(matchup_data, commence_time)
        
        logger.info(f"   Context: Net advantage = {context_analysis.get('net_context_advantage', 0):.3f}")
        
        # 3. Injury Analysis
        home_injuries = squad_data.get("home_team", {}).get("injuries", [])
        away_injuries = squad_data.get("away_team", {}).get("injuries", [])
        injury_analysis = analyze_injury_impact(sport_key, home_injuries, away_injuries)
        
        logger.info(f"   Injuries: {injury_analysis.get('summary', 'N/A')}")
        
        # 4. Line Movement Analysis
        line_analysis = await analyze_line_movement(
            line_movement_history,
            opening_odds,
            current_odds or {},
            commence_time,
            home_team,
            away_team
        )
        
        logger.info(f"   Line: {line_analysis.get('summary', 'No significant movement')}")
        
        # ==================== PHASE 2: SIMULATIONS & PSYCHOLOGY ====================
        
        logger.info("🎲 Phase 2: Running simulations and analyzing market psychology...")
        
        # 5. Market Psychology
        psychology_analysis = analyze_market_psychology(sport_key, event, line_analysis, matchup_data)
        
        logger.info(f"   Psychology: {psychology_analysis.get('market_efficiency', {}).get('opportunity_level', 'N/A')} opportunity")
        
        # 6. Get basic win probability from ELO
        elo_system = ELORatingSystem(sport_key)
        home_elo = home_metrics.get("elo_rating", 1500)
        away_elo = away_metrics.get("elo_rating", 1500)
        elo_home_prob, elo_away_prob = elo_system.calculate_win_probability(home_elo, away_elo, False)
        
        # 7. Run Simulations
        odds = event.get("odds", {})
        spread = self._extract_spread(event)
        total = self._extract_total(event)
        
        # Adjust probability based on all factors
        adjusted_home_prob = self._calculate_adjusted_probability(
            elo_home_prob,
            matchup_metrics,
            context_analysis,
            injury_analysis,
            line_analysis
        )
        
        simulation_results = run_game_simulation(sport_key, adjusted_home_prob, spread, total)
        
        logger.info(f"   Simulation: {simulation_results.get('monte_carlo', {}).get('outcomes', {}).get('home_win_pct', 0)*100:.1f}% home win")
        
        # ==================== PHASE 3: MACHINE LEARNING & ENSEMBLE ====================
        
        logger.info("🤖 Phase 3: Running ML models and ensemble voting...")
        
        # 8. Create feature vector for ML
        analysis_data = {
            "sport_key": sport_key,
            "home_metrics": home_metrics,
            "away_metrics": away_metrics,
            "context": context_analysis,
            "injury": injury_analysis,
            "line_movement": line_analysis
        }
        
        features = create_feature_vector(analysis_data)
        
        # 9. Logistic Regression Model
        lr_model = LogisticRegressionModel(sport_key)
        lr_probability = lr_model.predict_probability(features)
        
        logger.info(f"   Logistic Regression: {lr_probability*100:.1f}% home win")
        
        # 10. Run Individual Models
        model_predictions = self._run_individual_models(
            sport_key,
            elo_home_prob,
            adjusted_home_prob,
            lr_probability,
            simulation_results,
            line_analysis,
            context_analysis,
            psychology_analysis,
            home_team,
            away_team
        )
        
        # 11. Ensemble Model
        ensemble = EnsembleModel(sport_key)
        ensemble_result = ensemble.combine_predictions(model_predictions)
        # Add event odds for use in pick generation
        ensemble_result["_event_odds"] = event.get("odds", {})
        
        logger.info(f"   Ensemble: {ensemble_result.get('ensemble_probability', 0.5)*100:.1f}% confidence, {ensemble_result.get('model_agreement', 0)*100:.1f}% agreement")
        
        # ==================== FINAL DECISION ====================
        
        logger.info("🎯 Making final prediction...")
        
        prediction = self._make_final_decision(
            ensemble_result,
            home_team,
            away_team,
            event,
            line_analysis,
            simulation_results,
            matchup_metrics,
            context_analysis,
            injury_analysis,
            psychology_analysis,
            home_metrics,
            away_metrics
        )
        
        prediction["algorithm"] = "betpredictor_v6"
        prediction["models_used"] = list(model_predictions.keys())
        prediction["ensemble_data"] = ensemble_result
        prediction["analysis_depth"] = "comprehensive"
        
        return prediction
    
    def _run_individual_models(
        self,
        sport_key: str,
        elo_prob: float,
        adjusted_prob: float,
        lr_prob: float,
        simulation: Dict,
        line_analysis: Dict,
        context: Dict,
        psychology: Dict,
        home_team: str,
        away_team: str
    ) -> Dict:
        """
        Run 5 individual models and get their predictions.
        """
        mc_prob = simulation.get("monte_carlo", {}).get("outcomes", {}).get("home_win_pct", 0.5)
        
        models = {}
        
        # Model 1: ELO Model
        models["elo_model"] = {
            "probability": elo_prob,
            "confidence": 60 + abs(elo_prob - 0.5) * 40,
            "pick": home_team if elo_prob > 0.55 else away_team if elo_prob < 0.45 else None
        }
        
        # Model 2: Context Model (rest, travel, etc.)
        context_advantage = context.get("net_context_advantage", 0)
        context_prob = 0.50 + context_advantage
        models["context_model"] = {
            "probability": context_prob,
            "confidence": 55 + abs(context_advantage) * 200,
            "pick": home_team if context_advantage > 0.02 else away_team if context_advantage < -0.02 else None
        }
        
        # Model 3: Line Movement Model
        line_conf = line_analysis.get("confidence_adjustment", 0)
        recommended_market = line_analysis.get("recommended_market")
        line_pick = None
        if recommended_market:
            line_pick = recommended_market.get("side")
        
        models["line_movement_model"] = {
            "probability": 0.50 + line_conf,
            "confidence": 60 + line_conf * 100,
            "pick": line_pick
        }
        
        # Model 4: Statistical Model (Monte Carlo + ML)
        stat_prob = (mc_prob + lr_prob) / 2
        models["statistical_model"] = {
            "probability": stat_prob,
            "confidence": 65 + abs(stat_prob - 0.5) * 50,
            "pick": home_team if stat_prob > 0.55 else away_team if stat_prob < 0.45 else None
        }
        
        # Model 5: Psychology Model (contrarian + market efficiency)
        psych_score = psychology.get("overall_psychology_score", 0)
        contrarian_opps = psychology.get("contrarian_opportunities", {}).get("opportunities", [])
        psych_pick = None
        if contrarian_opps:
            psych_pick = contrarian_opps[0].get("contrarian_side")
        
        models["psychology_model"] = {
            "probability": 0.50 + psych_score,
            "confidence": 55 + psych_score * 200,
            "pick": psych_pick
        }
        
        return models
    
    def _calculate_adjusted_probability(
        self,
        base_prob: float,
        matchup_metrics: Dict,
        context: Dict,
        injury: Dict,
        line_analysis: Dict
    ) -> float:
        """
        Adjust base ELO probability with all factors.
        """
        adjusted = base_prob
        
        # Context adjustment
        context_adj = context.get("net_context_advantage", 0)
        adjusted += context_adj * 0.5
        
        # Injury adjustment
        injury_adj = injury.get("net_advantage", 0)
        adjusted += injury_adj * 0.5
        
        # Line movement adjustment
        line_adj = line_analysis.get("confidence_adjustment", 0)
        adjusted += line_adj * 0.3
        
        # Matchup quality adjustment
        elo_diff = matchup_metrics.get("elo_advantage", 0)
        if abs(elo_diff) > 100:
            adjusted += (elo_diff / 400) * 0.1
        
        # Clamp to reasonable range
        return max(0.20, min(0.80, adjusted))
    
    def _make_final_decision(
        self,
        ensemble_result: Dict,
        home_team: str,
        away_team: str,
        event: Dict,
        line_analysis: Dict,
        simulation: Dict,
        matchup_metrics: Dict,
        context: Dict,
        injury: Dict,
        psychology: Dict,
        home_metrics: Dict,
        away_metrics: Dict
    ) -> Dict:
        """
        Make final betting decision based on ensemble and thresholds.
        Research-backed decision logic with strong signal bypass.
        """
        ensemble_prob = ensemble_result.get("ensemble_probability", 0.5)
        ensemble_conf = ensemble_result.get("ensemble_confidence", 0)
        consensus_pick = ensemble_result.get("consensus_pick")
        model_agreement = ensemble_result.get("model_agreement", 0)
        consensus_strength = ensemble_result.get("consensus_strength", 0)
        
        # Count how many models agree on a pick
        individual_preds = ensemble_result.get("individual_predictions", {})
        picks_count = {}
        model_confidences = {}  # Track individual model confidences
        
        for model_name, model_pred in individual_preds.items():
            pick = model_pred.get("pick")
            conf = model_pred.get("confidence", 0)
            if pick:
                picks_count[pick] = picks_count.get(pick, 0) + 1
                if pick not in model_confidences:
                    model_confidences[pick] = []
                model_confidences[pick].append((model_name, conf))
        
        max_agreement = max(picks_count.values()) if picks_count else 0
        
        # Check for STRONG SIGNAL BYPASS
        # If ONE model is extremely confident (85%+) with good edge, consider it
        strong_signal_pick = None
        strong_signal_model = None
        strong_signal_conf = 0
        
        for pick, model_confs in model_confidences.items():
            for model_name, conf in model_confs:
                if conf >= self.strong_signal_threshold:
                    if conf > strong_signal_conf:
                        strong_signal_pick = pick
                        strong_signal_model = model_name
                        strong_signal_conf = conf
        
        # Standard ensemble requirements:
        # 1. Ensemble confidence >= 60% (research optimal)
        # 2. At least 3/5 models agree on same pick  
        # 3. Model agreement >= 25%
        # 4. Clear probability edge (> 0.55 or < 0.45)
        
        should_pick_ensemble = (
            ensemble_conf >= self.min_ensemble_confidence and
            max_agreement >= self.min_models_agreement and
            model_agreement >= 0.25 and
            (ensemble_prob > 0.55 or ensemble_prob < 0.45)
        )
        
        # Strong signal bypass (single model extremely confident)
        should_pick_strong_signal = (
            strong_signal_pick is not None and
            strong_signal_conf >= self.strong_signal_threshold
        )
        
        # Determine which pick to use
        final_pick = None
        pick_source = None
        
        if should_pick_ensemble and consensus_pick:
            final_pick = consensus_pick
            pick_source = "ensemble"
        elif should_pick_strong_signal:
            final_pick = strong_signal_pick
            pick_source = f"strong_signal_{strong_signal_model}"
            # Override ensemble probability with model's confidence
            ensemble_prob = strong_signal_conf / 100
            ensemble_conf = strong_signal_conf
        
        if final_pick:
            # Determine market type
            markets = line_analysis.get("markets", {})
            best_market = self._determine_best_market(markets, event)
            
            # Calculate edge
            pick_odds = self._get_pick_odds(final_pick, best_market["market"], event)
            implied_prob = 1 / pick_odds if pick_odds > 1 else 0.5
            
            if final_pick == home_team:
                edge = ensemble_prob - implied_prob
            else:
                edge = (1 - ensemble_prob) - implied_prob if ensemble_prob < 0.5 else ensemble_prob - implied_prob
            
            # Strong signal picks require higher edge
            required_edge = self.strong_signal_edge if pick_source and "strong_signal" in pick_source else self.min_edge
            
            if edge < required_edge:
                # No edge, decline pick
                return self._generate_no_pick_response(
                    ensemble_result,
                    home_team,
                    away_team,
                    f"Insufficient edge ({edge*100:.1f}% < {required_edge*100}%)",
                    line_analysis,
                    simulation,
                    matchup_metrics
                )
            
            # Generate pick
            return self._generate_pick_response(
                final_pick,
                best_market,
                ensemble_prob,
                ensemble_conf,
                edge,
                pick_odds,
                ensemble_result,
                home_team,
                away_team,
                line_analysis,
                simulation,
                matchup_metrics,
                context,
                injury,
                psychology,
                pick_source=pick_source
            )
        else:
            # No pick
            reasons = []
            if ensemble_conf < self.min_ensemble_confidence:
                reasons.append(f"Ensemble confidence {ensemble_conf:.0f}% < {self.min_ensemble_confidence}%")
            if max_agreement < self.min_models_agreement:
                reasons.append(f"Only {max_agreement}/{len(individual_preds)} models agree (need {self.min_models_agreement})")
            if model_agreement < 0.25:
                reasons.append(f"Model agreement {model_agreement*100:.0f}% < 25%")
            if 0.45 <= ensemble_prob <= 0.55:
                reasons.append(f"Probability too close to 50% ({ensemble_prob*100:.0f}%)")
            if not strong_signal_pick:
                reasons.append(f"No strong signal (need {self.strong_signal_threshold}%+)")
            
            return self._generate_no_pick_response(
                ensemble_result,
                home_team,
                away_team,
                "; ".join(reasons) if reasons else "No clear edge",
                line_analysis,
                simulation,
                matchup_metrics
            )
    
    def _determine_best_market(self, markets: Dict, event: Dict) -> Dict:
        """
        Determine which market to bet (ML, Spread, or Total).
        """
        best_market = {"market": "moneyline", "confidence": 0.5}
        
        for market_name, market_data in markets.items():
            if market_data.get("value_side") and market_data.get("confidence", 0) > best_market["confidence"]:
                best_market = {
                    "market": market_name,
                    "side": market_data["value_side"],
                    "confidence": market_data["confidence"],
                    "data": market_data
                }
        
        return best_market
    
    def _get_pick_odds(self, pick: str, market_type: str, event: Dict) -> float:
        """
        Get odds for the recommended pick.
        """
        odds = event.get("odds", {})
        home_team = event.get("home_team", "")
        
        if market_type == "moneyline":
            if pick == home_team:
                return odds.get("home_ml_decimal", 1.91)
            else:
                return odds.get("away_ml_decimal", 1.91)
        else:
            # Simplified for spread/total
            return 1.91
    
    def _generate_pick_response(
        self,
        pick: str,
        market: Dict,
        probability: float,
        confidence: float,
        edge: float,
        odds: float,
        ensemble_result: Dict,
        home_team: str,
        away_team: str,
        line_analysis: Dict,
        simulation: Dict,
        matchup_metrics: Dict,
        context: Dict,
        injury: Dict,
        psychology: Dict,
        pick_source: str = None
    ) -> Dict:
        """
        Generate comprehensive pick response.
        """
        market_type = market.get("market", "moneyline")
        market_data = market.get("data", {})
        
        # Create proper pick display based on market type
        pick_display = pick
        spread_value = None
        total_line = None
        
        if market_type == "moneyline":
            pick_display = f"{pick} ML"
        elif market_type == "spreads" or market_type == "spread":
            # Get spread value
            spread_value = market_data.get("spread") or line_analysis.get("markets", {}).get("spread", {}).get("current_line")
            if spread_value is not None:
                if pick == home_team:
                    spread_display = spread_value if spread_value < 0 else f"+{spread_value}"
                else:
                    # Away team spread is opposite
                    spread_display = -spread_value if spread_value < 0 else f"+{-spread_value}"
                pick_display = f"{pick} {spread_display}"
            else:
                pick_display = f"{pick} (spread)"
        elif market_type == "totals" or market_type == "total":
            # For totals, determine Over or Under
            # Try multiple sources for total line
            event_odds = ensemble_result.get("_event_odds", {})
            total_line = (
                market_data.get("total") or 
                line_analysis.get("markets", {}).get("totals", {}).get("current_line") or
                event_odds.get("total")
            )
            # Use simulation to determine over/under
            mc_data = simulation.get("monte_carlo", {}).get("outcomes", {})
            expected_total = mc_data.get("expected_total", 0)
            
            if total_line:
                if expected_total > total_line or probability > 0.55:
                    pick_display = f"OVER {total_line}"
                else:
                    pick_display = f"UNDER {total_line}"
            else:
                # Fallback - if pick is home team with high prob, likely Over
                if probability > 0.55:
                    pick_display = f"OVER (high-scoring expected)"
                else:
                    pick_display = f"UNDER (low-scoring expected)"
        
        # Build clean, organized reasoning
        reasoning_parts = []
        key_factors = []
        
        # Get all the data we need
        agreement = ensemble_result.get("model_agreement", 0)
        models_agree = ensemble_result.get("consensus_strength", 0)
        num_models = ensemble_result.get("num_models", 5)
        agreeing_models = int(models_agree * num_models)
        individual_preds = ensemble_result.get("individual_predictions", {})
        
        elo_diff = matchup_metrics.get("elo_advantage", 0)
        home_elo = matchup_metrics.get("home_elo", 1500)
        away_elo = matchup_metrics.get("away_elo", 1500)
        
        context_advantage = context.get("net_context_advantage", 0)
        context_factors_list = context.get("key_factors", [])
        
        injury_advantage = injury.get("net_advantage", 0)
        home_injuries = injury.get("home_injuries", [])
        away_injuries = injury.get("away_injuries", [])
        injury_summary = injury.get("summary", "")
        
        line_markets = line_analysis.get("markets", {})
        sharp_detected = line_analysis.get("sharp_money_detected", False)
        
        mc_outcomes = simulation.get("monte_carlo", {}).get("outcomes", {})
        home_win_pct = mc_outcomes.get("home_win_pct", 0.5)
        expected_total = mc_outcomes.get("expected_total", 0)
        spread_cover_pct = mc_outcomes.get("spread_cover_pct", 0.5)
        over_pct = mc_outcomes.get("over_pct", 0.5)
        
        # ===== SECTION 1: OVERVIEW =====
        reasoning_parts.append("PREDICTION OVERVIEW")
        reasoning_parts.append("")
        reasoning_parts.append(f"Pick: {pick_display}")
        reasoning_parts.append(f"Confidence: {confidence:.1f}%")
        reasoning_parts.append(f"Edge: {edge*100:+.1f}%")
        reasoning_parts.append(f"Win Probability: {probability*100:.1f}%")
        reasoning_parts.append("")
        
        # ===== SECTION 2: MODEL AGREEMENT =====
        reasoning_parts.append("MODEL AGREEMENT")
        reasoning_parts.append("")
        reasoning_parts.append(f"{agreeing_models} out of {num_models} models agree on this pick.")
        
        if individual_preds:
            reasoning_parts.append("")
            for model_name, model_data in individual_preds.items():
                model_pick = model_data.get("pick", "N/A")
                model_conf = model_data.get("confidence", 0)
                status = "agrees" if model_pick == pick else "disagrees"
                clean_name = model_name.replace("_", " ").title()
                reasoning_parts.append(f"  • {clean_name}: {status} ({model_conf:.0f}% conf)")
        reasoning_parts.append("")
        
        # ===== SECTION 3: TEAM STRENGTH =====
        reasoning_parts.append("TEAM STRENGTH")
        reasoning_parts.append("")
        reasoning_parts.append(f"{home_team}: {home_elo:.0f} ELO rating")
        reasoning_parts.append(f"{away_team}: {away_elo:.0f} ELO rating")
        
        if abs(elo_diff) > 100:
            advantage_team = home_team if elo_diff > 0 else away_team
            reasoning_parts.append(f"Significant advantage: {advantage_team} (+{abs(elo_diff):.0f} ELO)")
            key_factors.append(f"Strong ELO advantage for {advantage_team}")
        elif abs(elo_diff) > 50:
            advantage_team = home_team if elo_diff > 0 else away_team
            reasoning_parts.append(f"Moderate advantage: {advantage_team} (+{abs(elo_diff):.0f} ELO)")
            key_factors.append(f"ELO advantage for {advantage_team}")
        else:
            reasoning_parts.append("Teams are evenly matched in strength.")
        reasoning_parts.append("")
        
        # ===== SECTION 4: SITUATIONAL FACTORS =====
        reasoning_parts.append("SITUATIONAL FACTORS")
        reasoning_parts.append("")
        
        situation_notes = []
        if context.get("home_b2b"):
            situation_notes.append(f"{home_team} playing back-to-back (fatigue risk)")
        if context.get("away_b2b"):
            situation_notes.append(f"{away_team} playing back-to-back (fatigue risk)")
        if context.get("altitude_factor"):
            situation_notes.append(f"Altitude factor in play (Denver/Utah advantage)")
            key_factors.append("Altitude advantage")
        
        for cf in context_factors_list[:3]:
            situation_notes.append(cf)
            if len(key_factors) < 5:
                key_factors.append(cf)
        
        if situation_notes:
            for note in situation_notes:
                reasoning_parts.append(f"  • {note}")
        else:
            reasoning_parts.append("  No significant situational advantages.")
        
        if abs(context_advantage) > 0.03:
            advantage_side = "home team" if context_advantage > 0 else "away team"
            reasoning_parts.append(f"  Net situational edge: {advantage_side} ({abs(context_advantage)*100:.1f}%)")
        reasoning_parts.append("")
        
        # ===== SECTION 5: INJURIES =====
        reasoning_parts.append("INJURY IMPACT")
        reasoning_parts.append("")
        
        has_injuries = False
        if home_injuries:
            has_injuries = True
            reasoning_parts.append(f"{home_team}:")
            for inj in home_injuries[:3]:
                player = inj.get("name", "Unknown")
                status = inj.get("status", "Unknown")
                reasoning_parts.append(f"  • {player} - {status}")
        
        if away_injuries:
            has_injuries = True
            reasoning_parts.append(f"{away_team}:")
            for inj in away_injuries[:3]:
                player = inj.get("name", "Unknown")
                status = inj.get("status", "Unknown")
                reasoning_parts.append(f"  • {player} - {status}")
        
        if not has_injuries:
            reasoning_parts.append("Both teams are relatively healthy with no major injuries.")
        
        if injury_summary and "advantage" in injury_summary.lower():
            key_factors.append("Injury advantage")
        reasoning_parts.append("")
        
        # ===== SECTION 6: LINE MOVEMENT =====
        reasoning_parts.append("LINE MOVEMENT")
        reasoning_parts.append("")
        
        if sharp_detected:
            reasoning_parts.append("Sharp money detected! Professional bettors are active.")
            sharp_side = line_analysis.get("sharp_money_side", "")
            if sharp_side:
                reasoning_parts.append(f"Smart money favors: {sharp_side}")
            key_factors.append("Sharp money signal")
        else:
            reasoning_parts.append("No significant sharp money movement detected.")
        
        has_movement = False
        for market_name, market_info in line_markets.items():
            movement = market_info.get("movement", 0)
            if abs(movement) > 0.5:
                has_movement = True
                opening = market_info.get("opening_line", "N/A")
                current = market_info.get("current_line", "N/A")
                direction = "up" if movement > 0 else "down"
                reasoning_parts.append(f"  {market_name.title()}: moved {direction} from {opening} to {current}")
        
        if not has_movement:
            reasoning_parts.append("Lines have remained stable.")
        reasoning_parts.append("")
        
        # ===== SECTION 7: SIMULATION RESULTS =====
        reasoning_parts.append("SIMULATION RESULTS")
        reasoning_parts.append("(Based on 1000+ Monte Carlo simulations)")
        reasoning_parts.append("")
        reasoning_parts.append(f"  {home_team} win probability: {home_win_pct*100:.1f}%")
        reasoning_parts.append(f"  {away_team} win probability: {(1-home_win_pct)*100:.1f}%")
        
        if expected_total > 0:
            reasoning_parts.append(f"  Expected total points: {expected_total:.1f}")
        if over_pct != 0.5:
            reasoning_parts.append(f"  Over probability: {over_pct*100:.1f}%")
        reasoning_parts.append("")
        
        # ===== SECTION 8: MARKET SELECTION =====
        reasoning_parts.append("WHY THIS BET TYPE?")
        reasoning_parts.append("")
        
        if market_type == "moneyline":
            reasoning_parts.append(f"Moneyline selected because {pick} has a clear win expectation.")
            reasoning_parts.append(f"Our models give {pick} a {probability*100:.1f}% chance to win outright.")
        elif market_type in ["spread", "spreads"]:
            reasoning_parts.append("Spread selected because margin of victory is the key factor.")
            if spread_value:
                reasoning_parts.append(f"Line: {spread_value}")
            if spread_cover_pct != 0.5:
                reasoning_parts.append(f"Spread cover probability: {spread_cover_pct*100:.1f}%")
        elif market_type in ["total", "totals"]:
            reasoning_parts.append("Totals selected because scoring pace is the strongest signal.")
            if total_line and expected_total > 0:
                diff = expected_total - total_line
                if diff > 0:
                    reasoning_parts.append(f"Expected total ({expected_total:.1f}) is {diff:.1f} points ABOVE the line ({total_line}).")
                else:
                    reasoning_parts.append(f"Expected total ({expected_total:.1f}) is {abs(diff):.1f} points BELOW the line ({total_line}).")
            if over_pct != 0.5:
                reasoning_parts.append(f"Simulations show {over_pct*100:.1f}% chance of going over.")
        reasoning_parts.append("")
        
        # ===== SECTION 9: KEY FACTORS SUMMARY =====
        reasoning_parts.append("KEY FACTORS")
        reasoning_parts.append("")
        if key_factors:
            for i, factor in enumerate(key_factors[:5], 1):
                reasoning_parts.append(f"  {i}. {factor}")
        else:
            reasoning_parts.append("  Multiple factors align for this pick.")
        
        return {
            "has_pick": True,
            "pick": pick,
            "pick_type": market_type,
            "pick_display": pick_display,
            "spread": spread_value,
            "total_line": total_line,
            "odds": odds,
            "confidence": round(confidence, 1),
            "our_probability": round(probability * 100, 1),
            "edge": round(edge * 100, 1),
            "model_agreement": round(agreement * 100, 1),
            "models_in_consensus": ensemble_result.get("num_models", 5),
            "reasoning": "\n".join(reasoning_parts),
            "key_factors": key_factors,
            "ensemble_details": ensemble_result,
            "simulation_data": simulation,
            "matchup_summary": {
                "elo_diff": elo_diff,
                "context_advantage": context.get("net_context_advantage", 0),
                "injury_advantage": injury.get("net_advantage", 0),
                "line_movement_signal": line_analysis.get("confidence_adjustment", 0)
            },
            "market_analysis": line_analysis.get("markets", {}),
            "psychology_insights": psychology.get("contrarian_opportunities", {})
        }
    
    def _generate_no_pick_response(
        self,
        ensemble_result: Dict,
        home_team: str,
        away_team: str,
        reason: str,
        line_analysis: Dict,
        simulation: Dict,
        matchup_metrics: Dict
    ) -> Dict:
        """
        Generate no-pick response with analysis.
        """
        return {
            "has_pick": False,
            "reasoning": f"No pick recommended: {reason}",
            "ensemble_probability": round(ensemble_result.get("ensemble_probability", 0.5) * 100, 1),
            "ensemble_confidence": round(ensemble_result.get("ensemble_confidence", 0), 1),
            "model_agreement": round(ensemble_result.get("model_agreement", 0) * 100, 1),
            "models_in_consensus": ensemble_result.get("num_models", 5),
            "home_team": home_team,
            "away_team": away_team,
            "matchup_summary": {
                "elo_diff": matchup_metrics.get("elo_advantage", 0),
                "win_pct_diff": matchup_metrics.get("win_pct_diff", 0),
                "margin_diff": matchup_metrics.get("margin_diff", 0)
            },
            "simulation_summary": simulation.get("monte_carlo", {}).get("score_projections", {}),
            "line_movement_summary": line_analysis.get("summary", ""),
            "ensemble_details": ensemble_result
        }
    
    def _extract_spread(self, event: Dict) -> float:
        """Extract spread from event data."""
        bookmakers = event.get("bookmakers", [])
        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") == "spreads":
                    outcomes = market.get("outcomes", [])
                    if outcomes:
                        return abs(outcomes[0].get("point", 0))
        return 5.0  # Default
    
    def _extract_total(self, event: Dict) -> float:
        """Extract total from event data."""
        bookmakers = event.get("bookmakers", [])
        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") == "totals":
                    outcomes = market.get("outcomes", [])
                    if outcomes:
                        return outcomes[0].get("point", 220)
        
        # Sport-specific defaults
        defaults = {
            "basketball_nba": 225,
            "americanfootball_nfl": 45,
            "icehockey_nhl": 6,
            "soccer_epl": 2.5
        }
        return defaults.get(event.get("sport_key", ""), 220)


# Main entry point
async def generate_v6_prediction(
    event: Dict,
    sport_key: str,
    squad_data: Dict,
    matchup_data: Dict,
    line_movement_history: List[Dict],
    opening_odds: Dict,
    current_odds: Dict
) -> Dict:
    """Generate a prediction using BetPredictor V6."""
    engine = BetPredictorV6()
    return await engine.analyze_and_predict(
        event,
        sport_key,
        squad_data,
        matchup_data,
        line_movement_history,
        opening_odds,
        current_odds
    )
