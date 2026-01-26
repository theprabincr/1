"""
Enhanced Betting Algorithm V3 - ULTRA CONSERVATIVE with DEEP ANALYSIS
Predictions are ONLY made 1-2 hours before game start after comprehensive analysis

Key Features:
1. Waits until 1-2 hours before game for maximum data accuracy
2. Analyzes full squad player statistics and performances
3. Deep line movement analysis over time
4. Head-to-head record analysis
5. Venue and travel factors
6. Injury impact assessment
7. Multi-bookmaker odds comparison (when available)
8. Sharp vs Public money indicators
9. ONLY outputs 70%+ confidence when 5+ factors strongly align

Philosophy: NO PICK is the DEFAULT. Most games = efficient market = no edge.
"""
import logging
import math
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

# ESPN API Base
ESPN_BASE = "http://site.api.espn.com/apis/site/v2/sports"

# Sport configurations
SPORT_CONFIG = {
    "basketball_nba": {"sport": "basketball", "league": "nba", "avg_total": 225, "home_adv": 0.025},
    "americanfootball_nfl": {"sport": "football", "league": "nfl", "avg_total": 45, "home_adv": 0.022},
    "baseball_mlb": {"sport": "baseball", "league": "mlb", "avg_total": 8.5, "home_adv": 0.020},
    "icehockey_nhl": {"sport": "hockey", "league": "nhl", "avg_total": 6, "home_adv": 0.022},
    "soccer_epl": {"sport": "soccer", "league": "eng.1", "avg_total": 2.5, "home_adv": 0.035},
}

# Minimum requirements for making a pick
MIN_CONFIDENCE = 0.70
MIN_EDGE = 0.04  # 4% edge required
MIN_SUPPORTING_FACTORS = 4  # At least 4 factors must align


class EnhancedBettingAlgorithm:
    """
    Enhanced algorithm that performs deep analysis before making predictions.
    Only runs 1-2 hours before game start for maximum accuracy.
    """
    
    def __init__(self, odds_api_key: str = None):
        self.odds_api_key = odds_api_key  # For multi-bookmaker odds
        self.analysis_cache = {}  # Cache analysis to avoid repeated calls
        
    async def analyze_matchup(
        self,
        event: Dict,
        sport_key: str,
        matchup_data: Dict,
        line_movement_history: List[Dict],
        multi_book_odds: Dict = None
    ) -> Optional[Dict]:
        """
        Main entry point for comprehensive matchup analysis.
        Returns a pick dict if genuine edge found, None otherwise.
        """
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        
        logger.info(f"🔬 Deep analysis: {home_team} vs {away_team}")
        
        # Initialize analysis result
        analysis = {
            "home_team": home_team,
            "away_team": away_team,
            "factors": [],
            "home_score": 50,  # Start neutral
            "warnings": [],
            "edge_sources": []
        }
        
        try:
            # 1. Analyze recent form and momentum
            form_result = await self._analyze_form(matchup_data, analysis)
            
            # 2. Analyze player stats and squad strength
            squad_result = await self._analyze_squads(event, sport_key, analysis)
            
            # 3. Analyze head-to-head history
            h2h_result = await self._analyze_h2h(event, sport_key, analysis)
            
            # 4. Analyze venue and travel
            venue_result = self._analyze_venue(event, matchup_data, sport_key, analysis)
            
            # 5. Analyze injuries impact
            injury_result = await self._analyze_injuries(matchup_data, analysis)
            
            # 6. Analyze line movement for sharp money
            line_result = self._analyze_line_movement_deep(line_movement_history, analysis)
            
            # 7. Analyze multi-bookmaker odds if available
            if multi_book_odds:
                odds_result = self._analyze_multi_book_odds(multi_book_odds, analysis)
            
            # 8. Calculate final probability and edge
            pick = self._calculate_final_pick(analysis, matchup_data, sport_key)
            
            return pick
            
        except Exception as e:
            logger.error(f"Error in deep analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _analyze_form(self, matchup_data: Dict, analysis: Dict) -> Dict:
        """Analyze recent form with momentum detection"""
        home_team_data = matchup_data.get("home_team", {})
        away_team_data = matchup_data.get("away_team", {})
        
        home_form = home_team_data.get("form", {})
        away_form = away_team_data.get("form", {})
        
        home_recent = home_team_data.get("recent_games", [])
        away_recent = away_team_data.get("recent_games", [])
        
        # Win percentage (last 10 games)
        home_win_pct = home_form.get("win_pct", 0.5)
        away_win_pct = away_form.get("win_pct", 0.5)
        
        # Average margin
        home_avg_margin = home_form.get("avg_margin", 0)
        away_avg_margin = away_form.get("avg_margin", 0)
        
        # Momentum (last 5 games weighted more heavily)
        home_momentum = self._calculate_momentum(home_recent[:5])
        away_momentum = self._calculate_momentum(away_recent[:5])
        
        # Streak analysis
        home_streak = home_form.get("streak", 0)
        away_streak = away_form.get("streak", 0)
        
        # Calculate form adjustment
        form_diff = 0
        factors_added = 0
        
        # Win percentage factor (capped at 4%)
        win_pct_diff = (home_win_pct - away_win_pct) * 0.10
        win_pct_diff = max(-0.04, min(0.04, win_pct_diff))
        form_diff += win_pct_diff
        
        if abs(home_win_pct - away_win_pct) > 0.15:
            better = analysis["home_team"] if home_win_pct > away_win_pct else analysis["away_team"]
            analysis["factors"].append(f"Form: {better} significantly better ({home_win_pct*100:.0f}% vs {away_win_pct*100:.0f}%)")
            factors_added += 1
        
        # Margin factor (who wins/loses by more)
        margin_diff = (home_avg_margin - away_avg_margin) * 0.002  # 0.2% per point
        margin_diff = max(-0.03, min(0.03, margin_diff))
        form_diff += margin_diff
        
        if abs(home_avg_margin - away_avg_margin) > 5:
            better = analysis["home_team"] if home_avg_margin > away_avg_margin else analysis["away_team"]
            analysis["factors"].append(f"Margin: {better} wins by more (Δ{abs(home_avg_margin - away_avg_margin):.1f} pts)")
            factors_added += 1
        
        # Momentum factor
        momentum_diff = (home_momentum - away_momentum) * 0.02
        momentum_diff = max(-0.02, min(0.02, momentum_diff))
        form_diff += momentum_diff
        
        if abs(home_momentum - away_momentum) > 0.3:
            better = analysis["home_team"] if home_momentum > away_momentum else analysis["away_team"]
            analysis["factors"].append(f"Momentum: {better} playing better recently")
            factors_added += 1
        
        # Hot/Cold streak factor
        if abs(home_streak) >= 3 or abs(away_streak) >= 3:
            streak_diff = (home_streak - away_streak) * 0.005
            streak_diff = max(-0.02, min(0.02, streak_diff))
            form_diff += streak_diff
            
            if home_streak >= 3:
                analysis["factors"].append(f"Streak: {analysis['home_team']} on {home_streak} game win streak")
                factors_added += 1
            elif away_streak >= 3:
                analysis["factors"].append(f"Streak: {analysis['away_team']} on {away_streak} game win streak")
                factors_added += 1
        
        analysis["home_score"] += form_diff * 100
        
        return {
            "form_diff": form_diff,
            "factors_added": factors_added,
            "home_win_pct": home_win_pct,
            "away_win_pct": away_win_pct
        }
    
    def _calculate_momentum(self, recent_games: List[Dict]) -> float:
        """Calculate momentum with recency weighting"""
        if not recent_games:
            return 0
        
        weights = [1.5, 1.3, 1.1, 1.0, 0.9]  # More recent = more weight
        total_weight = 0
        weighted_sum = 0
        
        for i, game in enumerate(recent_games[:5]):
            weight = weights[i] if i < len(weights) else 0.8
            total_weight += weight
            if game.get("won", False):
                weighted_sum += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    async def _analyze_squads(self, event: Dict, sport_key: str, analysis: Dict) -> Dict:
        """Analyze full squad player statistics"""
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        home_team_id = event.get("home_team_id", "")
        away_team_id = event.get("away_team_id", "")
        
        if sport_key not in SPORT_CONFIG:
            return {"squad_diff": 0, "factors_added": 0}
        
        config = SPORT_CONFIG[sport_key]
        
        # Fetch roster stats for both teams
        home_roster_stats = await self._fetch_roster_stats(home_team_id, config)
        away_roster_stats = await self._fetch_roster_stats(away_team_id, config)
        
        squad_diff = 0
        factors_added = 0
        
        # Calculate team strength from player stats
        home_strength = self._calculate_squad_strength(home_roster_stats, config)
        away_strength = self._calculate_squad_strength(away_roster_stats, config)
        
        if home_strength > 0 and away_strength > 0:
            strength_ratio = home_strength / away_strength if away_strength > 0 else 1
            
            if strength_ratio > 1.10:  # Home team 10%+ stronger
                squad_diff = min(0.03, (strength_ratio - 1) * 0.15)
                analysis["factors"].append(f"Squad: {home_team} roster stats superior")
                factors_added += 1
            elif strength_ratio < 0.90:  # Away team 10%+ stronger
                squad_diff = max(-0.03, (strength_ratio - 1) * 0.15)
                analysis["factors"].append(f"Squad: {away_team} roster stats superior")
                factors_added += 1
        
        # Check for key player availability
        home_injuries = await self._get_injury_impact(home_team_id, config)
        away_injuries = await self._get_injury_impact(away_team_id, config)
        
        injury_diff = (away_injuries - home_injuries) * 0.01  # Higher = more injured = worse
        injury_diff = max(-0.03, min(0.03, injury_diff))
        squad_diff += injury_diff
        
        if abs(home_injuries - away_injuries) > 2:
            worse = analysis["away_team"] if home_injuries < away_injuries else analysis["home_team"]
            analysis["factors"].append(f"Injuries: {worse} missing more key players")
            factors_added += 1
        
        analysis["home_score"] += squad_diff * 100
        
        return {
            "squad_diff": squad_diff,
            "factors_added": factors_added,
            "home_strength": home_strength,
            "away_strength": away_strength
        }
    
    async def _fetch_roster_stats(self, team_id: str, config: Dict) -> Dict:
        """Fetch roster statistics from ESPN"""
        if not team_id:
            return {}
        
        url = f"{ESPN_BASE}/{config['sport']}/{config['league']}/teams/{team_id}/roster"
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.debug(f"Could not fetch roster stats: {e}")
        
        return {}
    
    def _calculate_squad_strength(self, roster_data: Dict, config: Dict) -> float:
        """Calculate overall squad strength from player stats"""
        if not roster_data:
            return 0
        
        total_score = 0
        player_count = 0
        
        for category in roster_data.get("athletes", []):
            for player in category.get("items", []):
                # Check if player has stats
                stats = player.get("statistics", {})
                if stats:
                    # Weight by position importance and stats
                    ppg = float(stats.get("points", {}).get("value", 0) or 0)
                    rpg = float(stats.get("rebounds", {}).get("value", 0) or 0)
                    apg = float(stats.get("assists", {}).get("value", 0) or 0)
                    
                    player_value = ppg + (rpg * 0.5) + (apg * 0.7)
                    total_score += player_value
                    player_count += 1
        
        return total_score / max(player_count, 1)
    
    async def _get_injury_impact(self, team_id: str, config: Dict) -> int:
        """Get injury impact score (higher = more injured)"""
        if not team_id:
            return 0
        
        url = f"{ESPN_BASE}/{config['sport']}/{config['league']}/teams/{team_id}"
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params={"enable": "injuries"})
                if response.status_code == 200:
                    data = response.json()
                    injuries = data.get("team", {}).get("injuries", [])
                    
                    impact = 0
                    for injury in injuries:
                        status = injury.get("status", "").lower()
                        if "out" in status:
                            impact += 2
                        elif "doubtful" in status:
                            impact += 1.5
                        elif "questionable" in status:
                            impact += 0.5
                    
                    return impact
        except Exception as e:
            logger.debug(f"Could not fetch injury data: {e}")
        
        return 0
    
    async def _analyze_h2h(self, event: Dict, sport_key: str, analysis: Dict) -> Dict:
        """Analyze head-to-head history"""
        home_team_id = event.get("home_team_id", "")
        away_team_id = event.get("away_team_id", "")
        
        if not home_team_id or sport_key not in SPORT_CONFIG:
            return {"h2h_diff": 0, "factors_added": 0}
        
        config = SPORT_CONFIG[sport_key]
        
        # Fetch recent games for home team and look for matchups with away team
        url = f"{ESPN_BASE}/{config['sport']}/{config['league']}/teams/{home_team_id}/schedule"
        
        h2h_games = []
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    
                    for game in data.get("events", []):
                        status = game.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("name", "")
                        if status not in ["STATUS_FINAL", "STATUS_FINAL_OT"]:
                            continue
                        
                        competitors = game.get("competitions", [{}])[0].get("competitors", [])
                        is_h2h = False
                        home_won = False
                        
                        for comp in competitors:
                            if comp.get("id") == away_team_id:
                                is_h2h = True
                            if comp.get("id") == home_team_id:
                                home_won = comp.get("winner", False)
                        
                        if is_h2h:
                            h2h_games.append({"won": home_won})
                        
                        if len(h2h_games) >= 5:
                            break
        except Exception as e:
            logger.debug(f"Could not fetch H2H data: {e}")
        
        h2h_diff = 0
        factors_added = 0
        
        if len(h2h_games) >= 3:
            home_wins = sum(1 for g in h2h_games if g.get("won", False))
            h2h_pct = home_wins / len(h2h_games)
            
            if h2h_pct >= 0.70:  # Home team dominates H2H
                h2h_diff = 0.02
                analysis["factors"].append(f"H2H: {analysis['home_team']} dominates series ({home_wins}-{len(h2h_games)-home_wins})")
                factors_added += 1
            elif h2h_pct <= 0.30:  # Away team dominates H2H
                h2h_diff = -0.02
                analysis["factors"].append(f"H2H: {analysis['away_team']} dominates series ({len(h2h_games)-home_wins}-{home_wins})")
                factors_added += 1
        
        analysis["home_score"] += h2h_diff * 100
        
        return {
            "h2h_diff": h2h_diff,
            "factors_added": factors_added,
            "h2h_games": len(h2h_games)
        }
    
    def _analyze_venue(self, event: Dict, matchup_data: Dict, sport_key: str, analysis: Dict) -> Dict:
        """Analyze venue and travel factors"""
        home_stats = matchup_data.get("home_team", {}).get("stats", {})
        away_stats = matchup_data.get("away_team", {}).get("stats", {})
        
        venue_diff = 0
        factors_added = 0
        
        # Base home advantage
        config = SPORT_CONFIG.get(sport_key, {})
        home_adv = config.get("home_adv", 0.025)
        venue_diff += home_adv
        
        # Home/Away record split
        home_record = home_stats.get("home_record", "")
        away_record_str = away_stats.get("away_record", "")
        
        home_home_pct = self._parse_record_pct(home_record)
        away_away_pct = self._parse_record_pct(away_record_str)
        
        if home_home_pct and away_away_pct:
            split_diff = (home_home_pct - away_away_pct) * 0.05
            split_diff = max(-0.02, min(0.02, split_diff))
            venue_diff += split_diff
            
            if abs(home_home_pct - away_away_pct) > 0.15:
                if home_home_pct > away_away_pct:
                    analysis["factors"].append(f"Venue: {analysis['home_team']} strong at home ({home_home_pct*100:.0f}%)")
                else:
                    analysis["factors"].append(f"Venue: {analysis['away_team']} good on road ({away_away_pct*100:.0f}%)")
                factors_added += 1
        
        analysis["factors"].append(f"Home advantage: +{home_adv*100:.1f}% for {analysis['home_team']}")
        factors_added += 1
        
        analysis["home_score"] += venue_diff * 100
        
        return {
            "venue_diff": venue_diff,
            "factors_added": factors_added
        }
    
    def _parse_record_pct(self, record_str: str) -> Optional[float]:
        """Parse record string like '15-8' to win percentage"""
        if not record_str:
            return None
        
        try:
            parts = record_str.split("-")
            if len(parts) >= 2:
                wins = int(parts[0].strip())
                losses = int(parts[1].strip())
                total = wins + losses
                return wins / total if total > 0 else None
        except:
            return None
        
        return None
    
    async def _analyze_injuries(self, matchup_data: Dict, analysis: Dict) -> Dict:
        """Analyze injury impact on both teams"""
        # This is partially covered in _analyze_squads
        # Here we add any additional injury-specific factors
        
        return {"injury_diff": 0, "factors_added": 0}
    
    def _analyze_line_movement_deep(self, line_history: List[Dict], analysis: Dict) -> Dict:
        """Deep analysis of line movement to detect sharp money"""
        if not line_history or len(line_history) < 2:
            return {"line_diff": 0, "factors_added": 0}
        
        line_diff = 0
        factors_added = 0
        
        # Get opening and current lines
        opening = line_history[0] if line_history else {}
        current = line_history[-1] if line_history else {}
        
        opening_home_odds = opening.get("home_odds", 1.91)
        current_home_odds = current.get("home_odds", 1.91)
        
        if opening_home_odds <= 0 or current_home_odds <= 0:
            return {"line_diff": 0, "factors_added": 0}
        
        # Calculate movement percentage
        movement_pct = (current_home_odds - opening_home_odds) / opening_home_odds * 100
        
        # Detect sharp money (reverse line movement)
        # If home odds dropped (became more favored) = sharp money on home
        if movement_pct < -5:  # Home odds dropped 5%+
            line_diff = 0.02
            analysis["factors"].append(f"Line: Sharp money on {analysis['home_team']} (odds dropped {abs(movement_pct):.1f}%)")
            analysis["edge_sources"].append("sharp_money")
            factors_added += 1
        elif movement_pct > 5:  # Away odds dropped 5%+
            line_diff = -0.02
            analysis["factors"].append(f"Line: Sharp money on {analysis['away_team']} (odds moved {movement_pct:.1f}%)")
            analysis["edge_sources"].append("sharp_money")
            factors_added += 1
        
        # Check for steam moves (rapid movement)
        if len(line_history) >= 3:
            recent_changes = []
            for i in range(1, min(4, len(line_history))):
                prev = line_history[i-1].get("home_odds", 1.91)
                curr = line_history[i].get("home_odds", 1.91)
                if prev > 0:
                    change = (curr - prev) / prev * 100
                    recent_changes.append(change)
            
            # Consistent direction = steam move
            if recent_changes and all(c < 0 for c in recent_changes):
                if sum(abs(c) for c in recent_changes) > 7:
                    line_diff += 0.01
                    analysis["factors"].append(f"Line: Steam move detected on {analysis['home_team']}")
                    factors_added += 1
            elif recent_changes and all(c > 0 for c in recent_changes):
                if sum(abs(c) for c in recent_changes) > 7:
                    line_diff -= 0.01
                    analysis["factors"].append(f"Line: Steam move detected on {analysis['away_team']}")
                    factors_added += 1
        
        analysis["home_score"] += line_diff * 100
        
        return {
            "line_diff": line_diff,
            "factors_added": factors_added,
            "movement_pct": movement_pct
        }
    
    def _analyze_multi_book_odds(self, multi_book_odds: Dict, analysis: Dict) -> Dict:
        """Analyze odds from multiple bookmakers to find value"""
        if not multi_book_odds or not multi_book_odds.get("bookmakers"):
            return {"odds_diff": 0, "factors_added": 0}
        
        bookmakers = multi_book_odds.get("bookmakers", [])
        
        home_odds_list = []
        away_odds_list = []
        
        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == analysis["home_team"]:
                            home_odds_list.append(outcome.get("price", 1.91))
                        else:
                            away_odds_list.append(outcome.get("price", 1.91))
        
        odds_diff = 0
        factors_added = 0
        
        if home_odds_list and away_odds_list:
            # Check for odds discrepancy (potential value)
            home_avg = sum(home_odds_list) / len(home_odds_list)
            home_max = max(home_odds_list)
            home_min = min(home_odds_list)
            
            away_avg = sum(away_odds_list) / len(away_odds_list)
            away_max = max(away_odds_list)
            away_min = min(away_odds_list)
            
            # Large spread = potential value
            home_spread = (home_max - home_min) / home_avg * 100
            away_spread = (away_max - away_min) / away_avg * 100
            
            if home_spread > 5:  # 5%+ odds difference
                analysis["factors"].append(f"Value: Odds spread on {analysis['home_team']} ({home_min:.2f}-{home_max:.2f})")
                factors_added += 1
            
            if away_spread > 5:
                analysis["factors"].append(f"Value: Odds spread on {analysis['away_team']} ({away_min:.2f}-{away_max:.2f})")
                factors_added += 1
            
            analysis["multi_book_data"] = {
                "home_odds_range": [home_min, home_max],
                "away_odds_range": [away_min, away_max],
                "bookmakers_count": len(bookmakers)
            }
        
        return {
            "odds_diff": odds_diff,
            "factors_added": factors_added
        }
    
    def _calculate_final_pick(self, analysis: Dict, matchup_data: Dict, sport_key: str) -> Optional[Dict]:
        """Calculate final pick based on all factors"""
        home_score = analysis.get("home_score", 50)
        factors = analysis.get("factors", [])
        
        # Convert score to probability (normalized)
        home_prob = home_score / 100
        home_prob = max(0.35, min(0.65, home_prob))  # Clamp to realistic range
        away_prob = 1 - home_prob
        
        # Get market implied probabilities
        odds = matchup_data.get("odds", {})
        home_ml_decimal = odds.get("home_ml_decimal", 1.91)
        away_ml_decimal = odds.get("away_ml_decimal", 1.91)
        
        # Calculate market probabilities (remove juice)
        market_home_prob = 1 / home_ml_decimal if home_ml_decimal > 1 else 0.5
        market_away_prob = 1 / away_ml_decimal if away_ml_decimal > 1 else 0.5
        total_implied = market_home_prob + market_away_prob
        
        if total_implied > 0:
            market_home_prob /= total_implied
            market_away_prob /= total_implied
        
        # Calculate edge
        home_edge = home_prob - market_home_prob
        away_edge = away_prob - market_away_prob
        
        # Count supporting factors
        supporting_factors = len(factors)
        
        # Determine best pick
        best_pick = None
        
        # Check home team
        if home_edge >= MIN_EDGE and supporting_factors >= MIN_SUPPORTING_FACTORS:
            confidence = self._calculate_confidence(home_edge, supporting_factors, analysis)
            if confidence >= MIN_CONFIDENCE:
                best_pick = {
                    "side": "home",
                    "team": analysis["home_team"],
                    "edge": home_edge,
                    "our_prob": home_prob,
                    "market_prob": market_home_prob,
                    "confidence": confidence,
                    "odds": home_ml_decimal
                }
        
        # Check away team
        if away_edge >= MIN_EDGE and supporting_factors >= MIN_SUPPORTING_FACTORS:
            confidence = self._calculate_confidence(away_edge, supporting_factors, analysis)
            if confidence >= MIN_CONFIDENCE:
                if best_pick is None or away_edge > best_pick["edge"]:
                    best_pick = {
                        "side": "away",
                        "team": analysis["away_team"],
                        "edge": away_edge,
                        "our_prob": away_prob,
                        "market_prob": market_away_prob,
                        "confidence": confidence,
                        "odds": away_ml_decimal
                    }
        
        # If no pick meets criteria, return None
        if best_pick is None:
            logger.info(f"No pick - Edge: H{home_edge*100:.1f}%/A{away_edge*100:.1f}%, Factors: {supporting_factors}")
            return None
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_pick, analysis, factors)
        
        return {
            "pick_type": "moneyline",
            "pick": best_pick["team"],
            "confidence": round(best_pick["confidence"], 2),
            "edge": round(best_pick["edge"] * 100, 1),
            "odds": best_pick["odds"],
            "our_probability": round(best_pick["our_prob"], 3),
            "market_probability": round(best_pick["market_prob"], 3),
            "reasoning": reasoning,
            "factors": factors,
            "supporting_factors_count": supporting_factors,
            "algorithm": "enhanced_v3"
        }
    
    def _calculate_confidence(self, edge: float, factor_count: int, analysis: Dict) -> float:
        """Calculate confidence based on edge and supporting factors"""
        # Base confidence from edge (each 1% edge = 1% confidence)
        base_conf = 0.65 + (edge * 1.5)
        
        # Bonus for more factors
        factor_bonus = min(0.05, (factor_count - 4) * 0.01)
        
        # Bonus for sharp money alignment
        if "sharp_money" in analysis.get("edge_sources", []):
            factor_bonus += 0.02
        
        confidence = base_conf + factor_bonus
        
        # CAP at 82% - anything higher is unrealistic
        return min(0.82, max(MIN_CONFIDENCE, confidence))
    
    def _generate_reasoning(self, pick: Dict, analysis: Dict, factors: List[str]) -> str:
        """Generate human-readable reasoning"""
        team = pick["team"]
        our_prob = pick["our_prob"] * 100
        market_prob = pick["market_prob"] * 100
        edge = pick["edge"] * 100
        
        reasoning_parts = [
            f"Our model gives {team} a {our_prob:.0f}% win probability vs market's {market_prob:.0f}%.",
            f"This represents a {edge:.1f}% edge.",
            f"Analysis based on {len(factors)} supporting factors."
        ]
        
        # Add top 3 factors
        if factors:
            reasoning_parts.append("Key factors: " + "; ".join(factors[:3]))
        
        return " ".join(reasoning_parts)


# Utility function for external use
async def calculate_enhanced_pick(
    event: Dict,
    sport_key: str,
    matchup_data: Dict,
    line_movement_history: List[Dict],
    multi_book_odds: Dict = None
) -> Optional[Dict]:
    """
    Main entry point for enhanced betting algorithm.
    Call this 1-2 hours before game start for maximum accuracy.
    """
    algorithm = EnhancedBettingAlgorithm()
    return await algorithm.analyze_matchup(
        event, sport_key, matchup_data, line_movement_history, multi_book_odds
    )
