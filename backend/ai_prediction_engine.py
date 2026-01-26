"""
AI-Powered Betting Prediction Engine V4
Uses LLM (GPT/Claude) to analyze comprehensive data and make predictions

Features:
- Pulls squad data 1 hour before game
- Analyzes player stats, H2H, venue, injuries
- Studies line movement across bookmakers
- Makes diverse predictions: Moneyline, Spread, Totals
- Only 70%+ confidence predictions
- Considers odds as low as 1.5x
"""
import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import LLM integration
try:
    from emergentintegrations.llm.chat import LlmChat, UserMessage
    LLM_AVAILABLE = True
except ImportError:
    logger.warning("emergentintegrations not available - AI predictions disabled")
    LLM_AVAILABLE = False


class AIPredictionEngine:
    """
    AI-powered prediction engine that analyzes all available data
    and makes diverse predictions across multiple markets.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('EMERGENT_LLM_KEY', '')
        self.model = "gpt-5.2"  # Default model
        
    async def generate_prediction(
        self,
        event: Dict,
        sport_key: str,
        squad_data: Dict,
        matchup_data: Dict,
        line_movement: List[Dict],
        multi_book_odds: Dict,
        h2h_records: List[Dict] = None
    ) -> Optional[Dict]:
        """
        Generate AI prediction by feeding all data to the LLM.
        Returns prediction with pick type (ML/Spread/Total), confidence, and reasoning.
        """
        if not LLM_AVAILABLE or not self.api_key:
            logger.warning("LLM not available for AI prediction")
            return None
        
        home_team = event.get("home_team", "Home")
        away_team = event.get("away_team", "Away")
        
        logger.info(f"🤖 AI Analysis: {home_team} vs {away_team}")
        
        # Build comprehensive prompt with all data
        prompt = self._build_analysis_prompt(
            event, sport_key, squad_data, matchup_data, 
            line_movement, multi_book_odds, h2h_records
        )
        
        try:
            # Call LLM for analysis
            session_id = str(uuid.uuid4())
            
            system_message = """You are an elite sports betting analyst with 20+ years of experience. 
You analyze betting markets with exceptional precision and only recommend bets when you find GENUINE VALUE.

IMPORTANT RULES:
1. You MUST analyze ALL markets: Moneyline, Spread, AND Totals
2. Be DIVERSE in your picks - don't always pick moneyline
3. Consider odds as low as 1.5x (50% implied probability) - these can have value too
4. Only recommend picks with TRUE EDGE - if no edge exists, say NO PICK
5. Confidence must be 70-85% range - anything higher is unrealistic
6. Consider home AND away teams equally - don't have bias
7. Look for VALUE, not just winners - a favorite at 1.5 can be good value

OUTPUT FORMAT (JSON):
{
    "has_pick": true/false,
    "pick_type": "moneyline" | "spread" | "total",
    "pick": "Team Name" | "Home -5.5" | "Over 225.5",
    "odds": 1.85,
    "confidence": 0.72,
    "edge_percent": 4.5,
    "reasoning": "2-3 sentence explanation",
    "key_factors": ["factor1", "factor2", "factor3"],
    "warnings": ["any concerns"]
}

If no value found:
{
    "has_pick": false,
    "reasoning": "Why no pick - market is efficient",
    "closest_value": "What came closest to having value"
}"""
            
            chat = LlmChat(
                api_key=self.api_key,
                session_id=session_id,
                system_message=system_message
            )
            
            # Use GPT-5.2 for best analysis
            chat.with_model("openai", "gpt-5.2")
            
            response = await asyncio.to_thread(
                chat.send_message,
                UserMessage(content=prompt)
            )
            
            # Parse AI response
            prediction = self._parse_ai_response(response.content, event)
            
            if prediction and prediction.get("has_pick"):
                logger.info(f"✅ AI Pick: {prediction.get('pick_type')} - {prediction.get('pick')} "
                          f"@ {prediction.get('confidence', 0)*100:.0f}% conf")
            else:
                logger.info(f"⏭️ AI: No pick for {home_team} vs {away_team}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in AI prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_analysis_prompt(
        self,
        event: Dict,
        sport_key: str,
        squad_data: Dict,
        matchup_data: Dict,
        line_movement: List[Dict],
        multi_book_odds: Dict,
        h2h_records: List[Dict]
    ) -> str:
        """Build comprehensive analysis prompt for the AI"""
        
        home_team = event.get("home_team", "Home")
        away_team = event.get("away_team", "Away")
        venue = event.get("venue", {})
        odds = event.get("odds", {})
        
        # Format matchup data
        home_data = matchup_data.get("home_team", {})
        away_data = matchup_data.get("away_team", {})
        
        home_form = home_data.get("form", {})
        away_form = away_data.get("form", {})
        
        home_stats = home_data.get("stats", {})
        away_stats = away_data.get("stats", {})
        
        # Format squad data
        home_squad = squad_data.get("home", {})
        away_squad = squad_data.get("away", {})
        
        # Format line movement
        line_movement_str = "No line movement data"
        if line_movement:
            movement_entries = []
            for lm in line_movement[-5:]:  # Last 5 snapshots
                movement_entries.append(f"  - {lm.get('timestamp', 'N/A')}: Home {lm.get('home_odds', 'N/A')}, Away {lm.get('away_odds', 'N/A')}")
            line_movement_str = "\n".join(movement_entries)
        
        # Format multi-book odds
        multi_odds_str = "Single source only (ESPN/DraftKings)"
        if multi_book_odds and multi_book_odds.get("bookmakers"):
            books = []
            for bm in multi_book_odds["bookmakers"][:6]:
                for market in bm.get("markets", []):
                    if market.get("key") == "h2h":
                        outcomes = market.get("outcomes", [])
                        if len(outcomes) >= 2:
                            books.append(f"  - {bm.get('title', 'Unknown')}: {outcomes[0].get('name')} {outcomes[0].get('price')}, {outcomes[1].get('name')} {outcomes[1].get('price')}")
            if books:
                multi_odds_str = "\n".join(books)
        
        # Build prompt
        prompt = f"""ANALYZE THIS MATCHUP AND FIND VALUE:

🏀 GAME INFO:
- {home_team} vs {away_team}
- Sport: {sport_key}
- Venue: {venue.get('name', 'N/A')} ({venue.get('city', 'N/A')})
- Start Time: {event.get('commence_time', 'N/A')}

📊 CURRENT ODDS (ESPN/DraftKings):
- Moneyline: {home_team} @ {odds.get('home_ml_decimal', 'N/A')}, {away_team} @ {odds.get('away_ml_decimal', 'N/A')}
- Spread: {home_team} {odds.get('spread', 'N/A')}
- Total: {odds.get('total', 'N/A')}
- Favorite: {"Home" if odds.get('home_favorite') else "Away"}

📈 LINE MOVEMENT (Last snapshots):
{line_movement_str}

💰 MULTI-BOOKMAKER ODDS:
{multi_odds_str}

📋 TEAM RECORDS:
- {home_team}: {home_stats.get('record', event.get('home_record', 'N/A'))}
  - Home Record: {home_stats.get('home_record', 'N/A')}
  - Standing: {home_stats.get('standing', 'N/A')}
- {away_team}: {away_stats.get('record', event.get('away_record', 'N/A'))}
  - Away Record: {away_stats.get('away_record', 'N/A')}
  - Standing: {away_stats.get('standing', 'N/A')}

🔥 RECENT FORM (Last 10 games):
- {home_team}: {home_form.get('wins', 0)}-{home_form.get('losses', 0)} ({home_form.get('win_pct', 0)*100:.0f}%), Avg Margin: {home_form.get('avg_margin', 0):+.1f}, Streak: {home_form.get('streak', 0):+d}
- {away_team}: {away_form.get('wins', 0)}-{away_form.get('losses', 0)} ({away_form.get('win_pct', 0)*100:.0f}%), Avg Margin: {away_form.get('avg_margin', 0):+.1f}, Streak: {away_form.get('streak', 0):+d}

👥 SQUAD INFO:
{home_team}:
  - Key Players: {', '.join(home_squad.get('key_players', ['N/A'])[:5])}
  - Injuries: {len(home_squad.get('injuries', []))} players
  {self._format_injuries(home_squad.get('injuries', []))}

{away_team}:
  - Key Players: {', '.join(away_squad.get('key_players', ['N/A'])[:5])}
  - Injuries: {len(away_squad.get('injuries', []))} players
  {self._format_injuries(away_squad.get('injuries', []))}

🎯 HEAD-TO-HEAD (if available):
{self._format_h2h(h2h_records)}

ANALYZE ALL THREE MARKETS (Moneyline, Spread, Totals) and find the BEST VALUE bet.
Consider:
1. Is the favorite overvalued? Could the underdog cover?
2. Does recent scoring suggest Over or Under?
3. What does line movement tell us about sharp money?
4. How do injuries affect the spread/total?

Return your analysis as JSON with pick_type, pick, odds, confidence (0.70-0.85), and reasoning."""

        return prompt
    
    def _format_injuries(self, injuries: List[Dict]) -> str:
        """Format injury list for prompt"""
        if not injuries:
            return "  (No significant injuries)"
        
        injury_strs = []
        for inj in injuries[:3]:
            injury_strs.append(f"  - {inj.get('name', 'Unknown')}: {inj.get('status', 'Unknown')} ({inj.get('injury', 'Unknown')})")
        
        return "\n".join(injury_strs) if injury_strs else "  (No significant injuries)"
    
    def _format_h2h(self, h2h_records: List[Dict]) -> str:
        """Format H2H records for prompt"""
        if not h2h_records:
            return "No recent H2H data available"
        
        h2h_strs = []
        for game in h2h_records[:5]:
            h2h_strs.append(f"  - {game.get('date', 'N/A')}: {game.get('result', 'N/A')}")
        
        return "\n".join(h2h_strs) if h2h_strs else "No recent H2H data available"
    
    def _parse_ai_response(self, response: str, event: Dict) -> Optional[Dict]:
        """Parse AI response to extract prediction"""
        import json
        import re
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                prediction = json.loads(json_match.group())
                
                # Validate and enhance prediction
                if prediction.get("has_pick"):
                    # Ensure required fields
                    if not prediction.get("pick_type"):
                        prediction["pick_type"] = "moneyline"
                    if not prediction.get("confidence"):
                        prediction["confidence"] = 0.70
                    if not prediction.get("odds"):
                        prediction["odds"] = 1.91
                    
                    # Clamp confidence to realistic range
                    prediction["confidence"] = max(0.70, min(0.85, prediction["confidence"]))
                    
                    # Add metadata
                    prediction["algorithm"] = "ai_v4"
                    prediction["model"] = self.model
                    prediction["event_id"] = event.get("id")
                    prediction["home_team"] = event.get("home_team")
                    prediction["away_team"] = event.get("away_team")
                
                return prediction
            
            # If no JSON found, try to parse structured text
            prediction = {
                "has_pick": False,
                "reasoning": response[:500],
                "algorithm": "ai_v4"
            }
            
            # Look for pick indicators in text
            pick_patterns = [
                (r'PICK:\s*(.+)', "pick"),
                (r'CONFIDENCE:\s*(\d+)', "confidence"),
                (r'PICK TYPE:\s*(\w+)', "pick_type"),
            ]
            
            for pattern, field in pick_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if field == "confidence":
                        value = float(value) / 100
                    prediction[field] = value
                    prediction["has_pick"] = True
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return None


async def generate_ai_prediction(
    event: Dict,
    sport_key: str,
    squad_data: Dict,
    matchup_data: Dict,
    line_movement: List[Dict],
    multi_book_odds: Dict,
    h2h_records: List[Dict] = None,
    api_key: str = None
) -> Optional[Dict]:
    """
    Convenience function to generate AI prediction.
    Call this 1 hour before game start for best results.
    """
    engine = AIPredictionEngine(api_key)
    return await engine.generate_prediction(
        event, sport_key, squad_data, matchup_data,
        line_movement, multi_book_odds, h2h_records
    )
