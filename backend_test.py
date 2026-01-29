#!/usr/bin/env python3
"""
Backend Test Suite for BetPredictor API
Tests the Matchup, Starting Lineup, and Roster API endpoints with REAL ESPN data.
Also includes Adaptive Learning System tests.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any, List

# Backend URL from frontend .env
BASE_URL = "https://project-navigator-11.preview.emergentagent.com/api"

class BetPredictorTester:
    def __init__(self):
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, status: str, details: str = "", response_data: Any = None):
        """Log test result"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "response_data": response_data
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
        if response_data and status == "FAIL":
            print(f"   Response: {json.dumps(response_data, indent=2)}")
        print()
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> tuple[int, Dict]:
        """Make HTTP request and return status code and response data"""
        url = f"{BASE_URL}{endpoint}"
        try:
            async with self.session.request(method, url, **kwargs) as response:
                status_code = response.status
                try:
                    data = await response.json()
                except:
                    data = {"error": "Invalid JSON response", "text": await response.text()}
                return status_code, data
        except Exception as e:
            return 0, {"error": str(e)}
    
    # ==================== MATCHUP & LINEUP TESTS ====================
    
    async def test_matchup_endpoint(self):
        """Test GET /api/matchup/{event_id}?sport_key=basketball_nba with event_id 401810535"""
        event_id = "401810535"
        status_code, data = await self.make_request("GET", f"/matchup/{event_id}?sport_key=basketball_nba")
        
        if status_code != 200:
            self.log_test("Matchup API", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Check required top-level fields
        required_fields = ["event_id", "sport_key", "commence_time", "venue", "home_team", "away_team", "lineup_status", "lineup_message"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("Matchup API", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check event_id matches
        if data.get("event_id") != event_id:
            self.log_test("Matchup API", "FAIL", 
                         f"Expected event_id '{event_id}', got '{data.get('event_id')}'", data)
            return
        
        # Check sport_key matches
        if data.get("sport_key") != "basketball_nba":
            self.log_test("Matchup API", "FAIL", 
                         f"Expected sport_key 'basketball_nba', got '{data.get('sport_key')}'", data)
            return
        
        # Check home_team structure
        home_team = data.get("home_team", {})
        required_home_fields = ["name", "id", "stats", "form", "recent_games", "roster", "injuries", "starters", "starters_confirmed"]
        missing_home_fields = [field for field in required_home_fields if field not in home_team]
        
        if missing_home_fields:
            self.log_test("Matchup API", "FAIL", 
                         f"Missing home_team fields: {missing_home_fields}", data)
            return
        
        # Check away_team structure
        away_team = data.get("away_team", {})
        required_away_fields = ["name", "id", "stats", "form", "recent_games", "roster", "injuries", "starters", "starters_confirmed"]
        missing_away_fields = [field for field in required_away_fields if field not in away_team]
        
        if missing_away_fields:
            self.log_test("Matchup API", "FAIL", 
                         f"Missing away_team fields: {missing_away_fields}", data)
            return
        
        # Check venue structure
        venue = data.get("venue", {})
        if not isinstance(venue, dict):
            self.log_test("Matchup API", "FAIL", 
                         f"Expected venue to be a dict, got {type(venue)}", data)
            return
        
        # Check roster structure for both teams
        home_roster = home_team.get("roster", {})
        away_roster = away_team.get("roster", {})
        
        for team_name, roster in [("home", home_roster), ("away", away_roster)]:
            if not isinstance(roster, dict):
                self.log_test("Matchup API", "FAIL", 
                             f"Expected {team_name} roster to be a dict, got {type(roster)}", data)
                return
            
            roster_fields = ["players", "key_players", "total_players"]
            missing_roster_fields = [field for field in roster_fields if field not in roster]
            if missing_roster_fields:
                self.log_test("Matchup API", "FAIL", 
                             f"Missing {team_name} roster fields: {missing_roster_fields}", data)
                return
        
        # Check injuries are arrays
        home_injuries = home_team.get("injuries", [])
        away_injuries = away_team.get("injuries", [])
        
        if not isinstance(home_injuries, list):
            self.log_test("Matchup API", "FAIL", 
                         f"Expected home_team injuries to be a list, got {type(home_injuries)}", data)
            return
        
        if not isinstance(away_injuries, list):
            self.log_test("Matchup API", "FAIL", 
                         f"Expected away_team injuries to be a list, got {type(away_injuries)}", data)
            return
        
        # Check if we have REAL injury data (should contain real player names)
        all_injuries = home_injuries + away_injuries
        real_injury_indicators = ["Joel Embiid", "Paul George", "Out", "Day-To-Day", "Questionable", "Probable"]
        has_real_injuries = any(
            any(indicator in str(injury).replace("-", " ") for indicator in real_injury_indicators)
            for injury in all_injuries
        )
        
        # Check lineup_status is valid
        valid_lineup_statuses = ["not_available", "projected", "confirmed"]
        lineup_status = data.get("lineup_status")
        if lineup_status not in valid_lineup_statuses:
            self.log_test("Matchup API", "FAIL", 
                         f"Expected lineup_status to be one of {valid_lineup_statuses}, got '{lineup_status}'", data)
            return
        
        # Check commence_time is ISO format
        commence_time = data.get("commence_time", "")
        if not commence_time or "T" not in commence_time:
            self.log_test("Matchup API", "FAIL", 
                         f"Expected commence_time in ISO format, got '{commence_time}'", data)
            return
        
        injury_status = "with REAL injuries" if has_real_injuries else "no real injuries found"
        self.log_test("Matchup API", "PASS", 
                     f"Event: {data.get('event_id')}, Teams: {home_team.get('name')} vs {away_team.get('name')}, "
                     f"Lineup: {lineup_status}, Injuries: {len(all_injuries)} ({injury_status})")
    
    async def test_starting_lineup_endpoint(self):
        """Test GET /api/starting-lineup/{event_id}?sport_key=basketball_nba with event_id 401810535"""
        event_id = "401810535"
        status_code, data = await self.make_request("GET", f"/starting-lineup/{event_id}?sport_key=basketball_nba")
        
        if status_code != 200:
            self.log_test("Starting Lineup API", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Check required fields
        required_fields = ["home", "away", "lineup_status", "message", "event_id"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("Starting Lineup API", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check event_id matches
        if data.get("event_id") != event_id:
            self.log_test("Starting Lineup API", "FAIL", 
                         f"Expected event_id '{event_id}', got '{data.get('event_id')}'", data)
            return
        
        # Check home and away structure
        home = data.get("home", {})
        away = data.get("away", {})
        
        for team_name, team_data in [("home", home), ("away", away)]:
            if not isinstance(team_data, dict):
                self.log_test("Starting Lineup API", "FAIL", 
                             f"Expected {team_name} to be a dict, got {type(team_data)}", data)
                return
            
            team_fields = ["team", "starters", "confirmed"]
            missing_team_fields = [field for field in team_fields if field not in team_data]
            if missing_team_fields:
                self.log_test("Starting Lineup API", "FAIL", 
                             f"Missing {team_name} fields: {missing_team_fields}", data)
                return
            
            # Check starters is a list
            starters = team_data.get("starters", [])
            if not isinstance(starters, list):
                self.log_test("Starting Lineup API", "FAIL", 
                             f"Expected {team_name} starters to be a list, got {type(starters)}", data)
                return
            
            # Check confirmed is a boolean
            confirmed = team_data.get("confirmed")
            if not isinstance(confirmed, bool):
                self.log_test("Starting Lineup API", "FAIL", 
                             f"Expected {team_name} confirmed to be a boolean, got {type(confirmed)}", data)
                return
        
        # Check lineup_status is valid
        valid_lineup_statuses = ["not_available", "projected", "confirmed"]
        lineup_status = data.get("lineup_status")
        if lineup_status not in valid_lineup_statuses:
            self.log_test("Starting Lineup API", "FAIL", 
                         f"Expected lineup_status to be one of {valid_lineup_statuses}, got '{lineup_status}'", data)
            return
        
        # Check message is a string
        message = data.get("message", "")
        if not isinstance(message, str):
            self.log_test("Starting Lineup API", "FAIL", 
                         f"Expected message to be a string, got {type(message)}", data)
            return
        
        home_starters_count = len(home.get("starters", []))
        away_starters_count = len(away.get("starters", []))
        
        self.log_test("Starting Lineup API", "PASS", 
                     f"Event: {event_id}, Home: {home.get('team')} ({home_starters_count} starters, confirmed: {home.get('confirmed')}), "
                     f"Away: {away.get('team')} ({away_starters_count} starters, confirmed: {away.get('confirmed')}), "
                     f"Status: {lineup_status}")
    
    async def test_roster_endpoint(self):
        """Test GET /api/roster/{team_name}?sport_key=basketball_nba with Philadelphia 76ers"""
        team_name = "Philadelphia%2076ers"  # URL encoded
        status_code, data = await self.make_request("GET", f"/roster/{team_name}?sport_key=basketball_nba")
        
        if status_code != 200:
            self.log_test("Roster API", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Check required fields
        required_fields = ["team", "sport_key", "players", "injuries", "key_players", "total_players"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("Roster API", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check team name contains "Philadelphia 76ers"
        team = data.get("team", "")
        if "Philadelphia" not in team or "76ers" not in team:
            self.log_test("Roster API", "FAIL", 
                         f"Expected team to contain 'Philadelphia 76ers', got '{team}'", data)
            return
        
        # Check sport_key matches
        if data.get("sport_key") != "basketball_nba":
            self.log_test("Roster API", "FAIL", 
                         f"Expected sport_key 'basketball_nba', got '{data.get('sport_key')}'", data)
            return
        
        # Check players is an array
        players = data.get("players", [])
        if not isinstance(players, list):
            self.log_test("Roster API", "FAIL", 
                         f"Expected players to be a list, got {type(players)}", data)
            return
        
        # Check injuries is an array
        injuries = data.get("injuries", [])
        if not isinstance(injuries, list):
            self.log_test("Roster API", "FAIL", 
                         f"Expected injuries to be a list, got {type(injuries)}", data)
            return
        
        # Check key_players is an array
        key_players = data.get("key_players", [])
        if not isinstance(key_players, list):
            self.log_test("Roster API", "FAIL", 
                         f"Expected key_players to be a list, got {type(key_players)}", data)
            return
        
        # Check total_players is a number > 0
        total_players = data.get("total_players", 0)
        if not isinstance(total_players, (int, float)) or total_players <= 0:
            self.log_test("Roster API", "FAIL", 
                         f"Expected total_players to be a number > 0, got {total_players}", data)
            return
        
        # Check if we have REAL injury data (should contain real player names and statuses)
        real_injury_indicators = ["Joel Embiid", "Paul George", "Out", "Day-To-Day", "Questionable", "Probable"]
        has_real_injuries = any(
            any(indicator in str(injury).replace("-", " ") for indicator in real_injury_indicators)
            for injury in injuries
        )
        
        # Check if we have real player names (not just mock data)
        has_real_players = len(players) > 0 and any(
            len(str(player).split()) >= 2  # Real names typically have first and last name
            for player in players[:5]  # Check first 5 players
        )
        
        injury_status = "with REAL injuries" if has_real_injuries else "no real injuries found"
        player_status = "with real player names" if has_real_players else "no real player names"
        
        self.log_test("Roster API", "PASS", 
                     f"Team: {team}, Players: {total_players} ({player_status}), "
                     f"Injuries: {len(injuries)} ({injury_status}), Key players: {len(key_players)}")
    
    # ==================== ADAPTIVE LEARNING TESTS ====================
    def __init__(self):
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, status: str, details: str = "", response_data: Any = None):
        """Log test result"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "response_data": response_data
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
        if response_data and status == "FAIL":
            print(f"   Response: {json.dumps(response_data, indent=2)}")
        print()
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> tuple[int, Dict]:
        """Make HTTP request and return status code and response data"""
        url = f"{BASE_URL}{endpoint}"
        try:
            async with self.session.request(method, url, **kwargs) as response:
                status_code = response.status
                try:
                    data = await response.json()
                except:
                    data = {"error": "Invalid JSON response", "text": await response.text()}
                return status_code, data
        except Exception as e:
            return 0, {"error": str(e)}
    
    async def test_adaptive_learning_status(self):
        """Test GET /api/adaptive-learning/status"""
        status_code, data = await self.make_request("GET", "/adaptive-learning/status")
        
        if status_code != 200:
            self.log_test("Adaptive Learning Status", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Check required fields
        required_fields = ["status", "model_performance", "current_weights_by_sport"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("Adaptive Learning Status", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check status is "active"
        if data.get("status") != "active":
            self.log_test("Adaptive Learning Status", "FAIL", 
                         f"Expected status 'active', got '{data.get('status')}'", data)
            return
        
        # Check sports weights structure
        weights_by_sport = data.get("current_weights_by_sport", {})
        expected_sports = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl"]
        expected_models = ["elo_model", "context_model", "line_movement_model", "statistical_model", "psychology_model"]
        
        for sport in expected_sports:
            if sport not in weights_by_sport:
                self.log_test("Adaptive Learning Status", "FAIL", 
                             f"Missing sport weights for {sport}", data)
                return
            
            sport_weights = weights_by_sport[sport]
            for model in expected_models:
                if model not in sport_weights:
                    self.log_test("Adaptive Learning Status", "FAIL", 
                                 f"Missing {model} weight for {sport}", data)
                    return
        
        self.log_test("Adaptive Learning Status", "PASS", 
                     f"Status: {data.get('status')}, Sports: {list(weights_by_sport.keys())}")
    
    async def test_model_stats_by_sport(self):
        """Test GET /api/adaptive-learning/model-stats/basketball_nba"""
        status_code, data = await self.make_request("GET", "/adaptive-learning/model-stats/basketball_nba")
        
        if status_code != 200:
            self.log_test("Model Stats by Sport", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Check required fields
        required_fields = ["sport_key", "model_stats", "current_weights", "best_performer", "worst_performer"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("Model Stats by Sport", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check sport_key matches
        if data.get("sport_key") != "basketball_nba":
            self.log_test("Model Stats by Sport", "FAIL", 
                         f"Expected sport_key 'basketball_nba', got '{data.get('sport_key')}'", data)
            return
        
        # Check current_weights has all 5 models
        current_weights = data.get("current_weights", {})
        expected_models = ["elo_model", "context_model", "line_movement_model", "statistical_model", "psychology_model"]
        
        for model in expected_models:
            if model not in current_weights:
                self.log_test("Model Stats by Sport", "FAIL", 
                             f"Missing {model} in current_weights", data)
                return
        
        self.log_test("Model Stats by Sport", "PASS", 
                     f"Sport: {data.get('sport_key')}, Models: {list(current_weights.keys())}")
    
    async def test_rolling_performance(self):
        """Test GET /api/adaptive-learning/rolling-performance/elo_model?sport_key=basketball_nba&days=30"""
        status_code, data = await self.make_request("GET", "/adaptive-learning/rolling-performance/elo_model?sport_key=basketball_nba&days=30")
        
        if status_code != 200:
            self.log_test("Rolling Performance", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Handle case where no data exists yet (newly initialized system)
        if "error" in data and data.get("error") == "No data found":
            self.log_test("Rolling Performance", "PASS", 
                         "No historical data yet (newly initialized system)", data)
            return
        
        # Check required fields for normal response
        required_fields = ["model_name", "sport_key", "window_days"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("Rolling Performance", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check values match request
        if data.get("model_name") != "elo_model":
            self.log_test("Rolling Performance", "FAIL", 
                         f"Expected model_name 'elo_model', got '{data.get('model_name')}'", data)
            return
        
        if data.get("sport_key") != "basketball_nba":
            self.log_test("Rolling Performance", "FAIL", 
                         f"Expected sport_key 'basketball_nba', got '{data.get('sport_key')}'", data)
            return
        
        if data.get("window_days") != 30:
            self.log_test("Rolling Performance", "FAIL", 
                         f"Expected window_days 30, got {data.get('window_days')}", data)
            return
        
        # Should have either data or "No results in window" message
        has_data = "total_in_window" in data and data.get("total_in_window", 0) > 0
        has_message = "message" in data and "No results in window" in data.get("message", "")
        
        if not (has_data or has_message):
            self.log_test("Rolling Performance", "FAIL", 
                         "Expected either data with total_in_window > 0 or 'No results in window' message", data)
            return
        
        result_type = "with data" if has_data else "no results in window"
        self.log_test("Rolling Performance", "PASS", 
                     f"Model: {data.get('model_name')}, Sport: {data.get('sport_key')}, Days: {data.get('window_days')}, Result: {result_type}")
    
    async def test_calibration_report(self):
        """Test GET /api/adaptive-learning/calibration"""
        status_code, data = await self.make_request("GET", "/adaptive-learning/calibration")
        
        if status_code != 200:
            self.log_test("Calibration Report", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Check required fields
        required_fields = ["sport_key", "calibration"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("Calibration Report", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check sport_key is "all"
        if data.get("sport_key") != "all":
            self.log_test("Calibration Report", "FAIL", 
                         f"Expected sport_key 'all', got '{data.get('sport_key')}'", data)
            return
        
        # Check calibration structure
        calibration = data.get("calibration", {})
        expected_cal_fields = ["buckets", "brier_score", "interpretation", "total_samples"]
        
        for field in expected_cal_fields:
            if field not in calibration:
                self.log_test("Calibration Report", "FAIL", 
                             f"Missing {field} in calibration object", data)
                return
        
        # Check buckets is an array
        if not isinstance(calibration.get("buckets"), list):
            self.log_test("Calibration Report", "FAIL", 
                         "Expected 'buckets' to be an array", data)
            return
        
        self.log_test("Calibration Report", "PASS", 
                     f"Sport: {data.get('sport_key')}, Buckets: {len(calibration.get('buckets', []))}, Total samples: {calibration.get('total_samples', 0)}")
    
    async def test_lr_weights(self):
        """Test GET /api/adaptive-learning/lr-weights/basketball_nba"""
        status_code, data = await self.make_request("GET", "/adaptive-learning/lr-weights/basketball_nba")
        
        if status_code != 200:
            self.log_test("LR Weights", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Check required fields
        required_fields = ["sport_key", "weights", "updates_count", "status", "note"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("LR Weights", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check sport_key matches
        if data.get("sport_key") != "basketball_nba":
            self.log_test("LR Weights", "FAIL", 
                         f"Expected sport_key 'basketball_nba', got '{data.get('sport_key')}'", data)
            return
        
        # Check weights structure
        weights = data.get("weights", {})
        expected_weight_fields = ["elo_diff", "form_diff", "margin_diff", "rest_advantage", 
                                "home_advantage", "injury_impact", "line_movement", 
                                "four_factors", "intercept"]
        
        for field in expected_weight_fields:
            if field not in weights:
                self.log_test("LR Weights", "FAIL", 
                             f"Missing {field} in weights object", data)
                return
        
        # Check updates_count is a number
        updates_count = data.get("updates_count")
        if not isinstance(updates_count, (int, float)):
            self.log_test("LR Weights", "FAIL", 
                         f"Expected updates_count to be a number, got {type(updates_count)}", data)
            return
        
        # Check status is valid
        status = data.get("status")
        if status not in ["using_defaults", "learned"]:
            self.log_test("LR Weights", "FAIL", 
                         f"Expected status 'using_defaults' or 'learned', got '{status}'", data)
            return
        
        # Check note mentions online learning
        note = data.get("note", "")
        if "online learning" not in note.lower():
            self.log_test("LR Weights", "FAIL", 
                         f"Expected note to mention 'online learning', got '{note}'", data)
            return
        
        self.log_test("LR Weights", "PASS", 
                     f"Sport: {data.get('sport_key')}, Updates: {updates_count}, Status: {status}")
    
    async def run_all_tests(self):
        """Run all tests - Matchup/Lineup/Roster + Adaptive Learning"""
        print("🏀 TESTING MATCHUP, LINEUP & ROSTER API ENDPOINTS")
        print("=" * 60)
        print()
        
        # Test new endpoints first
        await self.test_matchup_endpoint()
        await self.test_starting_lineup_endpoint()
        await self.test_roster_endpoint()
        
        print()
        print("🧠 TESTING ADAPTIVE LEARNING SYSTEM ENDPOINTS")
        print("=" * 60)
        print()
        
        # Test adaptive learning endpoints
        await self.test_adaptive_learning_status()
        await self.test_model_stats_by_sport()
        await self.test_rolling_performance()
        await self.test_calibration_report()
        await self.test_lr_weights()
        
        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = len([r for r in self.test_results if r["status"] == "PASS"])
        failed = len([r for r in self.test_results if r["status"] == "FAIL"])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        print()
        
        if failed > 0:
            print("FAILED TESTS:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"❌ {result['test']}: {result['details']}")
            print()
        
        return passed, failed, total

async def main():
    """Main test runner"""
    async with BetPredictorTester() as tester:
        passed, failed, total = await tester.run_all_tests()
        
        if failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {failed} out of {total} tests failed")
        
        return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)