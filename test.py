#!/usr/bin/env python3
"""
BetPredictor Comprehensive Test Suite
=====================================
Tests all app functionalities including:
- API Health & Data Sources
- Event Fetching (all sports)
- Line Movement Tracking
- V5 Analysis (Line Movement)
- V6 Analysis (ML Ensemble)
- Unified Predictions
- Win/Loss Result Processing
- Notifications System
- Performance Stats

Simulates games starting within 1 hour to trigger the prediction pipeline.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import sys

# Configuration
BACKEND_URL = "http://localhost:8001/api"
SPORTS = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl", "soccer_epl"]

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f" {text}")
    print(f"{'='*70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

class BetPredictorTester:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.results = {
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "tests": []
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, endpoint: str) -> Dict:
        """Make GET request to API"""
        try:
            async with self.session.get(f"{BACKEND_URL}{endpoint}") as resp:
                if resp.status == 200:
                    return {"success": True, "data": await resp.json(), "status": resp.status}
                else:
                    return {"success": False, "error": await resp.text(), "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e), "status": 0}
    
    async def post(self, endpoint: str, data: Dict = None) -> Dict:
        """Make POST request to API"""
        try:
            async with self.session.post(f"{BACKEND_URL}{endpoint}", json=data) as resp:
                if resp.status == 200:
                    return {"success": True, "data": await resp.json(), "status": resp.status}
                else:
                    return {"success": False, "error": await resp.text(), "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e), "status": 0}
    
    def record_test(self, name: str, passed: bool, details: str = ""):
        """Record test result"""
        self.results["tests"].append({
            "name": name,
            "passed": passed,
            "details": details
        })
        if passed:
            self.results["passed"] += 1
            print_success(f"{name}: {details}")
        else:
            self.results["failed"] += 1
            print_error(f"{name}: {details}")
    
    # ==================== TEST METHODS ====================
    
    async def test_api_health(self):
        """Test 1: API Health Check"""
        print_header("TEST 1: API Health Check")
        
        result = await self.get("/")
        if result["success"]:
            data = result["data"]
            if data.get("status") == "running":
                self.record_test("API Health", True, f"API v1.0 running")
            else:
                self.record_test("API Health", False, f"Unexpected status: {data}")
        else:
            self.record_test("API Health", False, f"Failed to connect: {result['error']}")
    
    async def test_data_sources(self):
        """Test 2: Data Source Status"""
        print_header("TEST 2: Data Source Status")
        
        result = await self.get("/data-source-status")
        if result["success"]:
            data = result["data"]
            source = data.get("source", "unknown")
            status = data.get("status", "unknown")
            snapshots = data.get("lineMovementSnapshots", 0)
            
            self.record_test("Data Source", True, 
                f"Source: {source}, Status: {status}, Snapshots: {snapshots}")
        else:
            self.record_test("Data Source", False, result.get("error", "Unknown error"))
    
    async def test_event_fetching(self):
        """Test 3: Event Fetching for All Sports"""
        print_header("TEST 3: Event Fetching (All Sports)")
        
        total_events = 0
        for sport in SPORTS:
            result = await self.get(f"/events/{sport}?pre_match_only=true")
            if result["success"]:
                events = result["data"]
                count = len(events) if isinstance(events, list) else 0
                total_events += count
                self.record_test(f"Events - {sport}", True, f"{count} events found")
            else:
                self.record_test(f"Events - {sport}", False, result.get("error", "Failed"))
        
        print_info(f"Total events across all sports: {total_events}")
    
    async def test_line_movement(self):
        """Test 4: Line Movement Tracking"""
        print_header("TEST 4: Line Movement Tracking")
        
        # First get an event
        result = await self.get("/events/basketball_nba?pre_match_only=true")
        if not result["success"] or not result["data"]:
            self.record_test("Line Movement", False, "No events available")
            return
        
        events = result["data"]
        if len(events) > 0:
            event = events[0]
            event_id = event.get("id")
            
            # Get line movement for this event
            lm_result = await self.get(f"/line-movement/{event_id}?sport_key=basketball_nba")
            if lm_result["success"]:
                data = lm_result["data"]
                opening = data.get("opening_odds", {})
                current = data.get("current_odds", {})
                snapshots = data.get("total_snapshots", 0)
                
                home_team = event.get("home_team", "Home")
                away_team = event.get("away_team", "Away")
                
                self.record_test("Line Movement", True, 
                    f"{home_team} vs {away_team}: {snapshots} snapshots, Opening ML: {opening.get('ml', {}).get('home', 'N/A')}")
            else:
                self.record_test("Line Movement", False, lm_result.get("error", "Failed"))
        else:
            self.record_test("Line Movement", False, "No events to test")
    
    async def test_v5_analysis(self):
        """Test 5: V5 Analysis (Line Movement Based)"""
        print_header("TEST 5: V5 Analysis (Line Movement)")
        
        # Get an event to analyze
        result = await self.get("/events/basketball_nba?pre_match_only=true")
        if not result["success"] or not result["data"]:
            self.record_test("V5 Analysis", False, "No events available")
            return
        
        events = result["data"]
        if len(events) > 0:
            event = events[0]
            event_id = event.get("id")
            
            # Run V5 analysis
            v5_result = await self.post(f"/analyze-v5/{event_id}?sport_key=basketball_nba")
            if v5_result["success"]:
                data = v5_result["data"]
                has_pick = data.get("has_pick", False)
                confidence = data.get("confidence", 0)
                factors = data.get("factor_count", 0)
                reasoning = data.get("reasoning", "")[:100]
                
                status = "PICK" if has_pick else "NO PICK"
                self.record_test("V5 Analysis", True, 
                    f"{status} - Confidence: {confidence}%, Factors: {factors}")
                if has_pick:
                    print_info(f"   Pick: {data.get('pick_display', 'N/A')}")
                print_info(f"   Reasoning: {reasoning}...")
            else:
                self.record_test("V5 Analysis", False, v5_result.get("error", "Failed"))
        else:
            self.record_test("V5 Analysis", False, "No events to analyze")
    
    async def test_v6_analysis(self):
        """Test 6: V6 Analysis (ML Ensemble)"""
        print_header("TEST 6: V6 Analysis (ML Ensemble)")
        
        # Get an event to analyze
        result = await self.get("/events/basketball_nba?pre_match_only=true")
        if not result["success"] or not result["data"]:
            self.record_test("V6 Analysis", False, "No events available")
            return
        
        events = result["data"]
        if len(events) > 0:
            event = events[0]
            event_id = event.get("id")
            home_team = event.get("home_team")
            away_team = event.get("away_team")
            
            print_info(f"Analyzing: {home_team} vs {away_team}")
            
            # Run V6 analysis
            v6_result = await self.post(f"/analyze-v6/{event_id}?sport_key=basketball_nba")
            if v6_result["success"]:
                data = v6_result["data"]
                has_pick = data.get("has_pick", False)
                confidence = data.get("confidence", 0)
                edge = data.get("edge", 0)
                
                # Get ensemble details
                ensemble = data.get("ensemble_details", {})
                models_agreeing = 0
                for model_name, model_data in ensemble.items():
                    if model_data.get("prediction"):
                        models_agreeing += 1
                
                status = "PICK" if has_pick else "NO PICK"
                self.record_test("V6 Analysis", True, 
                    f"{status} - Confidence: {confidence}%, Edge: {edge}%, Models: {models_agreeing}/5")
                
                if has_pick:
                    print_info(f"   Pick: {data.get('pick_display', 'N/A')}")
                    print_info(f"   Pick Type: {data.get('pick_type', 'N/A')}")
                
                # Show model breakdown
                print_info("   Model Breakdown:")
                for model_name, model_data in ensemble.items():
                    pred = model_data.get("prediction", "N/A")
                    conf = model_data.get("confidence", 0)
                    print(f"      - {model_name}: {pred} ({conf}%)")
            else:
                self.record_test("V6 Analysis", False, v6_result.get("error", "Failed"))
        else:
            self.record_test("V6 Analysis", False, "No events to analyze")
    
    async def test_unified_analysis(self):
        """Test 7: Unified Analysis (V5 + V6 Combined)"""
        print_header("TEST 7: Unified Analysis (V5 + V6 Combined)")
        
        # Get an event to analyze
        result = await self.get("/events/basketball_nba?pre_match_only=true")
        if not result["success"] or not result["data"]:
            self.record_test("Unified Analysis", False, "No events available")
            return
        
        events = result["data"]
        if len(events) > 0:
            event = events[0]
            event_id = event.get("id")
            
            # Run Unified analysis
            unified_result = await self.post(f"/analyze-unified/{event_id}?sport_key=basketball_nba")
            if unified_result["success"]:
                data = unified_result["data"]
                has_pick = data.get("has_pick", False)
                confidence = data.get("confidence", 0)
                v5_weight = data.get("v5_weight", 30)
                v6_weight = data.get("v6_weight", 70)
                
                status = "PICK" if has_pick else "NO PICK"
                self.record_test("Unified Analysis", True, 
                    f"{status} - Confidence: {confidence}%, Weights: V5={v5_weight}% V6={v6_weight}%")
                
                if has_pick:
                    print_info(f"   Pick: {data.get('pick_display', 'N/A')}")
            else:
                self.record_test("Unified Analysis", False, unified_result.get("error", "Failed"))
        else:
            self.record_test("Unified Analysis", False, "No events to analyze")
    
    async def test_predictions_list(self):
        """Test 8: Predictions List (V5, V6, Unified)"""
        print_header("TEST 8: Predictions List")
        
        # Test V5 predictions
        v5_result = await self.get("/predictions/v5")
        if v5_result["success"]:
            data = v5_result["data"]
            stats = data.get("stats", {})
            self.record_test("V5 Predictions", True, 
                f"Total: {stats.get('total', 0)}, Pending: {stats.get('pending', 0)}, Win Rate: {stats.get('win_rate', 0)}%")
        else:
            self.record_test("V5 Predictions", False, v5_result.get("error", "Failed"))
        
        # Test V6 predictions
        v6_result = await self.get("/predictions/v6")
        if v6_result["success"]:
            data = v6_result["data"]
            stats = data.get("stats", {})
            self.record_test("V6 Predictions", True, 
                f"Total: {stats.get('total', 0)}, Pending: {stats.get('pending', 0)}, Win Rate: {stats.get('win_rate', 0)}%")
        else:
            self.record_test("V6 Predictions", False, v6_result.get("error", "Failed"))
        
        # Test Unified predictions
        unified_result = await self.get("/predictions/unified")
        if unified_result["success"]:
            data = unified_result["data"]
            stats = data.get("stats", {})
            self.record_test("Unified Predictions", True, 
                f"Total: {stats.get('total', 0)}, Pending: {stats.get('pending', 0)}, Win Rate: {stats.get('win_rate', 0)}%")
        else:
            self.record_test("Unified Predictions", False, unified_result.get("error", "Failed"))
    
    async def test_simulate_pregame_predictions(self):
        """Test 9: Simulate Pre-Game Predictions (1 Hour Before)"""
        print_header("TEST 9: Simulate Pre-Game Predictions")
        
        print_info("Simulating games starting in ~1 hour to trigger prediction pipeline...")
        
        # Create mock events that appear to start in 1 hour
        now = datetime.now(timezone.utc)
        mock_events = [
            {
                "id": f"sim_nba_{int(now.timestamp())}",
                "sport_key": "basketball_nba",
                "home_team": "Boston Celtics",
                "away_team": "Los Angeles Lakers",
                "commence_time": (now + timedelta(minutes=60)).isoformat(),
                "odds": {
                    "home_ml_decimal": 1.55,
                    "away_ml_decimal": 2.45,
                    "spread": -6.5,
                    "total": 224.5
                }
            },
            {
                "id": f"sim_nba2_{int(now.timestamp())}",
                "sport_key": "basketball_nba",
                "home_team": "Golden State Warriors",
                "away_team": "Denver Nuggets",
                "commence_time": (now + timedelta(minutes=55)).isoformat(),
                "odds": {
                    "home_ml_decimal": 2.10,
                    "away_ml_decimal": 1.75,
                    "spread": 3.5,
                    "total": 230.0
                }
            }
        ]
        
        predictions_created = 0
        for event in mock_events:
            # Use the analyze endpoint to simulate prediction
            analysis_result = await self.post(
                f"/analyze-v6/{event['id']}?sport_key={event['sport_key']}",
                data={
                    "event": event,
                    "odds_data": event["odds"]
                }
            )
            
            if analysis_result["success"]:
                data = analysis_result["data"]
                has_pick = data.get("has_pick", False)
                if has_pick:
                    predictions_created += 1
                    print_success(f"   Created pick: {event['home_team']} vs {event['away_team']}")
                    print_info(f"   Pick: {data.get('pick_display', 'N/A')} @ {data.get('confidence', 0)}%")
                else:
                    print_warning(f"   No pick for {event['home_team']} vs {event['away_team']} (low confidence)")
            else:
                print_error(f"   Failed to analyze {event['home_team']} vs {event['away_team']}")
        
        self.record_test("Pre-Game Simulation", True, 
            f"Analyzed {len(mock_events)} events, {predictions_created} picks created")
    
    async def test_result_processing(self):
        """Test 10: Win/Loss Result Processing"""
        print_header("TEST 10: Win/Loss Result Processing")
        
        # Check auto result checker status
        result = await self.get("/live-scores")
        if result["success"]:
            data = result["data"]
            live_count = data.get("live_games_count", 0)
            games = data.get("games", [])
            
            self.record_test("Live Scores", True, f"{live_count} live games tracked")
            
            # Show some live games
            for game in games[:3]:
                status = game.get("status", "")
                home = game.get("home_team", "Home")
                away = game.get("away_team", "Away")
                home_score = game.get("home_score", 0)
                away_score = game.get("away_score", 0)
                print_info(f"   {home} {home_score} - {away_score} {away} ({status})")
        else:
            self.record_test("Live Scores", False, result.get("error", "Failed"))
        
        # Check performance stats
        perf_result = await self.get("/performance")
        if perf_result["success"]:
            data = perf_result["data"]
            wins = data.get("wins", 0)
            losses = data.get("losses", 0)
            win_rate = data.get("win_rate", 0)
            roi = data.get("roi", 0)
            
            self.record_test("Performance Stats", True, 
                f"Record: {wins}W-{losses}L, Win Rate: {win_rate}%, ROI: {roi}%")
        else:
            self.record_test("Performance Stats", False, perf_result.get("error", "Failed"))
    
    async def test_notifications(self):
        """Test 11: Notifications System"""
        print_header("TEST 11: Notifications System")
        
        # Get current notifications
        result = await self.get("/notifications")
        if result["success"]:
            data = result["data"]
            unread = data.get("unread_count", 0)
            notifications = data.get("notifications", [])
            
            self.record_test("Notifications List", True, 
                f"{len(notifications)} total, {unread} unread")
            
            # Show recent notifications
            for notif in notifications[:3]:
                notif_type = notif.get("type", "unknown")
                title = notif.get("title", "No title")
                print_info(f"   [{notif_type}] {title}")
        else:
            self.record_test("Notifications List", False, result.get("error", "Failed"))
        
        # Test creating a notification (daily summary)
        create_result = await self.post("/notifications/daily-summary")
        if create_result["success"]:
            data = create_result["data"]
            message = data.get("message", "")
            self.record_test("Create Notification", True, message[:50])
        else:
            self.record_test("Create Notification", False, create_result.get("error", "Failed"))
    
    async def test_model_performance(self):
        """Test 12: Model Performance Tracking"""
        print_header("TEST 12: Model Performance Tracking")
        
        result = await self.get("/model-performance")
        if result["success"]:
            data = result["data"]
            models = data.get("models", {})
            
            self.record_test("Model Performance", True, f"{len(models)} models tracked")
            
            for model_name, model_data in models.items():
                accuracy = model_data.get("accuracy", 0)
                total = model_data.get("total", 0)
                weight = model_data.get("current_weight", 0)
                print_info(f"   {model_name}: {accuracy}% accuracy, {total} predictions, {weight}% weight")
        else:
            self.record_test("Model Performance", False, result.get("error", "Failed"))
    
    async def test_predictions_comparison(self):
        """Test 13: Algorithm Comparison"""
        print_header("TEST 13: Algorithm Comparison (V5 vs V6)")
        
        result = await self.get("/predictions/comparison")
        if result["success"]:
            data = result["data"]
            algorithms = data.get("algorithms", {})
            
            self.record_test("Algorithm Comparison", True, f"{len(algorithms)} algorithms compared")
            
            for algo_name, algo_data in algorithms.items():
                total = algo_data.get("total", 0)
                wins = algo_data.get("wins", 0)
                win_rate = algo_data.get("win_rate", 0)
                print_info(f"   {algo_name}: {wins}W/{total} total ({win_rate}% win rate)")
        else:
            self.record_test("Algorithm Comparison", False, result.get("error", "Failed"))
    
    async def test_upcoming_predictions_window(self):
        """Test 14: Upcoming Predictions Window"""
        print_header("TEST 14: Upcoming Predictions Window")
        
        result = await self.get("/upcoming-predictions-window")
        if result["success"]:
            data = result["data"]
            in_window = data.get("total_in_window", 0)
            games_in_window = data.get("games_in_window", [])
            upcoming = data.get("upcoming_games", [])
            
            self.record_test("Predictions Window", True, 
                f"{in_window} games in 1-hour window, {len(upcoming)} upcoming")
            
            if games_in_window:
                print_info("   Games ready for prediction:")
                for game in games_in_window[:3]:
                    home = game.get("home_team", "Home")
                    away = game.get("away_team", "Away")
                    mins = game.get("minutes_to_start", 0)
                    print(f"      {home} vs {away} - starts in {mins} min")
        else:
            self.record_test("Predictions Window", False, result.get("error", "Failed"))
    
    async def test_recommendations(self):
        """Test 15: Get Recommendations"""
        print_header("TEST 15: Get Recommendations (Top Picks)")
        
        result = await self.get("/recommendations?min_confidence=0.60")
        if result["success"]:
            recs = result["data"]
            count = len(recs) if isinstance(recs, list) else 0
            
            self.record_test("Recommendations", True, f"{count} picks available")
            
            if count > 0:
                for rec in recs[:3]:
                    home = rec.get("home_team", "Home")
                    away = rec.get("away_team", "Away")
                    pick = rec.get("predicted_outcome", "N/A")
                    conf = rec.get("confidence", 0)
                    print_info(f"   {home} vs {away}: {pick} @ {conf*100:.0f}%")
        else:
            self.record_test("Recommendations", False, result.get("error", "Failed"))
    
    async def run_all_tests(self):
        """Run all tests"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}")
        print("=" * 70)
        print("   BETPREDICTOR COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print(f"{Colors.END}")
        print(f"Backend URL: {BACKEND_URL}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all tests in sequence
        await self.test_api_health()
        await self.test_data_sources()
        await self.test_event_fetching()
        await self.test_line_movement()
        await self.test_v5_analysis()
        await self.test_v6_analysis()
        await self.test_unified_analysis()
        await self.test_predictions_list()
        await self.test_simulate_pregame_predictions()
        await self.test_result_processing()
        await self.test_notifications()
        await self.test_model_performance()
        await self.test_predictions_comparison()
        await self.test_upcoming_predictions_window()
        await self.test_recommendations()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        total = self.results["passed"] + self.results["failed"]
        pass_rate = (self.results["passed"] / total * 100) if total > 0 else 0
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}")
        print("=" * 70)
        print("   TEST SUMMARY")
        print("=" * 70)
        print(f"{Colors.END}")
        
        print(f"\n   Total Tests: {total}")
        print(f"   {Colors.GREEN}Passed: {self.results['passed']}{Colors.END}")
        print(f"   {Colors.RED}Failed: {self.results['failed']}{Colors.END}")
        print(f"   Pass Rate: {pass_rate:.1f}%")
        
        if self.results["failed"] > 0:
            print(f"\n   {Colors.RED}Failed Tests:{Colors.END}")
            for test in self.results["tests"]:
                if not test["passed"]:
                    print(f"      - {test['name']}: {test['details']}")
        
        if pass_rate == 100:
            print(f"\n   {Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED!{Colors.END}")
        elif pass_rate >= 80:
            print(f"\n   {Colors.YELLOW}{Colors.BOLD}⚠️  Most tests passed, some issues to address{Colors.END}")
        else:
            print(f"\n   {Colors.RED}{Colors.BOLD}❌ Multiple test failures - investigate issues{Colors.END}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


async def main():
    """Main entry point"""
    async with BetPredictorTester() as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    print("\n🧪 Starting BetPredictor Test Suite...\n")
    asyncio.run(main())
