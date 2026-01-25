"""
Test suite for Line Movement feature in BetPredictor app
Tests: Events API, Line Movement API, Sport selector, Bookmakers
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestEventsAPI:
    """Test events endpoint for different sports"""
    
    def test_events_basketball_nba(self):
        """Test NBA events endpoint returns events with bookmakers"""
        response = requests.get(f"{BASE_URL}/api/events/basketball_nba")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list), "Events should be a list"
        
        if len(data) > 0:
            event = data[0]
            assert "id" in event, "Event should have id"
            assert "home_team" in event, "Event should have home_team"
            assert "away_team" in event, "Event should have away_team"
            assert "bookmakers" in event, "Event should have bookmakers"
            assert len(event["bookmakers"]) > 0, "Event should have at least one bookmaker"
            
            # Check bookmaker structure
            bookmaker = event["bookmakers"][0]
            assert "key" in bookmaker, "Bookmaker should have key"
            assert "title" in bookmaker, "Bookmaker should have title"
            assert "markets" in bookmaker, "Bookmaker should have markets"
            
            # Check market structure
            if len(bookmaker["markets"]) > 0:
                market = bookmaker["markets"][0]
                assert "key" in market, "Market should have key"
                assert "outcomes" in market, "Market should have outcomes"
                assert len(market["outcomes"]) >= 2, "Market should have at least 2 outcomes"
                
                # Check outcome structure
                outcome = market["outcomes"][0]
                assert "name" in outcome, "Outcome should have name"
                assert "price" in outcome, "Outcome should have price"
                assert 1.0 < outcome["price"] < 100, f"Price should be valid odds: {outcome['price']}"
    
    def test_events_americanfootball_nfl(self):
        """Test NFL events endpoint"""
        response = requests.get(f"{BASE_URL}/api/events/americanfootball_nfl")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list), "Events should be a list"
    
    def test_events_baseball_mlb(self):
        """Test MLB events endpoint"""
        response = requests.get(f"{BASE_URL}/api/events/baseball_mlb")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list), "Events should be a list"
    
    def test_events_icehockey_nhl(self):
        """Test NHL events endpoint"""
        response = requests.get(f"{BASE_URL}/api/events/icehockey_nhl")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list), "Events should be a list"
    
    def test_events_soccer_epl(self):
        """Test EPL events endpoint"""
        response = requests.get(f"{BASE_URL}/api/events/soccer_epl")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list), "Events should be a list"


class TestLineMovementAPI:
    """Test line movement endpoint"""
    
    @pytest.fixture
    def nba_event_id(self):
        """Get first NBA event ID for testing"""
        response = requests.get(f"{BASE_URL}/api/events/basketball_nba")
        if response.status_code == 200 and len(response.json()) > 0:
            return response.json()[0]["id"]
        pytest.skip("No NBA events available for testing")
    
    def test_line_movement_returns_data(self, nba_event_id):
        """Test line movement endpoint returns proper structure"""
        response = requests.get(f"{BASE_URL}/api/line-movement/{nba_event_id}?sport_key=basketball_nba")
        assert response.status_code == 200
        
        data = response.json()
        assert "event_id" in data, "Response should have event_id"
        assert data["event_id"] == nba_event_id, "Event ID should match"
        
        # Check event_info
        assert "event_info" in data, "Response should have event_info"
        if data["event_info"]:
            assert "home_team" in data["event_info"], "event_info should have home_team"
            assert "away_team" in data["event_info"], "event_info should have away_team"
        
        # Check opening_odds
        assert "opening_odds" in data, "Response should have opening_odds"
        
        # Check current_odds
        assert "current_odds" in data, "Response should have current_odds"
        if data["current_odds"]:
            assert "home" in data["current_odds"], "current_odds should have home"
            assert "away" in data["current_odds"], "current_odds should have away"
        
        # Check bookmakers
        assert "bookmakers" in data, "Response should have bookmakers"
        assert isinstance(data["bookmakers"], list), "bookmakers should be a list"
        
        # Check chart_data
        assert "chart_data" in data, "Response should have chart_data"
        assert isinstance(data["chart_data"], list), "chart_data should be a list"
        
        # Check total_snapshots
        assert "total_snapshots" in data, "Response should have total_snapshots"
    
    def test_line_movement_chart_data_structure(self, nba_event_id):
        """Test chart data has proper structure for graphing"""
        response = requests.get(f"{BASE_URL}/api/line-movement/{nba_event_id}?sport_key=basketball_nba")
        assert response.status_code == 200
        
        data = response.json()
        chart_data = data.get("chart_data", [])
        
        if len(chart_data) > 0:
            point = chart_data[0]
            assert "timestamp" in point, "Chart point should have timestamp"
            assert "home_odds" in point, "Chart point should have home_odds"
            assert "away_odds" in point, "Chart point should have away_odds"
            
            # Validate odds are reasonable
            if point["home_odds"]:
                assert 1.0 < point["home_odds"] < 100, f"home_odds should be valid: {point['home_odds']}"
            if point["away_odds"]:
                assert 1.0 < point["away_odds"] < 100, f"away_odds should be valid: {point['away_odds']}"
    
    def test_line_movement_bookmakers_structure(self, nba_event_id):
        """Test bookmakers data has proper structure"""
        response = requests.get(f"{BASE_URL}/api/line-movement/{nba_event_id}?sport_key=basketball_nba")
        assert response.status_code == 200
        
        data = response.json()
        bookmakers = data.get("bookmakers", [])
        
        if len(bookmakers) > 0:
            bm = bookmakers[0]
            assert "bookmaker" in bm, "Bookmaker should have bookmaker key"
            assert "bookmaker_title" in bm, "Bookmaker should have bookmaker_title"
            assert "snapshots" in bm, "Bookmaker should have snapshots"
            
            if len(bm["snapshots"]) > 0:
                snap = bm["snapshots"][0]
                assert "timestamp" in snap, "Snapshot should have timestamp"
                assert "home_odds" in snap, "Snapshot should have home_odds"
                assert "away_odds" in snap, "Snapshot should have away_odds"
    
    def test_line_movement_invalid_event(self):
        """Test line movement with invalid event ID"""
        response = requests.get(f"{BASE_URL}/api/line-movement/INVALID_EVENT_ID?sport_key=basketball_nba")
        assert response.status_code == 200  # Returns empty data, not 404
        
        data = response.json()
        assert data["event_id"] == "INVALID_EVENT_ID"
        # Should have empty or null data for invalid event
        assert data.get("event_info") is None or data.get("chart_data") == []


class TestSportsAPI:
    """Test sports listing endpoint"""
    
    def test_sports_list(self):
        """Test sports endpoint returns list of sports"""
        response = requests.get(f"{BASE_URL}/api/sports")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list), "Sports should be a list"
        assert len(data) >= 5, "Should have at least 5 sports"
        
        # Check required sports are present
        sport_keys = [s["key"] for s in data]
        assert "basketball_nba" in sport_keys, "NBA should be in sports list"
        assert "americanfootball_nfl" in sport_keys, "NFL should be in sports list"
        assert "baseball_mlb" in sport_keys, "MLB should be in sports list"
        assert "icehockey_nhl" in sport_keys, "NHL should be in sports list"
        assert "soccer_epl" in sport_keys, "EPL should be in sports list"
        
        # Check sport structure
        sport = data[0]
        assert "key" in sport, "Sport should have key"
        assert "title" in sport, "Sport should have title"
        assert "active" in sport, "Sport should have active flag"


class TestScraperStatus:
    """Test scraper status endpoint"""
    
    def test_scraper_status(self):
        """Test scraper status endpoint"""
        response = requests.get(f"{BASE_URL}/api/scraper-status")
        assert response.status_code == 200
        
        data = response.json()
        assert "source" in data, "Status should have source"
        assert data["source"] == "oddsportal", "Source should be oddsportal"
        assert "status" in data, "Status should have status field"
        assert "cachedEvents" in data, "Status should have cachedEvents"


class TestMultipleBookmakers:
    """Test that multiple bookmakers are returned from OddsPortal"""
    
    def test_events_have_multiple_bookmakers(self):
        """Test that events have multiple bookmakers from OddsPortal"""
        response = requests.get(f"{BASE_URL}/api/events/basketball_nba")
        assert response.status_code == 200
        
        data = response.json()
        if len(data) > 0:
            event = data[0]
            bookmakers = event.get("bookmakers", [])
            
            # Should have multiple bookmakers
            assert len(bookmakers) >= 3, f"Should have at least 3 bookmakers, got {len(bookmakers)}"
            
            # Check bookmaker names
            bm_titles = [bm["title"] for bm in bookmakers]
            print(f"Found bookmakers: {bm_titles}")
            
            # Verify some expected bookmakers are present
            expected_bookmakers = ["bet365", "Pinnacle", "DraftKings", "FanDuel", "BetMGM", "Unibet", "Betway"]
            found_expected = [bm for bm in expected_bookmakers if bm in bm_titles]
            assert len(found_expected) >= 2, f"Should have at least 2 expected bookmakers, found: {found_expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
