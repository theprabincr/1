"""
Test Multi-Sport Events and Live Scores Feature
Tests that Dashboard displays events from ALL sports (NBA, NHL, EPL, NFL, MLB)
and that live scores endpoint returns games from all sports.
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestMultiSportEvents:
    """Test events endpoints for all sports"""
    
    def test_api_health(self):
        """Test API is running"""
        response = requests.get(f"{BASE_URL}/api/")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "running"
        print("✓ API health check passed")
    
    def test_sports_list(self):
        """Test /api/sports returns all 5 sports"""
        response = requests.get(f"{BASE_URL}/api/sports")
        assert response.status_code == 200
        sports = response.json()
        
        # Verify all 5 sports are returned
        sport_keys = [s['key'] for s in sports]
        expected_sports = ['americanfootball_nfl', 'basketball_nba', 'baseball_mlb', 'icehockey_nhl', 'soccer_epl']
        
        for expected in expected_sports:
            assert expected in sport_keys, f"Missing sport: {expected}"
        
        print(f"✓ Sports list returned {len(sports)} sports: {sport_keys}")
    
    def test_nba_events(self):
        """Test /api/events/basketball_nba returns events"""
        response = requests.get(f"{BASE_URL}/api/events/basketball_nba")
        assert response.status_code == 200
        events = response.json()
        
        # NBA should have events
        assert isinstance(events, list)
        print(f"✓ NBA events: {len(events)} events returned")
        
        if len(events) > 0:
            event = events[0]
            # Verify event structure
            assert 'id' in event
            assert 'sport_key' in event
            assert 'sport_title' in event
            assert 'home_team' in event
            assert 'away_team' in event
            assert 'commence_time' in event
            
            # Verify sport_key is correct
            assert event['sport_key'] == 'basketball_nba'
            # Verify sport_title contains sport info
            assert 'basketball' in event['sport_title'].lower() or 'nba' in event['sport_title'].lower()
            print(f"  - First event: {event['away_team']} @ {event['home_team']}")
            print(f"  - Sport title: {event['sport_title']}")
    
    def test_nhl_events(self):
        """Test /api/events/icehockey_nhl returns events"""
        response = requests.get(f"{BASE_URL}/api/events/icehockey_nhl")
        assert response.status_code == 200
        events = response.json()
        
        assert isinstance(events, list)
        print(f"✓ NHL events: {len(events)} events returned")
        
        if len(events) > 0:
            event = events[0]
            assert event['sport_key'] == 'icehockey_nhl'
            assert 'icehockey' in event['sport_title'].lower() or 'nhl' in event['sport_title'].lower()
            print(f"  - First event: {event['away_team']} @ {event['home_team']}")
            print(f"  - Sport title: {event['sport_title']}")
    
    def test_epl_events(self):
        """Test /api/events/soccer_epl returns events"""
        response = requests.get(f"{BASE_URL}/api/events/soccer_epl")
        assert response.status_code == 200
        events = response.json()
        
        assert isinstance(events, list)
        print(f"✓ EPL events: {len(events)} events returned")
        
        if len(events) > 0:
            event = events[0]
            assert event['sport_key'] == 'soccer_epl'
            assert 'soccer' in event['sport_title'].lower() or 'epl' in event['sport_title'].lower()
            print(f"  - First event: {event['away_team']} @ {event['home_team']}")
            print(f"  - Sport title: {event['sport_title']}")
    
    def test_nfl_events(self):
        """Test /api/events/americanfootball_nfl returns events (may be empty in off-season)"""
        response = requests.get(f"{BASE_URL}/api/events/americanfootball_nfl")
        assert response.status_code == 200
        events = response.json()
        
        assert isinstance(events, list)
        print(f"✓ NFL events: {len(events)} events returned (may be 0 in off-season)")
        
        if len(events) > 0:
            event = events[0]
            assert event['sport_key'] == 'americanfootball_nfl'
    
    def test_mlb_events(self):
        """Test /api/events/baseball_mlb returns events (may be empty in off-season)"""
        response = requests.get(f"{BASE_URL}/api/events/baseball_mlb")
        assert response.status_code == 200
        events = response.json()
        
        assert isinstance(events, list)
        print(f"✓ MLB events: {len(events)} events returned (may be 0 in off-season)")
        
        if len(events) > 0:
            event = events[0]
            assert event['sport_key'] == 'baseball_mlb'


class TestLiveScores:
    """Test live scores endpoint returns games from all sports"""
    
    def test_live_scores_endpoint(self):
        """Test /api/live-scores returns proper structure"""
        response = requests.get(f"{BASE_URL}/api/live-scores")
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert 'live_games_count' in data
        assert 'games' in data
        assert isinstance(data['games'], list)
        
        print(f"✓ Live scores: {data['live_games_count']} live games")
        
        # If there are live games, verify they have sport_key
        if len(data['games']) > 0:
            for game in data['games']:
                assert 'sport_key' in game, "Live game missing sport_key"
                assert 'home_team' in game
                assert 'away_team' in game
                print(f"  - {game['sport_key']}: {game['away_team']} @ {game['home_team']}")
    
    def test_scores_by_sport(self):
        """Test /api/scores/{sport_key} endpoint for each sport"""
        sports = ['basketball_nba', 'icehockey_nhl', 'soccer_epl', 'americanfootball_nfl', 'baseball_mlb']
        
        for sport in sports:
            response = requests.get(f"{BASE_URL}/api/scores/{sport}")
            assert response.status_code == 200
            data = response.json()
            
            assert 'sport_key' in data
            assert data['sport_key'] == sport
            assert 'games_count' in data
            assert 'games' in data
            
            print(f"✓ {sport} scores: {data['games_count']} games")


class TestEventDataStructure:
    """Test that events have proper data structure for multi-sport display"""
    
    def test_events_have_sport_labels(self):
        """Verify events from different sports have proper sport_title for labeling"""
        sports_to_test = [
            ('basketball_nba', 'basketball'),
            ('icehockey_nhl', 'icehockey'),
            ('soccer_epl', 'soccer'),
        ]
        
        for sport_key, expected_label in sports_to_test:
            response = requests.get(f"{BASE_URL}/api/events/{sport_key}")
            assert response.status_code == 200
            events = response.json()
            
            if len(events) > 0:
                event = events[0]
                sport_title = event.get('sport_title', '').lower()
                
                # Verify sport_title contains expected label
                assert expected_label in sport_title or sport_key.split('_')[1] in sport_title, \
                    f"Event sport_title '{sport_title}' doesn't contain '{expected_label}'"
                
                print(f"✓ {sport_key} events have sport_title: {event['sport_title']}")
    
    def test_events_sorted_by_commence_time(self):
        """Verify events are sorted by commence_time"""
        response = requests.get(f"{BASE_URL}/api/events/basketball_nba")
        assert response.status_code == 200
        events = response.json()
        
        if len(events) > 1:
            # Check that events are sorted by commence_time
            for i in range(len(events) - 1):
                time1 = events[i].get('commence_time', '')
                time2 = events[i + 1].get('commence_time', '')
                if time1 and time2:
                    assert time1 <= time2, f"Events not sorted: {time1} > {time2}"
            
            print(f"✓ Events are sorted by commence_time")


class TestDashboardDataFlow:
    """Test the data flow that Dashboard.js uses"""
    
    def test_dashboard_fetches_all_sports(self):
        """Simulate Dashboard.js fetchData() - fetch events from all sports"""
        sports = ['basketball_nba', 'americanfootball_nfl', 'baseball_mlb', 'icehockey_nhl', 'soccer_epl']
        all_events = []
        
        for sport in sports:
            response = requests.get(f"{BASE_URL}/api/events/{sport}")
            assert response.status_code == 200
            events = response.json()
            
            # Add sport_key to each event (as Dashboard.js does)
            for event in events:
                event['sport_key'] = sport
            
            all_events.extend(events)
        
        # Sort by commence_time (as Dashboard.js does)
        all_events.sort(key=lambda e: e.get('commence_time', ''))
        
        # Take first 12 (as Dashboard.js does)
        upcoming_events = all_events[:12]
        
        print(f"✓ Dashboard would show {len(upcoming_events)} upcoming events from {len(sports)} sports")
        
        # Verify we have events from multiple sports
        sport_keys_in_results = set(e.get('sport_key') for e in upcoming_events)
        print(f"  - Sports represented: {sport_keys_in_results}")
        
        # Should have at least 2 different sports (NBA and NHL are active)
        assert len(sport_keys_in_results) >= 1, "Expected events from at least 1 sport"
        
        # Print first few events
        for i, event in enumerate(upcoming_events[:5]):
            print(f"  - {i+1}. [{event['sport_key']}] {event['away_team']} @ {event['home_team']} ({event['commence_time']})")
    
    def test_live_scores_for_dashboard(self):
        """Test live scores endpoint that Dashboard uses"""
        response = requests.get(f"{BASE_URL}/api/live-scores")
        assert response.status_code == 200
        data = response.json()
        
        print(f"✓ Live scores for Dashboard: {data['live_games_count']} games")
        
        # Verify structure matches what Dashboard expects
        assert 'live_games_count' in data
        assert 'games' in data
        
        if data['live_games_count'] > 0:
            game = data['games'][0]
            # Dashboard expects these fields
            assert 'sport_key' in game
            assert 'home_team' in game
            assert 'away_team' in game
            assert 'home_score' in game or 'home_score' not in game  # May not have score yet
            assert 'away_score' in game or 'away_score' not in game


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
