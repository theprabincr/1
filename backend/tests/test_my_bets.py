"""
Test suite for BetPredictor My Bets feature
Tests: POST /api/my-bets, GET /api/my-bets, PUT /api/my-bets/{id}, DELETE /api/my-bets/{id}
"""
import pytest
import requests
import os
import uuid

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestMyBetsAPI:
    """Test My Bets CRUD operations"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data"""
        self.test_bet_ids = []
        yield
        # Cleanup: Delete test bets
        for bet_id in self.test_bet_ids:
            try:
                requests.delete(f"{BASE_URL}/api/my-bets/{bet_id}")
            except:
                pass
    
    def test_get_my_bets_returns_200(self):
        """GET /api/my-bets should return 200 and bets array"""
        response = requests.get(f"{BASE_URL}/api/my-bets")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "bets" in data, "Response should contain 'bets' key"
        assert isinstance(data["bets"], list), "bets should be a list"
        print(f"✓ GET /api/my-bets returned {len(data['bets'])} bets")
    
    def test_post_my_bet_creates_bet(self):
        """POST /api/my-bets should create a new bet"""
        test_bet = {
            "event_name": "TEST_Lakers vs Celtics",
            "selection": "Lakers ML",
            "stake": 50,
            "odds": 1.85,
            "result": "pending"
        }
        
        response = requests.post(f"{BASE_URL}/api/my-bets", json=test_bet)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert data.get("success") == True, "Response should have success=True"
        assert "bet" in data, "Response should contain 'bet' key"
        
        created_bet = data["bet"]
        assert created_bet.get("event_name") == test_bet["event_name"]
        assert created_bet.get("selection") == test_bet["selection"]
        assert created_bet.get("stake") == test_bet["stake"]
        assert created_bet.get("odds") == test_bet["odds"]
        assert "id" in created_bet, "Created bet should have an id"
        
        self.test_bet_ids.append(created_bet["id"])
        print(f"✓ POST /api/my-bets created bet with id: {created_bet['id']}")
        
        # Verify persistence with GET
        get_response = requests.get(f"{BASE_URL}/api/my-bets")
        assert get_response.status_code == 200
        bets = get_response.json().get("bets", [])
        found = any(b.get("id") == created_bet["id"] for b in bets)
        assert found, "Created bet should be found in GET response"
        print("✓ Verified bet persisted in database")
    
    def test_put_my_bet_updates_result(self):
        """PUT /api/my-bets/{id} should update bet result"""
        # First create a bet
        test_bet = {
            "event_name": "TEST_Heat vs Bulls",
            "selection": "Heat -3.5",
            "stake": 25,
            "odds": 1.91,
            "result": "pending"
        }
        
        create_response = requests.post(f"{BASE_URL}/api/my-bets", json=test_bet)
        assert create_response.status_code == 200
        bet_id = create_response.json()["bet"]["id"]
        self.test_bet_ids.append(bet_id)
        
        # Update the bet result
        update_data = {"result": "win"}
        update_response = requests.put(f"{BASE_URL}/api/my-bets/{bet_id}", json=update_data)
        assert update_response.status_code == 200, f"Expected 200, got {update_response.status_code}"
        
        data = update_response.json()
        assert data.get("success") == True
        print(f"✓ PUT /api/my-bets/{bet_id} updated result to 'win'")
        
        # Verify update persisted
        get_response = requests.get(f"{BASE_URL}/api/my-bets")
        bets = get_response.json().get("bets", [])
        updated_bet = next((b for b in bets if b.get("id") == bet_id), None)
        assert updated_bet is not None, "Updated bet should exist"
        assert updated_bet.get("result") == "win", "Result should be updated to 'win'"
        print("✓ Verified update persisted in database")
    
    def test_delete_my_bet_removes_bet(self):
        """DELETE /api/my-bets/{id} should remove the bet"""
        # First create a bet
        test_bet = {
            "event_name": "TEST_Nets vs Knicks",
            "selection": "Over 220.5",
            "stake": 30,
            "odds": 1.87
        }
        
        create_response = requests.post(f"{BASE_URL}/api/my-bets", json=test_bet)
        assert create_response.status_code == 200
        bet_id = create_response.json()["bet"]["id"]
        
        # Delete the bet
        delete_response = requests.delete(f"{BASE_URL}/api/my-bets/{bet_id}")
        assert delete_response.status_code == 200, f"Expected 200, got {delete_response.status_code}"
        
        data = delete_response.json()
        assert data.get("success") == True
        print(f"✓ DELETE /api/my-bets/{bet_id} succeeded")
        
        # Verify deletion
        get_response = requests.get(f"{BASE_URL}/api/my-bets")
        bets = get_response.json().get("bets", [])
        found = any(b.get("id") == bet_id for b in bets)
        assert not found, "Deleted bet should not be found"
        print("✓ Verified bet removed from database")


class TestHealthAndBasicEndpoints:
    """Test basic API health and endpoints"""
    
    def test_root_endpoint(self):
        """GET / should return 200 (basic connectivity check)"""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200, f"Root endpoint failed: {response.status_code}"
        print("✓ Root endpoint OK")
    
    def test_performance_endpoint(self):
        """GET /api/performance should return 200"""
        response = requests.get(f"{BASE_URL}/api/performance")
        assert response.status_code == 200, f"Performance endpoint failed: {response.status_code}"
        
        data = response.json()
        assert "total_predictions" in data
        assert "win_rate" in data
        print(f"✓ Performance endpoint OK - {data.get('total_predictions')} predictions, {data.get('win_rate')}% win rate")
    
    def test_recommendations_endpoint(self):
        """GET /api/recommendations should return 200"""
        response = requests.get(f"{BASE_URL}/api/recommendations")
        assert response.status_code == 200, f"Recommendations endpoint failed: {response.status_code}"
        
        data = response.json()
        assert isinstance(data, list), "Recommendations should be a list"
        print(f"✓ Recommendations endpoint OK - {len(data)} recommendations")
    
    def test_events_endpoint(self):
        """GET /api/events/basketball_nba should return 200"""
        response = requests.get(f"{BASE_URL}/api/events/basketball_nba")
        assert response.status_code == 200, f"Events endpoint failed: {response.status_code}"
        
        data = response.json()
        assert isinstance(data, list), "Events should be a list"
        print(f"✓ Events endpoint OK - {len(data)} NBA events")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
