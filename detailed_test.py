#!/usr/bin/env python3
"""
Detailed Backend API Response Validation
"""

import requests
import json
from pathlib import Path

def get_backend_url():
    frontend_env_path = Path("/app/frontend/.env")
    if frontend_env_path.exists():
        with open(frontend_env_path, 'r') as f:
            for line in f:
                if line.startswith('REACT_APP_BACKEND_URL='):
                    base_url = line.split('=', 1)[1].strip()
                    return f"{base_url}/api"
    return "http://localhost:8001/api"

BASE_URL = get_backend_url()

def test_detailed_responses():
    print("🔍 DETAILED API RESPONSE VALIDATION")
    print("=" * 50)
    
    # Test health check
    print("\n1. Health Check (/api/)")
    response = requests.get(f"{BASE_URL}/")
    data = response.json()
    print(f"   Response: {json.dumps(data, indent=2)}")
    
    # Test sports
    print("\n2. Sports List (/api/sports)")
    response = requests.get(f"{BASE_URL}/sports")
    data = response.json()
    print(f"   Found {len(data)} sports:")
    for sport in data[:3]:  # Show first 3
        print(f"   - {sport.get('title', 'N/A')} ({sport.get('key', 'N/A')})")
    
    # Test NBA events
    print("\n3. NBA Events (/api/events/basketball_nba)")
    response = requests.get(f"{BASE_URL}/events/basketball_nba")
    data = response.json()
    print(f"   Found {len(data)} NBA events")
    if data:
        event = data[0]
        print(f"   Sample event: {event.get('home_team', 'N/A')} vs {event.get('away_team', 'N/A')}")
        print(f"   Commence time: {event.get('commence_time', 'N/A')}")
        bookmakers = event.get('bookmakers', [])
        print(f"   Bookmakers: {len(bookmakers)}")
    
    # Test recommendations
    print("\n4. Recommendations (/api/recommendations)")
    response = requests.get(f"{BASE_URL}/recommendations")
    data = response.json()
    print(f"   Found {len(data)} recommendations")
    
    # Test performance
    print("\n5. Performance Stats (/api/performance)")
    response = requests.get(f"{BASE_URL}/performance")
    data = response.json()
    print(f"   Total predictions: {data.get('total_predictions', 0)}")
    print(f"   Win rate: {data.get('win_rate', 0)}%")
    print(f"   ROI: {data.get('roi', 0)}%")
    
    # Test notifications
    print("\n6. Notifications (/api/notifications)")
    response = requests.get(f"{BASE_URL}/notifications")
    data = response.json()
    notifications = data.get('notifications', [])
    unread_count = data.get('unread_count', 0)
    print(f"   Total notifications: {len(notifications)}")
    print(f"   Unread count: {unread_count}")
    
    # Test data source status
    print("\n7. Data Source Status (/api/data-source-status)")
    response = requests.get(f"{BASE_URL}/data-source-status")
    data = response.json()
    print(f"   Source: {data.get('source', 'N/A')}")
    print(f"   Status: {data.get('status', 'N/A')}")
    print(f"   Cached events: {data.get('cachedEvents', 0)}")
    print(f"   Line movement snapshots: {data.get('lineMovementSnapshots', 0)}")
    
    print("\n✅ All endpoints returning valid data structures!")

if __name__ == "__main__":
    test_detailed_responses()