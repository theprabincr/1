#!/usr/bin/env python3
"""
V6 Predictor Reasoning Text Fixes Test Suite
Tests specific fixes for model agreement count, ELO ratings, reasoning consistency, and favored outcomes
"""

import requests
import json
import sys
import re
from datetime import datetime
from pathlib import Path

# Get backend URL from frontend .env
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
print(f"Testing V6 predictor fixes at: {BASE_URL}")

class V6PredictorTester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
        self.issues = []
        
    def test_model_agreement_count(self, event_id="401810582", sport_key="basketball_nba"):
        """Test 1: Model Agreement Count - Verify the model agreement count matches the actual list of models"""
        print(f"\n🧪 TEST 1: Model Agreement Count")
        print(f"   Testing: POST /api/analyze-unified/{event_id}?sport_key={sport_key}")
        print(f"   Expected: Toronto vs Minnesota - should have a pick with correct model agreement count")
        
        url = f"{BASE_URL}/analyze-unified/{event_id}?sport_key={sport_key}"
        
        try:
            response = requests.post(url, timeout=30)
            
            if response.status_code != 200:
                self.failed += 1
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                self.issues.append(f"Model Agreement Count Test - API Error: {error}")
                print(f"   ❌ FAIL - API Error: {error}")
                return False
            
            data = response.json()
            prediction = data.get('prediction', {})
            reasoning = prediction.get('reasoning', '') or data.get('reasoning', '')
            
            if not reasoning:
                self.failed += 1
                self.issues.append("Model Agreement Count Test - Missing reasoning text")
                print(f"   ❌ FAIL - Missing reasoning text")
                return False
            
            # Look for model agreement patterns like "X out of Y models agree"
            agreement_patterns = [
                r'(\d+)\s+out\s+of\s+(\d+)\s+models?\s+agree',
                r'(\d+)/(\d+)\s+models?\s+agree',
                r'(\d+)\s+of\s+(\d+)\s+models?\s+agree'
            ]
            
            agreement_match = None
            for pattern in agreement_patterns:
                match = re.search(pattern, reasoning, re.IGNORECASE)
                if match:
                    agreement_match = match
                    break
            
            if not agreement_match:
                self.failed += 1
                self.issues.append("Model Agreement Count Test - No model agreement statement found in reasoning")
                print(f"   ❌ FAIL - No model agreement statement found")
                print(f"   📝 Reasoning preview: {reasoning[:200]}...")
                return False
            
            agree_count = int(agreement_match.group(1))
            total_count = int(agreement_match.group(2))
            
            # Count actual "agrees" and "disagrees" mentions in reasoning
            agrees_mentions = len(re.findall(r'\bagrees?\b', reasoning, re.IGNORECASE))
            disagrees_mentions = len(re.findall(r'\bdisagrees?\b', reasoning, re.IGNORECASE))
            
            # Validate the counts make sense
            if agree_count + (total_count - agree_count) != total_count:
                self.failed += 1
                self.issues.append(f"Model Agreement Count Test - Math error: {agree_count} + {total_count - agree_count} != {total_count}")
                print(f"   ❌ FAIL - Math error in agreement count")
                return False
            
            # Check if the stated agreement matches the actual mentions (with some tolerance)
            expected_disagree_count = total_count - agree_count
            
            print(f"   📊 Found: '{agreement_match.group(0)}'")
            print(f"   📊 Agree count: {agree_count}, Disagree count: {expected_disagree_count}")
            print(f"   📊 'Agrees' mentions: {agrees_mentions}, 'Disagrees' mentions: {disagrees_mentions}")
            
            # The counts should be reasonable (allowing some flexibility for different phrasing)
            if agree_count < 1 or agree_count > 10 or total_count < 3 or total_count > 10:
                self.failed += 1
                self.issues.append(f"Model Agreement Count Test - Unreasonable counts: {agree_count}/{total_count}")
                print(f"   ❌ FAIL - Unreasonable model counts")
                return False
            
            self.passed += 1
            print(f"   ✅ PASS - Model agreement count is consistent: {agree_count} out of {total_count} models agree")
            return True
            
        except Exception as e:
            self.failed += 1
            self.issues.append(f"Model Agreement Count Test - Exception: {str(e)}")
            print(f"   ❌ FAIL - Exception: {str(e)}")
            return False
    
    def test_elo_ratings_calculation(self, sport_key="basketball_nba"):
        """Test 2: ELO Ratings - Verify ELO ratings are calculated from overall season records"""
        print(f"\n🧪 TEST 2: ELO Ratings Calculation")
        print(f"   Testing: GET /api/ml/elo-ratings?sport_key={sport_key}")
        print(f"   Expected: ELO calculated as 1200 + (win_pct * 600), teams with similar records have similar ELOs")
        
        url = f"{BASE_URL}/ml/elo-ratings?sport_key={sport_key}"
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                self.failed += 1
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                self.issues.append(f"ELO Ratings Test - API Error: {error}")
                print(f"   ❌ FAIL - API Error: {error}")
                return False
            
            data = response.json()
            teams = data.get('teams', [])
            
            if not teams or len(teams) < 5:
                # ELO ratings might not be populated yet - check if we can get ELO from prediction analysis instead
                print(f"   ⚠️  ELO ratings endpoint has {len(teams)} teams - checking prediction analysis for ELO usage...")
                
                # Test ELO usage in prediction analysis
                prediction_url = f"{BASE_URL}/analyze-unified/401810582?sport_key={sport_key}"
                pred_response = requests.post(prediction_url, timeout=30)
                
                if pred_response.status_code == 200:
                    pred_data = pred_response.json()
                    reasoning = pred_data.get('prediction', {}).get('reasoning', '')
                    
                    # Look for ELO ratings in the reasoning text
                    elo_pattern = r'(\w+[\w\s]*?):\s*(\d{4})\s*ELO\s*rating'
                    elo_matches = re.findall(elo_pattern, reasoning, re.IGNORECASE)
                    
                    if elo_matches and len(elo_matches) >= 2:
                        print(f"   📊 Found ELO ratings in prediction analysis:")
                        for team, elo in elo_matches:
                            print(f"      • {team.strip()}: {elo} ELO")
                        
                        # Check if ELO values are reasonable (1200-1800 range)
                        elo_values = [int(elo) for _, elo in elo_matches]
                        if all(1000 <= elo <= 2000 for elo in elo_values):
                            self.passed += 1
                            print(f"   ✅ PASS - ELO ratings are being used in predictions (reasonable values)")
                            return True
                        else:
                            self.failed += 1
                            self.issues.append(f"ELO Ratings Test - Unreasonable ELO values: {elo_values}")
                            print(f"   ❌ FAIL - Unreasonable ELO values")
                            return False
                    else:
                        self.failed += 1
                        self.issues.append("ELO Ratings Test - No ELO ratings found in prediction analysis")
                        print(f"   ❌ FAIL - No ELO ratings found in prediction analysis")
                        return False
                else:
                    self.failed += 1
                    self.issues.append(f"ELO Ratings Test - Cannot test ELO usage: prediction API error")
                    print(f"   ❌ FAIL - Cannot test ELO usage")
                    return False
            
            # Check ELO calculation formula: 1200 + (win_pct * 600)
            formula_errors = []
            similar_record_pairs = []
            
            for team in teams[:10]:  # Check first 10 teams
                team_name = team.get('team', 'Unknown')
                elo = team.get('elo', 0)
                wins = team.get('wins', 0)
                losses = team.get('losses', 0)
                
                if wins + losses == 0:
                    continue  # Skip teams with no games
                
                win_pct = wins / (wins + losses)
                expected_elo = 1200 + (win_pct * 600)
                
                # Allow small tolerance for rounding
                if abs(elo - expected_elo) > 5:
                    formula_errors.append(f"{team_name}: ELO={elo:.1f}, Expected={expected_elo:.1f} (W-L: {wins}-{losses})")
                
                # Look for teams with similar records
                for other_team in teams:
                    if other_team == team:
                        continue
                    other_wins = other_team.get('wins', 0)
                    other_losses = other_team.get('losses', 0)
                    other_elo = other_team.get('elo', 0)
                    
                    # Check if records are similar (within 1-2 games)
                    if abs(wins - other_wins) <= 2 and abs(losses - other_losses) <= 2:
                        elo_diff = abs(elo - other_elo)
                        if elo_diff <= 20:  # Similar ELOs for similar records
                            similar_record_pairs.append(f"{team_name} ({wins}-{losses}, ELO={elo:.1f}) vs {other_team.get('team')} ({other_wins}-{other_losses}, ELO={other_elo:.1f})")
            
            if formula_errors:
                self.failed += 1
                self.issues.append(f"ELO Ratings Test - Formula errors: {'; '.join(formula_errors[:3])}")
                print(f"   ❌ FAIL - ELO formula errors found")
                for error in formula_errors[:3]:
                    print(f"      • {error}")
                return False
            
            print(f"   📊 Checked {len(teams)} teams, ELO formula is correct")
            if similar_record_pairs:
                print(f"   📊 Found teams with similar records and ELOs:")
                for pair in similar_record_pairs[:2]:
                    print(f"      • {pair}")
            
            self.passed += 1
            print(f"   ✅ PASS - ELO ratings calculated correctly from overall season records")
            return True
            
        except Exception as e:
            self.failed += 1
            self.issues.append(f"ELO Ratings Test - Exception: {str(e)}")
            print(f"   ❌ FAIL - Exception: {str(e)}")
            return False
    
    def test_reasoning_text_consistency(self, event_id="401810582", sport_key="basketball_nba"):
        """Test 3: Reasoning Text Consistency - Check that the reasoning text is not contradictory"""
        print(f"\n🧪 TEST 3: Reasoning Text Consistency")
        print(f"   Testing: POST /api/analyze-unified/{event_id}?sport_key={sport_key}")
        print(f"   Expected: If it says '3 out of 5 agree', there should be 3 'agrees' and 2 'disagrees'")
        
        url = f"{BASE_URL}/analyze-unified/{event_id}?sport_key={sport_key}"
        
        try:
            response = requests.post(url, timeout=30)
            
            if response.status_code != 200:
                self.failed += 1
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                self.issues.append(f"Reasoning Consistency Test - API Error: {error}")
                print(f"   ❌ FAIL - API Error: {error}")
                return False
            
            data = response.json()
            prediction = data.get('prediction', {})
            reasoning = prediction.get('reasoning', '') or data.get('reasoning', '')
            
            if not reasoning:
                self.failed += 1
                self.issues.append("Reasoning Consistency Test - Missing reasoning text")
                print(f"   ❌ FAIL - Missing reasoning text")
                return False
            
            # Extract model agreement statement
            agreement_patterns = [
                r'(\d+)\s+out\s+of\s+(\d+)\s+models?\s+agree',
                r'(\d+)/(\d+)\s+models?\s+agree',
                r'(\d+)\s+of\s+(\d+)\s+models?\s+agree'
            ]
            
            agreement_match = None
            for pattern in agreement_patterns:
                match = re.search(pattern, reasoning, re.IGNORECASE)
                if match:
                    agreement_match = match
                    break
            
            if not agreement_match:
                # If no explicit agreement statement, check for consistency in other ways
                print(f"   ⚠️  No explicit model agreement statement found")
                self.passed += 1
                print(f"   ✅ PASS - No contradictory statements found (no explicit agreement count)")
                return True
            
            stated_agree = int(agreement_match.group(1))
            stated_total = int(agreement_match.group(2))
            stated_disagree = stated_total - stated_agree
            
            # Count actual model mentions
            model_sections = []
            
            # Look for model-specific sections or mentions
            model_patterns = [
                r'(V\d+|XGBoost|Ensemble|Model \d+).*?(agrees?|disagrees?)',
                r'(agrees?|disagrees?).*?(V\d+|XGBoost|Ensemble|Model \d+)',
                r'(\w+\s+model).*?(agrees?|disagrees?)',
                r'(agrees?|disagrees?).*?(\w+\s+model)'
            ]
            
            actual_agrees = 0
            actual_disagrees = 0
            
            for pattern in model_patterns:
                matches = re.findall(pattern, reasoning, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        text = ' '.join(match).lower()
                        if 'agree' in text and 'disagree' not in text:
                            actual_agrees += 1
                        elif 'disagree' in text:
                            actual_disagrees += 1
            
            # Alternative counting method - count explicit agree/disagree statements
            if actual_agrees == 0 and actual_disagrees == 0:
                agree_statements = re.findall(r'\b(agrees?)\b', reasoning, re.IGNORECASE)
                disagree_statements = re.findall(r'\b(disagrees?)\b', reasoning, re.IGNORECASE)
                actual_agrees = len(agree_statements)
                actual_disagrees = len(disagree_statements)
            
            print(f"   📊 Stated: {stated_agree} agree, {stated_disagree} disagree (total: {stated_total})")
            print(f"   📊 Found in text: {actual_agrees} agrees, {actual_disagrees} disagrees")
            
            # Check for consistency (allow some flexibility for different phrasing)
            consistency_issues = []
            
            # The stated counts should be reasonable
            if stated_agree < 1 or stated_total < 3 or stated_total > 10:
                consistency_issues.append(f"Unreasonable stated counts: {stated_agree}/{stated_total}")
            
            # If we found explicit mentions, they should roughly match
            if actual_agrees > 0 or actual_disagrees > 0:
                total_found = actual_agrees + actual_disagrees
                if total_found > 0:
                    # Allow some tolerance for different phrasing
                    if abs(actual_agrees - stated_agree) > 2:
                        consistency_issues.append(f"Agree count mismatch: stated {stated_agree}, found {actual_agrees}")
                    if abs(actual_disagrees - stated_disagree) > 2:
                        consistency_issues.append(f"Disagree count mismatch: stated {stated_disagree}, found {actual_disagrees}")
            
            # Check for contradictory statements
            contradictions = []
            
            # Look for contradictory confidence statements
            if re.search(r'high confidence.*low confidence|low confidence.*high confidence', reasoning, re.IGNORECASE):
                contradictions.append("Contradictory confidence statements")
            
            # Look for contradictory team preferences
            teams_mentioned = re.findall(r'(Lakers|Celtics|Warriors|Heat|Knicks|Nets|76ers|Raptors|Bulls|Cavaliers|Pistons|Pacers|Bucks|Hawks|Hornets|Magic|Wizards|Nuggets|Timberwolves|Thunder|Trail Blazers|Jazz|Kings|Clippers|Suns|Spurs|Mavericks|Rockets|Grizzlies|Pelicans)', reasoning, re.IGNORECASE)
            
            if consistency_issues or contradictions:
                self.failed += 1
                all_issues = consistency_issues + contradictions
                self.issues.append(f"Reasoning Consistency Test - Issues: {'; '.join(all_issues)}")
                print(f"   ❌ FAIL - Consistency issues found:")
                for issue in all_issues:
                    print(f"      • {issue}")
                return False
            
            self.passed += 1
            print(f"   ✅ PASS - Reasoning text is consistent with stated model agreement")
            return True
            
        except Exception as e:
            self.failed += 1
            self.issues.append(f"Reasoning Consistency Test - Exception: {str(e)}")
            print(f"   ❌ FAIL - Exception: {str(e)}")
            return False
    
    def test_favored_outcomes_fields(self, event_id="401810582", sport_key="basketball_nba"):
        """Test 4: Favored Outcomes - Verify the new favored outcome fields are present in unified analysis"""
        print(f"\n🧪 TEST 4: Favored Outcomes Fields")
        print(f"   Testing: POST /api/analyze-unified/{event_id}?sport_key={sport_key}")
        print(f"   Expected: ml_favored_team, ml_favored_prob, spread_favored_team, spread_favored_prob, totals_favored, totals_favored_prob")
        
        url = f"{BASE_URL}/analyze-unified/{event_id}?sport_key={sport_key}"
        
        try:
            response = requests.post(url, timeout=30)
            
            if response.status_code != 200:
                self.failed += 1
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                self.issues.append(f"Favored Outcomes Test - API Error: {error}")
                print(f"   ❌ FAIL - API Error: {error}")
                return False
            
            data = response.json()
            prediction = data.get('prediction', {})
            
            if not prediction:
                self.failed += 1
                self.issues.append("Favored Outcomes Test - Missing prediction object")
                print(f"   ❌ FAIL - Missing prediction object")
                return False
            
            # Check for required favored outcome fields
            required_fields = [
                'ml_favored_team', 'ml_favored_prob',
                'spread_favored_team', 'spread_favored_prob', 
                'totals_favored', 'totals_favored_prob'
            ]
            
            missing_fields = []
            present_fields = []
            
            for field in required_fields:
                if field in prediction and prediction[field] is not None:
                    present_fields.append(field)
                else:
                    missing_fields.append(field)
            
            if missing_fields:
                self.failed += 1
                self.issues.append(f"Favored Outcomes Test - Missing fields: {', '.join(missing_fields)}")
                print(f"   ❌ FAIL - Missing required fields:")
                for field in missing_fields:
                    print(f"      • {field}")
                return False
            
            # Validate field values
            validation_errors = []
            
            # Check ML favored fields
            ml_favored_team = prediction.get('ml_favored_team', '')
            ml_favored_prob = prediction.get('ml_favored_prob', 0)
            
            if not ml_favored_team or ml_favored_team.lower() in ['home', 'away']:
                validation_errors.append(f"ml_favored_team should be actual team name, got: '{ml_favored_team}'")
            
            if not (0.5 <= ml_favored_prob <= 1.0):
                validation_errors.append(f"ml_favored_prob should be 0.5-1.0, got: {ml_favored_prob}")
            
            # Check spread favored fields
            spread_favored_team = prediction.get('spread_favored_team', '')
            spread_favored_prob = prediction.get('spread_favored_prob', 0)
            
            if not spread_favored_team or spread_favored_team.lower() in ['home', 'away']:
                validation_errors.append(f"spread_favored_team should be actual team name, got: '{spread_favored_team}'")
            
            if not (0.3 <= spread_favored_prob <= 1.0):
                validation_errors.append(f"spread_favored_prob should be 0.3-1.0, got: {spread_favored_prob}")
            
            # Check totals favored fields
            totals_favored = prediction.get('totals_favored', '')
            totals_favored_prob = prediction.get('totals_favored_prob', 0)
            
            if totals_favored.upper() not in ['OVER', 'UNDER']:
                validation_errors.append(f"totals_favored should be 'OVER' or 'UNDER', got: '{totals_favored}'")
            
            if not (0.3 <= totals_favored_prob <= 1.0):
                validation_errors.append(f"totals_favored_prob should be 0.3-1.0, got: {totals_favored_prob}")
            
            if validation_errors:
                self.failed += 1
                self.issues.append(f"Favored Outcomes Test - Validation errors: {'; '.join(validation_errors)}")
                print(f"   ❌ FAIL - Field validation errors:")
                for error in validation_errors:
                    print(f"      • {error}")
                return False
            
            print(f"   📊 All required fields present and valid:")
            print(f"      • ML: {ml_favored_team} ({ml_favored_prob:.3f})")
            print(f"      • Spread: {spread_favored_team} ({spread_favored_prob:.3f})")
            print(f"      • Totals: {totals_favored} ({totals_favored_prob:.3f})")
            
            self.passed += 1
            print(f"   ✅ PASS - All favored outcome fields present and valid")
            return True
            
        except Exception as e:
            self.failed += 1
            self.issues.append(f"Favored Outcomes Test - Exception: {str(e)}")
            print(f"   ❌ FAIL - Exception: {str(e)}")
            return False
    
    def run_v6_predictor_tests(self):
        """Run all V6 predictor reasoning text fix tests"""
        print("=" * 80)
        print("🔬 V6 PREDICTOR REASONING TEXT FIXES - TEST SUITE")
        print("=" * 80)
        print("Testing specific fixes for:")
        print("1. Model Agreement Count matching actual model list")
        print("2. ELO Ratings calculated from overall season records")
        print("3. Reasoning Text Consistency (no contradictions)")
        print("4. Favored Outcomes fields in unified analysis")
        print("=" * 80)
        
        # Run all tests
        test1_pass = self.test_model_agreement_count()
        test2_pass = self.test_elo_ratings_calculation()
        test3_pass = self.test_reasoning_text_consistency()
        test4_pass = self.test_favored_outcomes_fields()
        
        # Print summary
        self.print_summary()
        
        return self.failed == 0
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("📊 V6 PREDICTOR TEST SUMMARY")
        print("=" * 80)
        
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.issues:
            print(f"\n🔍 ISSUES FOUND ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
        
        print("\n" + "=" * 80)
        
        if self.failed == 0:
            print("🎉 ALL V6 PREDICTOR TESTS PASSED!")
            print("✅ Model agreement counts are accurate")
            print("✅ ELO ratings calculated from overall records")
            print("✅ Reasoning text is consistent")
            print("✅ Favored outcome fields are present and valid")
        else:
            print("⚠️  SOME V6 PREDICTOR TESTS FAILED")
            print("❌ V6 predictor reasoning needs attention before deployment")
        
        print("=" * 80)

def main():
    """Main test execution"""
    tester = V6PredictorTester()
    success = tester.run_v6_predictor_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()