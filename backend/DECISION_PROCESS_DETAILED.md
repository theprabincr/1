"""
COMPLETE STEP-BY-STEP DECISION PROCESS
Every Factor the Unified Predictor Considers to Make a Pick
"""

# UNIFIED PREDICTOR - COMPLETE DECISION FLOW

## STAGE 1: DATA COLLECTION (Pre-Analysis)

### 1.1 Team Data Collection
- **Team IDs:** ESPN team IDs for both teams
- **Season Records:** Wins, losses, win percentage
- **Recent Form:** Last 10 games with scores, margins, home/away
- **Roster Data:** Active players, positions, jersey numbers
- **Injury Reports:** Player name, position, status (Out/Doubtful/Questionable), injury type

### 1.2 Game Context Data
- **Commence Time:** Exact game start time (UTC)
- **Venue:** Home/away designation, city, state
- **Days Since Last Game:** For rest analysis
- **Last 7 Days Schedule:** Count of games played
- **Last 14 Days Schedule:** For congestion detection
- **Travel Distance:** City-to-city calculation (Haversine formula)

### 1.3 Odds & Line Movement Data
- **Opening Odds:** Moneyline, spread, total (from 3+ days before game)
- **Current Odds:** Latest moneyline, spread, total
- **Odds History:** All snapshots taken (every 5 minutes)
- **Bookmakers:** Multiple books if available (DraftKings, FanDuel, etc.)

---

## STAGE 2: V6 (ML ENSEMBLE) ANALYSIS

### PHASE 1: ADVANCED METRICS

#### 2.1 ELO Rating System
**Input:**
- Current team ELO ratings (1200-1800 range)
- Home advantage bonus (+50 to +100 based on sport)

**Calculation:**
```python
home_elo = get_team_elo(home_team) + home_advantage_bonus
away_elo = get_team_elo(away_team)
elo_diff = home_elo - away_elo

# Convert to win probability
home_prob_elo = 1 / (1 + 10^(-elo_diff/400))
```

**Decision Factor:**
- ELO diff > 150: Strong favorite identified
- ELO diff < 50: Skip (too close)

#### 2.2 NBA Four Factors (NBA Only)
**Input:**
- Recent game stats (last 7 games minimum)

**Calculation:**
```python
1. eFG% (Effective Field Goal): Points/possessions adjusted for 3PT
2. TOV% (Turnover Rate): Turnovers per 100 possessions
3. ORB% (Offensive Rebound Rate): % of available offensive boards
4. FT Rate: Free throws attempted per field goal attempt

four_factors_score = (
    (eFG% - 0.50) × 100 × 0.40 +    # 40% weight
    (0.14 - TOV%) × 100 × 0.25 +    # 25% weight (lower is better)
    (ORB% - 0.25) × 100 × 0.20 +    # 20% weight
    (FT_rate - 0.20) × 100 × 0.15   # 15% weight
) + 50  # Baseline
```

**Decision Factor:**
- Four Factors diff > 10: Strong advantage
- Considers offensive efficiency vs defensive weakness

#### 2.3 NFL Efficiency Metrics (NFL Only)
**Input:**
- Scoring averages, points allowed

**Calculation:**
```python
offensive_efficiency = 40 + ((avg_points - 24) / 20) × 30
defensive_efficiency = 40 + ((24 - avg_opp_points) / 20) × 30
overall_efficiency = (off_eff + def_eff) / 2
```

**Decision Factor:**
- Efficiency diff > 15: Clear advantage
- Combined with QB injury weight (2.0x multiplier)

#### 2.4 NHL Possession Metrics (NHL Only)
**Input:**
- Win rate, goal differential, recent performance

**Calculation:**
```python
possession_rating = 45 + (win_pct × 15) + (avg_margin × 3)
PDO = shooting% + save%  # Luck factor (1.000 = league avg)
```

**Decision Factor:**
- Possession diff > 10: Control advantage
- PDO > 1.020: Unsustainably lucky (fade)
- PDO < 0.980: Unlucky (value potential)

#### 2.5 Context Analysis
**Rest Days:**
```python
days_since_last_game = (game_time - last_game_time).days

if days_since == 0 or 1:
    impact = -0.08  # Back-to-back penalty
elif days_since >= 3:
    impact = +0.04  # Well-rested bonus
else:
    impact = 0.0    # Normal rest
```

**Travel Distance:**
```python
distance = haversine_distance(home_city, away_city)
timezone_change = (away_lon - home_lon) / 15  # Hours

if distance > 2000:  # Cross-country
    impact = -0.05
if abs(timezone_change) >= 3:
    impact -= 0.015 × abs(timezone_change)
```

**Altitude:**
```python
if home_city in ["Denver", "Utah"]:
    altitude_diff = home_altitude - away_altitude
    if altitude_diff > 3000:
        impact = +0.03  # Home advantage
```

**Schedule Congestion:**
```python
games_in_7_days = count_games_last_7_days(recent_games)

if games_in_7_days >= 4:  # NBA/NHL
    impact = -0.03  # Fatigue penalty
```

#### 2.6 Smart Injury Analysis
**For Each Injured Player:**
```python
# Position importance weight
position_weight = {
    "QB": 2.0,      # NFL
    "Goalie": 1.5,  # NHL
    "Point Guard": 1.1,  # NBA
    "Center": 0.95  # NBA
}

# Injury severity
severity = {
    "Out": 1.0,
    "Doubtful": 0.7,
    "Questionable": 0.4,
    "Probable": 0.2
}

player_impact = position_weight × severity × max_impact_per_sport
total_team_impact = sum(all_player_impacts)

# Compare teams
net_advantage = away_injuries - home_injuries
```

**Decision Factor:**
- Net advantage > 0.05: Significant health edge
- QB out (NFL) = automatic +0.15 impact
- Goalie out (NHL) = automatic +0.10 impact

### PHASE 2: SIMULATIONS & PSYCHOLOGY

#### 2.7 Monte Carlo Simulation
**Run 1,000 game simulations:**
```python
for sim in range(1000):
    # Expected scores based on ELO probability
    home_share = 0.5 + (home_prob - 0.5) × 0.5
    expected_home = total_points × home_share
    expected_away = total_points × (1 - home_share)
    
    # Add randomness (Gaussian for NBA, Poisson for NFL/NHL/Soccer)
    if sport == "NBA":
        home_score = random.gauss(expected_home, std_dev)
    else:
        home_score = poisson_sample(expected_home)
    
    # Record outcomes
    if home_score > away_score: home_wins++
    if home_score - away_score > -spread: spread_covers++
    if home_score + away_score > total: over_hits++

# Calculate probabilities
home_win_prob_mc = home_wins / 1000
spread_cover_prob = spread_covers / 1000
over_prob = over_hits / 1000
```

**Decision Factor:**
- MC win prob > 70%: Strong favorite
- Spread cover prob > 60%: Good spread value
- Compare to odds for edge calculation

#### 2.8 Market Psychology
**Public Bias Detection:**
```python
biases_detected = []

# Home favorite bias
if home_favorite and home_ml < 1.70:
    biases.append("home_favorite_bias")
    public_side = home_team

# Overs bias
if total_movement_direction == "over":
    biases.append("overs_bias")
    
# Recency bias
if home_streak >= 4:
    biases.append("recency_bias")
    public_side = home_team

# Popular team bias
if home_team in ["Lakers", "Cowboys", "Yankees"]:
    biases.append("popular_team_bias")
    public_side = home_team

total_bias_score = len(biases) × 0.05
```

**Contrarian Opportunities:**
```python
if sharp_side != public_side:
    contrarian_opportunity = True
    contrarian_score = +0.10
```

**Decision Factor:**
- Bias score > 0.10: Public overvaluing one side
- Contrarian opportunity: Fade the public potential

### PHASE 3: MACHINE LEARNING

#### 2.9 Logistic Regression
**Feature Vector:**
```python
features = {
    "elo_diff": home_elo - away_elo,              # Weight: 0.005-0.006
    "form_diff": home_win% - away_win%,            # Weight: 0.22-0.30
    "margin_diff": home_avg_margin - away_avg_margin,  # Weight: 0.04-0.08
    "rest_advantage": context_net_advantage,        # Weight: 0.12-0.15
    "home_advantage": 1.0,                          # Weight: 0.048-0.055
    "injury_impact": injury_net_advantage,          # Weight: -0.45 to -0.60
    "line_movement": line_confidence_adj,           # Weight: 0.20-0.25
    "four_factors_diff": home_ff - away_ff         # Weight: 0.015 (NBA only)
}

# Logistic regression formula
z = intercept
for feature, value in features.items():
    z += value × weight[feature]

probability = 1 / (1 + e^(-z))
```

**Decision Factor:**
- LR probability > 70%: Strong ML signal
- Combined with other models for ensemble

#### 2.10 Five Sub-Models
**Model 1: ELO Model**
```python
probability = elo_home_prob
confidence = 60 + abs(probability - 0.5) × 40
pick = home if prob > 0.55 else away if prob < 0.45 else None
```

**Model 2: Context Model**
```python
context_advantage = rest + travel + altitude + schedule
probability = 0.50 + context_advantage
confidence = 55 + abs(context_advantage) × 200
pick = based on context_advantage threshold
```

**Model 3: Line Movement Model**
```python
line_confidence = line_movement_analysis.confidence_adjustment
probability = 0.50 + line_confidence
pick = recommended_side_from_line_movement
```

**Model 4: Statistical Model**
```python
# Average of Monte Carlo and Logistic Regression
probability = (mc_prob + lr_prob) / 2
confidence = 65 + abs(probability - 0.5) × 50
pick = based on probability threshold
```

**Model 5: Psychology Model**
```python
psychology_score = contrarian_score + market_efficiency_score
probability = 0.50 + psychology_score
pick = contrarian_side if opportunity exists
```

#### 2.11 Ensemble Voting
```python
# Collect all model predictions
model_predictions = {
    "elo_model": {probability, confidence, pick},
    "context_model": {probability, confidence, pick},
    "line_movement_model": {probability, confidence, pick},
    "statistical_model": {probability, confidence, pick},
    "psychology_model": {probability, confidence, pick}
}

# Weighted average (based on historical performance)
weights = get_dynamic_weights()  # Auto-adjusted based on accuracy
ensemble_prob = sum(prob × weight for each model)

# Calculate agreement
prob_std = stdev([all model probabilities])
agreement = 1.0 - min(1.0, prob_std / 0.25)

# Count picks
pick_counts = count_picks_per_team()
consensus_pick = most_common_pick()
consensus_strength = max_count / total_models

# Combined confidence
ensemble_confidence = (agreement × 0.6 + consensus_strength × 0.4) × 100
```

**V6 Decision Thresholds:**
```python
if (
    ensemble_confidence >= 55% AND
    models_agreeing >= 3 AND
    model_agreement >= 0.25 AND
    (ensemble_prob > 0.55 OR ensemble_prob < 0.45)
):
    V6_HAS_PICK = True
else:
    V6_HAS_PICK = False
```

---

## STAGE 3: V5 (LINE MOVEMENT) ANALYSIS

### 3.1 Line Movement Tracking
```python
# Compare opening to current
opening_ml = opening_odds.home_ml
current_ml = current_odds.home_ml

ml_movement = ((1/current_ml) - (1/opening_ml)) / (1/opening_ml) × 100
spread_movement = current_spread - opening_spread
total_movement = current_total - opening_total
```

### 3.2 Sharp Money Detection
**Criteria:**
```python
# Early phase (3+ days before)
if phase == "early" and abs(ml_movement) > 3%:
    sharp_money_detected = True
    
# Reverse Line Movement (RLM)
if line_moved_one_direction and public_on_opposite_side:
    rlm_detected = True
    sharp_side = direction_of_line_move

# Steam move (multiple books move quickly)
if all_books_moved_together_in_2hr_window:
    steam_detected = True
```

### 3.3 Market Phase Analysis
```python
time_until_game = game_time - now

if time_until_game > 48_hours:
    phase = "early"      # Sharp money phase
elif time_until_game > 24_hours:
    phase = "middle"     # Mixed action
else:
    phase = "late"       # Public betting phase
```

**V5 Decision:**
```python
factors = []

if sharp_money_detected: factors.append("sharp")
if rlm_detected: factors.append("rlm")
if steam_detected: factors.append("steam")
if significant_line_move: factors.append("value")

if len(factors) >= 2:
    V5_HAS_PICK = True
else:
    V5_HAS_PICK = False
```

---

## STAGE 4: UNIFIED COMBINATION

### 4.1 Combine V5 and V6
```python
v5_has_pick = V5_result.has_pick
v6_has_pick = V6_result.has_pick

if v5_has_pick and v6_has_pick:
    # Check if they agree
    if v5_pick == v6_pick:
        # STRONG CONSENSUS
        confidence = (v6_conf × 0.70 + v5_conf × 0.30 + 0.10)  # +10% bonus
        edge = (v6_edge × 0.70 + v5_edge × 0.30)
        UNIFIED_PICK = v6_pick
        
    else:
        # CONFLICT - Use V6 but penalize
        confidence = v6_conf × 0.70 - 0.15  # -15% conflict penalty
        if confidence >= 0.60:
            UNIFIED_PICK = v6_pick
        else:
            NO_PICK()
            
elif v6_has_pick and not v5_has_pick:
    # V6 ONLY
    confidence = v6_conf × 0.70  # Weighted
    if confidence >= 0.60:
        UNIFIED_PICK = v6_pick
    else:
        NO_PICK()
        
elif v5_has_pick and not v6_has_pick:
    # V5 ONLY - Insufficient
    NO_PICK()  # V5 alone not enough
    
else:
    # NEITHER HAS PICK
    NO_PICK()
```

### 4.2 Final Validation
```python
if UNIFIED_PICK:
    # Calculate edge
    implied_prob = 1 / odds
    edge = unified_prob - implied_prob
    
    # Final checks
    if edge < 0.04:
        NO_PICK()  # Insufficient edge
    
    if confidence < 0.60:
        NO_PICK()  # Insufficient confidence
    
    # PASS ALL CHECKS
    SAVE_PREDICTION_TO_DATABASE()
```

---

## STAGE 5: PREDICTION STORAGE

### 5.1 Database Entry
```python
prediction_document = {
    "id": uuid4(),
    "event_id": event_id,
    "sport_key": sport_key,
    "home_team": home_team,
    "away_team": away_team,
    "prediction": pick_team,
    "confidence": confidence,
    "odds": odds_at_prediction,
    "ai_model": "unified",
    "created_at": datetime.now(),
    "commence_time": game_commence_time,
    "prediction_type": pick_type,  # "moneyline", "spread", "total"
    "result": "pending",
    "reasoning": full_reasoning_text,
    "edge": calculated_edge,
    "consensus_level": "strong" | "weak" | "v6_only",
    "v5_agrees": True/False,
    "v6_agrees": True/False,
    
    # Full analysis data
    "v5_analysis": v5_full_result,
    "v6_analysis": v6_full_result,
    "matchup_data": matchup_summary,
    "context_data": context_summary,
    "injury_data": injury_summary
}

db.predictions.insert_one(prediction_document)
```

---

## SUMMARY: WHAT GETS CONSIDERED

**20+ Major Factors:**
1. ✅ Team ELO ratings (strength)
2. ✅ Recent form (win %, streak, margin)
3. ✅ Rest days (B2B, well-rested, normal)
4. ✅ Travel distance & timezone changes
5. ✅ Altitude advantage
6. ✅ Schedule congestion
7. ✅ Injuries (position-weighted, severity-adjusted)
8. ✅ Advanced stats (Four Factors, efficiency, possession)
9. ✅ Home court/ice/field advantage
10. ✅ Line movement (opening to current)
11. ✅ Sharp money detection
12. ✅ Reverse line movement (RLM)
13. ✅ Market phase (early vs late)
14. ✅ Public bias (5 types)
15. ✅ Contrarian opportunities
16. ✅ Market efficiency
17. ✅ Monte Carlo simulation (1,000 runs)
18. ✅ Poisson modeling (low-scoring sports)
19. ✅ Logistic regression (10+ features)
20. ✅ 5-model ensemble consensus
21. ✅ Historical performance of each model
22. ✅ Odds comparison for edge calculation

**All must align for a pick to be generated!**
