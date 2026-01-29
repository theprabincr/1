# BetPredictor V6 - Advanced Algorithm Implementation

## Overview
BetPredictor V6 is a comprehensive betting algorithm that implements advanced analytics, machine learning, and ensemble methods to provide highly accurate betting predictions. It represents a significant upgrade from V5 with 3 implementation phases.

## What's New in V6

### PHASE 1: Context & Advanced Metrics
Advanced situational and statistical analysis for superior prediction accuracy.

#### 1. **ELO Rating System** (`advanced_metrics.py`)
- Sport-specific ELO configurations with custom K-factors
- Home advantage adjustments (NBA: 100 pts, NFL: 65 pts, NHL: 50 pts)
- Blowout multipliers for margin-of-victory impact
- Dynamic rating updates after each game
- Win probability calculation using ELO difference

#### 2. **Advanced Metrics by Sport**
- **NBA Four Factors**: eFG%, TOV%, ORB%, FT Rate (Dean Oliver's framework)
- **NFL Efficiency Metrics**: Offensive/Defensive efficiency ratings (DVOA-inspired)
- **NHL Possession Metrics**: Corsi-like ratings and PDO analysis
- **Pace & Net Rating**: Calculated for all sports

#### 3. **Context Analysis** (`context_analyzer.py`)
- **Rest Days Analysis**:
  - Back-to-back game detection (B2B penalty: -8%)
  - Well-rested bonus (3+ days: +4%)
  - Sport-specific rest impact weights
  
- **Travel Analysis**:
  - City-to-city distance calculation (Haversine formula)
  - Timezone change impact (per hour penalty)
  - Cross-country travel penalties
  - 47 major city coordinates database
  
- **Altitude Advantage**:
  - Denver (5,280 ft) and Utah (4,226 ft) advantages
  - Visiting team altitude adjustment
  
- **Schedule Congestion**:
  - Games in last 7/14 days tracking
  - Congested schedule penalties (4+ games in 7 days)

#### 4. **Smart Injury Analysis** (`injury_analyzer.py`)
- **Position-Based Importance Weights**:
  - NFL QB: 2.0x (most critical)
  - NHL Goalie: 1.5x
  - NBA Point Guard: 1.1x
  - Position-specific multipliers for all sports
  
- **Injury Severity Multipliers**:
  - Out: 1.0 (full impact)
  - Doubtful: 0.7
  - Questionable: 0.4
  - Probable: 0.2
  
- **Smart Impact Scoring**:
  - Player importance × severity × max impact
  - Team health comparison
  - Net injury advantage calculation

### PHASE 2: Statistical Modeling & Psychology

#### 5. **Monte Carlo Simulation** (`simulation_engine.py`)
- **1,000+ simulations per game**
- Outcome probabilities (Win, Spread Cover, Over/Under)
- Score projections with confidence intervals
- Sport-specific scoring models:
  - Continuous (Basketball): Gaussian distribution
  - Discrete (Football/Hockey/Soccer): Poisson distribution

#### 6. **Poisson Modeling**
- Accurate for low-scoring sports (NFL, NHL, Soccer)
- Probability mass function calculations
- Most likely final scores (Top 5)
- Tie probability estimates

#### 7. **Market Psychology Analysis** (`market_psychology.py`)
- **Public Bias Detection** (5 types):
  1. Home Favorite Bias
  2. Overs Bias
  3. Recency Bias (hot teams overvalued)
  4. Popular Team Bias (Lakers, Cowboys, etc.)
  5. Primetime Bias
  
- **Contrarian Opportunities**:
  - Sharp vs Public money splits
  - Reverse Line Movement (RLM) detection
  - Fade-the-public opportunities
  
- **Market Efficiency Scoring**:
  - Efficiency rating (0-100)
  - Inefficiency indicators
  - Opportunity level classification

### PHASE 3: Machine Learning & Ensemble

#### 8. **Logistic Regression Model** (`ml_models.py`)
- **10+ Weighted Features**:
  - ELO difference (0.005-0.006 per point)
  - Form difference (0.22-0.30 weight)
  - Margin difference (0.04-0.08 weight)
  - Rest advantage (0.12-0.15 weight)
  - Home advantage (0.048-0.055 weight)
  - Injury impact (-0.45 to -0.60 weight)
  - Line movement signal (0.20-0.25 weight)
  - Sport-specific metrics (Four Factors, Efficiency, Possession)
  
- **Sport-Specific Calibration**:
  - Custom weight matrices for NBA, NFL, NHL
  - Intercept adjustments
  - Probability clamping (15%-85%)

#### 9. **5-Model Ensemble System**
The ensemble combines predictions from 5 independent models:

1. **ELO Model**: Pure ELO-based probability
2. **Context Model**: Rest/travel/altitude/schedule factors
3. **Line Movement Model**: Sharp money & RLM signals
4. **Statistical Model**: Monte Carlo + Logistic Regression average
5. **Psychology Model**: Contrarian opportunities & market efficiency

**Ensemble Logic**:
- Weighted voting based on historical performance
- Dynamic weight adjustment (better models get higher weights)
- Consensus requirement: **3 out of 5 models must agree**
- Model agreement threshold: **70%+**
- Confidence calculation based on agreement strength

#### 10. **Kelly Criterion**
- Optimal bet sizing calculation
- Edge-based bet recommendations
- Fractional Kelly (1/4 Kelly for safety)
- Capped at 5% of bankroll

#### 11. **Performance Tracking**
- Historical prediction storage
- Model accuracy tracking per sub-model
- Brier score calculation
- ROI tracking
- Automatic weight adjustment based on performance

## Decision Requirements

V6 is **conservative by design**. A pick is only recommended when:

1. ✅ **Ensemble confidence ≥ 65%**
2. ✅ **At least 3 out of 5 models agree on same pick**
3. ✅ **Model agreement ≥ 70%**
4. ✅ **Clear probability edge (>55% or <45%)**
5. ✅ **Minimum edge ≥ 4%**

If any requirement fails, **NO PICK** is returned with detailed reasoning.

## API Endpoints

### New V6 Endpoints

#### 1. `GET /api/predictions/v6`
Get all V6 predictions with stats
```json
{
  "predictions": [...],
  "stats": {
    "total": 10,
    "wins": 7,
    "losses": 2,
    "pending": 1,
    "win_rate": 77.8,
    "avg_confidence": 72.5
  },
  "algorithm": "betpredictor_v6"
}
```

#### 2. `POST /api/analyze-v6/{event_id}`
Analyze a specific event with V6
```bash
curl -X POST "http://localhost:8001/api/analyze-v6/401655350?sport_key=basketball_nba"
```

Response includes:
- Comprehensive prediction with reasoning
- Ensemble details (5 sub-models)
- Simulation results (Monte Carlo)
- Matchup summary (ELO, context, injuries)
- Market analysis (line movement, psychology)

#### 3. `GET /api/predictions/comparison`
Compare V5 vs V6 performance
```json
{
  "algorithms": {
    "betpredictor_v5": {...},
    "betpredictor_v6": {...}
  },
  "recommendation": "V6 uses advanced ML and ensemble methods for higher accuracy"
}
```

#### 4. `GET /api/model-performance`
Get individual sub-model performance
```json
{
  "sub_models": {
    "elo_model": {"accuracy": 0.583, "weight": 0.21},
    "context_model": {"accuracy": 0.550, "weight": 0.18},
    "line_movement_model": {"accuracy": 0.625, "weight": 0.26},
    "statistical_model": {"accuracy": 0.600, "weight": 0.22},
    "psychology_model": {"accuracy": 0.533, "weight": 0.13}
  }
}
```

## Sport Coverage

### Primary Focus (Optimized)
1. **NBA** - Full Four Factors, pace analysis, ELO system
2. **NFL** - Efficiency metrics, travel impact, injury weighting (QB critical)
3. **NHL** - Possession metrics, goalie impact, PDO analysis

### Secondary Support
4. **Soccer (EPL)** - Poisson modeling, home advantage, basic metrics
5. **Other Sports** - Fallback to general algorithm

## Algorithm Workflow

```
1. DATA COLLECTION
   ├─ ESPN: Events, odds, team stats, recent games
   ├─ Line Movement: Historical odds snapshots
   └─ Squad Data: Injuries, rosters

2. PHASE 1: ADVANCED METRICS
   ├─ ELO Ratings (home vs away)
   ├─ Advanced Metrics (Four Factors, efficiency, possession)
   ├─ Context Analysis (rest, travel, altitude, schedule)
   └─ Smart Injury Analysis (position weights × severity)

3. PHASE 2: SIMULATIONS & PSYCHOLOGY
   ├─ Monte Carlo (1000 sims → win%, spread cover%, over%)
   ├─ Poisson Modeling (for NFL/NHL/Soccer)
   ├─ Market Psychology (bias detection, contrarian opps)
   └─ Market Efficiency Scoring

4. PHASE 3: ML & ENSEMBLE
   ├─ Feature Vector Creation (10+ features)
   ├─ Logistic Regression (probability calculation)
   ├─ 5 Sub-Models Run Independently:
   │  ├─ ELO Model
   │  ├─ Context Model
   │  ├─ Line Movement Model
   │  ├─ Statistical Model
   │  └─ Psychology Model
   ├─ Ensemble Voting (weighted by performance)
   └─ Consensus Check (3/5 agree? 70% agreement?)

5. FINAL DECISION
   ├─ Check all requirements (confidence, edge, consensus)
   ├─ If ALL met → RECOMMEND PICK
   └─ If ANY fail → NO PICK (with detailed reasoning)

6. RESULT TRACKING
   ├─ Save prediction to database
   ├─ Update model performance metrics
   └─ Adjust model weights for future predictions
```

## Performance Advantages Over V5

| Feature | V5 | V6 |
|---------|----|----|
| **Metrics** | Basic (form, margin) | Advanced (ELO, Four Factors, efficiency) |
| **Context** | None | Full (rest, travel, altitude, schedule) |
| **Injuries** | Simple count | Smart weighting by position/severity |
| **Simulations** | None | Monte Carlo 1000+ + Poisson |
| **Psychology** | None | Full (bias, contrarian, efficiency) |
| **ML Models** | None | Logistic regression + 5-model ensemble |
| **Consensus** | Single algorithm | 5 models must agree (3/5 threshold) |
| **Bet Sizing** | Fixed | Kelly Criterion |
| **Performance Tracking** | Limited | Full (Brier scores, ROI, auto-tuning) |

## Technical Implementation

### Files Created
1. `advanced_metrics.py` (430 lines) - ELO & sport-specific metrics
2. `context_analyzer.py` (380 lines) - Situational analysis
3. `injury_analyzer.py` (220 lines) - Smart injury impact
4. `market_psychology.py` (290 lines) - Bias & contrarian detection
5. `simulation_engine.py` (310 lines) - Monte Carlo & Poisson
6. `ml_models.py` (430 lines) - Logistic regression & ensemble
7. `betpredictor_v6.py` (650 lines) - Main orchestration engine

**Total: ~2,700 lines of advanced betting intelligence**

### Dependencies
- All existing dependencies (no new packages required)
- Uses: statistics, math, datetime, logging, typing
- Integrates with: ESPN API, MongoDB, existing line movement system

## Usage Recommendations

1. **Start with V6** for most games (higher accuracy expected)
2. **Use V5** if you want simpler line movement focus
3. **Compare both** via `/api/predictions/comparison` endpoint
4. **Monitor** individual model performance via `/api/model-performance`
5. **Track ROI** over time to validate improvements

## Future Enhancements (Potential)

1. Real betting percentage data integration
2. Player prop analysis
3. Live betting model (in-game)
4. Deep learning neural network models
5. Automated backtesting system
6. Multi-league arbitrage detection

## Conclusion

BetPredictor V6 represents a **comprehensive upgrade** with:
- ✅ 3 implementation phases completed
- ✅ 7 new specialized modules
- ✅ 10+ advanced features per phase
- ✅ Conservative, multi-model consensus approach
- ✅ Automatic performance tracking & tuning
- ✅ Sport-specific optimizations (NBA, NFL, NHL focus)

**V5 is preserved** - both algorithms available for comparison and use.
