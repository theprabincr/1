# BetPredictor - AI-Powered Sports Betting Predictor

A sophisticated full-stack application that provides ML-powered sports betting recommendations by analyzing real-time odds, team statistics, line movements, and multiple predictive models including **XGBoost Machine Learning**.

## 🎯 Overview

BetPredictor uses a **Unified Prediction Engine** that combines THREE algorithms:

| Algorithm | Weight | Description |
|-----------|--------|-------------|
| **XGBoost ML** | 40% | Real trained machine learning models (Moneyline, Spread, Totals) |
| **V6 (Rule-based Ensemble)** | 35% | 5 rule-based analytical models |
| **V5 (Line Movement)** | 25% | Sharp money and line movement analysis |

**Key Principle**: Only when multiple algorithms align (or XGBoost shows strong confidence) does the system generate a pick.

---

## ✨ Features

### Core Functionality
- **Real-time Data**: Live odds, scores, and team stats from ESPN
- **XGBoost ML**: Trained models predicting Moneyline, Spread, AND Totals
- **Favored Outcome Display**: Shows which team/side is favored with win probability (not just home team)
- **Multi-Market Predictions**: Recommends the best market (ML, Spread, or Totals) per game
- **Smart Predictions**: Auto-generates picks 40 minutes before game time after lineup confirmations
- **Line Movement Tracking**: Monitors odds changes with 5-minute snapshots
- **Auto Result Tracking**: Checks game results every 15 minutes via ESPN API
- **Adaptive Learning**: Models self-adjust weights based on historical accuracy
- **Weekly Retraining**: XGBoost models automatically retrain every Sunday at 3 AM UTC
- **Duplicate Prevention**: Upsert logic prevents duplicate predictions in database

### Prediction Analysis Includes
- **Favored Team/Side**: Shows the predicted winner/cover with probability (e.g., "Lakers @ 78.5%")
- **ELO Ratings**: Calculated from overall season record (not just last 10 games)
- **Win Probability**: XGBoost-predicted probability for favored team
- **Spread Cover Probability**: Likelihood of favored team covering the spread
- **Over/Under Probability**: Prediction for totals market (OVER or UNDER)
- **Predicted Total Points**: Regression model estimates actual total
- **Model Consensus**: Agreement level between XGBoost, V6, and V5

### V6 Detailed Analysis Sections
Each prediction includes comprehensive analysis:
- **Team Strength**: ELO ratings based on season record
- **Recent Form & Records**: Season/home/away records, streaks, margins
- **Situational Factors**: Rest days, schedule congestion
- **Injury Impact**: Team health assessment
- **Simulation Results**: Monte Carlo win probabilities
- **Key Factors**: Top reasons for the prediction

### Sports Covered
| Sport | Key | ML Accuracy | Spread Accuracy | Totals Accuracy |
|-------|-----|-------------|-----------------|-----------------|
| 🏀 NBA | `basketball_nba` | 65.4% | 52.1% | 55.5% |
| 🏈 NFL | `americanfootball_nfl` | 77.6% | 53.4% | - |
| 🏒 NHL | `icehockey_nhl` | 64.6% | 56.5% | - |
| ⚽ EPL | `soccer_epl` | - | - | - |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      UNIFIED PREDICTOR                               │
├─────────────────────┬─────────────────────┬─────────────────────────┤
│  XGBOOST ML (40%)   │  V6 ENSEMBLE (35%)  │  V5 LINE MOVEMENT (25%) │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ • Moneyline Model   │ • ELO Model         │ • Sharp Money Detection │
│ • Spread Model      │ • Context Model     │ • Reverse Line Movement │
│ • Totals Model      │ • Line Movement     │ • Steam Move Detection  │
│ • Totals Regressor  │ • Statistical Model │ • Market Phase Analysis │
│                     │ • Psychology Model  │                         │
├─────────────────────┴─────────────────────┴─────────────────────────┤
│   FAVORED OUTCOME: Shows team/side with highest probability         │
│   BEST MARKET SELECTION: Chooses highest confidence market          │
│   DECISION: 60%+ Combined Confidence AND 4%+ Edge                   │
│   CONSENSUS: Strong (3/3), Moderate (2/3), or XGB Only              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 XGBoost Machine Learning System

### Model Architecture

The XGBoost system trains **4 models per sport**:

1. **Moneyline Classifier** (`XGBClassifier`)
   - Predicts: Win probability for each team
   - Output: `ml_favored_team`, `ml_favored_prob` (shows favored team, not just home)

2. **Spread Classifier** (`XGBClassifier`)
   - Predicts: Cover probability for each team
   - Output: `spread_favored_team`, `spread_favored_prob`, `spread_favored_line`

3. **Totals Classifier** (`XGBClassifier`)
   - Predicts: Over/Under probability
   - Output: `totals_favored` (OVER/UNDER), `totals_favored_prob`

4. **Totals Regressor** (`XGBRegressor`)
   - Predicts: Actual total points
   - Output: `predicted_total` (numeric)

### Favored Outcome Display

The system now shows **which team/side is favored**, not just home team probability:

```
📊 MARKET PREDICTIONS
  🏀 Moneyline: Toronto Raptors @ 87.9% (Acc: 65%)
  📏 Spread: Toronto Raptors +1.5 @ 59.9% (Acc: 52%)
  📈 Totals: OVER 225.5 @ 76.5% (Predicted: 230)
```

### Feature Engineering (35 Features)

```python
FEATURE_NAMES = [
    # Team Strength (6)
    "home_elo", "away_elo", "elo_diff",
    "home_win_pct", "away_win_pct", "win_pct_diff",
    
    # Recent Form (7)
    "home_last10_wins", "away_last10_wins",
    "home_streak", "away_streak",
    "home_avg_margin", "away_avg_margin", "margin_diff",
    
    # Scoring (6)
    "home_avg_pts", "away_avg_pts",
    "home_avg_pts_allowed", "away_avg_pts_allowed",
    "home_net_rating", "away_net_rating",
    
    # Context (5)
    "home_rest_days", "away_rest_days", "rest_advantage",
    "is_back_to_back_home", "is_back_to_back_away",
    
    # Odds/Market (5)
    "home_ml_odds", "away_ml_odds", "implied_home_prob",
    "spread", "total_line",
    
    # Head-to-Head (2)
    "h2h_home_wins", "h2h_total_games",
    
    # Totals-Specific (4)
    "combined_avg_pts", "combined_pts_allowed",
    "pace_factor", "defensive_rating_diff"
]
```

### Training Data Source
- **Source**: ESPN Historical Scoreboard API
- **NBA**: 1,313 games (2024 season)
- **NFL**: 286 games (2024 season)
- **NHL**: 1,354 games (2024 season)

### Model Persistence
Models are saved to `/app/backend/ml_models/`:
```
ml_models/
├── xgboost_ml_basketball_nba.joblib      # Moneyline model
├── xgboost_spread_basketball_nba.joblib  # Spread model
├── xgboost_totals_basketball_nba.joblib  # Totals model
├── xgboost_totals_reg_basketball_nba.joblib  # Totals regressor
├── scaler_basketball_nba.joblib          # Feature scaler
└── metadata_basketball_nba.json          # Training metrics
```

### Weekly Retraining
- **Schedule**: Every Sunday at 3 AM UTC
- **Process**: 
  1. Fetches latest game results from ESPN
  2. Retrains all 4 models per sport
  3. Saves new models to disk
  4. Creates notification about retraining

---

## 📊 ELO Rating System

### How ELO is Calculated
- **Source**: Overall season record (e.g., 32-18), NOT just last 10 games
- **Formula**: `ELO = 1200 + (win_pct × 600)`
- **Example**: 32-18 record = 64% win rate → ELO = 1584

### ELO Configuration
| Sport | K-Factor | Home Advantage | Initial Rating |
|-------|----------|----------------|----------------|
| NBA | 20 | +100 | 1500 |
| NFL | 25 | +65 | 1500 |
| NHL | 18 | +50 | 1500 |

### ELO Storage
- **Database**: MongoDB `elo_ratings` collection
- **Cache**: Loaded into memory on startup (`DB_ELO_CACHE`)
- **Updates**: After each game result is recorded

### Accessing ELO
```bash
# Get all team ELO ratings
curl "http://localhost:8001/api/ml/elo-ratings?sport_key=basketball_nba"
```

---

## 📁 Project Structure

```
/app
├── backend/
│   ├── server.py              # FastAPI main application
│   ├── unified_predictor.py   # Combines XGBoost + V5 + V6 algorithms
│   │   └── _build_xgb_reasoning()  # Consolidated reasoning builder
│   ├── betpredictor_v5.py     # Line movement analysis
│   ├── betpredictor_v6.py     # Rule-based ensemble engine
│   │   └── _build_recommendation_reasoning()  # V6 detailed analysis
│   ├── ml_xgboost.py          # ⭐ XGBoost ML system
│   │   ├── XGBoostPredictor   # Multi-market prediction with favored outcomes
│   │   ├── HistoricalDataCollector  # ESPN data fetcher
│   │   ├── Backtester         # Backtest validation
│   │   └── EnhancedELOSystem  # Database-backed ELO
│   ├── ml_models.py           # Legacy logistic regression
│   ├── advanced_metrics.py    # ELO from season record & sport-specific metrics
│   │   └── calculate_matchup_metrics()  # Returns home_elo, away_elo
│   ├── adaptive_learning.py   # Self-adjusting model weights
│   ├── context_analyzer.py    # Rest, travel, altitude analysis
│   ├── injury_analyzer.py     # Position-weighted injury impact
│   ├── market_psychology.py   # Bias detection & contrarian
│   ├── simulation_engine.py   # Monte Carlo & Poisson modeling
│   ├── line_movement_analyzer.py  # Sharp money detection
│   ├── espn_data_provider.py  # ESPN odds & stats fetcher
│   ├── espn_scores.py         # Live score tracking
│   ├── player_stats.py        # Player performance analysis
│   ├── ml_models/             # ⭐ Trained model storage
│   │   ├── *.joblib           # Serialized XGBoost models
│   │   └── metadata_*.json    # Training metrics
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main router
│   │   ├── pages/
│   │   │   ├── Dashboard.js   # Stats, compact ML status in header
│   │   │   ├── Events.js      # Events with favored outcome predictions
│   │   │   ├── LineMovement.js# Line movement charts
│   │   │   ├── Performance.js # Win/loss tracking
│   │   │   └── Settings.js    # App settings
│   │   └── App.css            # Tailwind styles
│   ├── package.json
│   └── .env                   # Frontend environment
├── memory/
│   └── PRD.md                 # Product requirements
├── test_result.md             # Testing documentation
└── README.md                  # This file
```

---

## 🔌 API Endpoints

### ML Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ml/status` | GET | Get status of all ML models |
| `/api/ml/collect-historical` | POST | Collect 1 season of data from ESPN |
| `/api/ml/train` | POST | Train XGBoost models |
| `/api/ml/predict/{event_id}` | POST | Get ML prediction with favored outcomes |
| `/api/ml/backtest` | POST | Run backtest validation |
| `/api/ml/elo-ratings` | GET | Get ELO ratings for all teams |
| `/api/ml/retrain-all` | POST | Manually trigger retraining |
| `/api/ml/update-elo-from-result` | POST | Update ELO from game result |

### Events & Odds
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/` | GET | Health check |
| `/api/sports` | GET | List available sports |
| `/api/events/{sport_key}` | GET | Get events with odds |
| `/api/line-movement/{event_id}` | GET | Line movement history |
| `/api/live-scores` | GET | Current live game scores |
| `/api/data-source-status` | GET | ESPN data source status |

### Predictions
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recommendations` | GET | Get AI recommendations (60%+ confidence) |
| `/api/analyze-unified/{event_id}` | POST | Full unified analysis with favored outcomes |
| `/api/analyze-v6/{event_id}` | POST | V6 rule-based analysis only |
| `/api/analyze-v5/{event_id}` | POST | V5 line movement only |
| `/api/predictions/unified` | GET | All unified predictions |

### Performance
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/performance` | GET | Win/loss statistics |
| `/api/notifications` | GET | System notifications |
| `/api/my-bets` | GET | User's tracked bets |

---

## 📊 Example API Response

### `/api/ml/predict/{event_id}` Response
```json
{
  "event_id": "401810582",
  "home_team": "Toronto Raptors",
  "away_team": "Minnesota Timberwolves",
  "prediction": {
    "ml_favored_team": "Toronto Raptors",
    "ml_favored_prob": 0.8789,
    "ml_underdog_team": "Minnesota Timberwolves",
    "ml_underdog_prob": 0.1211,
    
    "spread_favored_team": "Toronto Raptors",
    "spread_favored_prob": 0.5991,
    "spread_favored_line": 1.5,
    
    "totals_favored": "OVER",
    "totals_favored_prob": 0.7651,
    "totals_line": 225.5,
    "predicted_total": 229.6,
    
    "best_market": "moneyline",
    "best_pick": "Toronto Raptors",
    "confidence": 87.9,
    "model_accuracy": 0.654
  }
}
```

---

## ⚙️ Background Tasks

| Task | Frequency | Description |
|------|-----------|-------------|
| `scheduled_espn_odds_refresh` | 5 minutes | Snapshot odds for line movement |
| `scheduled_unified_predictor` | 1 minute | Generate picks 35-50 min before games |
| `scheduled_result_checker` | 15 minutes | Check game results via ESPN |
| `scheduled_ml_retraining` | Weekly (Sun 3AM) | Retrain XGBoost models |
| `scheduled_live_score_updater` | 10 seconds | Update live scores |
| `scheduled_player_stats_updater` | 6 hours | Update player statistics |
| `scheduled_daily_summary` | Daily 9PM | Send daily performance summary |

---

## 🗄️ Database Collections

### predictions
```javascript
{
  id: String (UUID),
  event_id: String,
  sport_key: String,
  home_team: String,
  away_team: String,
  prediction: String,           // Team name or "OVER"/"UNDER"
  pick_type: "moneyline" | "spread" | "totals",
  pick_display: String,         // e.g., "Lakers -3.5" or "OVER 220.5"
  confidence: Number (0-100),
  odds_at_prediction: Number,
  edge: Number,
  algorithm: "unified_xgboost" | "unified" | "v6_only",
  consensus_level: "strong_consensus" | "moderate_consensus" | "xgb_only",
  
  // Favored outcomes
  ml_favored_team: String,
  ml_favored_prob: Number,
  spread_favored_team: String,
  spread_favored_prob: Number,
  totals_favored: String,        // "OVER" or "UNDER"
  totals_favored_prob: Number,
  
  result: "pending" | "win" | "loss" | "push",
  reasoning: String,
  created_at: String (ISO),
  updated_at: String (ISO)       // For upsert tracking
}
```

### historical_games
```javascript
{
  event_id: String,
  sport_key: String,
  season: String,
  home_team: String,
  away_team: String,
  home_score: Number,
  away_score: Number,
  home_win: Boolean,
  home_covered: Boolean,        // Spread outcome
  went_over: Boolean,           // Totals outcome
  features: Object,             // 35 pre-game features
  is_complete: Boolean
}
```

### elo_ratings
```javascript
{
  sport_key: String,
  team_name: String,
  elo: Number,
  last_updated: String (ISO)
}
```

---

## 🔧 Environment Variables

### Backend (.env)
```env
MONGO_URL="mongodb://localhost:27017"
DB_NAME="test_database"
CORS_ORIGINS="*"
```

### Frontend (.env)
```env
REACT_APP_BACKEND_URL=<your-backend-url>
```

---

## 📊 Algorithm Decision Requirements

A pick is only recommended when ALL conditions are met:

1. ✅ **Weighted Confidence ≥ 60%** (XGBoost 40% + V6 35% + V5 25%)
2. ✅ **OR XGBoost Confidence ≥ 65% AND at least 1 model agrees**
3. ✅ **Minimum Edge ≥ 4%**
4. ✅ **Best market selected** (highest confidence among ML, Spread, Totals)

### Consensus Levels
- **Strong Consensus**: All 3 algorithms agree (+10% confidence bonus)
- **Moderate Consensus**: 2 out of 3 agree (+5% confidence bonus)
- **XGB Only**: Only XGBoost has a pick (no bonus)

---

## 🎨 UI Components

### Dashboard
- **Header**: Compact XGBoost ML status inline (NBA 65%, NHL 65%, NFL 78%)
- **Stats Grid**: Win Rate, ROI, Active Picks, Total Picks, Live Games
- **Live Games Section**: Real-time score updates (when games are live)
- **Today's Picks**: Current recommendations

### Events Page
- **Game Cards**: Odds comparison, team records
- **Analysis Modal**: Full prediction breakdown with:
  - Recommended Pick (favored team/side)
  - Market Predictions (ML, Spread, Totals with accuracies)
  - Model Agreement visualization
  - V6 Detailed Analysis (6 sections)

### Reasoning Display Format
```
==================================================
🤖 XGBOOST ML PREDICTION
==================================================

📊 MARKET PREDICTIONS
  🏀 Moneyline: Toronto Raptors @ 87.9% (Acc: 65%)
  📏 Spread: Toronto Raptors +1.5 @ 59.9% (Acc: 52%)
  📈 Totals: OVER 225.5 @ 76.5% (Predicted: 230)

💰 Confidence: 67.2%  |  🎯 Edge: +38.4%

📊 MODEL AGREEMENT
  📊 2 OF 3 MODELS AGREE (+5% boost)
  • XGBoost: Toronto Raptors ML (87.9%)
  • V6: Toronto Raptors (69%)
  • V5: No pick

==================================================

📋 V6 DETAILED ANALYSIS

TEAM STRENGTH
Toronto Raptors: 1553 ELO rating
Minnesota Timberwolves: 1565 ELO rating

RECENT FORM & RECORDS
Toronto Raptors: 30-21, 6W-4L last 10

SITUATIONAL FACTORS
• Home team well rested
• Away team congested schedule

INJURY IMPACT
Both teams are relatively healthy.

SIMULATION RESULTS
Toronto Raptors win probability: 60.0%

KEY FACTORS
1. Home team well rested
2. Away team congested schedule
```

---

## 🚀 Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11)
- **Database**: MongoDB (motor async driver)
- **ML Library**: XGBoost, scikit-learn, joblib
- **Data Source**: ESPN API (odds, scores, stats)

### Frontend
- **Framework**: React 18
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React

### ML Components
- XGBoost Classifier (Moneyline, Spread, Totals)
- XGBoost Regressor (Total Points)
- StandardScaler for feature normalization
- Train/Test split with stratification
- Cross-validation (5-fold)

---

## 📈 Training & Retraining

### Manual Training
```bash
# 1. Collect historical data (1 season)
curl -X POST "http://localhost:8001/api/ml/collect-historical?sport_key=basketball_nba&season=2024"

# 2. Train models (includes ELO rebuild)
curl -X POST "http://localhost:8001/api/ml/train?sport_key=basketball_nba&rebuild_elo=true"

# 3. Verify models loaded
curl "http://localhost:8001/api/ml/status"

# 4. Run backtest
curl -X POST "http://localhost:8001/api/ml/backtest?sport_key=basketball_nba&threshold=0.55"
```

### Automated Retraining
- Runs automatically every Sunday at 3 AM UTC
- Can be triggered manually: `POST /api/ml/retrain-all`

---

## 🛠️ Troubleshooting

### Common Issues

1. **ELO showing 1500 for all teams**
   - ELO is now calculated from season record (e.g., 32-18 → 1584)
   - Run: `POST /api/ml/train?rebuild_elo=true`

2. **Duplicate picks on dashboard**
   - Fixed: Upsert logic prevents duplicates
   - Clean existing: Check `/api/predictions/unified`

3. **V6 Analysis empty**
   - Fixed: Now includes 6 detailed sections
   - Check reasoning field in API response

4. **Model agreement count wrong**
   - Fixed: Now counts actual agreeing models, not consensus_strength ratio

5. **XGBoost models not loaded**
   - Run training: `POST /api/ml/train?sport_key=basketball_nba`
   - Check model files exist in `/app/backend/ml_models/`

6. **No picks generating**
   - Algorithm is conservative by design (needs 60%+ confidence, 4%+ edge)
   - Check if games are within 40-minute window
   - View `/api/analyze-unified/{event_id}` for detailed reasoning

### Logs
```bash
# Backend logs
tail -f /var/log/supervisor/backend.err.log

# Filter for XGBoost
grep -i "xgboost\|ml\|elo" /var/log/supervisor/backend.err.log

# Check supervisor status
sudo supervisorctl status

# Restart services
sudo supervisorctl restart backend
sudo supervisorctl restart frontend
sudo supervisorctl restart all
```

### Useful Debug Commands
```bash
# Check ML model status
curl "http://localhost:8001/api/ml/status" | python3 -m json.tool

# Get prediction with favored outcomes
curl -X POST "http://localhost:8001/api/ml/predict/401810582?sport_key=basketball_nba" | python3 -c "
import json, sys
d = json.load(sys.stdin)
p = d['prediction']
print(f\"ML: {p['ml_favored_team']} @ {p['ml_favored_prob']*100:.1f}%\")
print(f\"Spread: {p['spread_favored_team']} {p['spread_favored_line']:+.1f} @ {p['spread_favored_prob']*100:.1f}%\")
print(f\"Totals: {p['totals_favored']} {p['totals_line']} @ {p['totals_favored_prob']*100:.1f}%\")
"

# Check predictions for duplicates
curl "http://localhost:8001/api/predictions/unified" | python3 -c "
import json, sys
from collections import Counter
d = json.load(sys.stdin)
events = [p['event_id'] for p in d['predictions']]
dups = [(e, c) for e, c in Counter(events).items() if c > 1]
print(f'Total: {len(d[\"predictions\"])}, Duplicates: {len(dups)}')
"
```

---

## 📋 Key Files to Modify

### To Change ML Model Parameters
- **File**: `/app/backend/ml_xgboost.py`
- **Class**: `XGBoostPredictor.train()`
- **Settings**: `n_estimators`, `max_depth`, `learning_rate`

### To Change Favored Outcome Logic
- **File**: `/app/backend/ml_xgboost.py`
- **Method**: `XGBoostPredictor.predict()`
- **Variables**: `ml_favored_team`, `spread_favored_team`, `totals_favored`

### To Change Feature Engineering
- **File**: `/app/backend/ml_xgboost.py`
- **Variable**: `FEATURE_NAMES` (list of 35 features)
- **Class**: `FeatureEngineering.extract_features()`

### To Change Prediction Thresholds
- **File**: `/app/backend/unified_predictor.py`
- **Class**: `UnifiedBetPredictor.__init__()`
- **Variables**: 
  - `min_unified_confidence = 0.60`
  - `min_edge = 0.04`
  - `xgb_weight = 0.40`

### To Change ELO Configuration
- **File**: `/app/backend/advanced_metrics.py`
- **Variable**: `ELO_CONFIG`
- **Method**: `calculate_advanced_metrics()` - uses season record

### To Change Reasoning Display
- **File**: `/app/backend/unified_predictor.py`
- **Method**: `_build_xgb_reasoning()` - controls which V6 sections are included

### To Change Retraining Schedule
- **File**: `/app/backend/server.py`
- **Function**: `scheduled_ml_retraining()`
- **Current**: Sunday at 3 AM UTC (`weekday() == 6 and hour == 3`)

---

## 📝 License

MIT License - Feel free to modify and use for personal projects.

---

## 🙏 Credits

- **Data Source**: ESPN API
- **ML Framework**: XGBoost, scikit-learn
- **Icons**: [Lucide](https://lucide.dev/)
- **Charts**: [Recharts](https://recharts.org/)

---

## 📅 Changelog

### v2.1 (February 2026)
- ✅ **Favored Outcome Display**: Shows which team/side is favored (not just home team)
- ✅ **ELO from Season Record**: Now uses overall record (32-18) instead of last 10 games
- ✅ **Model Agreement Fix**: Correctly counts actual agreeing models
- ✅ **V6 Detailed Analysis**: Added 6 comprehensive sections (Team Strength, Form, Situational, Injury, Simulation, Key Factors)
- ✅ **Duplicate Prevention**: Upsert logic prevents duplicate predictions
- ✅ **Consolidated Reasoning**: Reduced from 35 sections to ~12 clean sections
- ✅ **Dashboard Compact ML Status**: XGBoost status moved to header (inline, smaller)
- ✅ **Live Games Section**: Now appears in its proper location

### v2.0 (February 2026)
- ✅ Added XGBoost ML models for Moneyline, Spread, and Totals
- ✅ Implemented multi-market prediction (best market selection)
- ✅ Added historical data collection from ESPN
- ✅ Implemented proper ELO tracking with database storage
- ✅ Added weekly automatic retraining
- ✅ Added backtesting infrastructure
- ✅ Updated UI to show ML status and predictions

### v1.0 (January 2026)
- Initial release with V5 + V6 ensemble
- Line movement tracking
- Automated prediction generation
- Result tracking and performance metrics
