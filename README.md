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
- **Multi-Market Predictions**: Recommends the best market (ML, Spread, or Totals) per game
- **Smart Predictions**: Auto-generates picks 40 minutes before game time after lineup confirmations
- **Line Movement Tracking**: Monitors odds changes with 5-minute snapshots
- **Auto Result Tracking**: Checks game results every 15 minutes via ESPN API
- **Adaptive Learning**: Models self-adjust weights based on historical accuracy
- **Weekly Retraining**: XGBoost models automatically retrain every Sunday at 3 AM UTC

### Prediction Analysis Includes
- **ELO Ratings**: Trained from historical games (not default 1500)
- **Win Probability**: XGBoost-predicted probability for each market
- **Spread Cover Probability**: Likelihood of home team covering the spread
- **Over/Under Probability**: Prediction for totals market
- **Predicted Total Points**: Regression model estimates actual total
- **Model Consensus**: Agreement level between XGBoost, V6, and V5

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
   - Predicts: Home win probability
   - Output: `home_win_prob` (0-1)

2. **Spread Classifier** (`XGBClassifier`)
   - Predicts: Home team covers spread probability
   - Output: `home_cover_prob` (0-1)

3. **Totals Classifier** (`XGBClassifier`)
   - Predicts: Over probability
   - Output: `over_prob` (0-1)

4. **Totals Regressor** (`XGBRegressor`)
   - Predicts: Actual total points
   - Output: `predicted_total` (numeric)

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

### How ELO Works
- **Initial Rating**: 1500 (for new teams)
- **K-Factor**: 20 (NBA), 25 (NFL), 18 (NHL)
- **Home Advantage**: +100 (NBA), +65 (NFL), +50 (NHL)
- **Margin of Victory**: Multiplier up to 1.5x for blowouts

### ELO Storage
- **Database**: MongoDB `elo_ratings` collection
- **Cache**: Loaded into memory on startup (`DB_ELO_CACHE`)
- **Updates**: After each game result is recorded

### Accessing ELO
```bash
# Get all team ELO ratings
curl "http://localhost:8001/api/ml/elo-ratings?sport_key=basketball_nba"

# Response:
{
  "teams": [
    {"team": "Boston Celtics", "elo": 1623},
    {"team": "Denver Nuggets", "elo": 1592},
    ...
  ]
}
```

---

## 📁 Project Structure

```
/app
├── backend/
│   ├── server.py              # FastAPI main application (4500+ lines)
│   ├── unified_predictor.py   # Combines XGBoost + V5 + V6 algorithms
│   ├── betpredictor_v5.py     # Line movement analysis
│   ├── betpredictor_v6.py     # Rule-based ensemble engine
│   ├── ml_xgboost.py          # ⭐ XGBoost ML system (NEW)
│   │   ├── XGBoostPredictor   # Multi-market prediction class
│   │   ├── HistoricalDataCollector  # ESPN data fetcher
│   │   ├── Backtester         # Backtest validation
│   │   └── EnhancedELOSystem  # Database-backed ELO
│   ├── ml_models.py           # Legacy logistic regression
│   ├── advanced_metrics.py    # ELO & sport-specific metrics
│   │   └── load_elo_cache_from_db()  # Startup ELO loader
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
│   │   │   ├── Dashboard.js   # Stats, ML status widget, top picks
│   │   │   ├── Events.js      # Events with XGBoost predictions
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

### ML Endpoints (NEW)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ml/status` | GET | Get status of all ML models |
| `/api/ml/collect-historical` | POST | Collect 1 season of data from ESPN |
| `/api/ml/train` | POST | Train XGBoost models |
| `/api/ml/predict/{event_id}` | POST | Get ML prediction for all markets |
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
| `/api/analyze-unified/{event_id}` | POST | Full unified analysis (XGBoost + V6 + V5) |
| `/api/analyze-v6/{event_id}` | POST | V6 rule-based analysis only |
| `/api/analyze-v5/{event_id}` | POST | V5 line movement only |

### Performance
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/performance` | GET | Win/loss statistics |
| `/api/notifications` | GET | System notifications |
| `/api/my-bets` | GET | User's tracked bets |

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
  xgb_probability: Number,      // XGBoost home win prob
  xgb_spread_probability: Number,
  xgb_over_probability: Number,
  result: "pending" | "win" | "loss" | "push",
  reasoning: String,
  created_at: String (ISO)
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

### elo_history
```javascript
{
  sport_key: String,
  game_date: String,
  home_team: String,
  away_team: String,
  pre_home_elo: Number,
  post_home_elo: Number,
  elo_change_home: Number
}
```

### opening_odds
Stores first-seen odds for each event (for line movement comparison)

### odds_history
5-minute snapshots of odds for line movement tracking

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

## 🎨 Design System

### Colors
- **Background**: #09090B (dark), #18181B (paper), #27272A (subtle)
- **Text**: #FAFAFA (primary), #A1A1AA (secondary), #71717A (muted)
- **Brand**: #CCFF00 (lime green accent)
- **Purple** (ML): #A855F7 (XGBoost indicators)
- **Semantic**: Success (#22C55E), Danger (#EF4444), Warning (#EAB308)

### Fonts
- **Data/Numbers**: JetBrains Mono
- **Body Text**: Manrope

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
   - Run: `POST /api/ml/train?rebuild_elo=true`
   - Or check: `/var/log/supervisor/backend.err.log` for "Loaded X ELO ratings"

2. **No picks generating**
   - Algorithm is conservative by design
   - Check if games are within 40-minute window
   - View `/api/analyze-unified/{event_id}` for detailed reasoning
   - Check edge requirement (needs 4%+)

3. **XGBoost models not loaded**
   - Run training: `POST /api/ml/train?sport_key=basketball_nba`
   - Check model files exist in `/app/backend/ml_models/`

4. **Line movement not showing**
   - Requires multiple snapshots over time (5-min intervals)
   - Check `/api/data-source-status` for ESPN connection

5. **Results not updating**
   - Background task runs every 15 minutes
   - Check `/var/log/supervisor/backend.err.log`

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

# Get prediction for specific game
curl -X POST "http://localhost:8001/api/ml/predict/401810581?sport_key=basketball_nba" | python3 -m json.tool

# Check ELO for specific teams
curl "http://localhost:8001/api/ml/elo-ratings?sport_key=basketball_nba" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for t in d['teams'][:10]:
    print(f\"{t['team']}: {t['elo']}\")
"
```

---

## 📋 Key Files to Modify

### To Change ML Model Parameters
- **File**: `/app/backend/ml_xgboost.py`
- **Class**: `XGBoostPredictor.train()`
- **Settings**: `n_estimators`, `max_depth`, `learning_rate`

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
- **Settings**: `k_factor`, `home_advantage`, `initial_elo`

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
