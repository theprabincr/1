# BetPredictor - AI-Powered Sports Betting Predictor

A sophisticated full-stack application that provides ML-powered sports betting recommendations by analyzing real-time odds, team statistics, line movements, and multiple predictive models.

## 🎯 Overview

BetPredictor uses a **Unified Prediction Engine** that combines two algorithms:
- **V6 (ML Ensemble)** - 5 machine learning models with 70% weight
- **V5 (Line Movement)** - Sharp money detection with 30% weight

Only when both algorithms align (or V6 shows strong confidence) does the system generate a pick.

## ✨ Features

### Core Functionality
- **Real-time Data**: Live odds, scores, and team stats from ESPN/DraftKings
- **ML Ensemble**: 5 independent models vote on each pick (ELO, Context, Line Movement, Statistical, Psychology)
- **Smart Predictions**: Auto-generates picks 40 minutes before game time after lineup confirmations
- **Line Movement Tracking**: Monitors odds changes with 5-minute snapshots
- **Auto Result Tracking**: Checks game results every 15 minutes via ESPN API
- **Adaptive Learning**: Models self-adjust weights based on historical accuracy

### Prediction Analysis Includes
- **Team Strength**: ELO ratings with home advantage adjustments
- **Recent Form & Records**: Season records, home/away splits, last 10 games, win streaks
- **Last 5 Games**: Game-by-game results with scores
- **Situational Factors**: Rest days, back-to-back detection, travel impact
- **Injury Impact**: Position-weighted, severity-adjusted analysis
- **Line Movement**: Sharp money detection, reverse line movement (RLM)
- **Monte Carlo Simulations**: 1,000+ game simulations
- **Market Psychology**: Public bias detection, contrarian opportunities

### Sports Covered
| Sport | Key | Features |
|-------|-----|----------|
| 🏀 NBA | `basketball_nba` | Four Factors, pace analysis, full ELO |
| 🏈 NFL | `americanfootball_nfl` | Efficiency metrics, QB injury weighting |
| 🏒 NHL | `hockey_nhl` | Possession metrics, goalie impact |
| ⚽ EPL | `soccer_epl` | Poisson modeling, home advantage |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED PREDICTOR                         │
├──────────────────────────┬──────────────────────────────────┤
│   V6 ML ENSEMBLE (70%)   │   V5 LINE MOVEMENT (30%)         │
├──────────────────────────┼──────────────────────────────────┤
│ • ELO Model              │ • Sharp Money Detection          │
│ • Context Model          │ • Reverse Line Movement (RLM)    │
│ • Line Movement Model    │ • Steam Move Detection           │
│ • Statistical Model      │ • Market Phase Analysis          │
│ • Psychology Model       │ • Opening vs Current Comparison  │
├──────────────────────────┴──────────────────────────────────┤
│              DECISION: 60%+ Combined Confidence              │
│              REQUIRES: 3/5 Models Agree (V6)                 │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
/app
├── backend/
│   ├── server.py              # FastAPI main application
│   ├── unified_predictor.py   # Combines V5 + V6 algorithms
│   ├── betpredictor_v5.py     # Line movement analysis
│   ├── betpredictor_v6.py     # ML ensemble engine
│   ├── ml_models.py           # Logistic regression & ensemble
│   ├── advanced_metrics.py    # ELO & sport-specific metrics
│   ├── context_analyzer.py    # Rest, travel, altitude analysis
│   ├── injury_analyzer.py     # Position-weighted injury impact
│   ├── market_psychology.py   # Bias detection & contrarian
│   ├── simulation_engine.py   # Monte Carlo & Poisson modeling
│   ├── adaptive_learning.py   # Self-adjusting model weights
│   ├── espn_data_provider.py  # ESPN odds & stats fetcher
│   ├── espn_scores.py         # Live score tracking
│   └── line_movement_analyzer.py
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main router
│   │   ├── pages/
│   │   │   ├── Dashboard.js   # Stats & top picks
│   │   │   ├── Events.js      # All events with odds
│   │   │   ├── LineMovement.js# Line movement charts
│   │   │   ├── Performance.js # Win/loss tracking
│   │   │   └── Settings.js    # App settings
│   │   └── App.css            # Styles
│   └── package.json
├── memory/
│   └── PRD.md                 # Product requirements
└── README.md
```

## 🔌 API Endpoints

### Events & Odds
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/` | GET | Health check |
| `/api/sports` | GET | List available sports |
| `/api/events/{sport_key}` | GET | Get events with odds |
| `/api/line-movement/{event_id}` | GET | Line movement history |
| `/api/data-source-status` | GET | ESPN data source status |

### Predictions
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recommendations` | GET | Get AI recommendations (60%+ confidence) |
| `/api/analyze-unified/{event_id}` | POST | Manual unified analysis |
| `/api/analyze-v6/{event_id}` | POST | V6 ML analysis only |
| `/api/analyze-v5/{event_id}` | POST | V5 line movement only |

### Performance
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/performance` | GET | Win/loss statistics |
| `/api/notifications` | GET | System notifications |

## ⚙️ Background Tasks

| Task | Frequency | Description |
|------|-----------|-------------|
| Line Movement Checker | 5 minutes | Snapshot odds for all events |
| Prediction Generator | 40 min before game | Auto-generate picks |
| Result Checker | 15 minutes | Check completed games via ESPN |
| Adaptive Learning | After each result | Update model weights |

## 🗄️ Database Collections

### predictions
```javascript
{
  id: String (UUID),
  event_id: String,
  sport_key: String,
  home_team: String,
  away_team: String,
  prediction: String,           // Team name
  confidence: Number (0-1),
  odds_at_prediction: Number,
  prediction_type: "moneyline" | "spread" | "total",
  result: "pending" | "win" | "loss" | "push",
  reasoning: String,            // Full detailed analysis
  edge: Number,
  consensus_level: "strong" | "weak" | "v6_only",
  created_at: String (ISO)
}
```

### opening_odds
Stores first-seen odds for each event (for line movement comparison)

### odds_history
Hourly snapshots of odds for line movement tracking

### model_performance
Tracks accuracy of each sub-model for adaptive weight adjustment

## 🎨 Design System

### Colors
- **Background**: #09090B (dark), #18181B (paper), #27272A (subtle)
- **Text**: #FAFAFA (primary), #A1A1AA (secondary), #71717A (muted)
- **Brand**: #CCFF00 (lime green accent)
- **Semantic**: Success (#22C55E), Danger (#EF4444), Warning (#EAB308)

### Fonts
- **Data/Numbers**: JetBrains Mono
- **Body Text**: Manrope

## 🚀 Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11)
- **Database**: MongoDB (motor async driver)
- **Data Source**: ESPN API (odds, scores, stats)

### Frontend
- **Framework**: React 18
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React

### ML Components
- Logistic Regression with sport-specific weights
- Monte Carlo Simulation (1,000+ iterations)
- Poisson Modeling for low-scoring sports
- 5-Model Ensemble with dynamic weight adjustment

## 📊 Algorithm Decision Requirements

A pick is only recommended when ALL conditions are met:

1. ✅ **Ensemble Confidence ≥ 60%** (combined V5+V6)
2. ✅ **At least 3 out of 5 V6 models agree**
3. ✅ **Model Agreement ≥ 25%**
4. ✅ **Clear probability edge (>55% or <45%)**
5. ✅ **Minimum edge ≥ 4%**

If any requirement fails, **NO PICK** is generated with detailed reasoning.

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

## 📈 Performance Tracking

The app automatically tracks:
- **Win Rate**: Percentage of winning predictions
- **ROI**: Return on investment based on $100 fixed bets
- **Model Accuracy**: Individual sub-model performance
- **Brier Score**: Probability calibration metric

## 🛠️ Troubleshooting

### Common Issues

1. **No picks generating**
   - Algorithm is conservative by design
   - Check if games are within 40-minute window
   - View `/api/analyze-unified/{event_id}` for detailed reasoning

2. **Line movement not showing**
   - Requires multiple snapshots over time
   - Check `/api/data-source-status` for ESPN connection

3. **Results not updating**
   - Background task runs every 15 minutes
   - Check `/var/log/supervisor/backend.err.log`

### Logs
```bash
# Backend logs
tail -f /var/log/supervisor/backend.err.log

# Check supervisor status
sudo supervisorctl status

# Restart services
sudo supervisorctl restart all
```

## 📝 License

MIT License - Feel free to modify and use for personal projects.

## 🙏 Credits

- **Data Source**: ESPN API
- **Icons**: [Lucide](https://lucide.dev/)
- **Charts**: [Recharts](https://recharts.org/)
