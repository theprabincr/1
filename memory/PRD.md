# BetPredictor - Product Requirements Document

## Overview

BetPredictor is an AI-powered sports betting prediction application that uses machine learning ensemble models to analyze sports events and provide data-driven betting recommendations.

## Original Problem Statement

> Build a betting predictor application that fetches data from all future and in-play sports events, takes into account various popular sportsbook odds and their movement. Acts wisely to track the line movement, does game analysis as to why the lines are moving, runs comparisons and recommends bets. Also after the events are over, keeps track of either that bet lost or won to keep record of the algorithm prediction for users' reference as to how the app is performing.

## User Personas

1. **Data-Driven Bettor** - Wants ML-powered insights backed by statistics and analysis
2. **Sharp Bettor** - Tracks line movement and identifies sharp money opportunities
3. **Casual Sports Fan** - Looks for easy-to-understand recommendations with reasoning

## Core Architecture

### Unified Prediction Engine

The application uses a **two-algorithm approach** that combines:

| Algorithm | Weight | Focus |
|-----------|--------|-------|
| **V6 (ML Ensemble)** | 70% | 5 independent ML models |
| **V5 (Line Movement)** | 30% | Sharp money & RLM detection |

### V6 ML Ensemble - 5 Sub-Models

1. **ELO Model** - Team strength ratings with home advantage
2. **Context Model** - Rest, travel, altitude, schedule factors
3. **Line Movement Model** - Odds movement signals
4. **Statistical Model** - Monte Carlo + Logistic Regression
5. **Psychology Model** - Market bias & contrarian opportunities

### Decision Requirements

A pick is ONLY generated when:
- Combined confidence ≥ 60%
- At least 3/5 V6 models agree
- Model agreement ≥ 25%
- Clear probability edge exists
- Minimum edge ≥ 4%

## Features Implemented

### Data Collection ✅
- [x] ESPN API integration for odds, scores, and team stats
- [x] DraftKings odds via ESPN
- [x] Real-time score tracking
- [x] Team records and recent form
- [x] Injury reports with position weighting

### Analysis Engine ✅
- [x] ELO rating system with sport-specific configurations
- [x] Advanced metrics (NBA Four Factors, NFL Efficiency, NHL Possession)
- [x] Context analysis (rest days, travel, altitude, schedule)
- [x] Smart injury analysis (position-weighted, severity-adjusted)
- [x] Monte Carlo simulations (1,000+ iterations)
- [x] Poisson modeling for low-scoring sports
- [x] Market psychology (bias detection, contrarian opportunities)
- [x] Logistic regression with sport-specific weights
- [x] 5-model ensemble voting system

### Prediction System ✅
- [x] Unified predictor combining V5 + V6
- [x] Auto-generate predictions 40 minutes before game
- [x] Comprehensive reasoning with all factors
- [x] Edge calculation against odds
- [x] Confidence scoring

### Line Movement ✅
- [x] Opening odds capture (when event first seen)
- [x] 5-minute snapshot intervals
- [x] Sharp money detection
- [x] Reverse line movement (RLM) identification
- [x] Steam move detection

### Result Tracking ✅
- [x] Auto-check results every 15 minutes
- [x] ESPN API score fetching
- [x] Win/loss/push determination
- [x] Performance statistics calculation
- [x] Adaptive learning (model weight adjustment)

### Frontend ✅
- [x] Dashboard with stats and top picks
- [x] Events page with live odds
- [x] Line Movement page with charts
- [x] Performance tracking with ROI
- [x] Collapsible, color-coded analysis sections
- [x] Notifications system
- [x] Settings page

## Technical Stack

### Backend
- **Framework**: FastAPI (Python 3.11)
- **Database**: MongoDB with motor async driver
- **Data Source**: ESPN API (free, no key required)

### Frontend
- **Framework**: React 18
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React

### ML Components
- Logistic Regression
- Monte Carlo Simulation
- Poisson Distribution Modeling
- Ensemble Voting System
- Adaptive Weight Learning

## Database Schema

### predictions
```javascript
{
  id: UUID,
  event_id: String,
  sport_key: String,
  home_team: String,
  away_team: String,
  prediction: String,
  confidence: Number (0-1),
  odds_at_prediction: Number,
  prediction_type: "moneyline" | "spread" | "total",
  result: "pending" | "win" | "loss" | "push",
  reasoning: String,  // Full detailed analysis
  edge: Number,
  consensus_level: String,
  created_at: ISO String
}
```

### opening_odds
First-seen odds for each event

### odds_history
Snapshots every 5 minutes for line movement

### model_performance
Sub-model accuracy tracking for weight adjustment

## API Endpoints

### Core
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/` | GET | Health check |
| `/api/sports` | GET | Available sports |
| `/api/events/{sport_key}` | GET | Events with odds |
| `/api/recommendations` | GET | AI picks (60%+ confidence) |
| `/api/performance` | GET | Win/loss statistics |

### Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze-unified/{event_id}` | POST | Full unified analysis |
| `/api/analyze-v6/{event_id}` | POST | V6 ML ensemble only |
| `/api/analyze-v5/{event_id}` | POST | V5 line movement only |
| `/api/line-movement/{event_id}` | GET | Line movement history |

## Background Tasks

| Task | Frequency | Purpose |
|------|-----------|---------|
| Line Movement Checker | 5 min | Snapshot odds |
| Prediction Generator | 40 min before game | Auto-generate picks |
| Result Checker | 15 min | Update win/loss |
| Adaptive Learning | Per result | Adjust model weights |

## Analysis Sections (Frontend Display)

The prediction analysis is displayed with collapsible, color-coded sections:

1. **Prediction Overview** (green) - Pick, confidence, edge
2. **Model Agreement** (blue) - All 5 model votes
3. **Team Strength** (purple) - ELO ratings
4. **Recent Form & Records** (purple) - Season record, streaks, last 5 games
5. **Situational Factors** (gray) - Rest, travel, back-to-back
6. **Injury Impact** (red) - Key injuries
7. **Line Movement** (cyan) - Sharp money, RLM
8. **Simulation Results** (orange) - Monte Carlo probabilities
9. **Key Factors** (green) - Summary of most important factors

## Future Enhancements (Backlog)

### P1 - High Priority
- [ ] Push notifications for significant line movements (>5%)
- [ ] Player prop analysis
- [ ] More sports coverage

### P2 - Medium Priority
- [ ] Historical performance export
- [ ] Bankroll management tools
- [ ] Social sharing

### P3 - Nice to Have
- [ ] Live in-game betting model
- [ ] Deep learning neural networks
- [ ] Multi-league arbitrage detection

## Design Principles

1. **Conservative by Design** - Only recommend high-confidence picks
2. **Transparent Analysis** - Show all factors considered
3. **Adaptive** - Self-improve based on results
4. **User-Friendly** - Clean, collapsible UI for complex data

## Changelog

### Latest Update (January 2026)
- Enhanced analysis display with collapsible, color-coded sections
- Added Recent Form & Records section with last 5 games
- Full V6 reasoning now included in unified analysis
- Fixed schema mismatch for predictions (odds_at_prediction)
- Lowered confidence threshold from 70% to 60% for dashboard

### Previous Updates
- Unified Predictor (V5+V6) integration
- ESPN API data source
- Adaptive learning system
- Monte Carlo simulations
- 5-model ensemble system
