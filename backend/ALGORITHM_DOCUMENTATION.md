# BetPredictor Algorithm Documentation

## Overview

BetPredictor uses a **Unified Prediction Engine** that combines two sophisticated algorithms to provide high-confidence betting recommendations.

## Algorithm Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED PREDICTOR                             │
│              (unified_predictor.py)                              │
├───────────────────────────┬─────────────────────────────────────┤
│    V6 ML ENSEMBLE (70%)   │    V5 LINE MOVEMENT (30%)           │
│    (betpredictor_v6.py)   │    (betpredictor_v5.py)             │
├───────────────────────────┼─────────────────────────────────────┤
│ 5 Independent Models:     │ Sharp Money Detection:              │
│ • ELO Model               │ • Opening vs Current Odds           │
│ • Context Model           │ • Reverse Line Movement (RLM)       │
│ • Line Movement Model     │ • Steam Move Detection              │
│ • Statistical Model       │ • Market Phase Analysis             │
│ • Psychology Model        │ • Sharp vs Public Money             │
└───────────────────────────┴─────────────────────────────────────┘
```

## V6 ML Ensemble - Detailed Breakdown

### 1. ELO Model (`advanced_metrics.py`)

**Purpose**: Measure team strength using chess-style ratings

**Features**:
- Sport-specific K-factors (how much ratings change per game)
- Home advantage adjustments:
  - NBA: +100 points
  - NFL: +65 points
  - NHL: +50 points
- Win probability calculation: `P = 1 / (1 + 10^(-diff/400))`

### 2. Context Model (`context_analyzer.py`)

**Purpose**: Analyze situational factors affecting performance

**Factors Analyzed**:
| Factor | Impact | Notes |
|--------|--------|-------|
| Back-to-back games | -8% | Playing consecutive days |
| Well-rested (3+ days) | +4% | Extra rest bonus |
| Cross-country travel | -5% | >2000 miles |
| Timezone change | -1.2% per hour | West-to-East harder |
| Altitude (Denver/Utah) | +3% home | Visiting teams struggle |
| Schedule congestion | -3% | 4+ games in 7 days |

### 3. Line Movement Model (`line_movement_analyzer.py`)

**Purpose**: Detect professional bettor activity

**Signals Detected**:
- **Sharp Money**: Early significant moves (>3% before game)
- **Reverse Line Movement (RLM)**: Line moves opposite public betting
- **Steam Moves**: Multiple books move together quickly

### 4. Statistical Model (`ml_models.py` + `simulation_engine.py`)

**Purpose**: Pure statistical probability calculation

**Components**:
1. **Logistic Regression**: 10+ weighted features
   - ELO difference (0.005-0.006 per point)
   - Form difference (0.22-0.30 weight)
   - Rest advantage (0.12-0.15 weight)
   - Injury impact (-0.45 to -0.60 weight)

2. **Monte Carlo Simulation**: 1,000+ game simulations
   - Gaussian distribution for NBA (high-scoring)
   - Poisson distribution for NFL/NHL/Soccer (low-scoring)
   - Outputs: Win %, Spread Cover %, Over/Under %

### 5. Psychology Model (`market_psychology.py`)

**Purpose**: Identify market inefficiencies and biases

**Biases Detected**:
| Bias Type | Description |
|-----------|-------------|
| Home Favorite | Public overvalues home favorites |
| Overs | Public prefers betting over |
| Recency | Hot teams overvalued |
| Popular Team | Lakers, Cowboys, etc. inflated |
| Primetime | National TV games skewed |

**Contrarian Opportunities**: Fade the public when sharp money disagrees

## V5 Line Movement Algorithm

### Sharp Money Detection

```python
# Early phase detection (3+ days before)
if phase == "early" and abs(ml_movement) > 3%:
    sharp_money_detected = True

# Reverse Line Movement
if line_moved_one_direction and public_on_opposite_side:
    rlm_detected = True
    sharp_side = direction_of_line_move

# Steam move (synchronized book movement)
if all_books_moved_together_in_2hr_window:
    steam_detected = True
```

### Market Phase Analysis

| Phase | Time Until Game | Characteristics |
|-------|-----------------|-----------------|
| Early | >48 hours | Sharp money dominates |
| Middle | 24-48 hours | Mixed action |
| Late | <24 hours | Public betting heavy |

## Unified Combination Logic

```python
# Weight distribution
V6_WEIGHT = 0.70  # ML Ensemble
V5_WEIGHT = 0.30  # Line Movement

# Combination scenarios
if v5_has_pick and v6_has_pick:
    if v5_pick == v6_pick:
        # STRONG CONSENSUS (+10% bonus)
        confidence = (v6_conf * 0.70 + v5_conf * 0.30) + 0.10
    else:
        # CONFLICT (-15% penalty)
        confidence = v6_conf * 0.70 - 0.15

elif v6_has_pick and not v5_has_pick:
    # V6 ONLY (no line movement signal)
    confidence = v6_conf * 0.70

else:
    # NO PICK (insufficient signals)
    return None
```

## Decision Requirements

A pick is ONLY generated when ALL conditions are met:

| Requirement | Threshold | Reason |
|-------------|-----------|--------|
| Combined Confidence | ≥ 60% | High confidence only |
| Models Agreeing | ≥ 3/5 | Consensus required |
| Model Agreement | ≥ 25% | Low variance in predictions |
| Probability Edge | >55% or <45% | Clear directional signal |
| Minimum Edge | ≥ 4% | Profitable vs odds |

## Analysis Output Structure

### Reasoning Sections

1. **PREDICTION OVERVIEW**
   - Pick, Confidence, Edge, Win Probability

2. **MODEL AGREEMENT**
   - Each of 5 models: agrees/disagrees + confidence %

3. **TEAM STRENGTH**
   - ELO ratings for both teams
   - Advantage calculation

4. **RECENT FORM & RECORDS**
   - Season Record (e.g., 18-27)
   - Home/Away Record
   - Last 10 Games (6W-4L)
   - Current Streak
   - Average Point Margin

5. **LAST 5 GAMES**
   - Game-by-game results with scores
   - W/L @ Opponent (Score)

6. **SITUATIONAL FACTORS**
   - Rest days, back-to-back
   - Travel impact

7. **INJURY IMPACT**
   - Key injuries with position weights

8. **LINE MOVEMENT**
   - Opening vs Current odds
   - Sharp money signals
   - RLM detection

9. **SIMULATION RESULTS**
   - Monte Carlo probabilities
   - Expected scores
   - Over/Under probability

10. **KEY FACTORS**
    - Top 5 most important factors

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `unified_predictor.py` | ~400 | Combines V5+V6 |
| `betpredictor_v6.py` | ~850 | ML ensemble engine |
| `betpredictor_v5.py` | ~300 | Line movement analysis |
| `ml_models.py` | ~430 | Logistic regression + ensemble |
| `advanced_metrics.py` | ~430 | ELO & sport metrics |
| `context_analyzer.py` | ~380 | Situational analysis |
| `injury_analyzer.py` | ~220 | Position-weighted injuries |
| `market_psychology.py` | ~290 | Bias & contrarian |
| `simulation_engine.py` | ~310 | Monte Carlo + Poisson |
| `adaptive_learning.py` | ~200 | Self-improving weights |

**Total: ~3,800+ lines of betting intelligence**

## Performance Tracking

### Metrics Tracked
- **Win Rate**: % of winning predictions
- **ROI**: Return on $100 fixed bets
- **Brier Score**: Probability calibration
- **Model Accuracy**: Per sub-model accuracy

### Adaptive Learning

After each result:
1. Update model accuracy metrics
2. Recalculate model weights
3. Better-performing models get higher weights
4. Continuously improves over time

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/analyze-unified/{event_id}` | Full unified analysis |
| `POST /api/analyze-v6/{event_id}` | V6 ML ensemble only |
| `POST /api/analyze-v5/{event_id}` | V5 line movement only |
| `GET /api/recommendations` | All picks ≥60% confidence |
| `GET /api/performance` | Win/loss statistics |
