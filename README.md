# BetPredictor - AI-Powered Sports Betting Predictor

A full-stack application that provides AI-powered sports betting recommendations by analyzing real-time odds from multiple sportsbooks.

## Features

### Core Functionality
- **Real-time Odds Tracking**: Fetches live odds from The Odds API across 7 major sportsbooks
- **AI-Powered Analysis**: Uses GPT-5.2 to analyze all markets (Moneyline, Spreads, Totals) and recommend best value bets
- **Auto-Generated Recommendations**: Automatically generates picks every 6 hours sorted by confidence
- **Line Movement Tracking**: Monitors odds changes hourly and updates confidence accordingly
- **Auto Result Tracking**: Automatically fetches game scores and updates prediction results
- **Multi-Sport Coverage**: NBA, NFL, MLB, NHL, Soccer (EPL, La Liga), MMA, Tennis

### Sportsbooks Tracked
1. DraftKings
2. FanDuel
3. BetMGM
4. Pinnacle
5. Unibet
6. Betway
7. BetOnline

### Market Types Analyzed
- **Moneyline (ML)**: Straight win/loss bets
- **Spread (SPR)**: Point spread bets
- **Totals (O/U)**: Over/Under bets

## Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11)
- **Database**: MongoDB
- **AI Integration**: Emergent Integrations (GPT-5.2)
- **Odds Data**: The Odds API

### Frontend
- **Framework**: React 18
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React
- **Date Handling**: date-fns

## Project Structure

```
/app
├── backend/
│   ├── server.py           # Main FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js         # Main React component with routing
│   │   ├── App.css        # Component styles
│   │   ├── index.css      # Global styles (Tailwind)
│   │   └── pages/
│   │       ├── Dashboard.js      # Main dashboard with stats & picks
│   │       ├── Events.js         # Events listing with odds
│   │       ├── LineMovement.js   # Line movement charts
│   │       ├── OddsComparison.js # Side-by-side odds comparison
│   │       ├── Predictions.js    # AI predictions page
│   │       └── Performance.js    # Win/loss tracking
│   ├── package.json       # Node dependencies
│   ├── tailwind.config.js # Tailwind configuration
│   └── .env              # Frontend environment variables
├── memory/
│   └── PRD.md            # Product Requirements Document
└── README.md             # This file
```

## API Endpoints

### Events & Odds
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sports` | GET | List available sports |
| `/api/events/{sport_key}` | GET | Get events with odds (cached 30 min) |
| `/api/event/{event_id}` | GET | Get specific event details |
| `/api/odds-comparison/{event_id}` | GET | Compare odds across sportsbooks |
| `/api/line-movement/{event_id}` | GET | Get line movement history |
| `/api/scores/{sport_key}` | GET | Get recent game scores |

### Predictions
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recommendations` | GET | Get AI recommendations (sorted by confidence) |
| `/api/recommendations` | POST | Create new recommendation |
| `/api/generate-recommendations` | POST | Generate AI picks for a sport |
| `/api/analyze` | POST | Run AI analysis on specific event |

### System
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/` | GET | Health check |
| `/api/api-usage` | GET | API usage stats and background task status |
| `/api/performance` | GET | Win/loss performance statistics |
| `/api/check-results` | POST | Trigger result checking |

## Background Tasks

The application runs three background tasks automatically:

1. **Recommendation Generator** (Every 6 hours)
   - Analyzes upcoming events across all sports
   - Generates AI-powered picks for best value bets
   - Skips generation if API calls < 30 remaining

2. **Line Movement Checker** (Every hour)
   - Monitors odds changes for pending predictions
   - Updates confidence based on movement direction
   - Adds movement notes to analysis

3. **Result Checker** (Every 2 hours)
   - Checks for completed games (5+ hours after start)
   - Fetches scores from The Odds API
   - Updates prediction results (win/loss/push)

## Environment Variables

### Backend (.env)
```env
MONGO_URL="mongodb://localhost:27017"
DB_NAME="test_database"
CORS_ORIGINS="*"
EMERGENT_LLM_KEY=your_emergent_key
ODDS_API_KEY=your_odds_api_key
```

### Frontend (.env)
```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB
- The Odds API key (free tier: 500 requests/month)
- Emergent LLM key (for AI analysis)

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Add your API keys to .env
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend Setup
```bash
cd frontend
yarn install
# Update REACT_APP_BACKEND_URL in .env if needed
yarn start
```

## API Rate Limiting

The app includes smart API usage management:

- **30-minute cache** for events data
- **Auto-skip** recommendation generation when < 30 API calls remaining
- **API usage tracking** displayed in sidebar
- **Free tier**: 500 requests/month

## Odds Format

All odds are displayed in **European/Decimal format**:
- 1.50 = -200 American (heavy favorite)
- 2.00 = +100 American (even money)
- 3.00 = +200 American (underdog)

**Odds Range Filter**: 1.5 - 20 (excludes extreme values)

## Database Schema

### Predictions Collection
```javascript
{
  id: String (UUID),
  event_id: String,
  sport_key: String,
  home_team: String,
  away_team: String,
  commence_time: String (ISO),
  prediction_type: "moneyline" | "spread" | "total",
  predicted_outcome: String,
  confidence: Number (0-1),
  analysis: String,
  ai_model: String,
  odds_at_prediction: Number,
  result: "pending" | "win" | "loss" | "push",
  created_at: String (ISO),
  line_movement_pct: Number (optional),
  current_odds: Number (optional)
}
```

### Odds History Collection
```javascript
{
  event_id: String,
  timestamp: String (ISO),
  bookmaker: String,
  bookmaker_title: String,
  market: String,
  outcomes: Array
}
```

## Design System

### Colors
- **Background**: #09090B (default), #18181B (paper), #27272A (subtle)
- **Text**: #FAFAFA (primary), #A1A1AA (secondary), #71717A (muted)
- **Brand**: #CCFF00 (lime green)
- **Semantic**: Success (#22C55E), Danger (#EF4444), Warning (#EAB308)

### Fonts
- **Headings/Data**: JetBrains Mono
- **Body**: Manrope

## Future Improvements

### P1 - High Priority
- Push notifications for significant line movements
- Historical odds tracking (requires API upgrade)
- Prop bets support

### P2 - Medium Priority
- More sports coverage
- Bankroll management
- Social sharing
- User accounts

## Troubleshooting

### Common Issues

1. **No events showing**
   - Check if ODDS_API_KEY is valid
   - Verify the sport has upcoming events
   - Check API usage limits

2. **AI analysis not working**
   - Verify EMERGENT_LLM_KEY is set
   - Check backend logs for errors

3. **Predictions not auto-updating**
   - Background tasks run on schedule (2-6 hours)
   - Check `/api/api-usage` for task status

### Logs
```bash
# Backend logs
tail -f /var/log/supervisor/backend.err.log

# Check supervisor status
sudo supervisorctl status
```

## License

MIT License - Feel free to modify and use for personal projects.

## Credits

- **Odds Data**: [The Odds API](https://the-odds-api.com/)
- **AI Analysis**: OpenAI GPT-5.2 via Emergent Integrations
- **Icons**: [Lucide](https://lucide.dev/)
