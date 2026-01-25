# BetPredictor - AI-Powered Sports Betting Predictor

## Original Problem Statement
Build a betting predictor application that fetches data from all future and in-play sports events, takes into account various popular sportsbook odds and their movement. Acts wisely to track the line movement, does game analysis as to why the lines are moving, runs comparisons and recommends bets. Also after the events are over, keeps track of either that bet lost or won to keep record of the algorithm prediction for users' reference as to how the app is performing.

## User Personas
1. **Sports Bettor** - Wants data-driven insights to make informed betting decisions
2. **Sharp Bettor** - Tracks line movement and odds arbitrage opportunities
3. **Casual Fan** - Looks for AI-powered recommendations with analysis

## Core Requirements (Static)
- Real-time odds from OddsPortal (FREE - no API key required)
- European/decimal odds format
- Track all bookmakers available on OddsPortal (bet365, Pinnacle, FanDuel, DraftKings, BetMGM, Unibet, Betway, 1xBet, etc.)
- AI analysis using GPT-5.2
- Auto-generate recommendations without user interaction
- Line movement tracking with hourly updates
- Auto-track results 5 hours after event completion
- No authentication required

## What's Been Implemented (January 2025)

### Data Source Migration ✅
- **Removed**: The Odds API (paid service)
- **Added**: OddsPortal web scraper (FREE)
  - Playwright-based scraping with httpx fallback
  - Captures opening odds when market first seen
  - Stores hourly snapshots for line movement tracking
  - Imports all bookmakers from OddsPortal dynamically

### Backend (FastAPI + MongoDB)
- [x] GET /api/sports - List available sports (NBA, NFL, MLB, NHL, EPL, La Liga, MMA)
- [x] GET /api/events/{sport_key} - Get upcoming events with odds from OddsPortal
- [x] GET /api/line-movement/{event_id} - Line movement history with opening/current odds
- [x] GET /api/odds-comparison/{event_id} - Compare odds across books
- [x] POST /api/analyze - AI analysis using GPT-5.2
- [x] GET /api/recommendations - Get auto-generated AI picks (70%+ confidence filter, 3-day window)
- [x] PUT /api/result - Track win/loss/push
- [x] GET /api/performance - Performance statistics
- [x] GET /api/scraper-status - OddsPortal scraper status
- [x] POST /api/scrape-odds - Manual scrape trigger
- [x] GET /api/scores/{sport_key} - Real-time scores from ESPN API (NEW)
- [x] GET /api/live-scores - All in-progress games across sports (NEW)
- [x] GET /api/pending-results - Pending predictions by status (NEW)
- [x] POST /api/check-results - Trigger result checking (NEW)

### Background Tasks (Automatic)
- [x] Auto-scrape OddsPortal every hour for odds updates
- [x] Store opening odds when event first seen
- [x] Store hourly odds snapshots for line movement
- [x] Auto-generate recommendations every 4 hours (70%+ confidence, 3-day window)
- [x] Check line movements hourly and update confidence
- [x] **Auto-check results every 15 MINUTES using ESPN API** (UPDATED)
- [x] **Automatic win/loss calculation based on final scores** (NEW)

### Frontend (React + Tailwind)
- [x] Dashboard with auto-generated Top Picks (filtered to 70%+ confidence)
- [x] Events page with decimal odds from multiple bookmakers
- [x] **Line Movement page** with:
  - Sport selector (NBA, NFL, MLB, NHL, EPL)
  - Event list with bookmaker counts
  - Interactive chart showing odds movement over time
  - Opening vs Current odds comparison with % change
  - Bookmakers list with individual odds
  - Source: OddsPortal badge
- [x] Odds Comparison with best odds highlighted
- [x] Performance tracking with win/loss history (auto-updated)
- [x] Notifications system (includes result notifications)
- [x] Settings page

### Key Features (January 2025 Update)
- **Real-time Score Tracking**: ESPN API integration for live and final scores
- **Automatic Result Calculation**: Win/loss/push automatically determined based on:
  - Moneyline: Which team won
  - Spread: Did team cover the spread
  - Total (O/U): Combined score vs line
- **70%+ Confidence Filter**: Only high-confidence recommendations shown
- **Time Window Filter**: Pre-match bets only (today to 3 days out)
- **15-Minute Result Checking**: Background task checks completed games frequently

### Removed Features
- [x] API Keys management (no longer needed - OddsPortal is free)
- [x] Bankroll feature
- [x] Standalone Analytics page (merged into Performance)

### Design
- Dark terminal/HFT theme with lime green (#ADFF2F) and cyan (#00CED1) accents
- JetBrains Mono for data, Manrope for body
- All bookmakers from OddsPortal displayed dynamically

## Database Collections
- `events_cache` - Cached scraped events
- `opening_odds` - First seen odds for each event
- `odds_history` - Hourly odds snapshots for line movement
- `predictions` - AI-generated betting predictions
- `notifications` - System alerts

## Tech Stack
- **Backend**: FastAPI, MongoDB (motor), Playwright, httpx
- **Frontend**: React, Tailwind CSS, Recharts, Lucide Icons, Shadcn UI
- **AI Model**: GPT-5.2 (OpenAI via Emergent LLM Key)
- **Data Source**: OddsPortal (FREE web scraping)

## Available Bookmakers (from OddsPortal)
- bet365, Pinnacle, 1xBet, Betfair, Unibet, Betway
- William Hill, Betsson, bwin, 888sport
- DraftKings, FanDuel, BetMGM, Caesars, PointsBet
- Bovada, BetOnline.ag, MyBookie, Betcris, Cloudbet, Stake

## Prioritized Backlog

### P0 - Critical ✅ DONE
- [x] OddsPortal integration (replaces paid API)
- [x] Line movement tracking with opening/current odds
- [x] Hourly odds snapshots
- [x] Auto-generate recommendations

### P1 - Important
- [ ] Push notifications for sharp line movements (>5% change)
- [ ] More robust error handling for OddsPortal scraping
- [ ] Spreads and totals in line movement chart

### P2 - Nice to Have
- [ ] More sports coverage
- [ ] Historical performance export
- [ ] Social sharing

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/sports | GET | List available sports |
| /api/events/{sport_key} | GET | Get events with odds |
| /api/line-movement/{event_id} | GET | Line movement history |
| /api/odds-comparison/{event_id} | GET | Compare bookmaker odds |
| /api/analyze | POST | AI game analysis |
| /api/recommendations | GET | AI betting picks |
| /api/result | PUT | Update prediction result |
| /api/performance | GET | Performance stats |
| /api/scraper-status | GET | Scraper health check |
| /api/scrape-odds | POST | Manual scrape trigger |
