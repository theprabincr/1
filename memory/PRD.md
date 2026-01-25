# BetPredictor - AI-Powered Sports Betting Predictor

## Original Problem Statement
Build a betting predictor application that fetches data from all future and in-play sports events, takes into account various popular sportsbook odds and their movement. Acts wisely to track the line movement, does game analysis as to why the lines are moving, runs comparisons and recommends bets. Also after the events are over, keeps track of either that bet lost or won to keep record of the algorithm prediction for users' reference as to how the app is performing.

## User Personas
1. **Sports Bettor** - Wants data-driven insights to make informed betting decisions
2. **Sharp Bettor** - Tracks line movement and odds arbitrage opportunities
3. **Casual Fan** - Looks for AI-powered recommendations with analysis

## Core Requirements (Static)
- Real-time odds from The Odds API (API key configured)
- European/decimal odds format
- Track 7 active sportsbooks: DraftKings, FanDuel, BetMGM, Pinnacle, Unibet, Betway, BetOnline
- AI analysis using GPT-5.2
- Auto-generate recommendations without user interaction
- Line movement tracking with hourly updates
- Auto-track results 5 hours after event completion
- No authentication required

## What's Been Implemented (January 2025)

### Backend (FastAPI + MongoDB)
- [x] GET /api/sports - List available sports
- [x] GET /api/events/{sport_key} - Get upcoming events with decimal odds
- [x] GET /api/api-usage - Track API calls remaining
- [x] GET /api/line-movement/{event_id} - Line movement history
- [x] GET /api/odds-comparison/{event_id} - Compare odds across books
- [x] POST /api/analyze - AI analysis using GPT-5.2
- [x] GET /api/recommendations - Get auto-generated AI picks
- [x] PUT /api/result - Track win/loss/push
- [x] GET /api/performance - Performance statistics
- [x] GET /api/scores/{sport} - Fetch completed game scores

### Background Tasks (Automatic)
- [x] Auto-generate recommendations every 2 hours
- [x] Check line movements hourly and update confidence
- [x] Auto-check results hourly for completed events
- [x] Track API usage from response headers

### Frontend (React + Tailwind)
- [x] Dashboard with auto-generated Top Picks
- [x] API usage counter in sidebar (Calls Left: XXX)
- [x] Events page with decimal odds from 7 active sportsbooks
- [x] Line Movement page with charts
- [x] Odds Comparison with best odds highlighted
- [x] Performance tracking with win/loss history

### Design
- Dark terminal/HFT theme with lime green accents
- JetBrains Mono for data, Manrope for body
- Only active sportsbooks displayed (removed bet365, Caesars, PointsBet)

## Prioritized Backlog

### P0 - Critical
- [x] Real API integration ✅
- [x] Auto-generate recommendations ✅
- [x] API usage tracking ✅

### P1 - Important
- [ ] Push notifications for sharp line movements
- [ ] Historical odds tracking (requires API upgrade)
- [ ] Prop bets support

### P2 - Nice to Have
- [ ] More sports coverage
- [ ] Bankroll management
- [ ] Social sharing

## Tech Stack
- **Backend**: FastAPI, MongoDB, emergentintegrations (LLM)
- **Frontend**: React, Tailwind CSS, Recharts, Lucide Icons
- **AI Model**: GPT-5.2 (OpenAI via Emergent)
- **Data Source**: The Odds API (241 calls remaining)

## Active Sportsbooks
1. DraftKings
2. FanDuel
3. BetMGM
4. Pinnacle
5. Unibet
6. Betway
7. BetOnline

## Next Tasks
1. Monitor API usage (241 calls remaining)
2. Add push notifications for line movement alerts
3. Consider API tier upgrade for historical data
