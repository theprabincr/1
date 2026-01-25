# BetPredictor - AI-Powered Sports Betting Predictor

## Original Problem Statement
Build a betting predictor application that fetches data from all future and in-play sports events, takes into account various popular sportsbook odds and their movement. Acts wisely to track the line movement, does game analysis as to why the lines are moving, runs comparisons and recommends bets. Also after the events are over, keeps track of either that bet lost or won to keep record of the algorithm prediction for users' reference as to how the app is performing.

## User Personas
1. **Sports Bettor** - Wants data-driven insights to make informed betting decisions
2. **Sharp Bettor** - Tracks line movement and odds arbitrage opportunities
3. **Casual Fan** - Looks for AI-powered recommendations with analysis

## Core Requirements (Static)
- Fetch sports events and odds from The Odds API (or mock data)
- Track 10 major sportsbooks: bet365, DraftKings, FanDuel, BetMGM, Caesars, Pinnacle, Unibet, Betway, PointsBet, BetOnline
- AI analysis using GPT-5.2 and Claude Sonnet 4.5
- Line movement tracking with visual charts
- Odds comparison across sportsbooks
- Win/loss tracking for algorithm performance
- No authentication required

## What's Been Implemented (December 2025)
### Backend (FastAPI + MongoDB)
- [x] GET /api/sports - List available sports
- [x] GET /api/events/{sport_key} - Get events with odds
- [x] GET /api/line-movement/{event_id} - Line movement history
- [x] GET /api/odds-comparison/{event_id} - Compare odds across books
- [x] POST /api/analyze - AI analysis using GPT-5.2 + Claude
- [x] GET /api/recommendations - Get AI picks
- [x] POST /api/generate-recommendations - Generate new picks
- [x] PUT /api/result - Track win/loss/push
- [x] GET /api/performance - Performance statistics
- [x] Mock data generation when API key not provided

### Frontend (React + Tailwind)
- [x] Dashboard with stats, top picks, live events
- [x] Events page with sport filters, odds display
- [x] Line Movement page with Recharts visualization
- [x] Odds Comparison with 10 sportsbook comparison
- [x] Predictions page with AI analysis
- [x] Performance tracking with charts

### Design
- Dark terminal/HFT theme with lime green accents
- JetBrains Mono for data, Manrope for body
- Responsive sidebar navigation

## Prioritized Backlog

### P0 - Critical (Blocking Core Functionality)
- [ ] Add real The Odds API integration (requires API key from user)
- [ ] Add scheduled job to auto-track event results

### P1 - Important
- [ ] Add notifications for sharp line movements
- [ ] Add prop bets support
- [ ] Add bankroll management tracking
- [ ] Add user preferences/favorites

### P2 - Nice to Have
- [ ] Add more sports (Tennis, Golf, Cricket)
- [ ] Add historical odds analysis
- [ ] Add social sharing of picks
- [ ] Add dark/light theme toggle

## Tech Stack
- **Backend**: FastAPI, MongoDB, emergentintegrations (LLM)
- **Frontend**: React, Tailwind CSS, Recharts, Lucide Icons
- **AI Models**: GPT-5.2 (OpenAI), Claude Sonnet 4.5 (Anthropic)
- **Data Source**: The Odds API (mock data if no key)

## Next Tasks
1. User to provide The Odds API key for real data
2. Implement automated result tracking
3. Add push notifications for line movements
