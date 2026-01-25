import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import axios from "axios";
import { 
  BarChart3, RefreshCw, Check, TrendingUp, 
  TrendingDown, Star, ExternalLink
} from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const SPORTSBOOK_LOGOS = {
  'bet365': 'https://logo.clearbit.com/bet365.com',
  'draftkings': 'https://logo.clearbit.com/draftkings.com',
  'fanduel': 'https://logo.clearbit.com/fanduel.com',
  'betmgm': 'https://logo.clearbit.com/betmgm.com',
  'williamhill_us': 'https://logo.clearbit.com/caesars.com',
  'pinnacle': 'https://logo.clearbit.com/pinnacle.com',
  'unibet': 'https://logo.clearbit.com/unibet.com',
  'betway': 'https://logo.clearbit.com/betway.com',
  'pointsbetus': 'https://logo.clearbit.com/pointsbet.com',
  'betonlineag': 'https://logo.clearbit.com/betonline.ag',
};

const SPORTSBOOK_NAMES = {
  'bet365': 'Bet365',
  'draftkings': 'DraftKings',
  'fanduel': 'FanDuel',
  'betmgm': 'BetMGM',
  'williamhill_us': 'Caesars',
  'pinnacle': 'Pinnacle',
  'unibet': 'Unibet',
  'betway': 'Betway',
  'pointsbetus': 'PointsBet',
  'betonlineag': 'BetOnline'
};

// Odds Cell Component
const OddsCell = ({ value, isBest, point }) => {
  const isPositive = value > 0;
  
  return (
    <div className={`text-center py-3 px-2 rounded-lg transition-all ${
      isBest ? 'bg-brand-primary/10 ring-1 ring-brand-primary' : 'hover:bg-zinc-800'
    }`}>
      {point !== undefined && (
        <p className="text-xs text-text-muted mb-1">
          {point > 0 ? '+' : ''}{point}
        </p>
      )}
      <p className={`font-mono font-bold text-lg ${
        isBest ? 'text-brand-primary' : 
        isPositive ? 'text-semantic-success' : 'text-text-primary'
      }`}>
        {isPositive ? '+' : ''}{value}
      </p>
      {isBest && (
        <div className="flex items-center justify-center gap-1 mt-1">
          <Star className="w-3 h-3 text-brand-primary fill-brand-primary" />
          <span className="text-xs text-brand-primary">Best</span>
        </div>
      )}
    </div>
  );
};

// Market Table Component
const MarketTable = ({ title, data, homeTeam, awayTeam, marketKey }) => {
  if (!data || data.length === 0) return null;

  // Find best odds for each outcome
  const findBestOdds = () => {
    let bestHome = { price: -9999, book: '' };
    let bestAway = { price: -9999, book: '' };

    data.forEach(bm => {
      bm.outcomes?.forEach(outcome => {
        if (outcome.name === homeTeam || outcome.name === 'Over') {
          if (outcome.price > bestHome.price) {
            bestHome = { price: outcome.price, book: bm.bookmaker };
          }
        } else {
          if (outcome.price > bestAway.price) {
            bestAway = { price: outcome.price, book: bm.bookmaker };
          }
        }
      });
    });

    return { bestHome, bestAway };
  };

  const { bestHome, bestAway } = findBestOdds();

  return (
    <div className="stat-card" data-testid={`market-${marketKey}`}>
      <h3 className="font-mono font-bold text-lg text-text-primary mb-4">{title}</h3>
      
      {/* Headers */}
      <div className="grid grid-cols-3 gap-2 mb-4 pb-3 border-b border-zinc-800">
        <div className="text-text-muted text-sm">Sportsbook</div>
        <div className="text-center text-text-muted text-sm">
          {marketKey === 'totals' ? 'Over' : homeTeam}
        </div>
        <div className="text-center text-text-muted text-sm">
          {marketKey === 'totals' ? 'Under' : awayTeam}
        </div>
      </div>

      {/* Rows */}
      <div className="space-y-2">
        {data.map((bm) => {
          const outcomes = bm.outcomes || [];
          const homeOutcome = outcomes.find(o => o.name === homeTeam || o.name === 'Over') || outcomes[0];
          const awayOutcome = outcomes.find(o => o.name === awayTeam || o.name === 'Under') || outcomes[1];

          return (
            <div 
              key={bm.bookmaker} 
              className="grid grid-cols-3 gap-2 items-center"
              data-testid={`odds-row-${bm.bookmaker}`}
            >
              <div className="flex items-center gap-3">
                <img 
                  src={SPORTSBOOK_LOGOS[bm.bookmaker]} 
                  alt={bm.title}
                  className="w-6 h-6 rounded"
                  onError={(e) => e.target.style.display = 'none'}
                />
                <span className="text-text-secondary text-sm truncate">
                  {SPORTSBOOK_NAMES[bm.bookmaker] || bm.title}
                </span>
              </div>
              
              <OddsCell 
                value={homeOutcome?.price || 0}
                point={homeOutcome?.point}
                isBest={bm.bookmaker === bestHome.book}
              />
              
              <OddsCell 
                value={awayOutcome?.price || 0}
                point={awayOutcome?.point}
                isBest={bm.bookmaker === bestAway.book}
              />
            </div>
          );
        })}
      </div>

      {/* Best Odds Summary */}
      <div className="mt-4 pt-4 border-t border-zinc-800">
        <div className="flex items-center justify-between text-sm">
          <span className="text-text-muted">Best Available:</span>
          <div className="flex gap-4">
            <span className="font-mono text-brand-primary">
              {homeTeam || 'Home'}: {bestHome.price > 0 ? '+' : ''}{bestHome.price} @ {SPORTSBOOK_NAMES[bestHome.book]}
            </span>
            <span className="font-mono text-brand-primary">
              {awayTeam || 'Away'}: {bestAway.price > 0 ? '+' : ''}{bestAway.price} @ {SPORTSBOOK_NAMES[bestAway.book]}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

const OddsComparison = () => {
  const location = useLocation();
  const [loading, setLoading] = useState(true);
  const [events, setEvents] = useState([]);
  const [selectedEvent, setSelectedEvent] = useState(location.state?.event || null);
  const [comparison, setComparison] = useState(null);
  const [selectedSport, setSelectedSport] = useState(location.state?.sportKey || "basketball_nba");

  const sports = [
    { key: "basketball_nba", label: "NBA" },
    { key: "americanfootball_nfl", label: "NFL" },
    { key: "baseball_mlb", label: "MLB" },
    { key: "icehockey_nhl", label: "NHL" },
    { key: "soccer_epl", label: "EPL" },
  ];

  const fetchEvents = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/events/${selectedSport}`);
      setEvents(response.data);
      
      // If no event selected, select first one
      if (!selectedEvent && response.data.length > 0) {
        setSelectedEvent(response.data[0]);
      }
    } catch (error) {
      console.error("Error fetching events:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchComparison = async () => {
    if (!selectedEvent?.id) return;
    
    try {
      const response = await axios.get(
        `${API}/odds-comparison/${selectedEvent.id}?sport_key=${selectedSport}`
      );
      setComparison(response.data);
    } catch (error) {
      console.error("Error fetching comparison:", error);
      // Use event data directly if comparison API fails
      if (selectedEvent?.bookmakers) {
        setComparison({
          event_id: selectedEvent.id,
          home_team: selectedEvent.home_team,
          away_team: selectedEvent.away_team,
          h2h: selectedEvent.bookmakers.map(bm => ({
            bookmaker: bm.key,
            title: bm.title,
            outcomes: bm.markets?.find(m => m.key === 'h2h')?.outcomes || []
          })),
          spreads: selectedEvent.bookmakers.map(bm => ({
            bookmaker: bm.key,
            title: bm.title,
            outcomes: bm.markets?.find(m => m.key === 'spreads')?.outcomes || []
          })),
          totals: selectedEvent.bookmakers.map(bm => ({
            bookmaker: bm.key,
            title: bm.title,
            outcomes: bm.markets?.find(m => m.key === 'totals')?.outcomes || []
          }))
        });
      }
    }
  };

  useEffect(() => {
    fetchEvents();
  }, [selectedSport]);

  useEffect(() => {
    if (selectedEvent) {
      fetchComparison();
    }
  }, [selectedEvent]);

  return (
    <div className="space-y-6" data-testid="odds-comparison-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary">Odds Comparison</h1>
          <p className="text-text-muted text-sm mt-1">Compare odds across 10 sportsbooks</p>
        </div>
        <button 
          onClick={fetchEvents}
          className="btn-outline flex items-center gap-2"
          data-testid="refresh-comparison-btn"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Sport Filters */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {sports.map(sport => (
          <button
            key={sport.key}
            onClick={() => {
              setSelectedSport(sport.key);
              setSelectedEvent(null);
            }}
            className={`px-4 py-2 rounded-lg font-mono text-sm whitespace-nowrap transition-all ${
              selectedSport === sport.key
                ? "bg-brand-primary text-zinc-950 font-bold"
                : "bg-zinc-800 text-text-secondary hover:bg-zinc-700"
            }`}
            data-testid={`comparison-sport-${sport.key}`}
          >
            {sport.label}
          </button>
        ))}
      </div>

      {/* Event Selector */}
      <div className="stat-card">
        <label className="text-text-muted text-sm mb-2 block">Select Event</label>
        <select
          value={selectedEvent?.id || ''}
          onChange={(e) => {
            const event = events.find(ev => ev.id === e.target.value);
            setSelectedEvent(event);
          }}
          className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 text-text-primary font-mono focus:outline-none focus:ring-2 focus:ring-brand-primary"
          data-testid="comparison-event-selector"
        >
          <option value="">Select an event...</option>
          {events.map(event => (
            <option key={event.id} value={event.id}>
              {event.home_team} vs {event.away_team}
            </option>
          ))}
        </select>
      </div>

      {/* Comparison Tables */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
        </div>
      ) : comparison ? (
        <div className="space-y-6">
          {/* Match Header */}
          <div className="stat-card">
            <div className="flex items-center justify-center gap-8">
              <div className="text-center">
                <h2 className="text-xl font-bold text-text-primary">{comparison.home_team}</h2>
                <p className="text-text-muted text-sm">Home</p>
              </div>
              <div className="w-16 h-16 rounded-full bg-zinc-800 flex items-center justify-center">
                <span className="text-text-muted font-mono">VS</span>
              </div>
              <div className="text-center">
                <h2 className="text-xl font-bold text-text-primary">{comparison.away_team}</h2>
                <p className="text-text-muted text-sm">Away</p>
              </div>
            </div>
          </div>

          {/* Market Tables */}
          <MarketTable 
            title="Moneyline (H2H)"
            data={comparison.h2h}
            homeTeam={comparison.home_team}
            awayTeam={comparison.away_team}
            marketKey="h2h"
          />
          
          <MarketTable 
            title="Point Spread"
            data={comparison.spreads}
            homeTeam={comparison.home_team}
            awayTeam={comparison.away_team}
            marketKey="spreads"
          />
          
          <MarketTable 
            title="Totals (Over/Under)"
            data={comparison.totals}
            homeTeam={comparison.home_team}
            awayTeam={comparison.away_team}
            marketKey="totals"
          />
        </div>
      ) : (
        <div className="stat-card text-center py-12">
          <BarChart3 className="w-12 h-12 text-text-muted mx-auto mb-4" />
          <h3 className="text-text-primary font-semibold mb-2">Select an Event</h3>
          <p className="text-text-muted text-sm">Choose an event to compare odds across sportsbooks</p>
        </div>
      )}
    </div>
  );
};

export default OddsComparison;
