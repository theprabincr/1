import { useState, useEffect } from "react";
import axios from "axios";
import { 
  Calendar, Clock, RefreshCw, ChevronRight, 
  TrendingUp, BarChart3, Zap, Filter
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { format, parseISO } from "date-fns";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Sportsbook logos mapping
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

// Event Card Component
const EventCard = ({ event, onAnalyze, onCompare }) => {
  const [expanded, setExpanded] = useState(false);
  const bookmakers = event.bookmakers || [];
  const bestOdds = getBestOdds(bookmakers, event.home_team, event.away_team);
  
  const eventTime = event.commence_time ? 
    format(parseISO(event.commence_time), "MMM d, h:mm a") : "TBD";

  return (
    <div className="event-card" data-testid={`event-card-${event.id}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-text-muted uppercase">
            {event.sport_title || event.sport_key?.replace(/_/g, ' ')}
          </span>
        </div>
        <div className="flex items-center gap-2 text-text-muted text-sm">
          <Clock className="w-4 h-4" />
          {eventTime}
        </div>
      </div>

      {/* Teams & Odds */}
      <div className="grid grid-cols-3 gap-4 items-center mb-4">
        <div>
          <p className="text-text-primary font-semibold">{event.home_team}</p>
          <div className="mt-2">
            <p className="text-xs text-text-muted">Best ML</p>
            <p className="font-mono text-xl font-bold text-brand-primary">
              {bestOdds.home?.toFixed(2)}
            </p>
            <p className="text-xs text-text-muted">{bestOdds.homeBk}</p>
          </div>
        </div>

        <div className="text-center">
          <div className="w-12 h-12 mx-auto rounded-full bg-zinc-800 flex items-center justify-center">
            <span className="text-text-muted font-mono text-sm">VS</span>
          </div>
        </div>

        <div className="text-right">
          <p className="text-text-primary font-semibold">{event.away_team}</p>
          <div className="mt-2">
            <p className="text-xs text-text-muted">Best ML</p>
            <p className="font-mono text-xl font-bold text-brand-primary">
              {bestOdds.away?.toFixed(2)}
            </p>
            <p className="text-xs text-text-muted">{bestOdds.awayBk}</p>
          </div>
        </div>
      </div>

      {/* Spreads & Totals Summary */}
      <div className="grid grid-cols-2 gap-4 p-3 bg-zinc-800/50 rounded-lg mb-4">
        <div>
          <p className="text-xs text-text-muted mb-1">Spread</p>
          <p className="font-mono text-sm text-text-primary">
            {bestOdds.spread ? `${event.home_team} ${bestOdds.spread > 0 ? '+' : ''}${bestOdds.spread}` : 'N/A'}
          </p>
        </div>
        <div>
          <p className="text-xs text-text-muted mb-1">Total</p>
          <p className="font-mono text-sm text-text-primary">
            {bestOdds.total ? `O/U ${bestOdds.total}` : 'N/A'}
          </p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <button 
          onClick={() => onCompare(event)}
          className="btn-outline flex-1 flex items-center justify-center gap-2 text-sm"
          data-testid={`compare-odds-${event.id}`}
        >
          <BarChart3 className="w-4 h-4" />
          Compare Odds
        </button>
        <button 
          onClick={() => onAnalyze(event)}
          className="btn-primary flex-1 flex items-center justify-center gap-2 text-sm"
          data-testid={`analyze-${event.id}`}
        >
          <Zap className="w-4 h-4" />
          AI Analysis
        </button>
      </div>

      {/* Expanded Odds View */}
      {expanded && (
        <div className="mt-4 pt-4 border-t border-zinc-800">
          <h4 className="text-sm font-mono text-text-muted mb-3">All Sportsbooks</h4>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {bookmakers.slice(0, 8).map((bm) => {
              const h2h = bm.markets?.find(m => m.key === 'h2h');
              if (!h2h) return null;
              
              return (
                <div key={bm.key} className="flex items-center justify-between py-2 border-b border-zinc-800/50">
                  <div className="flex items-center gap-2">
                    <img 
                      src={SPORTSBOOK_LOGOS[bm.key]} 
                      alt={bm.title}
                      className="w-5 h-5 rounded"
                      onError={(e) => e.target.style.display = 'none'}
                    />
                    <span className="text-sm text-text-secondary">{SPORTSBOOK_NAMES[bm.key] || bm.title}</span>
                  </div>
                  <div className="flex gap-6 font-mono text-sm">
                    {h2h.outcomes?.map((o, i) => (
                      <span key={i} className={o.price > 0 ? "text-semantic-success" : "text-text-primary"}>
                        {o.price > 0 ? "+" : ""}{o.price}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <button 
        onClick={() => setExpanded(!expanded)}
        className="w-full mt-3 text-center text-sm text-text-muted hover:text-text-primary transition-colors"
        data-testid={`toggle-odds-${event.id}`}
      >
        {expanded ? "Hide" : "Show"} all sportsbooks
      </button>
    </div>
  );
};

// Helper to get best odds (decimal format)
const getBestOdds = (bookmakers, homeTeam, awayTeam) => {
  let bestHome = 1, bestAway = 1;
  let homeBk = '', awayBk = '';
  let spread = null, total = null;

  bookmakers.forEach(bm => {
    bm.markets?.forEach(market => {
      if (market.key === 'h2h') {
        market.outcomes?.forEach(outcome => {
          if (outcome.name === homeTeam && outcome.price > bestHome) {
            bestHome = outcome.price;
            homeBk = SPORTSBOOK_NAMES[bm.key] || bm.title;
          }
          if (outcome.name === awayTeam && outcome.price > bestAway) {
            bestAway = outcome.price;
            awayBk = SPORTSBOOK_NAMES[bm.key] || bm.title;
          }
        });
      }
      if (market.key === 'spreads' && !spread) {
        const homeSpread = market.outcomes?.find(o => o.name === homeTeam);
        if (homeSpread) spread = homeSpread.point;
      }
      if (market.key === 'totals' && !total) {
        const over = market.outcomes?.find(o => o.name === 'Over');
        if (over) total = over.point;
      }
    });
  });

  return { 
    home: bestHome > 1 ? bestHome : 1.91, 
    away: bestAway > 1 ? bestAway : 1.91,
    homeBk: homeBk || 'N/A',
    awayBk: awayBk || 'N/A',
    spread,
    total
  };
};

const Events = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [events, setEvents] = useState([]);
  const [selectedSport, setSelectedSport] = useState("basketball_nba");
  const [analyzing, setAnalyzing] = useState(null);

  const sports = [
    { key: "basketball_nba", label: "NBA" },
    { key: "americanfootball_nfl", label: "NFL" },
    { key: "baseball_mlb", label: "MLB" },
    { key: "icehockey_nhl", label: "NHL" },
    { key: "soccer_epl", label: "EPL" },
    { key: "soccer_spain_la_liga", label: "La Liga" },
    { key: "mma_mixed_martial_arts", label: "MMA" },
  ];

  const fetchEvents = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/events/${selectedSport}`);
      setEvents(response.data);
    } catch (error) {
      console.error("Error fetching events:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, [selectedSport]);

  const handleAnalyze = async (event) => {
    setAnalyzing(event.id);
    try {
      // Navigate to predictions page with analysis
      navigate('/predictions', { 
        state: { 
          analyzeEvent: event,
          sportKey: selectedSport 
        } 
      });
    } finally {
      setAnalyzing(null);
    }
  };

  const handleCompare = (event) => {
    navigate('/odds-comparison', { 
      state: { 
        event,
        sportKey: selectedSport 
      } 
    });
  };

  return (
    <div className="space-y-6" data-testid="events-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary">Events</h1>
          <p className="text-text-muted text-sm mt-1">Live and upcoming matches</p>
        </div>
        <button 
          onClick={fetchEvents}
          className="btn-outline flex items-center gap-2"
          data-testid="refresh-events-btn"
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
            onClick={() => setSelectedSport(sport.key)}
            className={`px-4 py-2 rounded-lg font-mono text-sm whitespace-nowrap transition-all ${
              selectedSport === sport.key
                ? "bg-brand-primary text-zinc-950 font-bold"
                : "bg-zinc-800 text-text-secondary hover:bg-zinc-700"
            }`}
            data-testid={`sport-btn-${sport.key}`}
          >
            {sport.label}
          </button>
        ))}
      </div>

      {/* Events Grid */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
        </div>
      ) : events.length > 0 ? (
        <div className="grid md:grid-cols-2 gap-4">
          {events.map((event) => (
            <EventCard 
              key={event.id} 
              event={event} 
              onAnalyze={handleAnalyze}
              onCompare={handleCompare}
            />
          ))}
        </div>
      ) : (
        <div className="stat-card text-center py-12">
          <Calendar className="w-12 h-12 text-text-muted mx-auto mb-4" />
          <h3 className="text-text-primary font-semibold mb-2">No Events Found</h3>
          <p className="text-text-muted text-sm">
            No upcoming events for {sports.find(s => s.key === selectedSport)?.label || selectedSport}
          </p>
        </div>
      )}
    </div>
  );
};

export default Events;
