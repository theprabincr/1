import { useState, useEffect } from "react";
import axios from "axios";
import { 
  Calendar, Clock, RefreshCw, ChevronRight, 
  TrendingUp, BarChart3, Zap, Filter
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { format, parseISO } from "date-fns";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Active sportsbooks only (ones that return data from API)
const SPORTSBOOK_LOGOS = {
  'draftkings': 'https://logo.clearbit.com/draftkings.com',
  'fanduel': 'https://logo.clearbit.com/fanduel.com',
  'betmgm': 'https://logo.clearbit.com/betmgm.com',
  'pinnacle': 'https://logo.clearbit.com/pinnacle.com',
  'unibet': 'https://logo.clearbit.com/unibet.com',
  'betway': 'https://logo.clearbit.com/betway.com',
  'betonlineag': 'https://logo.clearbit.com/betonline.ag',
};

const SPORTSBOOK_NAMES = {
  'draftkings': 'DraftKings',
  'fanduel': 'FanDuel',
  'betmgm': 'BetMGM',
  'pinnacle': 'Pinnacle',
  'unibet': 'Unibet',
  'betway': 'Betway',
  'betonlineag': 'BetOnline'
};

// Event Card Component
const EventCard = ({ event, onCompare }) => {
  const [expanded, setExpanded] = useState(false);
  const bookmakers = event.bookmakers || [];
  const bestOdds = getBestOdds(bookmakers, event.home_team, event.away_team);
  const espnOdds = event.odds || {};
  
  // Format date and time
  const eventTime = event.commence_time ? 
    format(parseISO(event.commence_time), "MMM d, h:mm a") : "TBD";
  const eventDate = event.commence_time ?
    format(parseISO(event.commence_time), "EEE, MMM d") : "";

  // Use ESPN odds directly (decimal format)
  const homeML = espnOdds.home_ml_decimal || bestOdds.home;
  const awayML = espnOdds.away_ml_decimal || bestOdds.away;
  const spread = espnOdds.spread ?? bestOdds.spread;
  const total = espnOdds.total ?? bestOdds.total;
  const isFavorite = espnOdds.home_favorite ?? (spread && spread < 0);

  return (
    <div className="event-card" data-testid={`event-card-${event.id}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-text-muted uppercase">
            {event.sport_title || event.sport_key?.replace(/_/g, ' ')}
          </span>
          <span className="text-xs px-2 py-0.5 bg-zinc-800 rounded text-brand-primary">
            {event.source === 'espn' ? 'ESPN' : 'Live'}
          </span>
        </div>
        <div className="flex flex-col items-end">
          <div className="flex items-center gap-1 text-text-muted text-sm">
            <Calendar className="w-3 h-3" />
            <span className="text-xs">{eventDate}</span>
          </div>
          <div className="flex items-center gap-1 text-brand-primary text-sm font-mono">
            <Clock className="w-3 h-3" />
            {eventTime}
          </div>
        </div>
      </div>

      {/* Teams & Odds */}
      <div className="space-y-3 mb-4">
        {/* Away Team */}
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <p className="text-text-primary font-semibold">{event.away_team}</p>
            {event.away_record && <p className="text-xs text-text-muted">{event.away_record}</p>}
          </div>
          <div className="flex items-center gap-4 font-mono">
            <span className="text-text-muted text-xs w-12 text-right">ML</span>
            <span className={`font-bold ${awayML < homeML ? 'text-semantic-success' : 'text-text-primary'}`}>
              {awayML?.toFixed(2) || '-'}
            </span>
          </div>
        </div>
        
        {/* Home Team */}
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <p className="text-text-primary font-semibold">{event.home_team}</p>
            {event.home_record && <p className="text-xs text-text-muted">{event.home_record}</p>}
          </div>
          <div className="flex items-center gap-4 font-mono">
            <span className="text-text-muted text-xs w-12 text-right">ML</span>
            <span className={`font-bold ${homeML < awayML ? 'text-semantic-success' : 'text-text-primary'}`}>
              {homeML?.toFixed(2) || '-'}
            </span>
          </div>
        </div>
      </div>

      {/* Spread & Total */}
      <div className="grid grid-cols-2 gap-4 py-3 border-t border-zinc-800">
        <div className="text-center">
          <p className="text-xs text-text-muted mb-1">SPREAD</p>
          <p className="font-mono font-bold text-brand-primary">
            {spread !== null ? `${isFavorite ? event.home_team.split(' ').pop() : event.away_team.split(' ').pop()} ${spread > 0 ? '+' : ''}${spread}` : '-'}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-text-muted mb-1">TOTAL</p>
          <p className="font-mono font-bold text-brand-primary">
            {total ? `O/U ${total}` : '-'}
          </p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2 mt-4">
        <button 
          onClick={() => onCompare(event)}
          className="btn-secondary flex-1 flex items-center justify-center gap-2 text-sm"
          data-testid={`compare-${event.id}`}
        >
          <BarChart3 className="w-4 h-4" />
          View Details
        </button>
      </div>

      {/* Venue Info */}
      {event.venue?.name && (
        <div className="mt-3 pt-3 border-t border-zinc-800">
          <p className="text-xs text-text-muted text-center">
            📍 {event.venue.name}{event.venue.city ? `, ${event.venue.city}` : ''}
          </p>
        </div>
      )}
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
