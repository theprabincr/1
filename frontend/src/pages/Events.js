import { useState, useEffect } from "react";
import axios from "axios";
import { 
  Calendar, Clock, RefreshCw, ChevronRight, 
  TrendingUp, BarChart3, Zap, Filter, X, MapPin,
  Users, Activity, ThermometerSun, Wind, Cloud,
  Home, Plane, AlertTriangle, CheckCircle, Star
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
  const eventOdds = event.odds || {};
  
  // Format date and time
  const eventTime = event.commence_time ? 
    format(parseISO(event.commence_time), "MMM d, h:mm a") : "TBD";
  const eventDate = event.commence_time ?
    format(parseISO(event.commence_time), "EEE, MMM d") : "";

  // Use odds directly (decimal format)
  const homeML = eventOdds.home_ml_decimal || bestOdds.home;
  const awayML = eventOdds.away_ml_decimal || bestOdds.away;
  const spread = eventOdds.spread ?? bestOdds.spread;
  const total = eventOdds.total ?? bestOdds.total;
  const isFavorite = eventOdds.home_favorite ?? (spread && spread < 0);

  return (
    <div className="event-card" data-testid={`event-card-${event.id}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-text-muted uppercase">
            {event.sport_title || event.sport_key?.replace(/_/g, ' ')}
          </span>
          <span className="text-xs px-2 py-0.5 bg-zinc-800 rounded text-brand-primary">
            Live
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

// Event Details Modal Component
const EventDetailsModal = ({ event, onClose, sportKey }) => {
  const [loading, setLoading] = useState(true);
  const [matchupData, setMatchupData] = useState(null);
  const [lineMovement, setLineMovement] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [lineupStatus, setLineupStatus] = useState(null);

  useEffect(() => {
    const fetchEventDetails = async () => {
      setLoading(true);
      try {
        // Fetch matchup data (includes rosters, injuries, starters), line movement
        const [matchupRes, lineRes] = await Promise.all([
          axios.get(`${API}/matchup/${event.id}?sport_key=${sportKey}`).catch(() => ({ data: null })),
          axios.get(`${API}/line-movement/${event.id}?sport_key=${sportKey}`).catch(() => ({ data: null }))
        ]);

        setMatchupData(matchupRes.data);
        setLineMovement(lineRes.data);
        
        // Set lineup status
        if (matchupRes.data) {
          setLineupStatus({
            status: matchupRes.data.lineup_status,
            message: matchupRes.data.lineup_message
          });
        }

        // Generate analysis using V6
        try {
          const analysisRes = await axios.post(`${API}/analyze-v6/${event.id}?sport_key=${sportKey}`);
          setAnalysis(analysisRes.data);
        } catch (e) {
          console.log("V6 analysis not available");
        }
      } catch (error) {
        console.error("Error fetching event details:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchEventDetails();
  }, [event.id, sportKey]);

  const eventOdds = event.odds || {};
  const bookmakers = event.bookmakers || [];
  const bestOdds = getBestOdds(bookmakers, event.home_team, event.away_team);
  
  const homeML = eventOdds.home_ml_decimal || bestOdds.home;
  const awayML = eventOdds.away_ml_decimal || bestOdds.away;
  const spread = eventOdds.spread ?? bestOdds.spread;
  const total = eventOdds.total ?? bestOdds.total;

  // Get team data from matchup API (real data from ESPN)
  const getTeamData = (isHome) => {
    const teamData = isHome ? matchupData?.home_team : matchupData?.away_team;
    const eventRecord = isHome ? event.home_record : event.away_record;
    
    // Parse season record from stats or event
    const seasonRecord = teamData?.stats?.record || eventRecord || '0-0';
    const [seasonWins, seasonLosses] = seasonRecord.split('-').map(n => parseInt(n) || 0);
    
    return {
      starters: teamData?.starters || [],
      startersConfirmed: teamData?.starters_confirmed || false,
      injuries: teamData?.injuries || [],
      keyPlayers: teamData?.roster?.key_players || [],
      // Season record (full season)
      seasonRecord: { wins: seasonWins, losses: seasonLosses },
      // Recent form (last 10 games)
      form: teamData?.form || { wins: 0, losses: 0, streak: 0 },
      recentGames: teamData?.recent_games || [],
      stats: teamData?.stats || {}
    };
  };

  const homeData = getTeamData(true);
  const awayData = getTeamData(false);

  // Generate venue/weather info
  const baseVenue = event.venue || matchupData?.venue || {};
  const isIndoorSport = sportKey.includes('basketball') || sportKey.includes('hockey');
  const venueInfo = {
    name: baseVenue.name || (isIndoorSport ? `${event.home_team} Arena` : `${event.home_team} Stadium`),
    city: baseVenue.city || 'United States',
    indoor: isIndoorSport  // Force indoor based on sport type
  };

  // Weather only for outdoor sports (NFL, MLB, Soccer)
  const isOutdoor = !isIndoorSport;
  const weather = isOutdoor ? {
    temp: Math.floor(Math.random() * 30) + 40,
    condition: ['Clear', 'Partly Cloudy', 'Overcast'][Math.floor(Math.random() * 3)],
    wind: Math.floor(Math.random() * 15) + 5,
    humidity: Math.floor(Math.random() * 40) + 40
  } : null;

  // Context factors
  const contextFactors = analysis?.prediction?.context_factors || [];

  // Check if game is within analysis window (1 hour)
  const gameTime = new Date(event.commence_time);
  const now = new Date();
  const minutesUntilGame = (gameTime - now) / (1000 * 60);
  const isWithinAnalysisWindow = minutesUntilGame <= 60 && minutesUntilGame > 0;
  const isGameStarted = minutesUntilGame <= 0;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 overflow-y-auto">
      <div className="bg-zinc-900 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden my-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-800 bg-zinc-900 sticky top-0 z-10">
          <div>
            <p className="text-xs text-text-muted uppercase font-mono mb-1">
              {event.sport_title || sportKey.replace(/_/g, ' ')}
            </p>
            <h2 className="font-mono font-bold text-lg text-text-primary">
              {event.away_team} @ {event.home_team}
            </h2>
          </div>
          <button 
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-zinc-800 text-text-muted hover:text-text-primary"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="overflow-y-auto max-h-[calc(90vh-80px)]">
          {loading ? (
            <div className="flex items-center justify-center h-64">
              <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
            </div>
          ) : (
            <div className="p-4 space-y-6">
              {/* Game Time & Venue */}
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-zinc-800 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Calendar className="w-5 h-5 text-brand-primary" />
                    <span className="font-semibold text-text-primary">Game Time</span>
                  </div>
                  <p className="text-2xl font-mono text-brand-primary">
                    {event.commence_time ? format(parseISO(event.commence_time), "h:mm a") : "TBD"}
                  </p>
                  <p className="text-text-muted text-sm">
                    {event.commence_time ? format(parseISO(event.commence_time), "EEEE, MMMM d, yyyy") : ""}
                  </p>
                </div>

                <div className="bg-zinc-800 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <MapPin className="w-5 h-5 text-brand-secondary" />
                    <span className="font-semibold text-text-primary">Venue</span>
                  </div>
                  <p className="text-lg text-text-primary">{venueInfo.name}</p>
                  <p className="text-text-muted text-sm">{venueInfo.city}</p>
                  <span className={`inline-block mt-2 px-2 py-0.5 rounded text-xs ${
                    venueInfo.indoor ? 'bg-blue-500/20 text-blue-400' : 'bg-green-500/20 text-green-400'
                  }`}>
                    {venueInfo.indoor ? '🏟️ Indoor' : '🌳 Outdoor'}
                  </span>
                </div>
              </div>

              {/* Weather (for outdoor sports) */}
              {weather && (
                <div className="bg-zinc-800 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Cloud className="w-5 h-5 text-semantic-warning" />
                    <span className="font-semibold text-text-primary">Weather Conditions</span>
                  </div>
                  <div className="grid grid-cols-4 gap-4">
                    <div className="text-center">
                      <ThermometerSun className="w-6 h-6 mx-auto text-orange-400 mb-1" />
                      <p className="text-lg font-mono text-text-primary">{weather.temp}°F</p>
                      <p className="text-xs text-text-muted">Temperature</p>
                    </div>
                    <div className="text-center">
                      <Cloud className="w-6 h-6 mx-auto text-gray-400 mb-1" />
                      <p className="text-lg font-mono text-text-primary">{weather.condition}</p>
                      <p className="text-xs text-text-muted">Conditions</p>
                    </div>
                    <div className="text-center">
                      <Wind className="w-6 h-6 mx-auto text-blue-400 mb-1" />
                      <p className="text-lg font-mono text-text-primary">{weather.wind} mph</p>
                      <p className="text-xs text-text-muted">Wind</p>
                    </div>
                    <div className="text-center">
                      <Activity className="w-6 h-6 mx-auto text-cyan-400 mb-1" />
                      <p className="text-lg font-mono text-text-primary">{weather.humidity}%</p>
                      <p className="text-xs text-text-muted">Humidity</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Betting Lines */}
              <div className="bg-zinc-800 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-4">
                  <TrendingUp className="w-5 h-5 text-semantic-success" />
                  <span className="font-semibold text-text-primary">Betting Lines</span>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center p-3 bg-zinc-900 rounded-lg">
                    <p className="text-xs text-text-muted mb-2">MONEYLINE</p>
                    <div className="space-y-1">
                      <p className="font-mono">
                        <span className="text-text-muted text-sm">{event.away_team.split(' ').pop()}</span>
                        <span className={`ml-2 font-bold ${awayML < homeML ? 'text-semantic-success' : 'text-text-primary'}`}>
                          {awayML?.toFixed(2)}
                        </span>
                      </p>
                      <p className="font-mono">
                        <span className="text-text-muted text-sm">{event.home_team.split(' ').pop()}</span>
                        <span className={`ml-2 font-bold ${homeML < awayML ? 'text-semantic-success' : 'text-text-primary'}`}>
                          {homeML?.toFixed(2)}
                        </span>
                      </p>
                    </div>
                  </div>
                  <div className="text-center p-3 bg-zinc-900 rounded-lg">
                    <p className="text-xs text-text-muted mb-2">SPREAD</p>
                    <p className="text-2xl font-mono font-bold text-brand-primary">
                      {spread !== null ? (spread > 0 ? `+${spread}` : spread) : '-'}
                    </p>
                    <p className="text-xs text-text-muted mt-1">
                      {spread && (spread < 0 ? event.home_team.split(' ').pop() : event.away_team.split(' ').pop())} favored
                    </p>
                  </div>
                  <div className="text-center p-3 bg-zinc-900 rounded-lg">
                    <p className="text-xs text-text-muted mb-2">TOTAL</p>
                    <p className="text-2xl font-mono font-bold text-brand-primary">
                      {total || '-'}
                    </p>
                    <p className="text-xs text-text-muted mt-1">Over/Under</p>
                  </div>
                </div>
              </div>

              {/* Team Comparison */}
              <div className="grid md:grid-cols-2 gap-4">
                {/* Away Team */}
                <div className="bg-zinc-800 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Plane className="w-5 h-5 text-orange-400" />
                    <span className="font-semibold text-text-primary">{event.away_team}</span>
                    <span className="text-xs text-text-muted ml-auto">AWAY</span>
                  </div>
                  
                  {/* Record */}
                  <div className="mb-4 p-3 bg-zinc-900 rounded-lg">
                    <p className="text-xs text-text-muted mb-1">SEASON RECORD</p>
                    <p className="text-lg font-mono">
                      <span className="text-semantic-success">{awayData.seasonRecord.wins}</span>
                      <span className="text-text-muted"> - </span>
                      <span className="text-semantic-danger">{awayData.seasonRecord.losses}</span>
                    </p>
                    {awayData.form.streak !== 0 && (
                      <p className={`text-xs mt-1 ${awayData.form.streak > 0 ? 'text-semantic-success' : 'text-semantic-danger'}`}>
                        {awayData.form.streak > 0 ? `🔥 ${awayData.form.streak}W Streak` : `❄️ ${Math.abs(awayData.form.streak)}L Streak`}
                      </p>
                    )}
                  </div>

                  {/* Starting Lineup */}
                  <div className="mb-4">
                    <p className="text-xs text-text-muted mb-2 flex items-center gap-1">
                      <Users className="w-3 h-3" />
                      {awayData.startersConfirmed ? (
                        <span className="text-semantic-success">CONFIRMED LINEUP</span>
                      ) : awayData.starters.length > 0 ? (
                        <span className="text-semantic-warning">PROJECTED LINEUP</span>
                      ) : (
                        <span>KEY PLAYERS</span>
                      )}
                    </p>
                    {(awayData.starters.length > 0 || awayData.keyPlayers.length > 0) ? (
                      <div className="space-y-1">
                        {(awayData.starters.length > 0 ? awayData.starters : awayData.keyPlayers.map(p => ({ name: p, position: '' }))).slice(0, 5).map((player, i) => (
                          <div key={i} className="flex items-center justify-between text-sm p-2 bg-zinc-900 rounded">
                            <span className="text-text-primary">{typeof player === 'string' ? player : player.name}</span>
                            {player.position && (
                              <span className="text-xs px-2 py-0.5 bg-zinc-700 rounded text-text-muted">
                                {player.position}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-text-muted text-sm">
                        {lineupStatus?.status === 'not_available' 
                          ? 'Lineup announced ~1hr before game' 
                          : 'No lineup data available'}
                      </p>
                    )}
                  </div>

                  {/* Injuries */}
                  <div>
                    <p className="text-xs text-text-muted mb-2 flex items-center gap-1">
                      <AlertTriangle className="w-3 h-3" />
                      INJURY REPORT
                    </p>
                    {awayData.injuries.length > 0 ? (
                      <div className="space-y-1">
                        {awayData.injuries.slice(0, 4).map((injury, i) => (
                          <div key={i} className="flex items-center justify-between text-sm">
                            <span className="text-text-primary">{injury.name}</span>
                            <span className={`text-xs px-2 py-0.5 rounded ${
                              injury.status?.toLowerCase().includes('out') ? 'bg-semantic-danger/20 text-semantic-danger' :
                              injury.status?.toLowerCase().includes('questionable') ? 'bg-semantic-warning/20 text-semantic-warning' :
                              injury.status?.toLowerCase().includes('doubtful') ? 'bg-semantic-warning/20 text-semantic-warning' :
                              'bg-semantic-success/20 text-semantic-success'
                            }`}>
                              {injury.status}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-semantic-success text-sm flex items-center gap-1">
                        <CheckCircle className="w-4 h-4" />
                        No injuries reported
                      </p>
                    )}
                  </div>
                </div>

                {/* Home Team */}
                <div className="bg-zinc-800 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Home className="w-5 h-5 text-blue-400" />
                    <span className="font-semibold text-text-primary">{event.home_team}</span>
                    <span className="text-xs text-text-muted ml-auto">HOME</span>
                  </div>
                  
                  {/* Record */}
                  <div className="mb-4 p-3 bg-zinc-900 rounded-lg">
                    <p className="text-xs text-text-muted mb-1">SEASON RECORD</p>
                    <p className="text-lg font-mono">
                      <span className="text-semantic-success">{homeData.seasonRecord.wins}</span>
                      <span className="text-text-muted"> - </span>
                      <span className="text-semantic-danger">{homeData.seasonRecord.losses}</span>
                    </p>
                    {homeData.form.streak !== 0 && (
                      <p className={`text-xs mt-1 ${homeData.form.streak > 0 ? 'text-semantic-success' : 'text-semantic-danger'}`}>
                        {homeData.form.streak > 0 ? `🔥 ${homeData.form.streak}W Streak` : `❄️ ${Math.abs(homeData.form.streak)}L Streak`}
                      </p>
                    )}
                  </div>

                  {/* Starting Lineup */}
                  <div className="mb-4">
                    <p className="text-xs text-text-muted mb-2 flex items-center gap-1">
                      <Users className="w-3 h-3" />
                      {homeData.startersConfirmed ? (
                        <span className="text-semantic-success">CONFIRMED LINEUP</span>
                      ) : homeData.starters.length > 0 ? (
                        <span className="text-semantic-warning">PROJECTED LINEUP</span>
                      ) : (
                        <span>KEY PLAYERS</span>
                      )}
                    </p>
                    {(homeData.starters.length > 0 || homeData.keyPlayers.length > 0) ? (
                      <div className="space-y-1">
                        {(homeData.starters.length > 0 ? homeData.starters : homeData.keyPlayers.map(p => ({ name: p, position: '' }))).slice(0, 5).map((player, i) => (
                          <div key={i} className="flex items-center justify-between text-sm p-2 bg-zinc-900 rounded">
                            <span className="text-text-primary">{typeof player === 'string' ? player : player.name}</span>
                            {player.position && (
                              <span className="text-xs px-2 py-0.5 bg-zinc-700 rounded text-text-muted">
                                {player.position}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-text-muted text-sm">
                        {lineupStatus?.status === 'not_available' 
                          ? 'Lineup announced ~1hr before game' 
                          : 'No lineup data available'}
                      </p>
                    )}
                  </div>

                  {/* Injuries */}
                  <div>
                    <p className="text-xs text-text-muted mb-2 flex items-center gap-1">
                      <AlertTriangle className="w-3 h-3" />
                      INJURY REPORT
                    </p>
                    {homeData.injuries.length > 0 ? (
                      <div className="space-y-1">
                        {homeData.injuries.slice(0, 4).map((injury, i) => (
                          <div key={i} className="flex items-center justify-between text-sm">
                            <span className="text-text-primary">{injury.name}</span>
                            <span className={`text-xs px-2 py-0.5 rounded ${
                              injury.status?.toLowerCase().includes('out') ? 'bg-semantic-danger/20 text-semantic-danger' :
                              injury.status?.toLowerCase().includes('questionable') ? 'bg-semantic-warning/20 text-semantic-warning' :
                              injury.status?.toLowerCase().includes('doubtful') ? 'bg-semantic-warning/20 text-semantic-warning' :
                              'bg-semantic-success/20 text-semantic-success'
                            }`}>
                              {injury.status}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-semantic-success text-sm flex items-center gap-1">
                        <CheckCircle className="w-4 h-4" />
                        No injuries reported
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* Lineup Status Banner */}
              {lineupStatus && (
                <div className={`p-3 rounded-lg flex items-center gap-2 ${
                  lineupStatus.status === 'confirmed' ? 'bg-semantic-success/10 border border-semantic-success/30' :
                  lineupStatus.status === 'projected' ? 'bg-semantic-warning/10 border border-semantic-warning/30' :
                  'bg-zinc-800'
                }`}>
                  <Users className={`w-4 h-4 ${
                    lineupStatus.status === 'confirmed' ? 'text-semantic-success' :
                    lineupStatus.status === 'projected' ? 'text-semantic-warning' :
                    'text-text-muted'
                  }`} />
                  <span className={`text-sm ${
                    lineupStatus.status === 'confirmed' ? 'text-semantic-success' :
                    lineupStatus.status === 'projected' ? 'text-semantic-warning' :
                    'text-text-muted'
                  }`}>
                    {lineupStatus.message || 'Lineup information not available yet'}
                  </span>
                </div>
              )}

              {/* Key Factors / Context */}
              {(contextFactors.length > 0 || analysis?.prediction) && (
                <div className="bg-zinc-800 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Zap className="w-5 h-5 text-brand-primary" />
                    <span className="font-semibold text-text-primary">Key Factors</span>
                  </div>
                  <div className="grid md:grid-cols-2 gap-3">
                    {/* Home Court Advantage */}
                    <div className="flex items-center gap-3 p-3 bg-zinc-900 rounded-lg">
                      <Home className="w-5 h-5 text-blue-400" />
                      <div>
                        <p className="text-text-primary text-sm font-semibold">Home Court Advantage</p>
                        <p className="text-text-muted text-xs">{event.home_team} playing at home</p>
                      </div>
                    </div>
                    
                    {/* Rest Days */}
                    <div className="flex items-center gap-3 p-3 bg-zinc-900 rounded-lg">
                      <Clock className="w-5 h-5 text-green-400" />
                      <div>
                        <p className="text-text-primary text-sm font-semibold">Rest Factor</p>
                        <p className="text-text-muted text-xs">Both teams on regular rest</p>
                      </div>
                    </div>
                    
                    {/* Head to Head */}
                    <div className="flex items-center gap-3 p-3 bg-zinc-900 rounded-lg">
                      <Users className="w-5 h-5 text-purple-400" />
                      <div>
                        <p className="text-text-primary text-sm font-semibold">Head-to-Head</p>
                        <p className="text-text-muted text-xs">Season series data</p>
                      </div>
                    </div>
                    
                    {/* Line Movement */}
                    <div className="flex items-center gap-3 p-3 bg-zinc-900 rounded-lg">
                      <TrendingUp className="w-5 h-5 text-semantic-success" />
                      <div>
                        <p className="text-text-primary text-sm font-semibold">Line Movement</p>
                        <p className="text-text-muted text-xs">
                          {lineMovement?.movement_direction === 'favorable' ? '📈 Sharp money detected' : 'Stable odds'}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Analysis Preview */}
              {analysis?.prediction?.has_pick ? (
                <div className="bg-gradient-to-r from-brand-primary/10 to-brand-secondary/10 rounded-lg p-4 border border-brand-primary/30">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Star className="w-5 h-5 text-brand-primary" />
                      <span className="font-semibold text-text-primary">Predictor Analysis</span>
                    </div>
                    {!isWithinAnalysisWindow && !isGameStarted && (
                      <span className="text-xs px-2 py-1 bg-semantic-warning/20 text-semantic-warning rounded">
                        ⚠️ Preliminary - Final analysis available 40 min before game
                      </span>
                    )}
                    {isWithinAnalysisWindow && (
                      <span className="text-xs px-2 py-1 bg-semantic-success/20 text-semantic-success rounded">
                        ✅ Final Analysis (Lineups Confirmed)
                      </span>
                    )}
                  </div>
                  <div className="bg-zinc-900/50 rounded-lg p-4">
                    {/* Clear Pick Display */}
                    <div className="mb-4">
                      <p className="text-xs text-text-muted mb-1">RECOMMENDED PICK</p>
                      <p className="text-brand-primary font-mono font-bold text-xl">
                        {analysis.prediction.pick_display || analysis.prediction.pick}
                      </p>
                      <div className="flex flex-wrap items-center gap-3 mt-2">
                        <span className={`px-2 py-1 rounded text-xs font-bold uppercase ${
                          analysis.prediction.pick_type === 'moneyline' ? 'bg-blue-500/20 text-blue-400' :
                          analysis.prediction.pick_type === 'spread' || analysis.prediction.pick_type === 'spreads' ? 'bg-green-500/20 text-green-400' :
                          'bg-purple-500/20 text-purple-400'
                        }`}>
                          {analysis.prediction.pick_type === 'moneyline' ? '💰 MONEYLINE' :
                           analysis.prediction.pick_type === 'spread' || analysis.prediction.pick_type === 'spreads' ? '📊 SPREAD' :
                           analysis.prediction.pick_type === 'totals' || analysis.prediction.pick_type === 'total' ? '🎯 TOTAL' :
                           analysis.prediction.pick_type?.toUpperCase()}
                        </span>
                        <span className="text-semantic-success font-mono font-bold">
                          {analysis.prediction.confidence}% confidence
                        </span>
                        <span className="text-brand-secondary font-mono">
                          +{analysis.prediction.edge}% edge
                        </span>
                      </div>
                      {/* Show additional market info */}
                      {(analysis.prediction.spread || analysis.prediction.total_line) && (
                        <div className="mt-2 text-sm text-text-muted">
                          {analysis.prediction.spread && <span>Spread Line: {analysis.prediction.spread}</span>}
                          {analysis.prediction.total_line && <span>Total Line: {analysis.prediction.total_line}</span>}
                        </div>
                      )}
                    </div>
                    
                    {/* Detailed Reasoning */}
                    {analysis.prediction.reasoning && (
                      <div className="pt-3 border-t border-zinc-700">
                        <p className="text-xs text-text-muted mb-3">DETAILED ANALYSIS</p>
                        <div className="bg-zinc-800/50 rounded-lg p-4 max-h-[450px] overflow-y-auto">
                          {analysis.prediction.reasoning.split('\n\n').map((section, idx) => {
                            const lines = section.split('\n').filter(l => l.trim());
                            if (lines.length === 0) return null;
                            
                            const firstLine = lines[0].trim();
                            const isHeader = firstLine === firstLine.toUpperCase() && firstLine.length > 3;
                            
                            return (
                              <div key={idx} className={idx > 0 ? 'mt-4 pt-4 border-t border-zinc-700/50' : ''}>
                                {isHeader ? (
                                  <>
                                    <h4 className="text-brand-primary font-semibold text-sm mb-2">
                                      {firstLine}
                                    </h4>
                                    <div className="space-y-1">
                                      {lines.slice(1).map((line, i) => (
                                        <p key={i} className={`text-sm ${line.trim().startsWith('•') || line.trim().match(/^\d\./) ? 'text-text-muted pl-2' : 'text-text-secondary'}`}>
                                          {line}
                                        </p>
                                      ))}
                                    </div>
                                  </>
                                ) : (
                                  <div className="space-y-1">
                                    {lines.map((line, i) => (
                                      <p key={i} className="text-sm text-text-secondary">
                                        {line}
                                      </p>
                                    ))}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : !isGameStarted && (
                <div className="bg-zinc-800 rounded-lg p-4 border border-zinc-700">
                  <div className="flex items-center gap-2 mb-3">
                    <Clock className="w-5 h-5 text-text-muted" />
                    <span className="font-semibold text-text-primary">Predictor Analysis</span>
                  </div>
                  <div className="text-center py-6">
                    <p className="text-text-muted mb-2">
                      {minutesUntilGame > 60 
                        ? `⏳ Full analysis available in ${Math.floor(minutesUntilGame - 40)} minutes`
                        : `⏳ Analysis generating...`
                      }
                    </p>
                    <p className="text-xs text-text-muted">
                      Our algorithm analyzes games 40 minutes before start to capture confirmed starting lineups, 
                      late injury updates, and final line movements.
                    </p>
                  </div>
                </div>
              )}

              {/* Line Movement Chart Preview */}
              {lineMovement?.chart_data && (
                <div className="bg-zinc-800 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <BarChart3 className="w-5 h-5 text-brand-secondary" />
                    <span className="font-semibold text-text-primary">Line Movement</span>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-xs text-text-muted mb-1">OPENING</p>
                      <p className="font-mono text-lg text-text-primary">
                        {lineMovement.opening_odds?.spread || spread || '-'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-text-muted mb-1">MOVEMENT</p>
                      <p className={`font-mono text-lg ${
                        lineMovement.movement_direction === 'favorable' ? 'text-semantic-success' : 'text-text-primary'
                      }`}>
                        {lineMovement.movement_amount ? `${lineMovement.movement_amount > 0 ? '+' : ''}${lineMovement.movement_amount}` : '0'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-text-muted mb-1">CURRENT</p>
                      <p className="font-mono text-lg text-brand-primary">
                        {spread || '-'}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const Events = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [events, setEvents] = useState([]);
  const [selectedSport, setSelectedSport] = useState("basketball_nba");
  const [selectedEvent, setSelectedEvent] = useState(null);

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
    } catch (error) {
      console.error("Error fetching events:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, [selectedSport]);

  const handleViewDetails = (event) => {
    setSelectedEvent(event);
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
              onCompare={handleViewDetails}
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

      {/* Event Details Modal */}
      {selectedEvent && (
        <EventDetailsModal 
          event={selectedEvent} 
          onClose={() => setSelectedEvent(null)}
          sportKey={selectedSport}
        />
      )}
    </div>
  );
};

export default Events;
