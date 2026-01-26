import { useState, useEffect } from "react";
import axios from "axios";
import { 
  TrendingUp, TrendingDown, Trophy, Target, 
  DollarSign, Clock, ChevronRight, Zap,
  Activity, AlertCircle, RefreshCw, X
} from "lucide-react";
import { useNavigate } from "react-router-dom";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Stat Card Component - Now clickable
const StatCard = ({ title, value, subtitle, icon: Icon, color = "lime", onClick }) => {
  const colorClasses = {
    lime: "text-brand-primary",
    green: "text-semantic-success",
    red: "text-semantic-danger",
    blue: "text-brand-secondary",
    yellow: "text-semantic-warning"
  };

  return (
    <div 
      className={`stat-card card-hover ${onClick ? 'cursor-pointer' : ''}`} 
      data-testid={`stat-${title.toLowerCase().replace(/\s+/g, '-')}`}
      onClick={onClick}
    >
      <div className="flex items-start justify-between mb-3">
        <div className={`p-2 rounded-lg bg-zinc-800 ${colorClasses[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
        {onClick && (
          <ChevronRight className="w-5 h-5 text-text-muted" />
        )}
      </div>
      <h3 className="text-text-muted text-sm mb-1">{title}</h3>
      <p className="font-mono font-bold text-2xl text-text-primary">{value}</p>
      {subtitle && <p className="text-text-muted text-xs mt-1">{subtitle}</p>}
    </div>
  );
};

// Top Pick Card
const TopPickCard = ({ pick, onClick }) => {
  const confidence = pick.confidence * 100;
  
  // Get market type badge
  const marketBadge = {
    'moneyline': { label: 'ML', color: 'bg-blue-500/20 text-blue-400' },
    'spread': { label: 'SPR', color: 'bg-purple-500/20 text-purple-400' },
    'total': { label: 'O/U', color: 'bg-orange-500/20 text-orange-400' }
  }[pick.prediction_type] || { label: 'ML', color: 'bg-blue-500/20 text-blue-400' };
  
  // Format the pick display based on prediction type
  const formatPick = () => {
    const outcome = pick.predicted_outcome || '';
    if (pick.prediction_type === 'spread') {
      return outcome; // e.g., "Boston Celtics -5.5"
    } else if (pick.prediction_type === 'total') {
      return outcome; // e.g., "Over 225.5"
    }
    return outcome; // Moneyline just shows team name
  };
  
  // Get time until game
  const getTimeUntil = () => {
    if (!pick.commence_time) return '';
    const commence = new Date(pick.commence_time);
    const now = new Date();
    const diffMs = commence - now;
    if (diffMs < 0) return 'Started';
    const hours = Math.floor(diffMs / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);
    if (days > 0) return `${days}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h`;
    return 'Soon';
  };
  
  return (
    <div 
      className="event-card cursor-pointer" 
      onClick={onClick}
      data-testid={`pick-${pick.event_id}`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-text-muted uppercase">{pick.sport_key?.replace(/_/g, ' ')}</span>
          <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${marketBadge.color}`}>
            {marketBadge.label}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-muted">{getTimeUntil()}</span>
          <span className={`px-2 py-1 rounded text-xs font-bold ${
            confidence >= 70 ? "bg-semantic-success/20 text-semantic-success" :
            confidence >= 50 ? "bg-semantic-warning/20 text-semantic-warning" :
            "bg-text-muted/20 text-text-muted"
          }`}>
            {confidence.toFixed(0)}%
          </span>
        </div>
      </div>
      
      <div className="mb-3">
        <p className="text-text-primary font-semibold">{pick.home_team}</p>
        <p className="text-text-muted text-sm">vs {pick.away_team}</p>
      </div>
      
      <div className="flex items-center justify-between pt-3 border-t border-zinc-700/50">
        <div>
          <p className="text-xs text-text-muted mb-1">Pick ({marketBadge.label})</p>
          <p className="font-mono font-bold text-brand-primary text-sm">{formatPick()}</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-text-muted mb-1">Odds</p>
          <p className="font-mono font-bold text-lg text-brand-primary">
            {pick.odds_at_prediction?.toFixed(2)}
          </p>
        </div>
      </div>
      
      {/* Reasoning Section */}
      {pick.reasoning && (
        <div className="mt-3 pt-3 border-t border-zinc-700/50">
          <p className="text-xs text-text-muted mb-1">Analysis</p>
          <p className="text-text-secondary text-xs leading-relaxed">{pick.reasoning}</p>
        </div>
      )}
    </div>
  );
};

// Live Event Card
const LiveEventCard = ({ event }) => {
  const bestOdds = getBestOdds(event.bookmakers || []);
  
  return (
    <div className="event-card" data-testid={`event-${event.id}`}>
      <div className="flex items-center gap-2 mb-3">
        <span className="w-2 h-2 rounded-full bg-brand-primary"></span>
        <span className="text-xs font-mono text-brand-primary">UPCOMING</span>
        <span className="text-xs text-text-muted ml-auto">
          {event.sport_title || event.sport_key?.replace(/_/g, ' ')}
        </span>
      </div>
      
      <div className="grid grid-cols-3 gap-2 items-center">
        <div>
          <p className="text-text-primary font-semibold text-sm truncate">{event.home_team}</p>
          <p className="font-mono text-lg font-bold text-brand-primary">
            {bestOdds.home?.toFixed(2)}
          </p>
        </div>
        
        <div className="text-center">
          <p className="text-text-muted text-xs">VS</p>
        </div>
        
        <div className="text-right">
          <p className="text-text-primary font-semibold text-sm truncate">{event.away_team}</p>
          <p className="font-mono text-lg font-bold text-brand-primary">
            {bestOdds.away?.toFixed(2)}
          </p>
        </div>
      </div>
    </div>
  );
};

// Helper to get best odds (decimal format - higher is better)
const getBestOdds = (bookmakers) => {
  let bestHome = 1;
  let bestAway = 1;
  
  bookmakers.forEach(bm => {
    bm.markets?.forEach(market => {
      if (market.key === 'h2h') {
        market.outcomes?.forEach(outcome => {
          if (outcome.name === bookmakers[0]?.markets?.[0]?.outcomes?.[0]?.name) {
            if (outcome.price > bestHome) bestHome = outcome.price;
          } else {
            if (outcome.price > bestAway) bestAway = outcome.price;
          }
        });
      }
    });
  });
  
  return { home: bestHome || 1.91, away: bestAway || 1.91 };
};

const Dashboard = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [performance, setPerformance] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [events, setEvents] = useState([]);
  const [selectedSport, setSelectedSport] = useState("basketball_nba");
  const [showActivePicksModal, setShowActivePicksModal] = useState(false);
  const [liveScores, setLiveScores] = useState([]);
  const [showLiveScoresModal, setShowLiveScoresModal] = useState(false);

  const sports = [
    { key: "basketball_nba", label: "NBA" },
    { key: "americanfootball_nfl", label: "NFL" },
    { key: "baseball_mlb", label: "MLB" },
    { key: "icehockey_nhl", label: "NHL" },
    { key: "soccer_epl", label: "EPL" },
  ];

  const fetchData = async () => {
    setLoading(true);
    try {
      const [perfRes, recsRes, eventsRes, liveRes] = await Promise.all([
        axios.get(`${API}/performance`),
        axios.get(`${API}/recommendations?limit=50&min_confidence=0.70`),
        axios.get(`${API}/events/${selectedSport}`),
        axios.get(`${API}/live-scores`)
      ]);
      
      setPerformance(perfRes.data);
      setRecommendations(recsRes.data);
      setEvents(eventsRes.data);
      setLiveScores(liveRes.data.games || []);
    } catch (error) {
      console.error("Error fetching dashboard data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [selectedSport]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64" data-testid="loading">
        <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="dashboard">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary">Dashboard</h1>
          <p className="text-text-muted text-sm mt-1">AI-powered betting insights</p>
        </div>
        <button 
          onClick={fetchData}
          className="btn-outline flex items-center gap-2"
          data-testid="refresh-btn"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <StatCard
          title="Win Rate"
          value={`${performance?.win_rate || 0}%`}
          subtitle={`${performance?.wins || 0}W - ${performance?.losses || 0}L`}
          icon={Trophy}
          color="lime"
          onClick={() => navigate('/performance')}
        />
        <StatCard
          title="ROI"
          value={`${performance?.roi > 0 ? '+' : ''}${performance?.roi || 0}%`}
          subtitle="Return on investment"
          icon={DollarSign}
          color={performance?.roi >= 0 ? "green" : "red"}
          onClick={() => navigate('/performance')}
        />
        <StatCard
          title="Completed"
          value={performance?.total_predictions || 0}
          subtitle={`${performance?.wins || 0}W - ${performance?.losses || 0}L`}
          icon={Target}
          color="blue"
          onClick={() => navigate('/predictions')}
        />
        <StatCard
          title="Active Picks"
          value={recommendations.length}
          subtitle="Click to view all"
          icon={Clock}
          color="yellow"
          onClick={() => setShowActivePicksModal(true)}
        />
        <StatCard
          title="Live Games"
          value={liveScores.length}
          subtitle="Click for scores"
          icon={Activity}
          color={liveScores.length > 0 ? "green" : "blue"}
          onClick={() => setShowLiveScoresModal(true)}
        />
      </div>

      {/* Live Scores Modal */}
      {showLiveScoresModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-zinc-900 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-zinc-800">
              <h2 className="font-mono font-bold text-lg text-text-primary flex items-center gap-2">
                <Activity className="w-5 h-5 text-semantic-success animate-pulse" />
                Live Games ({liveScores.length})
                <span className="text-xs text-text-muted ml-2">Auto-updates every 10s</span>
              </h2>
              <button 
                onClick={() => setShowLiveScoresModal(false)}
                className="p-2 rounded-lg hover:bg-zinc-800 text-text-muted hover:text-text-primary"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto max-h-[60vh] space-y-3">
              {liveScores.length === 0 ? (
                <p className="text-text-muted text-center py-8">No live games at the moment.</p>
              ) : (
                liveScores.map((game, i) => (
                  <div key={game.espn_id || i} className="bg-zinc-800 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-mono text-brand-primary uppercase flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-semantic-success animate-pulse"></span>
                        {game.sport_key?.replace(/_/g, ' ')}
                      </span>
                      <span className="text-xs text-text-muted">
                        {game.clock} - {game.period ? `Q${game.period}` : 'Live'}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 items-center gap-4">
                      <div className="text-left">
                        <p className="text-text-primary font-semibold text-sm">{game.away_team}</p>
                      </div>
                      <div className="text-center">
                        <p className="font-mono text-2xl font-bold text-brand-primary">
                          {game.away_score} - {game.home_score}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-text-primary font-semibold text-sm">{game.home_team}</p>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="p-4 border-t border-zinc-800 text-center">
              <p className="text-text-muted text-xs">
                Scores powered by ESPN • Updates automatically
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Active Picks Modal */}
      {showActivePicksModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-zinc-900 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-zinc-800">
              <h2 className="font-mono font-bold text-lg text-text-primary flex items-center gap-2">
                <Clock className="w-5 h-5 text-semantic-warning" />
                Active Picks ({recommendations.length})
              </h2>
              <button 
                onClick={() => setShowActivePicksModal(false)}
                className="p-2 rounded-lg hover:bg-zinc-800 text-text-muted hover:text-text-primary"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto max-h-[60vh] space-y-3">
              {recommendations.length === 0 ? (
                <p className="text-text-muted text-center py-8">No active picks at the moment.</p>
              ) : (
                recommendations.map((pick, i) => (
                  <div key={pick.id || i} className="bg-zinc-800 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <p className="text-text-primary font-semibold">{pick.home_team} vs {pick.away_team}</p>
                        <p className="text-text-muted text-sm">
                          {new Date(pick.commence_time).toLocaleDateString()} at {new Date(pick.commence_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                        </p>
                      </div>
                      <div className="text-right">
                        <span className={`px-2 py-1 rounded text-xs font-bold ${
                          pick.confidence >= 0.7 ? 'bg-semantic-success/20 text-semantic-success' :
                          pick.confidence >= 0.5 ? 'bg-semantic-warning/20 text-semantic-warning' :
                          'bg-text-muted/20 text-text-muted'
                        }`}>
                          {(pick.confidence * 100).toFixed(0)}% Confidence
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between mt-3 pt-3 border-t border-zinc-700">
                      <div>
                        <p className="text-brand-primary font-bold">{pick.predicted_outcome}</p>
                        <p className="text-text-muted text-xs">{pick.prediction_type}</p>
                      </div>
                      <p className="font-mono text-lg text-text-primary">@ {pick.odds_at_prediction?.toFixed(2)}</p>
                    </div>
                    {/* Reasoning Section */}
                    {pick.reasoning && (
                      <div className="mt-3 pt-3 border-t border-zinc-700">
                        <p className="text-xs text-text-muted mb-1">Why this pick?</p>
                        <p className="text-text-secondary text-xs leading-relaxed">{pick.reasoning}</p>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
            <div className="p-4 border-t border-zinc-800">
              <button 
                onClick={() => { setShowActivePicksModal(false); navigate('/predictions'); }}
                className="btn-primary w-full"
              >
                View All Predictions
              </button>
            </div>
          </div>
        </div>
      )}

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
            data-testid={`sport-filter-${sport.key}`}
          >
            {sport.label}
          </button>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Top Picks */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-mono font-bold text-lg text-text-primary flex items-center gap-2">
              <Zap className="w-5 h-5 text-brand-primary" />
              Top Picks
            </h2>
            <button 
              onClick={() => navigate('/predictions')}
              className="text-brand-primary text-sm flex items-center gap-1 hover:underline"
              data-testid="view-all-picks"
            >
              View All <ChevronRight className="w-4 h-4" />
            </button>
          </div>
          
          {recommendations.length > 0 ? (
            <div className="grid md:grid-cols-2 gap-4">
              {recommendations.slice(0, 4).map((pick) => (
                <TopPickCard 
                  key={pick.id} 
                  pick={pick} 
                  onClick={() => navigate(`/predictions`)}
                />
              ))}
            </div>
          ) : (
            <div className="stat-card text-center py-8">
              <AlertCircle className="w-12 h-12 text-text-muted mx-auto mb-3" />
              <p className="text-text-muted mb-2">No picks available yet</p>
              <p className="text-text-muted text-xs">Picks are generated automatically using our algorithm.</p>
              <p className="text-brand-primary text-xs mt-2">Check back soon!</p>
            </div>
          )}
        </div>

        {/* Upcoming Events */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-mono font-bold text-lg text-text-primary flex items-center gap-2">
              <Calendar className="w-5 h-5 text-brand-primary" />
              Upcoming Events
            </h2>
          </div>
          
          <div className="space-y-3">
            {events.slice(0, 4).map((event) => (
              <LiveEventCard key={event.id} event={event} />
            ))}
            
            {events.length === 0 && (
              <div className="stat-card text-center py-6">
                <p className="text-text-muted text-sm">No live events</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Performance by Sport */}
      {performance?.by_sport && Object.keys(performance.by_sport).length > 0 && (
        <div className="stat-card">
          <h2 className="font-mono font-bold text-lg text-text-primary mb-4">Performance by Sport</h2>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Sport</th>
                  <th>Wins</th>
                  <th>Losses</th>
                  <th>Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(performance.by_sport).map(([sport, stats]) => {
                  const winRate = stats.total > 0 ? ((stats.wins / stats.total) * 100).toFixed(1) : 0;
                  return (
                    <tr key={sport}>
                      <td className="text-text-primary">{sport.replace(/_/g, ' ')}</td>
                      <td className="font-mono text-semantic-success">{stats.wins}</td>
                      <td className="font-mono text-semantic-danger">{stats.losses}</td>
                      <td className={`font-mono font-bold ${
                        winRate >= 55 ? "text-semantic-success" : 
                        winRate >= 45 ? "text-semantic-warning" : "text-semantic-danger"
                      }`}>
                        {winRate}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
