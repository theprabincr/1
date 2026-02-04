import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import { 
  TrendingUp, TrendingDown, Trophy, Target, 
  DollarSign, Clock, ChevronRight, Zap,
  Activity, AlertCircle, RefreshCw, X, Calendar,
  ChevronDown, ChevronUp
} from "lucide-react";
import { useNavigate } from "react-router-dom";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Analysis Section Component - Collapsible with proper formatting
const AnalysisSection = ({ analysisText, defaultExpanded = false }) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  
  if (!analysisText) return null;
  
  // Parse analysis into sections
  const parseAnalysis = (text) => {
    const sections = [];
    const lines = text.split('\n');
    let currentSection = null;
    let currentContent = [];
    
    const sectionHeaders = [
      'V6 STRONG SIGNAL', 'STRONG CONSENSUS', 'V6 ML ENSEMBLE',
      'PREDICTION OVERVIEW', 'MODEL AGREEMENT', 'TEAM STRENGTH',
      'RECENT FORM & RECORDS', 'SITUATIONAL FACTORS', 'INJURY IMPACT',
      'LINE MOVEMENT', 'SIMULATION RESULTS', 'WHY THIS BET TYPE',
      'KEY FACTORS', 'V5 LINE MOVEMENT', 'XGBOOST ML PREDICTION',
      'MODERATE CONSENSUS', 'XGB ONLY'
    ];
    
    lines.forEach((line) => {
      const trimmedLine = line.trim();
      
      const isHeader = sectionHeaders.some(header => 
        trimmedLine.toUpperCase().includes(header) && 
        !trimmedLine.startsWith('•') && 
        !trimmedLine.startsWith('-')
      );
      
      if (trimmedLine.match(/^[=\-]{10,}$/)) return;
      
      if (isHeader && trimmedLine.length > 0) {
        if (currentSection) {
          sections.push({ header: currentSection, content: currentContent });
        }
        currentSection = trimmedLine.replace(/[🎯📊💰✅⚠️📈🤖🎲]/g, '').trim();
        currentContent = [];
      } else if (trimmedLine.length > 0) {
        currentContent.push(trimmedLine);
      }
    });
    
    if (currentSection && currentContent.length > 0) {
      sections.push({ header: currentSection, content: currentContent });
    }
    
    return sections;
  };
  
  const sections = parseAnalysis(analysisText);
  
  const getSummary = () => {
    const overviewSection = sections.find(s => s.header.includes('OVERVIEW'));
    if (overviewSection) {
      return overviewSection.content.slice(0, 3).join(' • ');
    }
    return sections[0]?.content.slice(0, 2).join(' • ') || '';
  };
  
  const getSectionStyle = (header) => {
    if (header.includes('OVERVIEW')) return { bg: 'bg-brand-primary/10', border: 'border-brand-primary/30', text: 'text-brand-primary' };
    if (header.includes('MODEL') || header.includes('XGBOOST') || header.includes('CONSENSUS')) return { bg: 'bg-purple-500/10', border: 'border-purple-500/30', text: 'text-purple-400' };
    if (header.includes('STRENGTH') || header.includes('FORM')) return { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400' };
    if (header.includes('INJURY')) return { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400' };
    if (header.includes('LINE') || header.includes('MOVEMENT')) return { bg: 'bg-cyan-500/10', border: 'border-cyan-500/30', text: 'text-cyan-400' };
    if (header.includes('SIMULATION')) return { bg: 'bg-orange-500/10', border: 'border-orange-500/30', text: 'text-orange-400' };
    if (header.includes('KEY')) return { bg: 'bg-green-500/10', border: 'border-green-500/30', text: 'text-green-400' };
    return { bg: 'bg-zinc-700/50', border: 'border-zinc-600', text: 'text-text-secondary' };
  };
  
  return (
    <div className="mt-3 pt-3 border-t border-zinc-700/50">
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={(e) => { e.stopPropagation(); setIsExpanded(!isExpanded); }}
      >
        <p className="text-xs text-text-muted font-semibold uppercase tracking-wide">Analysis</p>
        <button className="flex items-center gap-1 text-xs text-text-muted hover:text-brand-primary transition-colors">
          {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </button>
      </div>
      
      {!isExpanded ? (
        <p className="mt-2 text-xs text-text-secondary line-clamp-2">{getSummary()}</p>
      ) : (
        <div className="mt-2 space-y-2 max-h-[300px] overflow-y-auto pr-1">
          {sections.slice(0, 6).map((section, idx) => {
            const style = getSectionStyle(section.header);
            return (
              <div key={idx} className={`p-2 rounded border ${style.bg} ${style.border}`}>
                <h4 className={`text-[10px] font-bold uppercase tracking-wide mb-1 ${style.text}`}>
                  {section.header}
                </h4>
                <div className="space-y-0.5">
                  {section.content.slice(0, 5).map((line, lineIdx) => {
                    if (line.startsWith('•') || line.startsWith('-')) {
                      return (
                        <p key={lineIdx} className="text-[11px] text-text-secondary pl-1">
                          • {line.replace(/^[•\-]\s*/, '')}
                        </p>
                      );
                    }
                    if (line.includes(':')) {
                      const [key, ...valueParts] = line.split(':');
                      const value = valueParts.join(':').trim();
                      return (
                        <p key={lineIdx} className="text-[11px]">
                          <span className="text-text-muted">{key}:</span>
                          <span className="text-text-primary ml-1">{value}</span>
                        </p>
                      );
                    }
                    return <p key={lineIdx} className="text-[11px] text-text-secondary">{line}</p>;
                  })}
                </div>
              </div>
            );
          })}
          {sections.length > 6 && (
            <p className="text-[10px] text-text-muted text-center">+{sections.length - 6} more sections...</p>
          )}
        </div>
      )}
    </div>
  );
};

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
      // For spread, extract spread value from outcome or analysis
      // Check if outcome already has spread value (e.g., "Boston Celtics -5.5")
      if (outcome.match(/-?\d+\.?\d*/)) {
        return outcome;
      }
      // Try to extract spread from analysis
      const spreadMatch = pick.analysis?.match(/(-?\d+\.?\d*)\s*(?:point|spread)/i) || 
                         pick.reasoning?.match(/(-?\d+\.?\d*)\s*(?:point|spread)/i);
      if (spreadMatch) {
        return `${outcome} ${spreadMatch[1]}`;
      }
      // Default: show team name with typical spread indicator
      const isHome = outcome.toLowerCase().includes(pick.home_team?.toLowerCase());
      return `${outcome} ${isHome ? '-' : '+'}X.X`;
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
      
      {/* Analysis Section - Collapsible with proper formatting */}
      {(pick.reasoning || pick.analysis) && pick.result === 'pending' && (
        <AnalysisSection analysisText={pick.reasoning || pick.analysis} />
      )}
    </div>
  );
};

// Reusable Pick Card for modals
const PickCard = ({ pick, showResult = false }) => {
  const FIXED_BANKROLL = 100;
  
  // Format the pick display based on prediction type
  const formatPick = () => {
    const outcome = pick.predicted_outcome || '';
    
    if (pick.prediction_type === 'spread') {
      // Check if outcome already has spread value (e.g., "Boston Celtics -5.5")
      if (outcome.match(/-?\d+\.?\d*/)) {
        return outcome;
      }
      // Try to extract spread from analysis or reasoning
      const analysis = pick.analysis || pick.reasoning || '';
      const spreadMatch = analysis.match(/(-?\d+\.?\d*)\s*(?:point|spread|pts)/i);
      if (spreadMatch) {
        const isHome = outcome.toLowerCase().includes(pick.home_team?.toLowerCase());
        return `${outcome} ${isHome ? '' : '+'}${spreadMatch[1]}`;
      }
      // Default: indicate it's a spread pick
      return `${outcome} (spread)`;
    } else if (pick.prediction_type === 'total') {
      return outcome; // e.g., "Over 225.5"
    }
    return outcome; // Moneyline just shows team name
  };
  
  // Market badge
  const marketBadge = {
    'moneyline': { label: 'ML', color: 'bg-blue-500/20 text-blue-400' },
    'spread': { label: 'SPR', color: 'bg-purple-500/20 text-purple-400' },
    'total': { label: 'O/U', color: 'bg-orange-500/20 text-orange-400' }
  }[pick.prediction_type] || { label: 'ML', color: 'bg-blue-500/20 text-blue-400' };
  
  // Calculate profit/loss
  const odds = pick.odds_at_prediction || 1.91;
  let profit = 0;
  if (pick.result === 'win') {
    profit = FIXED_BANKROLL * (odds - 1);
  } else if (pick.result === 'loss') {
    profit = -FIXED_BANKROLL;
  }
  
  // Result badge
  const resultBadge = {
    'win': { label: 'WON', color: 'bg-semantic-success/20 text-semantic-success' },
    'loss': { label: 'LOST', color: 'bg-semantic-danger/20 text-semantic-danger' },
    'push': { label: 'PUSH', color: 'bg-text-muted/20 text-text-muted' },
    'pending': { label: 'PENDING', color: 'bg-semantic-warning/20 text-semantic-warning' }
  }[pick.result] || { label: 'PENDING', color: 'bg-semantic-warning/20 text-semantic-warning' };
  
  return (
    <div className="bg-zinc-800 rounded-lg p-4">
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${marketBadge.color}`}>
              {marketBadge.label}
            </span>
            <p className="text-text-primary font-semibold">{pick.home_team} vs {pick.away_team}</p>
          </div>
          <p className="text-text-muted text-sm">
            {pick.commence_time ? new Date(pick.commence_time).toLocaleDateString() : ''} 
            {pick.commence_time ? ` at ${new Date(pick.commence_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}` : ''}
          </p>
        </div>
        <div className="text-right flex flex-col gap-1">
          <span className={`px-2 py-1 rounded text-xs font-bold ${
            pick.confidence >= 0.7 ? 'bg-semantic-success/20 text-semantic-success' :
            pick.confidence >= 0.5 ? 'bg-semantic-warning/20 text-semantic-warning' :
            'bg-text-muted/20 text-text-muted'
          }`}>
            {(pick.confidence * 100).toFixed(0)}%
          </span>
          {showResult && (
            <span className={`px-2 py-1 rounded text-xs font-bold ${resultBadge.color}`}>
              {resultBadge.label}
            </span>
          )}
        </div>
      </div>
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-zinc-700">
        <div>
          <p className="text-brand-primary font-bold font-mono">{formatPick()}</p>
          <p className="text-text-muted text-xs mt-1">$100 bet @ {odds.toFixed(2)}</p>
        </div>
        {showResult && pick.result !== 'pending' && (
          <p className={`font-mono text-lg font-bold ${
            profit >= 0 ? 'text-semantic-success' : 'text-semantic-danger'
          }`}>
            {profit >= 0 ? '+' : ''}${profit.toFixed(2)}
          </p>
        )}
      </div>
      {/* Analysis Section - Only show for pending picks */}
      {(pick.reasoning || pick.analysis) && pick.result === 'pending' && (
        <div className="mt-3 pt-3 border-t border-zinc-700">
          <p className="text-xs text-text-muted mb-2 font-semibold uppercase tracking-wide">Analysis</p>
          <p className="text-text-secondary text-sm leading-relaxed whitespace-pre-line">{pick.reasoning || pick.analysis}</p>
        </div>
      )}
    </div>
  );
};

// Sport display names mapping
const sportDisplayNames = {
  'basketball_nba': 'NBA',
  'americanfootball_nfl': 'NFL',
  'baseball_mlb': 'MLB',
  'icehockey_nhl': 'NHL',
  'soccer_epl': 'EPL'
};

// Format sport key to display name
const formatSportName = (sportKey) => {
  if (!sportKey) return '';
  return sportDisplayNames[sportKey] || sportKey.split('_').pop().toUpperCase();
};

// Compact Event Item - Simple info display without odds
const CompactEventItem = ({ event, showSport = false }) => {
  // Format time
  const formatTime = (datetime) => {
    if (!datetime) return 'TBD';
    const date = new Date(datetime);
    return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  };
  
  const formatDate = (datetime) => {
    if (!datetime) return '';
    const date = new Date(datetime);
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    
    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === tomorrow.toDateString()) {
      return 'Tomorrow';
    }
    return date.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
  };
  
  // Calculate time until game
  const getTimeUntil = () => {
    if (!event.commence_time) return null;
    const now = new Date();
    const gameTime = new Date(event.commence_time);
    const diffMs = gameTime - now;
    if (diffMs < 0) return 'Started';
    const hours = Math.floor(diffMs / (1000 * 60 * 60));
    const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
    if (hours < 1) return `${minutes}m`;
    if (hours < 24) return `${hours}h ${minutes}m`;
    return null;
  };
  
  const timeUntil = getTimeUntil();
  const isStartingSoon = timeUntil && !timeUntil.includes('Started') && 
    (timeUntil.includes('m') || (timeUntil.includes('h') && parseInt(timeUntil) < 3));
  
  return (
    <div className="flex items-center gap-3 p-3 bg-zinc-800/50 rounded-lg hover:bg-zinc-800 transition-colors">
      {/* Time Info */}
      <div className="flex flex-col items-center min-w-[60px]">
        <span className="text-xs text-text-muted">{formatDate(event.commence_time)}</span>
        <span className="text-sm font-mono font-bold text-text-primary">{formatTime(event.commence_time)}</span>
        {isStartingSoon && (
          <span className="text-[10px] text-semantic-warning font-bold mt-0.5">{timeUntil}</span>
        )}
      </div>
      
      {/* Divider */}
      <div className="w-px h-10 bg-zinc-700"></div>
      
      {/* Teams */}
      <div className="flex-1 min-w-0">
        <p className="text-sm text-text-primary font-medium truncate">{event.away_team}</p>
        <p className="text-xs text-text-muted">@</p>
        <p className="text-sm text-text-primary font-medium truncate">{event.home_team}</p>
      </div>
      
      {/* Sport Badge (optional) */}
      {showSport && (
        <span className="text-[10px] font-mono font-bold text-text-secondary bg-zinc-700 px-2 py-1 rounded">
          {formatSportName(event.sport_key)}
        </span>
      )}
    </div>
  );
};

// Helper to get best odds (decimal format - higher is better)
const getBestOdds = (bookmakers, homeTeam, awayTeam) => {
  let bestHome = 1;
  let bestAway = 1;
  let spread = null;
  let total = null;
  
  bookmakers.forEach(bm => {
    bm.markets?.forEach(market => {
      if (market.key === 'h2h') {
        market.outcomes?.forEach(outcome => {
          // Match by team name if available, otherwise use first outcome
          const isHomeTeam = homeTeam ? outcome.name === homeTeam : 
            outcome.name === bookmakers[0]?.markets?.[0]?.outcomes?.[0]?.name;
          
          if (isHomeTeam) {
            if (outcome.price > bestHome) bestHome = outcome.price;
          } else {
            if (outcome.price > bestAway) bestAway = outcome.price;
          }
        });
      }
      // Extract spread
      if (market.key === 'spreads' && spread === null) {
        const homeSpread = market.outcomes?.find(o => o.name === homeTeam);
        if (homeSpread) spread = homeSpread.point;
      }
      // Extract total
      if (market.key === 'totals' && total === null) {
        const over = market.outcomes?.find(o => o.name === 'Over');
        if (over) total = over.point;
      }
    });
  });
  
  return { 
    home: bestHome > 1 ? bestHome : 1.91, 
    away: bestAway > 1 ? bestAway : 1.91,
    spread,
    total
  };
};

const Dashboard = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [performance, setPerformance] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [allPicks, setAllPicks] = useState([]);
  const [events, setEvents] = useState([]);
  const [showActivePicksModal, setShowActivePicksModal] = useState(false);
  const [showAllPicksModal, setShowAllPicksModal] = useState(false);
  const [liveScores, setLiveScores] = useState([]);
  const [selectedLiveSport, setSelectedLiveSport] = useState('all');
  const [scoreChanges, setScoreChanges] = useState({});
  const [mlStatus, setMlStatus] = useState(null);
  
  // Use ref to track previous scores (persists across renders without causing re-renders)
  const previousScoresRef = useRef({});

  // All supported sports for fetching events
  const sportKeys = ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "soccer_epl"];

  // Sport display names
  const sportNames = {
    'basketball_nba': 'NBA',
    'americanfootball_nfl': 'NFL',
    'baseball_mlb': 'MLB',
    'icehockey_nhl': 'NHL',
    'soccer_epl': 'EPL'
  };

  // Detect score changes and trigger animations - memoized to prevent recreation
  const detectScoreChanges = useCallback((newScores) => {
    const changes = {};
    const prevScores = previousScoresRef.current;
    
    // Only detect changes if we have previous scores
    if (Object.keys(prevScores).length === 0) {
      console.log('[SCORE] First load - initializing previous scores');
    }
    
    newScores.forEach(game => {
      const gameId = game.espn_id || `${game.home_team}-${game.away_team}`;
      const prev = prevScores[gameId];
      
      if (prev) {
        // Check if home score changed
        if (game.home_score !== prev.home_score) {
          changes[`${gameId}-home`] = true;
          console.log(`[SCORE CHANGE] ${game.home_team}: ${prev.home_score} → ${game.home_score}`);
        }
        // Check if away score changed
        if (game.away_score !== prev.away_score) {
          changes[`${gameId}-away`] = true;
          console.log(`[SCORE CHANGE] ${game.away_team}: ${prev.away_score} → ${game.away_score}`);
        }
      }
    });
    
    // If any scores changed, trigger animation
    if (Object.keys(changes).length > 0) {
      console.log('[SCORE] Triggering animation for:', Object.keys(changes));
      setScoreChanges({...changes}); // Spread to ensure new object reference
    }
    
    // Clear the animation after 2.5 seconds
    if (Object.keys(changes).length > 0) {
      setTimeout(() => {
        console.log('[SCORE] Clearing animation');
        setScoreChanges({});
      }, 2500);
    }
    
    // Update previous scores ref for next comparison
    const newPrevScores = {};
    newScores.forEach(game => {
      const gameId = game.espn_id || `${game.home_team}-${game.away_team}`;
      newPrevScores[gameId] = {
        home_score: game.home_score,
        away_score: game.away_score
      };
    });
    previousScoresRef.current = newPrevScores;
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      // Fetch events from ALL sports
      const eventsPromises = sportKeys.map(key => 
        axios.get(`${API}/events/${key}`).catch(() => ({ data: [] }))
      );
      
      const [perfRes, recsRes, liveRes, allPicksRes, ...eventsResponses] = await Promise.all([
        axios.get(`${API}/performance`),
        axios.get(`${API}/recommendations?limit=50&min_confidence=0.60`),
        axios.get(`${API}/live-scores`),
        axios.get(`${API}/recommendations?limit=100&min_confidence=0`),
        ...eventsPromises
      ]);
      
      setPerformance(perfRes.data);
      setRecommendations(recsRes.data);
      
      // Detect score changes (works on every refresh after first load)
      const newLiveScores = liveRes.data.games || [];
      detectScoreChanges(newLiveScores);
      setLiveScores(newLiveScores);
      
      // Combine events from all sports and sort by commence time
      const allEvents = eventsResponses
        .flatMap((res, idx) => (res.data || []).map(e => ({ ...e, sport_key: sportKeys[idx] })))
        .sort((a, b) => new Date(a.commence_time) - new Date(b.commence_time))
        .slice(0, 50); // Show up to 50 upcoming events for better sport coverage
      
      setEvents(allEvents);
      
      // Combine all picks (from recommendations and performance history)
      const activePicks = allPicksRes.data || [];
      const completedPicks = perfRes.data.recent_predictions || [];
      
      // Merge and dedupe
      const allPicksMap = new Map();
      [...activePicks, ...completedPicks].forEach(p => {
        if (!allPicksMap.has(p.id)) {
          allPicksMap.set(p.id, p);
        }
      });
      
      // Sort by date (newest first)
      const sortedPicks = Array.from(allPicksMap.values()).sort((a, b) => {
        const dateA = new Date(a.commence_time || a.created_at || 0);
        const dateB = new Date(b.commence_time || b.created_at || 0);
        return dateB - dateA;
      });
      
      setAllPicks(sortedPicks);
    } catch (error) {
      console.error("Error fetching dashboard data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []); // Remove selectedSport dependency - fetch all sports on mount

  // Auto-refresh live scores every 10 seconds with score change detection
  useEffect(() => {
    const fetchLiveScoresWithAnimation = async () => {
      try {
        const res = await axios.get(`${API}/live-scores`);
        const newScores = res.data.games || [];
        
        // Detect score changes and trigger animations
        detectScoreChanges(newScores);
        
        // Update the live scores state
        setLiveScores(newScores);
      } catch (error) {
        console.error("Error fetching live scores:", error);
      }
    };

    // Set up interval for auto-refresh (don't fetch immediately - fetchData already does that)
    const interval = setInterval(fetchLiveScoresWithAnimation, 10000); // Every 10 seconds

    // Cleanup on unmount
    return () => clearInterval(interval);
  }, []);

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
          title="Active Picks"
          value={recommendations.length}
          subtitle="Click to view all"
          icon={Clock}
          color="yellow"
          onClick={() => setShowActivePicksModal(true)}
        />
        <StatCard
          title="Total Picks"
          value={allPicks.length}
          subtitle="All time picks"
          icon={Target}
          color="blue"
          onClick={() => setShowAllPicksModal(true)}
        />
        <StatCard
          title="Live Games"
          value={liveScores.length}
          subtitle="In progress"
          icon={Activity}
          color={liveScores.length > 0 ? "green" : "blue"}
        />
      </div>

      {/* Live Games Section - Only shows when games are live */}
      {liveScores.length > 0 && (
        <div className="bg-gradient-to-r from-zinc-900 to-zinc-800 border border-zinc-700 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-mono font-bold text-lg text-text-primary flex items-center gap-2">
              <Activity className="w-5 h-5 text-semantic-success animate-pulse" />
              Live Games
              <span className="text-xs text-text-muted ml-2 font-normal">Auto-updates every 10s</span>
            </h2>
          </div>
          
          {/* Sport Filter Tabs */}
          <div className="flex gap-2 mb-4 overflow-x-auto pb-2">
            <button
              onClick={() => setSelectedLiveSport('all')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all whitespace-nowrap ${
                selectedLiveSport === 'all'
                  ? 'bg-brand-primary text-zinc-900'
                  : 'bg-zinc-800 text-text-muted hover:bg-zinc-700 hover:text-text-primary'
              }`}
            >
              All ({liveScores.length})
            </button>
            {[...new Set(liveScores.map(g => g.sport_key))].map(sport => {
              const count = liveScores.filter(g => g.sport_key === sport).length;
              return (
                <button
                  key={sport}
                  onClick={() => setSelectedLiveSport(sport)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all whitespace-nowrap ${
                    selectedLiveSport === sport
                      ? 'bg-brand-primary text-zinc-900'
                      : 'bg-zinc-800 text-text-muted hover:bg-zinc-700 hover:text-text-primary'
                  }`}
                >
                  {sportNames[sport] || sport?.replace(/_/g, ' ').toUpperCase()} ({count})
                </button>
              );
            })}
          </div>
          
          {/* Filtered Live Games Grid */}
          <div className="grid md:grid-cols-2 gap-4">
            {liveScores
              .filter(game => selectedLiveSport === 'all' || game.sport_key === selectedLiveSport)
              .map((game, i) => {
                const gameId = game.espn_id || `${game.home_team}-${game.away_team}`;
                const homeChanged = scoreChanges[`${gameId}-home`];
                const awayChanged = scoreChanges[`${gameId}-away`];
                
                return (
                  <div 
                    key={gameId} 
                    className="bg-zinc-800/50 rounded-lg p-4 border border-zinc-700 hover:border-brand-primary/50 transition-all"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-mono text-brand-primary uppercase flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-semantic-success animate-pulse"></span>
                        {sportNames[game.sport_key] || game.sport_key?.replace(/_/g, ' ')}
                      </span>
                      <span className="text-xs text-text-muted font-mono">
                        {game.clock} • {game.period ? `Q${game.period}` : 'Live'}
                      </span>
                    </div>
                    
                    <div className="space-y-2">
                      <div className={`flex items-center justify-between p-1 rounded transition-all duration-500 ${
                        awayChanged ? 'bg-semantic-success/30' : ''
                      }`}>
                        <span className="text-sm text-text-primary font-medium">{game.away_team}</span>
                        <span className={`font-mono text-xl font-bold transition-all duration-500 ${
                          awayChanged 
                            ? 'text-semantic-success scale-110' 
                            : 'text-brand-primary'
                        }`}>
                          {game.away_score}
                        </span>
                      </div>
                      <div className={`flex items-center justify-between p-1 rounded transition-all duration-500 ${
                        homeChanged ? 'bg-semantic-success/30' : ''
                      }`}>
                        <span className="text-sm text-text-primary font-medium">{game.home_team}</span>
                        <span className={`font-mono text-xl font-bold transition-all duration-500 ${
                          homeChanged 
                            ? 'text-semantic-success scale-110' 
                            : 'text-brand-primary'
                        }`}>
                          {game.home_score}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
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
                  <PickCard key={pick.id || i} pick={pick} />
                ))
              )}
            </div>
            <div className="p-4 border-t border-zinc-800">
              <button 
                onClick={() => { setShowActivePicksModal(false); navigate('/performance'); }}
                className="btn-primary w-full"
              >
                View Performance
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Total Picks Modal */}
      {showAllPicksModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-zinc-900 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-zinc-800">
              <h2 className="font-mono font-bold text-lg text-text-primary flex items-center gap-2">
                <Target className="w-5 h-5 text-brand-secondary" />
                Total Picks ({allPicks.length})
              </h2>
              <button 
                onClick={() => setShowAllPicksModal(false)}
                className="p-2 rounded-lg hover:bg-zinc-800 text-text-muted hover:text-text-primary"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto max-h-[60vh] space-y-3">
              {allPicks.length === 0 ? (
                <p className="text-text-muted text-center py-8">No picks yet.</p>
              ) : (
                allPicks.map((pick, i) => (
                  <PickCard key={pick.id || i} pick={pick} showResult={true} />
                ))
              )}
            </div>
            <div className="p-4 border-t border-zinc-800">
              <button 
                onClick={() => { setShowAllPicksModal(false); navigate('/performance'); }}
                className="btn-primary w-full"
              >
                View Detailed Performance
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Auto-Pick Info Banner */}
      <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4 flex items-center gap-3">
        <div className="p-2 bg-brand-primary/20 rounded-lg">
          <Clock className="w-5 h-5 text-brand-primary" />
        </div>
        <div>
          <p className="text-text-primary text-sm font-medium">Picks Auto-Generate 40 Minutes Before Game Time</p>
          <p className="text-text-muted text-xs">Our algorithm analyzes confirmed lineups, injuries, and line movements to generate optimal picks.</p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Top Picks */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-mono font-bold text-lg text-text-primary flex items-center gap-2">
              <Zap className="w-5 h-5 text-brand-primary" />
              Top Picks (All Sports)
            </h2>
            <button 
              onClick={() => navigate('/performance')}
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
                  onClick={() => navigate('/performance')}
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

        {/* Upcoming Events - Organized by sections */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-mono font-bold text-lg text-text-primary flex items-center gap-2">
              <Calendar className="w-5 h-5 text-brand-primary" />
              Upcoming Events
            </h2>
            <button 
              onClick={() => navigate('/events')}
              className="text-brand-primary text-sm flex items-center gap-1 hover:underline"
            >
              View All <ChevronRight className="w-4 h-4" />
            </button>
          </div>
          
          {events.length > 0 ? (
            <div className="space-y-4 max-h-[500px] overflow-y-auto pr-1">
              {/* Starting Soon Section - Events within 3 hours */}
              {(() => {
                const now = new Date();
                const threeHoursFromNow = new Date(now.getTime() + 3 * 60 * 60 * 1000);
                const startingSoon = events.filter(e => {
                  const gameTime = new Date(e.commence_time);
                  return gameTime > now && gameTime <= threeHoursFromNow;
                });
                
                if (startingSoon.length > 0) {
                  return (
                    <div className="bg-gradient-to-r from-semantic-warning/10 to-transparent border border-semantic-warning/30 rounded-lg p-3">
                      <div className="flex items-center gap-2 mb-3">
                        <Clock className="w-4 h-4 text-semantic-warning" />
                        <span className="text-xs font-bold text-semantic-warning uppercase tracking-wide">Starting Soon</span>
                        <span className="text-xs text-text-muted">({startingSoon.length})</span>
                      </div>
                      <div className="space-y-2">
                        {startingSoon.slice(0, 3).map((event) => (
                          <CompactEventItem key={event.id} event={event} showSport={true} />
                        ))}
                      </div>
                    </div>
                  );
                }
                return null;
              })()}
              
              {/* Events by Sport */}
              {Object.entries(
                events.reduce((acc, event) => {
                  const sport = event.sport_key || 'other';
                  if (!acc[sport]) acc[sport] = [];
                  acc[sport].push(event);
                  return acc;
                }, {})
              ).map(([sportKey, sportEvents]) => (
                <div key={sportKey} className="bg-zinc-800/30 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-xs font-bold text-brand-primary uppercase tracking-wide">
                      {formatSportName(sportKey)}
                    </span>
                    <span className="text-xs text-text-muted">({sportEvents.length} games)</span>
                  </div>
                  <div className="space-y-2">
                    {sportEvents.slice(0, 3).map((event) => (
                      <CompactEventItem key={event.id} event={event} />
                    ))}
                    {sportEvents.length > 3 && (
                      <button 
                        onClick={() => navigate('/events')}
                        className="w-full text-center text-xs text-text-muted hover:text-brand-primary py-2"
                      >
                        +{sportEvents.length - 3} more {formatSportName(sportKey)} games
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="stat-card text-center py-6">
              <Calendar className="w-8 h-8 text-text-muted mx-auto mb-2" />
              <p className="text-text-muted text-sm">No upcoming events</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
