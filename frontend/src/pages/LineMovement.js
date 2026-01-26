import { useState, useEffect } from "react";
import axios from "axios";
import { 
  TrendingUp, TrendingDown, RefreshCw, Clock,
  AlertCircle, Activity, Calendar, ArrowRight
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid, ReferenceLine } from "recharts";
import { format, parseISO } from "date-fns";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Custom Tooltip for chart
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 shadow-lg">
        <p className="text-xs text-text-muted mb-2">{label}</p>
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center justify-between gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div 
                className="w-3 h-3 rounded-full" 
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-text-secondary">{entry.name}</span>
            </div>
            <span className="font-mono font-bold text-text-primary">
              {entry.value?.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

// Line Movement Chart Component - updated to handle new API structure
const MovementChart = ({ chartData, homeTeam, awayTeam, openingOdds }) => {
  // Handle both old and new API structure for opening odds
  const openingHomeOdds = openingOdds?.ml?.home || openingOdds?.home_odds;
  const openingAwayOdds = openingOdds?.ml?.away || openingOdds?.away_odds;
  
  if (!chartData || chartData.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 bg-zinc-900 rounded-lg">
        <Clock className="w-12 h-12 text-text-muted mb-3" />
        <p className="text-text-muted">No line movement data yet</p>
        <p className="text-text-muted text-sm mt-1">Data will appear as odds change</p>
      </div>
    );
  }

  // Calculate min/max for Y axis
  const allOdds = chartData.flatMap(d => [d.home_odds, d.away_odds]).filter(Boolean);
  const minOdds = Math.max(1, Math.floor(Math.min(...allOdds) * 10 - 1) / 10);
  const maxOdds = Math.ceil(Math.max(...allOdds) * 10 + 1) / 10;

  return (
    <div className="bg-zinc-900 rounded-lg p-4">
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis 
            dataKey="time" 
            stroke="#71717A"
            fontSize={11}
            tickLine={false}
            axisLine={false}
          />
          <YAxis 
            stroke="#71717A"
            fontSize={11}
            tickLine={false}
            axisLine={false}
            domain={[minOdds, maxOdds]}
            tickFormatter={(val) => val.toFixed(2)}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ paddingTop: '10px' }}
          />
          {/* Opening odds reference lines */}
          {openingHomeOdds && (
            <ReferenceLine 
              y={openingHomeOdds} 
              stroke="#ADFF2F" 
              strokeDasharray="5 5" 
              strokeOpacity={0.5}
            />
          )}
          {openingAwayOdds && (
            <ReferenceLine 
              y={openingAwayOdds} 
              stroke="#00CED1" 
              strokeDasharray="5 5" 
              strokeOpacity={0.5}
            />
          )}
          <Line
            type="monotone"
            dataKey="home_odds"
            name={homeTeam || "Home"}
            stroke="#ADFF2F"
            strokeWidth={2}
            dot={{ fill: '#ADFF2F', r: 4 }}
            activeDot={{ r: 6, fill: '#ADFF2F' }}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="away_odds"
            name={awayTeam || "Away"}
            stroke="#00CED1"
            strokeWidth={2}
            dot={{ fill: '#00CED1', r: 4 }}
            activeDot={{ r: 6, fill: '#00CED1' }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="flex justify-center gap-6 mt-2 text-xs text-text-muted">
        <span className="flex items-center gap-1">
          <span className="w-4 h-px bg-[#ADFF2F] inline-block" style={{ borderStyle: 'dashed' }}></span>
          Opening odds (dashed)
        </span>
      </div>
    </div>
  );
};

// Opening vs Current Odds Comparison - updated to handle new API structure
const OddsComparison = ({ opening, current, homeTeam, awayTeam }) => {
  // Handle both old and new API structure
  const openingHomeOdds = opening?.ml?.home || opening?.home_odds;
  const openingAwayOdds = opening?.ml?.away || opening?.away_odds;
  const openingTimestamp = opening?.timestamp;
  const openingSpread = opening?.spread;
  const openingTotal = opening?.total;
  
  const currentHomeOdds = current?.home;
  const currentAwayOdds = current?.away;
  
  const getChangeIndicator = (open, curr) => {
    if (!open || !curr) return null;
    const change = curr - open;
    const pctChange = ((curr - open) / open * 100).toFixed(1);
    
    if (Math.abs(change) < 0.01) return <span className="text-text-muted">—</span>;
    
    return (
      <span className={`flex items-center gap-1 text-sm ${change > 0 ? 'text-semantic-success' : 'text-semantic-danger'}`}>
        {change > 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
        {change > 0 ? '+' : ''}{pctChange}%
      </span>
    );
  };

  return (
    <div className="stat-card">
      <h3 className="font-mono font-bold text-text-primary mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5 text-brand-primary" />
        Odds Movement Summary
      </h3>
      
      <div className="space-y-4">
        {/* Home Team */}
        <div className="bg-zinc-800 p-4 rounded-lg">
          <p className="text-text-muted text-xs mb-2">{homeTeam || "Home"}</p>
          <div className="flex items-center justify-between">
            <div className="text-center">
              <p className="text-text-muted text-xs">Opening</p>
              <p className="font-mono text-xl font-bold text-text-primary">
                {openingHomeOdds?.toFixed(2) || "—"}
              </p>
            </div>
            <ArrowRight className="w-5 h-5 text-text-muted" />
            <div className="text-center">
              <p className="text-text-muted text-xs">Current</p>
              <p className="font-mono text-xl font-bold text-brand-primary">
                {currentHomeOdds?.toFixed(2) || "—"}
              </p>
            </div>
            <div className="text-center min-w-[60px]">
              <p className="text-text-muted text-xs">Change</p>
              {getChangeIndicator(openingHomeOdds, currentHomeOdds)}
            </div>
          </div>
        </div>
        
        {/* Away Team */}
        <div className="bg-zinc-800 p-4 rounded-lg">
          <p className="text-text-muted text-xs mb-2">{awayTeam || "Away"}</p>
          <div className="flex items-center justify-between">
            <div className="text-center">
              <p className="text-text-muted text-xs">Opening</p>
              <p className="font-mono text-xl font-bold text-text-primary">
                {openingAwayOdds?.toFixed(2) || "—"}
              </p>
            </div>
            <ArrowRight className="w-5 h-5 text-text-muted" />
            <div className="text-center">
              <p className="text-text-muted text-xs">Current</p>
              <p className="font-mono text-xl font-bold text-brand-secondary">
                {currentAwayOdds?.toFixed(2) || "—"}
              </p>
            </div>
            <div className="text-center min-w-[60px]">
              <p className="text-text-muted text-xs">Change</p>
              {getChangeIndicator(openingAwayOdds, currentAwayOdds)}
            </div>
          </div>
        </div>
        
        {/* Spread and Total if available */}
        {(openingSpread !== null || openingTotal !== null) && (
          <div className="flex gap-4 pt-2 border-t border-zinc-700">
            {openingSpread !== null && (
              <div className="text-center flex-1">
                <p className="text-text-muted text-xs">Opening Spread</p>
                <p className="font-mono text-lg font-bold text-text-primary">
                  {openingSpread > 0 ? '+' : ''}{openingSpread}
                </p>
              </div>
            )}
            {openingTotal !== null && (
              <div className="text-center flex-1">
                <p className="text-text-muted text-xs">Opening Total</p>
                <p className="font-mono text-lg font-bold text-text-primary">
                  {openingTotal}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
      
      {openingTimestamp && (
        <p className="text-text-muted text-xs mt-4">
          Market opened: {format(parseISO(openingTimestamp), 'MMM d, yyyy HH:mm')}
        </p>
      )}
    </div>
  );
};

// Bookmaker Breakdown Card
const BookmakerBreakdown = ({ bookmakers, currentBookmakers }) => {
  // Use current bookmaker data from event if snapshots are empty
  const displayBookmakers = bookmakers && bookmakers.length > 0 
    ? bookmakers 
    : currentBookmakers;
  
  if (!displayBookmakers || displayBookmakers.length === 0) {
    return null;
  }

  return (
    <div className="stat-card">
      <h3 className="font-mono font-bold text-text-primary mb-4 flex items-center gap-2">
        <Calendar className="w-5 h-5 text-brand-secondary" />
        Bookmakers ({displayBookmakers.length})
      </h3>
      <div className="space-y-2 max-h-72 overflow-y-auto">
        {displayBookmakers.map((bm, i) => {
          // Handle both snapshot format and current odds format
          const latestSnap = bm.snapshots?.[bm.snapshots.length - 1];
          const firstSnap = bm.snapshots?.[0];
          
          // For current bookmaker data from event
          const currentOdds = bm.markets?.[0]?.outcomes;
          
          let homeOdds, awayOdds, homeChange = null;
          
          if (latestSnap) {
            homeOdds = latestSnap.home_odds;
            awayOdds = latestSnap.away_odds;
            if (firstSnap) {
              homeChange = (latestSnap.home_odds - firstSnap.home_odds).toFixed(2);
            }
          } else if (currentOdds && currentOdds.length >= 2) {
            homeOdds = currentOdds[0].price;
            awayOdds = currentOdds[1].price;
          }
          
          return (
            <div key={bm.bookmaker || bm.key || i} className="bg-zinc-800 p-3 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-text-primary font-medium">
                  {bm.bookmaker_title || bm.title || bm.bookmaker}
                </span>
                {bm.snapshots && (
                  <span className="text-text-muted text-xs">
                    {bm.snapshots.length} updates
                  </span>
                )}
              </div>
              {(homeOdds || awayOdds) && (
                <div className="flex items-center justify-between text-sm">
                  <div className="flex gap-4">
                    <span className="text-brand-primary font-mono">
                      H: {homeOdds?.toFixed(2) || "—"}
                    </span>
                    <span className="text-brand-secondary font-mono">
                      A: {awayOdds?.toFixed(2) || "—"}
                    </span>
                  </div>
                  {homeChange && parseFloat(homeChange) !== 0 && (
                    <span className={`flex items-center gap-1 ${
                      parseFloat(homeChange) > 0 ? 'text-semantic-success' : 'text-semantic-danger'
                    }`}>
                      {parseFloat(homeChange) > 0 ? (
                        <TrendingUp className="w-3 h-3" />
                      ) : (
                        <TrendingDown className="w-3 h-3" />
                      )}
                      {homeChange}
                    </span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const LineMovement = () => {
  const [events, setEvents] = useState([]);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [lineData, setLineData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingLine, setLoadingLine] = useState(false);
  const [selectedSport, setSelectedSport] = useState("basketball_nba");
  const [lastUpdate, setLastUpdate] = useState(null);
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);

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
      
      // Auto-select first event
      if (response.data.length > 0 && !selectedEvent) {
        setSelectedEvent(response.data[0]);
        fetchLineMovement(response.data[0].id);
      }
    } catch (error) {
      console.error("Error fetching events:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSport]);

  // Real-time polling for line movement data (every 30 seconds)
  useEffect(() => {
    if (!selectedEvent || !isAutoRefresh) return;
    
    const pollInterval = setInterval(() => {
      fetchLineMovement(selectedEvent.id, true); // silent refresh
    }, 30000); // 30 seconds
    
    return () => clearInterval(pollInterval);
  }, [selectedEvent, isAutoRefresh, selectedSport]);

  const fetchLineMovement = async (eventId, silent = false) => {
    if (!silent) setLoadingLine(true);
    try {
      const response = await axios.get(`${API}/line-movement/${eventId}?sport_key=${selectedSport}`);
      setLineData(response.data);
      setLastUpdate(new Date());
    } catch (error) {
      console.error("Error fetching line movement:", error);
      if (!silent) setLineData(null);
    } finally {
      if (!silent) setLoadingLine(false);
    }
  };

  const handleEventSelect = (event) => {
    setSelectedEvent(event);
    fetchLineMovement(event.id);
  };

  // Process chart data for the chart - now handles the new structure with moneyline, spread, totals
  const processChartData = (marketType = 'moneyline') => {
    // chart_data is now an object: { moneyline: [], spread: [], totals: [] }
    const chartData = lineData?.chart_data?.[marketType] || lineData?.chart_data?.moneyline || [];
    
    if (!chartData || chartData.length === 0) {
      // If no stored snapshots, create data from current event odds
      if (selectedEvent?.bookmakers?.length > 0) {
        const now = new Date();
        const homeOdds = [];
        const awayOdds = [];
        
        selectedEvent.bookmakers.forEach(bm => {
          bm.markets?.forEach(market => {
            if (market.key === 'h2h' && market.outcomes?.length >= 2) {
              homeOdds.push(market.outcomes[0].price);
              awayOdds.push(market.outcomes[1].price);
            }
          });
        });
        
        if (homeOdds.length > 0) {
          const avgHome = homeOdds.reduce((a, b) => a + b, 0) / homeOdds.length;
          const avgAway = awayOdds.reduce((a, b) => a + b, 0) / awayOdds.length;
          
          return [{
            time: format(now, 'MM/dd HH:mm'),
            home_odds: parseFloat(avgHome.toFixed(2)),
            away_odds: parseFloat(avgAway.toFixed(2))
          }];
        }
      }
      return [];
    }
    
    if (marketType === 'moneyline') {
      return chartData.map(item => ({
        time: item.timestamp ? format(parseISO(item.timestamp), 'MM/dd HH:mm') : 'N/A',
        home_odds: item.home_odds,
        away_odds: item.away_odds
      }));
    } else if (marketType === 'spread') {
      return chartData.map(item => ({
        time: item.timestamp ? format(parseISO(item.timestamp), 'MM/dd HH:mm') : 'N/A',
        spread: item.spread
      }));
    } else if (marketType === 'totals') {
      return chartData.map(item => ({
        time: item.timestamp ? format(parseISO(item.timestamp), 'MM/dd HH:mm') : 'N/A',
        total: item.total
      }));
    }
    
    return [];
  };

  // Get current odds from selected event
  const getCurrentOdds = () => {
    // Handle new structure: current_odds.ml instead of current_odds directly
    if (lineData?.current_odds?.ml) {
      return lineData.current_odds.ml;
    }
    if (lineData?.current_odds?.home) {
      return lineData.current_odds;
    }
    
    if (selectedEvent?.bookmakers?.length > 0) {
      const homeOdds = [];
      const awayOdds = [];
      
      selectedEvent.bookmakers.forEach(bm => {
        bm.markets?.forEach(market => {
          if (market.key === 'h2h' && market.outcomes?.length >= 2) {
            homeOdds.push(market.outcomes[0].price);
            awayOdds.push(market.outcomes[1].price);
          }
        });
      });
      
      if (homeOdds.length > 0) {
        return {
          home: parseFloat((homeOdds.reduce((a, b) => a + b, 0) / homeOdds.length).toFixed(2)),
          away: parseFloat((awayOdds.reduce((a, b) => a + b, 0) / awayOdds.length).toFixed(2))
        };
      }
    }
    
    return null;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="line-movement-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-brand-primary" />
            Line Movement
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Track odds changes from opening to current — updated every 5 minutes via ESPN
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* Auto-refresh toggle */}
          <div className="flex items-center gap-2">
            <span className="text-text-muted text-sm">Auto-refresh</span>
            <button
              onClick={() => setIsAutoRefresh(!isAutoRefresh)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                isAutoRefresh ? 'bg-brand-primary' : 'bg-zinc-700'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  isAutoRefresh ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          {lastUpdate && (
            <span className="text-text-muted text-xs">
              Updated: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
          <button 
            onClick={() => {
              fetchEvents();
              if (selectedEvent) fetchLineMovement(selectedEvent.id);
            }}
            className="btn-outline flex items-center gap-2"
            data-testid="refresh-events-btn"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Sport Selector */}
      <div className="flex flex-wrap gap-2" data-testid="sport-selector">
        {sports.map(sport => (
          <button
            key={sport.key}
            onClick={() => {
              setSelectedSport(sport.key);
              setSelectedEvent(null);
              setLineData(null);
            }}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              selectedSport === sport.key
                ? 'bg-brand-primary text-zinc-950'
                : 'bg-zinc-800 text-text-secondary hover:bg-zinc-700'
            }`}
            data-testid={`sport-btn-${sport.key}`}
          >
            {sport.label}
          </button>
        ))}
      </div>

      {/* Event Selector */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4">Select Event</h2>
        <div className="grid gap-2 max-h-48 overflow-y-auto" data-testid="event-list">
          {events.length === 0 ? (
            <div className="flex items-center gap-2 text-text-muted p-4">
              <AlertCircle className="w-5 h-5" />
              <span>No upcoming events found for this sport</span>
            </div>
          ) : (
            events.map(event => (
              <button
                key={event.id}
                onClick={() => handleEventSelect(event)}
                className={`p-3 rounded-lg text-left transition-all ${
                  selectedEvent?.id === event.id
                    ? 'bg-brand-primary/20 border border-brand-primary'
                    : 'bg-zinc-800 hover:bg-zinc-700 border border-transparent'
                }`}
                data-testid={`event-btn-${event.id}`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-text-primary">
                    {event.home_team} vs {event.away_team}
                  </span>
                  <span className="text-text-muted text-xs">
                    {event.commence_time ? format(parseISO(event.commence_time), 'MMM d, HH:mm') : '—'}
                  </span>
                </div>
                {event.bookmakers?.length > 0 && (
                  <div className="text-xs text-text-muted mt-1">
                    {event.bookmakers.length} bookmaker{event.bookmakers.length > 1 ? 's' : ''}
                  </div>
                )}
              </button>
            ))
          )}
        </div>
      </div>

      {/* Line Movement Content */}
      {selectedEvent && (
        <>
          <div className="stat-card" data-testid="line-chart-card">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <h2 className="font-mono font-bold text-lg text-text-primary">
                  {selectedEvent.home_team} vs {selectedEvent.away_team}
                </h2>
                {isAutoRefresh && (
                  <span className="flex items-center gap-1 text-xs text-semantic-success">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-semantic-success opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-semantic-success"></span>
                    </span>
                    Live
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3">
                {lineData?.total_snapshots > 0 && (
                  <span className="text-text-muted text-sm">
                    {lineData.total_snapshots} snapshot{lineData.total_snapshots > 1 ? 's' : ''}
                  </span>
                )}
                <span className="text-xs text-brand-primary bg-brand-primary/10 px-2 py-1 rounded">
                  Source: ESPN
                </span>
              </div>
            </div>
            
            {loadingLine ? (
              <div className="flex items-center justify-center h-64">
                <RefreshCw className="w-6 h-6 text-brand-primary animate-spin" />
              </div>
            ) : (
              <MovementChart 
                chartData={processChartData()} 
                homeTeam={selectedEvent.home_team}
                awayTeam={selectedEvent.away_team}
                openingOdds={lineData?.opening_odds}
              />
            )}
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <OddsComparison 
              opening={lineData?.opening_odds}
              current={getCurrentOdds()}
              homeTeam={selectedEvent.home_team}
              awayTeam={selectedEvent.away_team}
            />
            <BookmakerBreakdown 
              bookmakers={lineData?.bookmakers} 
              currentBookmakers={selectedEvent?.bookmakers}
            />
          </div>
        </>
      )}

      {!selectedEvent && events.length > 0 && (
        <div className="stat-card text-center py-12">
          <TrendingUp className="w-16 h-16 text-text-muted mx-auto mb-4" />
          <h3 className="text-text-primary font-bold text-lg mb-2">Select an Event</h3>
          <p className="text-text-muted">
            Choose an event above to view line movement data
          </p>
        </div>
      )}
    </div>
  );
};

export default LineMovement;
