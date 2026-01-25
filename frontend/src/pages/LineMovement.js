import { useState, useEffect } from "react";
import axios from "axios";
import { 
  TrendingUp, TrendingDown, RefreshCw, Clock,
  AlertCircle, ChevronDown, Activity, Calendar
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid } from "recharts";
import { format, parseISO } from "date-fns";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Bookmaker colors for chart
const BOOKMAKER_COLORS = [
  '#61B510', '#1493FF', '#BFA165', '#D22630', '#147B45', 
  '#00A826', '#FF6600', '#9945FF', '#00CED1', '#FF69B4'
];

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

// Line Movement Chart Component
const MovementChart = ({ chartData, homeTeam, awayTeam }) => {
  if (!chartData || chartData.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 bg-zinc-900 rounded-lg">
        <Clock className="w-12 h-12 text-text-muted mb-3" />
        <p className="text-text-muted">No line movement data yet</p>
        <p className="text-text-muted text-sm mt-1">Data is collected hourly</p>
      </div>
    );
  }

  return (
    <div className="bg-zinc-900 rounded-lg p-4">
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={chartData}>
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
            domain={['auto', 'auto']}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ paddingTop: '10px' }}
            formatter={(value) => <span className="text-text-secondary text-sm">{value}</span>}
          />
          <Line
            type="monotone"
            dataKey="home_odds"
            name={homeTeam || "Home"}
            stroke="#ADFF2F"
            strokeWidth={2}
            dot={{ fill: '#ADFF2F', r: 3 }}
            activeDot={{ r: 5, fill: '#ADFF2F' }}
          />
          <Line
            type="monotone"
            dataKey="away_odds"
            name={awayTeam || "Away"}
            stroke="#00CED1"
            strokeWidth={2}
            dot={{ fill: '#00CED1', r: 3 }}
            activeDot={{ r: 5, fill: '#00CED1' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// Opening Odds Card
const OpeningOddsCard = ({ opening, homeTeam, awayTeam }) => {
  if (!opening) {
    return (
      <div className="stat-card">
        <h3 className="font-mono font-bold text-text-primary mb-3 flex items-center gap-2">
          <Calendar className="w-5 h-5 text-brand-secondary" />
          Opening Odds
        </h3>
        <p className="text-text-muted text-sm">Opening odds not yet captured</p>
      </div>
    );
  }

  return (
    <div className="stat-card">
      <h3 className="font-mono font-bold text-text-primary mb-3 flex items-center gap-2">
        <Calendar className="w-5 h-5 text-brand-secondary" />
        Opening Odds
      </h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-zinc-800 p-3 rounded-lg">
          <p className="text-text-muted text-xs mb-1">{homeTeam || "Home"}</p>
          <p className="font-mono text-2xl font-bold text-brand-primary">
            {opening.home_odds?.toFixed(2) || "—"}
          </p>
        </div>
        <div className="bg-zinc-800 p-3 rounded-lg">
          <p className="text-text-muted text-xs mb-1">{awayTeam || "Away"}</p>
          <p className="font-mono text-2xl font-bold text-brand-secondary">
            {opening.away_odds?.toFixed(2) || "—"}
          </p>
        </div>
      </div>
      {opening.timestamp && (
        <p className="text-text-muted text-xs mt-3">
          Captured: {format(parseISO(opening.timestamp), 'MMM d, yyyy HH:mm')}
        </p>
      )}
    </div>
  );
};

// Bookmaker Breakdown Card
const BookmakerBreakdown = ({ bookmakers }) => {
  if (!bookmakers || bookmakers.length === 0) {
    return null;
  }

  return (
    <div className="stat-card">
      <h3 className="font-mono font-bold text-text-primary mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5 text-brand-primary" />
        By Bookmaker ({bookmakers.length})
      </h3>
      <div className="space-y-3 max-h-64 overflow-y-auto">
        {bookmakers.map((bm, i) => {
          const latestSnap = bm.snapshots?.[bm.snapshots.length - 1];
          const firstSnap = bm.snapshots?.[0];
          
          const homeChange = latestSnap && firstSnap 
            ? (latestSnap.home_odds - firstSnap.home_odds).toFixed(2)
            : null;
          
          return (
            <div key={bm.bookmaker || i} className="bg-zinc-800 p-3 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-text-primary font-medium">
                  {bm.bookmaker_title || bm.bookmaker}
                </span>
                <span className="text-text-muted text-xs">
                  {bm.snapshots?.length || 0} snapshots
                </span>
              </div>
              {latestSnap && (
                <div className="flex items-center justify-between text-sm">
                  <div className="flex gap-4">
                    <span className="text-brand-primary font-mono">
                      H: {latestSnap.home_odds?.toFixed(2)}
                    </span>
                    <span className="text-brand-secondary font-mono">
                      A: {latestSnap.away_odds?.toFixed(2)}
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

  const sports = [
    { key: "basketball_nba", label: "NBA" },
    { key: "americanfootball_nfl", label: "NFL" },
    { key: "baseball_mlb", label: "MLB" },
    { key: "icehockey_nhl", label: "NHL" },
    { key: "soccer_epl", label: "EPL" },
  ];

  useEffect(() => {
    fetchEvents();
  }, [selectedSport]);

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

  const fetchLineMovement = async (eventId) => {
    setLoadingLine(true);
    try {
      const response = await axios.get(`${API}/line-movement/${eventId}?sport_key=${selectedSport}`);
      setLineData(response.data);
    } catch (error) {
      console.error("Error fetching line movement:", error);
      setLineData(null);
    } finally {
      setLoadingLine(false);
    }
  };

  const handleEventSelect = (event) => {
    setSelectedEvent(event);
    fetchLineMovement(event.id);
  };

  // Process chart data for the chart
  const processChartData = () => {
    if (!lineData?.chart_data) return [];
    
    return lineData.chart_data.map(item => ({
      time: item.timestamp ? format(parseISO(item.timestamp), 'MM/dd HH:mm') : 'N/A',
      home_odds: item.home_odds,
      away_odds: item.away_odds
    }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-brand-primary" />
            Line Movement
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Track odds changes from opening to current — updated hourly
          </p>
        </div>
        <button 
          onClick={fetchEvents}
          className="btn-outline flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Sport Selector */}
      <div className="flex flex-wrap gap-2">
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
          >
            {sport.label}
          </button>
        ))}
      </div>

      {/* Event Selector */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4">Select Event</h2>
        <div className="grid gap-2 max-h-48 overflow-y-auto">
          {events.length === 0 ? (
            <p className="text-text-muted">No upcoming events found</p>
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
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-text-primary">
                    {event.home_team} vs {event.away_team}
                  </span>
                  <span className="text-text-muted text-xs">
                    {event.commence_time ? format(parseISO(event.commence_time), 'MMM d, HH:mm') : '—'}
                  </span>
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Line Movement Content */}
      {selectedEvent && (
        <>
          <div className="stat-card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-mono font-bold text-lg text-text-primary">
                {selectedEvent.home_team} vs {selectedEvent.away_team}
              </h2>
              {lineData?.total_snapshots > 0 && (
                <span className="text-text-muted text-sm">
                  {lineData.total_snapshots} data points
                </span>
              )}
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
              />
            )}
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <OpeningOddsCard 
              opening={lineData?.opening_odds} 
              homeTeam={selectedEvent.home_team}
              awayTeam={selectedEvent.away_team}
            />
            <BookmakerBreakdown bookmakers={lineData?.bookmakers} />
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
