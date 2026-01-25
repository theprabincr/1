import { useState, useEffect } from "react";
import axios from "axios";
import { 
  TrendingUp, TrendingDown, RefreshCw, Clock,
  AlertCircle, ChevronDown, Activity
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { format, parseISO } from "date-fns";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Active sportsbooks only
const SPORTSBOOK_COLORS = {
  'draftkings': '#61B510',
  'fanduel': '#1493FF',
  'betmgm': '#BFA165',
  'pinnacle': '#D22630',
  'unibet': '#147B45',
  'betway': '#00A826',
  'betonlineag': '#FF6600'
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

// Custom Tooltip
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 shadow-lg">
        <p className="text-xs text-text-muted mb-2">{label}</p>
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center gap-2 text-sm">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-text-secondary">{entry.name}:</span>
            <span className={`font-mono font-bold ${
              entry.value > 0 ? 'text-semantic-success' : 'text-text-primary'
            }`}>
              {entry.value > 0 ? '+' : ''}{entry.value}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

// Line Movement Chart
const MovementChart = ({ data, selectedBooks }) => {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-zinc-900 rounded-lg">
        <p className="text-text-muted">No line movement data available</p>
      </div>
    );
  }

  // Process data for chart
  const chartData = processChartData(data, selectedBooks);

  return (
    <div className="bg-zinc-900 rounded-lg p-4" data-testid="line-movement-chart">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <XAxis 
            dataKey="time" 
            stroke="#71717A"
            fontSize={12}
            tickLine={false}
            axisLine={false}
          />
          <YAxis 
            stroke="#71717A"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => value > 0 ? `+${value}` : value}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ paddingTop: '10px' }}
            formatter={(value) => <span className="text-text-secondary text-sm">{value}</span>}
          />
          {selectedBooks.map((book) => (
            <Line
              key={book}
              type="monotone"
              dataKey={book}
              name={SPORTSBOOK_NAMES[book] || book}
              stroke={SPORTSBOOK_COLORS[book] || '#CCFF00'}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: SPORTSBOOK_COLORS[book] || '#CCFF00' }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// Process chart data
const processChartData = (data, selectedBooks) => {
  // Group by timestamp
  const timeGroups = {};
  
  data.forEach(item => {
    const time = item.timestamp ? format(parseISO(item.timestamp), 'HH:mm') : 'N/A';
    if (!timeGroups[time]) {
      timeGroups[time] = { time };
    }
    
    if (selectedBooks.includes(item.bookmaker)) {
      const homeOutcome = item.outcomes?.find(o => o.name === 'Home') || item.outcomes?.[0];
      if (homeOutcome) {
        timeGroups[time][item.bookmaker] = homeOutcome.price;
      }
    }
  });

  return Object.values(timeGroups).sort((a, b) => a.time.localeCompare(b.time));
};

// Movement Summary Card
const MovementSummary = ({ data }) => {
  if (!data || data.length === 0) return null;

  // Calculate movement summary
  const bookmakers = [...new Set(data.map(d => d.bookmaker))];
  const summaries = bookmakers.map(bm => {
    const bmData = data.filter(d => d.bookmaker === bm).sort((a, b) => 
      new Date(a.timestamp) - new Date(b.timestamp)
    );
    
    if (bmData.length < 2) return null;
    
    const first = bmData[0].outcomes?.[0]?.price || 0;
    const last = bmData[bmData.length - 1].outcomes?.[0]?.price || 0;
    const change = last - first;
    
    return {
      bookmaker: bm,
      name: SPORTSBOOK_NAMES[bm] || bm,
      opening: first,
      current: last,
      change,
      direction: change > 0 ? 'up' : change < 0 ? 'down' : 'flat'
    };
  }).filter(Boolean);

  const maxMove = Math.max(...summaries.map(s => Math.abs(s.change)));
  const sharpestMove = summaries.find(s => Math.abs(s.change) === maxMove);

  return (
    <div className="stat-card" data-testid="movement-summary">
      <h3 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5 text-brand-primary" />
        Movement Summary
      </h3>
      
      {sharpestMove && (
        <div className="p-3 bg-zinc-800 rounded-lg mb-4">
          <p className="text-text-muted text-sm mb-1">Sharpest Movement</p>
          <div className="flex items-center gap-2">
            {sharpestMove.direction === 'up' ? (
              <TrendingUp className="w-5 h-5 text-semantic-success" />
            ) : (
              <TrendingDown className="w-5 h-5 text-semantic-danger" />
            )}
            <span className="text-text-primary font-semibold">{sharpestMove.name}</span>
            <span className={`font-mono font-bold ${
              sharpestMove.change > 0 ? 'text-semantic-success' : 'text-semantic-danger'
            }`}>
              {sharpestMove.change > 0 ? '+' : ''}{sharpestMove.change}
            </span>
          </div>
        </div>
      )}

      <div className="space-y-2">
        {summaries.slice(0, 5).map((summary) => (
          <div key={summary.bookmaker} className="flex items-center justify-between py-2 border-b border-zinc-800">
            <span className="text-text-secondary text-sm">{summary.name}</span>
            <div className="flex items-center gap-4">
              <span className="font-mono text-sm text-text-muted">
                {summary.opening > 0 ? '+' : ''}{summary.opening}
              </span>
              <span className="text-text-muted">→</span>
              <span className={`font-mono text-sm font-bold ${
                summary.current > 0 ? 'text-semantic-success' : 'text-text-primary'
              }`}>
                {summary.current > 0 ? '+' : ''}{summary.current}
              </span>
              <span className={`font-mono text-xs px-2 py-1 rounded ${
                summary.change > 0 ? 'bg-semantic-success/20 text-semantic-success' :
                summary.change < 0 ? 'bg-semantic-danger/20 text-semantic-danger' :
                'bg-zinc-800 text-text-muted'
              }`}>
                {summary.change > 0 ? '+' : ''}{summary.change}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const LineMovement = () => {
  const [loading, setLoading] = useState(true);
  const [events, setEvents] = useState([]);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [lineMovement, setLineMovement] = useState([]);
  const [selectedSport, setSelectedSport] = useState("basketball_nba");
  const [selectedBooks, setSelectedBooks] = useState(['draftkings', 'fanduel', 'pinnacle', 'betmgm']);

  const sports = [
    { key: "basketball_nba", label: "NBA" },
    { key: "americanfootball_nfl", label: "NFL" },
    { key: "baseball_mlb", label: "MLB" },
    { key: "icehockey_nhl", label: "NHL" },
    { key: "soccer_epl", label: "EPL" },
  ];

  const allBooks = Object.keys(SPORTSBOOK_NAMES);

  const fetchEvents = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/events/${selectedSport}`);
      setEvents(response.data);
      if (response.data.length > 0) {
        setSelectedEvent(response.data[0]);
      }
    } catch (error) {
      console.error("Error fetching events:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchLineMovement = async (eventId) => {
    try {
      const response = await axios.get(`${API}/line-movement/${eventId}`);
      setLineMovement(response.data);
    } catch (error) {
      console.error("Error fetching line movement:", error);
      setLineMovement([]);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, [selectedSport]);

  useEffect(() => {
    if (selectedEvent?.id) {
      fetchLineMovement(selectedEvent.id);
    }
  }, [selectedEvent]);

  const toggleBook = (book) => {
    setSelectedBooks(prev => 
      prev.includes(book) 
        ? prev.filter(b => b !== book)
        : [...prev, book]
    );
  };

  return (
    <div className="space-y-6" data-testid="line-movement-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary">Line Movement</h1>
          <p className="text-text-muted text-sm mt-1">Track odds changes over time</p>
        </div>
        <button 
          onClick={fetchEvents}
          className="btn-outline flex items-center gap-2"
          data-testid="refresh-line-movement-btn"
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
            data-testid={`line-sport-${sport.key}`}
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
          data-testid="event-selector"
        >
          {events.map(event => (
            <option key={event.id} value={event.id}>
              {event.home_team} vs {event.away_team}
            </option>
          ))}
        </select>
      </div>

      {/* Sportsbook Filter */}
      <div className="stat-card">
        <label className="text-text-muted text-sm mb-3 block">Filter Sportsbooks</label>
        <div className="flex flex-wrap gap-2">
          {allBooks.map(book => (
            <button
              key={book}
              onClick={() => toggleBook(book)}
              className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                selectedBooks.includes(book)
                  ? "text-zinc-950 font-bold"
                  : "bg-zinc-800 text-text-muted hover:bg-zinc-700"
              }`}
              style={{
                backgroundColor: selectedBooks.includes(book) 
                  ? SPORTSBOOK_COLORS[book] 
                  : undefined
              }}
              data-testid={`filter-book-${book}`}
            >
              {SPORTSBOOK_NAMES[book]}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
        </div>
      ) : selectedEvent ? (
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="stat-card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-mono font-bold text-lg text-text-primary">
                  {selectedEvent.home_team} vs {selectedEvent.away_team}
                </h3>
                <span className="text-text-muted text-sm flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  Last 24h
                </span>
              </div>
              <MovementChart data={lineMovement} selectedBooks={selectedBooks} />
            </div>
          </div>
          
          <div>
            <MovementSummary data={lineMovement} />
          </div>
        </div>
      ) : (
        <div className="stat-card text-center py-12">
          <AlertCircle className="w-12 h-12 text-text-muted mx-auto mb-4" />
          <h3 className="text-text-primary font-semibold mb-2">No Events Selected</h3>
          <p className="text-text-muted text-sm">Select a sport and event to view line movement</p>
        </div>
      )}
    </div>
  );
};

export default LineMovement;
