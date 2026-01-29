import { useState, useEffect } from "react";
import axios from "axios";
import { 
  Trophy, RefreshCw, TrendingUp, TrendingDown,
  CheckCircle, XCircle, DollarSign, Target,
  BarChart3, Calendar, Clock, Zap
} from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine } from "recharts";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Stat Card
const StatCard = ({ title, value, subtitle, icon: Icon, color, trend }) => {
  const colorClasses = {
    lime: "text-brand-primary bg-brand-primary/10",
    green: "text-semantic-success bg-semantic-success/10",
    red: "text-semantic-danger bg-semantic-danger/10",
    blue: "text-brand-secondary bg-brand-secondary/10",
    yellow: "text-semantic-warning bg-semantic-warning/10"
  };

  return (
    <div className="stat-card" data-testid={`stat-${title.toLowerCase().replace(/\s+/g, '-')}`}>
      <div className="flex items-start justify-between mb-3">
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
        {trend !== undefined && trend !== 0 && (
          <span className={`flex items-center gap-1 text-sm font-mono ${
            trend >= 0 ? "text-semantic-success" : "text-semantic-danger"
          }`}>
            {trend >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
            {trend >= 0 ? '+' : ''}{trend}%
          </span>
        )}
      </div>
      <h3 className="text-text-muted text-sm mb-1">{title}</h3>
      <p className="font-mono font-bold text-2xl text-text-primary">{value}</p>
      {subtitle && <p className="text-text-muted text-xs mt-1">{subtitle}</p>}
    </div>
  );
};

// Prediction Row Component
const PredictionRow = ({ prediction }) => {
  const statusConfig = {
    pending: { color: "text-semantic-warning bg-semantic-warning/10", icon: Clock, label: "PENDING" },
    win: { color: "text-semantic-success bg-semantic-success/10", icon: CheckCircle, label: "WON" },
    loss: { color: "text-semantic-danger bg-semantic-danger/10", icon: XCircle, label: "LOST" },
    push: { color: "text-text-muted bg-text-muted/10", icon: Target, label: "PUSH" }
  };

  const status = statusConfig[prediction.result] || statusConfig.pending;
  const StatusIcon = status.icon;
  
  // Calculate profit/loss for completed predictions
  const odds = prediction.odds_at_prediction || 1.91;
  const unitStake = 100; // Assume $100 unit for display
  let profit = 0;
  if (prediction.result === 'win') {
    profit = unitStake * (odds - 1);
  } else if (prediction.result === 'loss') {
    profit = -unitStake;
  }

  const formatDate = (dateStr) => {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="p-4 bg-zinc-800 rounded-lg hover:bg-zinc-750 transition-colors" data-testid={`prediction-${prediction.event_id}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono text-text-muted uppercase">
              {prediction.sport_key?.replace(/_/g, ' ')}
            </span>
            <span className="text-xs text-text-muted">•</span>
            <span className="text-xs text-text-muted">
              {formatDate(prediction.commence_time || prediction.created_at)}
            </span>
          </div>
          <p className="text-text-primary font-semibold">
            {prediction.home_team} vs {prediction.away_team}
          </p>
          <div className="flex items-center gap-2 mt-2">
            <span className="text-brand-primary font-mono font-bold">
              {prediction.predicted_outcome}
            </span>
            <span className="text-text-muted">@</span>
            <span className="font-mono text-text-primary">{odds.toFixed(2)}</span>
          </div>
          {prediction.confidence && (
            <span className={`inline-block mt-2 px-2 py-0.5 rounded text-xs font-bold ${
              prediction.confidence >= 0.7 ? 'bg-semantic-success/20 text-semantic-success' :
              prediction.confidence >= 0.5 ? 'bg-semantic-warning/20 text-semantic-warning' :
              'bg-text-muted/20 text-text-muted'
            }`}>
              {(prediction.confidence * 100).toFixed(0)}% Confidence
            </span>
          )}
        </div>
        <div className="text-right flex flex-col items-end gap-2">
          <span className={`px-2 py-1 rounded text-xs font-bold flex items-center gap-1 ${status.color}`}>
            <StatusIcon className="w-3 h-3" />
            {status.label}
          </span>
          {prediction.result !== 'pending' && prediction.result !== 'push' && (
            <p className={`text-sm font-mono ${
              profit >= 0 ? 'text-semantic-success' : 'text-semantic-danger'
            }`}>
              {profit >= 0 ? '+' : ''}${profit.toFixed(2)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

// Profit History Chart with positive/negative bars
const ProfitHistoryChart = ({ predictions }) => {
  // Group predictions by date and calculate daily profit
  const dateGroups = {};
  
  predictions.forEach(p => {
    if (p.result === 'pending') return; // Skip pending
    
    const dateStr = p.commence_time || p.created_at;
    if (!dateStr) return;
    
    const date = new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    if (!dateGroups[date]) {
      dateGroups[date] = { date, profit: 0, wins: 0, losses: 0 };
    }
    
    const odds = p.odds_at_prediction || 1.91;
    const unitStake = 100;
    
    if (p.result === 'win') {
      dateGroups[date].profit += unitStake * (odds - 1);
      dateGroups[date].wins++;
    } else if (p.result === 'loss') {
      dateGroups[date].profit -= unitStake;
      dateGroups[date].losses++;
    }
  });

  // Get last 7 days of data
  const data = Object.values(dateGroups).slice(-7);

  if (data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 text-text-muted">
        <BarChart3 className="w-12 h-12 mb-2 opacity-30" />
        <p>No completed predictions yet</p>
        <p className="text-xs mt-1">Results will appear here after games finish</p>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 20, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272A" vertical={false} />
        <XAxis 
          dataKey="date" 
          stroke="#71717A" 
          style={{ fontSize: '11px' }}
          axisLine={{ stroke: '#27272A' }}
          tickLine={false}
        />
        <YAxis 
          stroke="#71717A" 
          style={{ fontSize: '11px' }}
          axisLine={{ stroke: '#27272A' }}
          tickLine={false}
          tickFormatter={(value) => `$${value}`}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#18181B', 
            border: '1px solid #27272A',
            borderRadius: '8px',
            padding: '10px'
          }}
          formatter={(value, name) => [`$${value.toFixed(2)}`, 'Profit/Loss']}
          labelStyle={{ color: '#A1A1AA', marginBottom: '5px' }}
        />
        <ReferenceLine y={0} stroke="#71717A" strokeDasharray="3 3" />
        <Bar dataKey="profit" radius={[4, 4, 0, 0]}>
          {data.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={entry.profit >= 0 ? '#22C55E' : '#EF4444'} 
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

const Performance = () => {
  const [performance, setPerformance] = useState(null);
  const [allPredictions, setAllPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // all, pending, completed

  useEffect(() => {
    fetchPerformanceData();
  }, []);

  const fetchPerformanceData = async () => {
    try {
      // Fetch performance stats and all predictions
      const [perfRes, recsRes] = await Promise.all([
        axios.get(`${API}/performance`),
        axios.get(`${API}/recommendations?limit=100&min_confidence=0`) // Get all predictions
      ]);
      
      setPerformance(perfRes.data);
      
      // Combine recent_predictions from performance with active recommendations
      const completedPredictions = perfRes.data.recent_predictions || [];
      const activePredictions = recsRes.data || [];
      
      // Merge and dedupe by event_id
      const allPreds = [...completedPredictions];
      activePredictions.forEach(p => {
        if (!allPreds.find(cp => cp.event_id === p.event_id)) {
          allPreds.push({ ...p, result: 'pending' });
        }
      });
      
      // Sort by date (newest first)
      allPreds.sort((a, b) => {
        const dateA = new Date(a.commence_time || a.created_at || 0);
        const dateB = new Date(b.commence_time || b.created_at || 0);
        return dateB - dateA;
      });
      
      setAllPredictions(allPreds);
    } catch (error) {
      console.error("Error fetching performance data:", error);
    } finally {
      setLoading(false);
    }
  };

  // Filter predictions
  const filteredPredictions = allPredictions.filter(p => {
    if (filter === 'pending') return p.result === 'pending';
    if (filter === 'completed') return p.result !== 'pending';
    return true;
  });

  // Calculate stats
  const totalPicks = allPredictions.length;
  const pendingCount = allPredictions.filter(p => p.result === 'pending').length;
  const completedPredictions = allPredictions.filter(p => p.result !== 'pending');
  
  const wins = performance?.wins || 0;
  const losses = performance?.losses || 0;
  const winRate = performance?.win_rate || 0;
  const roi = performance?.roi || 0;
  
  // Calculate total profit and avg odds
  const totalProfit = completedPredictions.reduce((sum, p) => {
    const odds = p.odds_at_prediction || 1.91;
    const unitStake = 100;
    if (p.result === 'win') return sum + unitStake * (odds - 1);
    if (p.result === 'loss') return sum - unitStake;
    return sum;
  }, 0);
  
  const avgOdds = completedPredictions.length > 0
    ? (completedPredictions.reduce((sum, p) => sum + (p.odds_at_prediction || 1.91), 0) / completedPredictions.length).toFixed(2)
    : '0.00';

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64" data-testid="loading">
        <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="performance-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary flex items-center gap-2">
            <Trophy className="w-6 h-6 text-brand-primary" />
            Algorithm Performance
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Track all picks made by the prediction algorithm
          </p>
        </div>
        <button 
          onClick={fetchPerformanceData}
          className="btn-outline flex items-center gap-2"
          data-testid="refresh-btn"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          title="Total Picks"
          value={totalPicks}
          subtitle={`${pendingCount} pending`}
          icon={Zap}
          color="lime"
        />
        <StatCard
          title="Win Rate"
          value={`${winRate}%`}
          subtitle={`${wins}W - ${losses}L`}
          icon={Target}
          color={winRate >= 55 ? "green" : winRate >= 45 ? "yellow" : "red"}
        />
        <StatCard
          title="Total Profit"
          value={`$${totalProfit.toFixed(2)}`}
          subtitle={`ROI: ${roi}%`}
          icon={DollarSign}
          color={totalProfit >= 0 ? "green" : "red"}
          trend={parseFloat(roi)}
        />
        <StatCard
          title="Avg Odds"
          value={avgOdds}
          subtitle={`${completedPredictions.length} completed`}
          icon={TrendingUp}
          color="blue"
        />
      </div>

      {/* Profit History Chart */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
          <Calendar className="w-5 h-5 text-brand-primary" />
          Profit History (Last 7 Days)
        </h2>
        <ProfitHistoryChart predictions={allPredictions} />
      </div>

      {/* Predictions List */}
      <div className="stat-card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-mono font-bold text-lg text-text-primary">
            All Picks
          </h2>
          <div className="flex gap-2">
            {['all', 'pending', 'completed'].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors ${
                  filter === f 
                    ? 'bg-brand-primary text-zinc-950 font-bold' 
                    : 'bg-zinc-800 text-text-secondary hover:bg-zinc-700'
                }`}
                data-testid={`filter-${f}`}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
                {f === 'pending' && pendingCount > 0 && (
                  <span className="ml-1">({pendingCount})</span>
                )}
              </button>
            ))}
          </div>
        </div>
        
        <div className="space-y-3">
          {filteredPredictions.length === 0 ? (
            <div className="text-center py-8 text-text-muted">
              <Zap className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p>No picks yet</p>
              <p className="text-xs mt-1">
                {filter === 'pending' 
                  ? 'No pending predictions' 
                  : filter === 'completed'
                  ? 'No completed predictions yet'
                  : 'The algorithm will generate picks automatically'}
              </p>
            </div>
          ) : (
            filteredPredictions.slice(0, 20).map((prediction, i) => (
              <PredictionRow key={prediction.event_id || i} prediction={prediction} />
            ))
          )}
        </div>
        
        {filteredPredictions.length > 20 && (
          <p className="text-text-muted text-sm text-center mt-4">
            Showing 20 of {filteredPredictions.length} picks
          </p>
        )}
      </div>

      {/* Performance by Sport */}
      {performance?.by_sport && Object.keys(performance.by_sport).length > 0 && (
        <div className="stat-card">
          <h2 className="font-mono font-bold text-lg text-text-primary mb-4">
            Performance by Sport
          </h2>
          <div className="overflow-x-auto">
            <table className="data-table w-full">
              <thead>
                <tr>
                  <th className="text-left">Sport</th>
                  <th className="text-center">Wins</th>
                  <th className="text-center">Losses</th>
                  <th className="text-center">Pushes</th>
                  <th className="text-right">Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(performance.by_sport).map(([sport, stats]) => {
                  const sportWinRate = stats.total > 0 
                    ? ((stats.wins / stats.total) * 100).toFixed(1) 
                    : 0;
                  return (
                    <tr key={sport}>
                      <td className="text-text-primary font-medium">
                        {sport.replace(/_/g, ' ').toUpperCase()}
                      </td>
                      <td className="text-center font-mono text-semantic-success">{stats.wins}</td>
                      <td className="text-center font-mono text-semantic-danger">{stats.losses}</td>
                      <td className="text-center font-mono text-text-muted">{stats.pushes}</td>
                      <td className={`text-right font-mono font-bold ${
                        sportWinRate >= 55 ? "text-semantic-success" : 
                        sportWinRate >= 45 ? "text-semantic-warning" : "text-semantic-danger"
                      }`}>
                        {sportWinRate}%
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

export default Performance;
