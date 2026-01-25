import { useState, useEffect } from "react";
import axios from "axios";
import { 
  Trophy, RefreshCw, TrendingUp, TrendingDown,
  CheckCircle, XCircle, Clock, DollarSign, Target,
  BarChart3, PieChart
} from "lucide-react";
import { PieChart as RePieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";

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
    <div className="stat-card" data-testid={`perf-stat-${title.toLowerCase().replace(/\s+/g, '-')}`}>
      <div className="flex items-start justify-between mb-3">
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
        {trend !== undefined && (
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

// Win/Loss Chart
const WinLossChart = ({ wins, losses, pushes }) => {
  const data = [
    { name: 'Wins', value: wins, color: '#22C55E' },
    { name: 'Losses', value: losses, color: '#EF4444' },
    { name: 'Pushes', value: pushes, color: '#71717A' }
  ].filter(d => d.value > 0);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-text-muted">
        No data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <RePieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={50}
          outerRadius={80}
          paddingAngle={2}
          dataKey="value"
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#18181B', 
            border: '1px solid #27272A',
            borderRadius: '8px'
          }}
        />
        <Legend 
          formatter={(value) => <span className="text-text-secondary text-sm">{value}</span>}
        />
      </RePieChart>
    </ResponsiveContainer>
  );
};

// Recent Prediction Row
const PredictionRow = ({ prediction }) => {
  const statusColors = {
    pending: "text-semantic-warning",
    win: "text-semantic-success",
    loss: "text-semantic-danger",
    push: "text-text-muted"
  };

  const statusIcons = {
    pending: Clock,
    win: CheckCircle,
    loss: XCircle,
    push: Target
  };

  const StatusIcon = statusIcons[prediction.result] || Clock;

  return (
    <tr className="border-b border-zinc-800 hover:bg-zinc-800/30">
      <td className="py-3 px-4">
        <div>
          <p className="text-text-primary text-sm font-medium">{prediction.home_team}</p>
          <p className="text-text-muted text-xs">vs {prediction.away_team}</p>
        </div>
      </td>
      <td className="py-3 px-4">
        <span className="text-xs font-mono text-text-muted uppercase">
          {prediction.sport_key?.replace(/_/g, ' ').slice(0, 15)}
        </span>
      </td>
      <td className="py-3 px-4">
        <span className="font-mono text-brand-primary text-sm">
          {prediction.predicted_outcome?.slice(0, 12)}
        </span>
      </td>
      <td className="py-3 px-4">
        <span className={`font-mono text-sm ${
          prediction.odds_at_prediction > 0 ? 'text-semantic-success' : 'text-text-primary'
        }`}>
          {typeof prediction.odds_at_prediction === 'number' ? prediction.odds_at_prediction.toFixed(2) : prediction.odds_at_prediction}
        </span>
      </td>
      <td className="py-3 px-4">
        <span className={`flex items-center gap-1 ${statusColors[prediction.result]}`}>
          <StatusIcon className="w-4 h-4" />
          <span className="text-sm capitalize">{prediction.result}</span>
        </span>
      </td>
    </tr>
  );
};

// Sport Performance Card
const SportPerformanceCard = ({ sport, stats }) => {
  const winRate = stats.total > 0 ? ((stats.wins / stats.total) * 100).toFixed(1) : 0;
  
  return (
    <div className="p-4 bg-zinc-800/50 rounded-lg">
      <div className="flex items-center justify-between mb-2">
        <span className="text-text-primary font-medium capitalize">
          {sport.replace(/_/g, ' ')}
        </span>
        <span className={`font-mono font-bold ${
          winRate >= 55 ? 'text-semantic-success' :
          winRate >= 45 ? 'text-semantic-warning' : 'text-semantic-danger'
        }`}>
          {winRate}%
        </span>
      </div>
      <div className="flex gap-4 text-sm">
        <span className="text-semantic-success">{stats.wins}W</span>
        <span className="text-semantic-danger">{stats.losses}L</span>
        {stats.pushes > 0 && <span className="text-text-muted">{stats.pushes}P</span>}
      </div>
      {/* Progress bar */}
      <div className="mt-2 h-1.5 bg-zinc-700 rounded-full overflow-hidden">
        <div 
          className="h-full bg-gradient-to-r from-semantic-success to-brand-primary"
          style={{ width: `${Math.min(winRate, 100)}%` }}
        />
      </div>
    </div>
  );
};

const Performance = () => {
  const [loading, setLoading] = useState(true);
  const [performance, setPerformance] = useState(null);

  const fetchPerformance = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/performance`);
      setPerformance(response.data);
    } catch (error) {
      console.error("Error fetching performance:", error);
    } finally {
      setLoading(false);
    }
  };

  const checkResults = async () => {
    try {
      await axios.post(`${API}/check-results`);
      // Refresh performance after checking results
      setTimeout(fetchPerformance, 2000);
    } catch (error) {
      console.error("Error checking results:", error);
    }
  };

  useEffect(() => {
    fetchPerformance();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="performance-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary">Performance</h1>
          <p className="text-text-muted text-sm mt-1">Track your prediction history</p>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={checkResults}
            className="btn-primary flex items-center gap-2"
            data-testid="check-results-btn"
          >
            <CheckCircle className="w-4 h-4" />
            Check Results
          </button>
          <button 
            onClick={fetchPerformance}
            className="btn-outline flex items-center gap-2"
            data-testid="refresh-performance-btn"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Main Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Win Rate"
          value={`${performance?.win_rate || 0}%`}
          subtitle="Overall accuracy"
          icon={Trophy}
          color={performance?.win_rate >= 55 ? "green" : performance?.win_rate >= 45 ? "yellow" : "red"}
        />
        <StatCard
          title="ROI"
          value={`${performance?.roi > 0 ? '+' : ''}${performance?.roi || 0}%`}
          subtitle="Return on investment"
          icon={DollarSign}
          color={performance?.roi >= 0 ? "green" : "red"}
        />
        <StatCard
          title="Total Picks"
          value={performance?.total_predictions || 0}
          subtitle={`${performance?.wins || 0}W - ${performance?.losses || 0}L`}
          icon={Target}
          color="blue"
        />
        <StatCard
          title="Pushes"
          value={performance?.pushes || 0}
          subtitle="No decision"
          icon={Clock}
          color="yellow"
        />
      </div>

      {/* Charts and Breakdown */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Win/Loss Chart */}
        <div className="stat-card">
          <h3 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
            <PieChart className="w-5 h-5 text-brand-primary" />
            Win/Loss Distribution
          </h3>
          <WinLossChart 
            wins={performance?.wins || 0}
            losses={performance?.losses || 0}
            pushes={performance?.pushes || 0}
          />
        </div>

        {/* Performance by Sport */}
        <div className="lg:col-span-2 stat-card">
          <h3 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-brand-primary" />
            Performance by Sport
          </h3>
          
          {performance?.by_sport && Object.keys(performance.by_sport).length > 0 ? (
            <div className="grid md:grid-cols-2 gap-4">
              {Object.entries(performance.by_sport).map(([sport, stats]) => (
                <SportPerformanceCard key={sport} sport={sport} stats={stats} />
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-text-muted">
              No sport-specific data available
            </div>
          )}
        </div>
      </div>

      {/* Recent Predictions Table */}
      <div className="stat-card">
        <h3 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
          <Target className="w-5 h-5 text-brand-primary" />
          Recent Predictions
        </h3>
        
        {performance?.recent_predictions?.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-zinc-700">
                  <th className="text-left py-3 px-4 text-text-muted text-xs font-mono uppercase">Match</th>
                  <th className="text-left py-3 px-4 text-text-muted text-xs font-mono uppercase">Sport</th>
                  <th className="text-left py-3 px-4 text-text-muted text-xs font-mono uppercase">Pick</th>
                  <th className="text-left py-3 px-4 text-text-muted text-xs font-mono uppercase">Odds</th>
                  <th className="text-left py-3 px-4 text-text-muted text-xs font-mono uppercase">Result</th>
                </tr>
              </thead>
              <tbody>
                {performance.recent_predictions.map((prediction) => (
                  <PredictionRow key={prediction.id} prediction={prediction} />
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12 text-text-muted">
            <Trophy className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No prediction history yet</p>
            <p className="text-sm mt-2">Start making predictions to see your performance</p>
          </div>
        )}
      </div>

      {/* Performance Tips */}
      <div className="stat-card bg-gradient-to-r from-zinc-900 to-zinc-800">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-lg bg-brand-primary/10">
            <TrendingUp className="w-6 h-6 text-brand-primary" />
          </div>
          <div>
            <h3 className="font-mono font-bold text-text-primary mb-2">Performance Insights</h3>
            <ul className="text-text-secondary text-sm space-y-1">
              <li>• Track all your bets to build an accurate performance history</li>
              <li>• Focus on sports where your win rate exceeds 55% for best ROI</li>
              <li>• Use AI analysis to identify value opportunities</li>
              <li>• Consider line movement as an indicator of sharp action</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Performance;
