import { useState, useEffect } from "react";
import axios from "axios";
import { 
  BarChart3, TrendingUp, TrendingDown, Trophy, Target,
  RefreshCw, Calendar, Activity, Flame, Download
} from "lucide-react";
import { 
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend 
} from 'recharts';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const COLORS = ['#ADFF2F', '#00CED1', '#FF6B6B', '#FFD93D', '#6BCB77'];

const Analytics = () => {
  const [loading, setLoading] = useState(true);
  const [trends, setTrends] = useState(null);
  const [streaks, setStreaks] = useState(null);
  const [dateRange, setDateRange] = useState(30);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [trendsRes, streaksRes] = await Promise.all([
        axios.get(`${API}/analytics/trends?days=${dateRange}`),
        axios.get(`${API}/analytics/streaks`)
      ]);
      setTrends(trendsRes.data);
      setStreaks(streaksRes.data);
    } catch (error) {
      console.error("Error fetching analytics:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [dateRange]);

  const handleExport = async () => {
    try {
      const response = await axios.get(`${API}/export/performance-report`);
      const dataStr = JSON.stringify(response.data, null, 2);
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'performance_report.json');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error("Error exporting report:", error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
      </div>
    );
  }

  // Prepare daily chart data
  const dailyChartData = trends?.daily_stats 
    ? Object.entries(trends.daily_stats).map(([date, stats]) => ({
        date: date.slice(5), // MM-DD format
        wins: stats.wins,
        losses: stats.losses,
        total: stats.total,
        winRate: stats.total > 0 ? ((stats.wins / stats.total) * 100).toFixed(1) : 0
      }))
    : [];

  // Prepare sport breakdown data
  const sportChartData = trends?.sport_stats
    ? Object.entries(trends.sport_stats).map(([sport, stats]) => ({
        name: sport.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        wins: stats.wins,
        losses: stats.losses,
        total: stats.total,
        winRate: stats.total > 0 ? ((stats.wins / stats.total) * 100).toFixed(1) : 0
      }))
    : [];

  // Prepare market type data for pie chart
  const marketChartData = trends?.market_stats
    ? Object.entries(trends.market_stats).map(([market, stats]) => ({
        name: market.charAt(0).toUpperCase() + market.slice(1),
        value: stats.total,
        wins: stats.wins,
        winRate: stats.total > 0 ? ((stats.wins / stats.total) * 100).toFixed(1) : 0
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-brand-primary" />
            Analytics
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Detailed performance insights and trends
          </p>
        </div>
        <div className="flex gap-2">
          <select
            value={dateRange}
            onChange={(e) => setDateRange(parseInt(e.target.value))}
            className="bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 text-text-primary focus:outline-none focus:border-brand-primary"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
            <option value={365}>Last year</option>
          </select>
          <button 
            onClick={handleExport}
            className="btn-outline flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      {/* Streak Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <Flame className={`w-5 h-5 ${
              streaks?.streak_type === 'win' ? 'text-semantic-success' : 
              streaks?.streak_type === 'loss' ? 'text-semantic-danger' : 'text-text-muted'
            }`} />
            <p className="text-text-muted text-sm">Current Streak</p>
          </div>
          <p className={`font-mono text-3xl font-bold ${
            streaks?.streak_type === 'win' ? 'text-semantic-success' : 
            streaks?.streak_type === 'loss' ? 'text-semantic-danger' : 'text-text-muted'
          }`}>
            {streaks?.current_streak || 0} {streaks?.streak_type === 'win' ? 'W' : streaks?.streak_type === 'loss' ? 'L' : ''}
          </p>
        </div>

        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-5 h-5 text-semantic-success" />
            <p className="text-text-muted text-sm">Best Win Streak</p>
          </div>
          <p className="font-mono text-3xl font-bold text-semantic-success">
            {streaks?.best_win_streak || 0}
          </p>
        </div>

        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-5 h-5 text-semantic-danger" />
            <p className="text-text-muted text-sm">Worst Loss Streak</p>
          </div>
          <p className="font-mono text-3xl font-bold text-semantic-danger">
            {streaks?.worst_loss_streak || 0}
          </p>
        </div>

        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-5 h-5 text-brand-primary" />
            <p className="text-text-muted text-sm">Total Predictions</p>
          </div>
          <p className="font-mono text-3xl font-bold text-brand-primary">
            {trends?.total_predictions || 0}
          </p>
        </div>
      </div>

      {/* Daily Performance Chart */}
      {dailyChartData.length > 0 && (
        <div className="stat-card">
          <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
            <Calendar className="w-5 h-5 text-brand-primary" />
            Daily Performance
          </h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dailyChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="date" stroke="#666" fontSize={12} />
                <YAxis stroke="#666" fontSize={12} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                  labelStyle={{ color: '#bbb' }}
                />
                <Legend />
                <Bar dataKey="wins" fill="#22c55e" name="Wins" />
                <Bar dataKey="losses" fill="#ef4444" name="Losses" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Win Rate Trend */}
      {dailyChartData.length > 1 && (
        <div className="stat-card">
          <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-brand-primary" />
            Win Rate Trend
          </h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dailyChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="date" stroke="#666" fontSize={12} />
                <YAxis stroke="#666" fontSize={12} domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                  labelStyle={{ color: '#bbb' }}
                  formatter={(value) => [`${value}%`, 'Win Rate']}
                />
                <Line 
                  type="monotone" 
                  dataKey="winRate" 
                  stroke="#ADFF2F" 
                  strokeWidth={2}
                  dot={{ fill: '#ADFF2F', strokeWidth: 2 }}
                  name="Win Rate"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Sport Performance */}
        {sportChartData.length > 0 && (
          <div className="stat-card">
            <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
              <Trophy className="w-5 h-5 text-brand-primary" />
              Performance by Sport
            </h2>
            <div className="space-y-3">
              {sportChartData.map((sport, i) => (
                <div key={sport.name} className="p-3 bg-zinc-800 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-text-primary font-medium">{sport.name}</span>
                    <span className={`font-mono font-bold ${
                      parseFloat(sport.winRate) >= 55 ? 'text-semantic-success' :
                      parseFloat(sport.winRate) >= 45 ? 'text-semantic-warning' : 'text-semantic-danger'
                    }`}>
                      {sport.winRate}%
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-zinc-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-brand-primary"
                        style={{ width: `${sport.winRate}%` }}
                      />
                    </div>
                    <span className="text-text-muted text-xs">
                      {sport.wins}W / {sport.losses}L
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Market Type Distribution */}
        {marketChartData.length > 0 && (
          <div className="stat-card">
            <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-brand-primary" />
              Predictions by Market Type
            </h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={marketChartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {marketChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                    formatter={(value, name, props) => [
                      `${value} picks (${props.payload.winRate}% win rate)`,
                      name
                    ]}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center gap-4 mt-4">
              {marketChartData.map((market, i) => (
                <div key={market.name} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                  <span className="text-text-muted text-sm">{market.name}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* No Data State */}
      {trends?.total_predictions === 0 && (
        <div className="stat-card text-center py-12">
          <BarChart3 className="w-16 h-16 text-text-muted mx-auto mb-4" />
          <h3 className="text-text-primary font-bold text-lg mb-2">No Analytics Data Yet</h3>
          <p className="text-text-muted">
            Start making predictions to see your performance analytics here.
          </p>
        </div>
      )}
    </div>
  );
};

export default Analytics;
