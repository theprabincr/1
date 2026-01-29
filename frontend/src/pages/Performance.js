import { useState, useEffect } from "react";
import axios from "axios";
import { 
  Trophy, RefreshCw, TrendingUp, TrendingDown,
  CheckCircle, XCircle, DollarSign, Target,
  BarChart3, PlusCircle, Calendar
} from "lucide-react";
import { PieChart as RePieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from "recharts";

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
    <div className="stat-card">
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

// My Bets Row
const MyBetRow = ({ bet }) => {
  const statusColors = {
    pending: "text-semantic-warning bg-semantic-warning/10",
    won: "text-semantic-success bg-semantic-success/10",
    lost: "text-semantic-danger bg-semantic-danger/10"
  };

  const profit = bet.result === 'won' ? (bet.stake * (bet.odds - 1)) : (bet.result === 'lost' ? -bet.stake : 0);
  const profitPercent = ((profit / bet.stake) * 100).toFixed(1);

  return (
    <div className="p-4 bg-zinc-800 rounded-lg hover:bg-zinc-700 transition-colors">
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1">
          <p className="text-text-primary font-semibold">{bet.event_name}</p>
          <p className="text-text-muted text-sm">{bet.selection} @ {bet.odds}</p>
          <p className="text-text-muted text-xs mt-1">
            {new Date(bet.created_at).toLocaleDateString()} • Stake: ${bet.stake}
          </p>
        </div>
        <div className="text-right">
          <span className={`px-2 py-1 rounded text-xs font-bold ${statusColors[bet.result]}`}>
            {bet.result.toUpperCase()}
          </span>
          {bet.result !== 'pending' && (
            <p className={`text-sm font-mono mt-2 ${
              profit >= 0 ? 'text-semantic-success' : 'text-semantic-danger'
            }`}>
              {profit >= 0 ? '+' : ''}${profit.toFixed(2)} ({profitPercent >= 0 ? '+' : ''}{profitPercent}%)
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

// Performance Chart
const PerformanceChart = ({ bets }) => {
  // Group by date
  const dateGroups = {};
  bets.forEach(bet => {
    const date = new Date(bet.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    if (!dateGroups[date]) {
      dateGroups[date] = { date, profit: 0, bets: 0 };
    }
    if (bet.result === 'won') {
      dateGroups[date].profit += bet.stake * (bet.odds - 1);
    } else if (bet.result === 'lost') {
      dateGroups[date].profit -= bet.stake;
    }
    dateGroups[date].bets++;
  });

  const data = Object.values(dateGroups).slice(-7); // Last 7 days

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-text-muted">
        No bet history available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272A" />
        <XAxis dataKey="date" stroke="#71717A" style={{ fontSize: '12px' }} />
        <YAxis stroke="#71717A" style={{ fontSize: '12px' }} />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#18181B', 
            border: '1px solid #27272A',
            borderRadius: '8px'
          }}
          formatter={(value) => ['$' + value.toFixed(2), 'Profit']}
        />
        <Bar dataKey="profit" fill="#22C55E" />
      </BarChart>
    </ResponsiveContainer>
  );
};

const Performance = () => {
  const [myBets, setMyBets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddBetModal, setShowAddBetModal] = useState(false);
  const [newBet, setNewBet] = useState({
    event_name: '',
    selection: '',
    stake: '',
    odds: '',
    result: 'pending'
  });

  useEffect(() => {
    fetchMyBets();
  }, []);

  const fetchMyBets = async () => {
    try {
      const res = await axios.get(`${API}/my-bets`);
      setMyBets(res.data.bets || []);
    } catch (error) {
      console.error("Error fetching my bets:", error);
      setMyBets([]);
    } finally {
      setLoading(false);
    }
  };

  const handleAddBet = async () => {
    try {
      await axios.post(`${API}/my-bets`, {
        ...newBet,
        stake: parseFloat(newBet.stake),
        odds: parseFloat(newBet.odds)
      });
      setShowAddBetModal(false);
      setNewBet({ event_name: '', selection: '', stake: '', odds: '', result: 'pending' });
      fetchMyBets();
    } catch (error) {
      console.error("Error adding bet:", error);
      alert("Failed to add bet");
    }
  };

  // Calculate stats
  const completedBets = myBets.filter(b => b.result !== 'pending');
  const wins = myBets.filter(b => b.result === 'won').length;
  const losses = myBets.filter(b => b.result === 'lost').length;
  const winRate = completedBets.length > 0 ? (wins / completedBets.length * 100).toFixed(1) : 0;
  
  const totalStaked = myBets.reduce((sum, b) => sum + (b.stake || 0), 0);
  const totalProfit = myBets.reduce((sum, b) => {
    if (b.result === 'won') return sum + (b.stake * (b.odds - 1));
    if (b.result === 'lost') return sum - b.stake;
    return sum;
  }, 0);
  const roi = totalStaked > 0 ? ((totalProfit / totalStaked) * 100).toFixed(1) : 0;

  const avgOdds = completedBets.length > 0 
    ? (completedBets.reduce((sum, b) => sum + b.odds, 0) / completedBets.length).toFixed(2)
    : 0;

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
            <Trophy className="w-6 h-6 text-brand-primary" />
            My Betting Performance
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Track your actual bets and analyze your performance
          </p>
        </div>
        <button 
          onClick={() => setShowAddBetModal(true)}
          className="btn-primary flex items-center gap-2"
        >
          <PlusCircle className="w-4 h-4" />
          Add Bet
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          title="Total Bets"
          value={myBets.length}
          subtitle={`${wins}W - ${losses}L`}
          icon={BarChart3}
          color="lime"
        />
        <StatCard
          title="Win Rate"
          value={`${winRate}%`}
          subtitle={`${completedBets.length} completed`}
          icon={Target}
          color="green"
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
          subtitle={`$${totalStaked.toFixed(2)} staked`}
          icon={TrendingUp}
          color="blue"
        />
      </div>

      {/* Performance Chart */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
          <Calendar className="w-5 h-5 text-brand-primary" />
          Profit History (Last 7 Days)
        </h2>
        <PerformanceChart bets={myBets} />
      </div>

      {/* My Bets List */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4">
          My Bets
        </h2>
        <div className="space-y-3">
          {myBets.length === 0 ? (
            <div className="text-center py-8 text-text-muted">
              <p>No bets tracked yet</p>
              <button 
                onClick={() => setShowAddBetModal(true)}
                className="btn-primary mt-4"
              >
                Add Your First Bet
              </button>
            </div>
          ) : (
            myBets.slice(0, 10).map((bet, i) => <MyBetRow key={i} bet={bet} />)
          )}
        </div>
      </div>

      {/* Add Bet Modal */}
      {showAddBetModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowAddBetModal(false)}>
          <div className="bg-zinc-900 rounded-xl p-6 w-full max-w-md border border-zinc-700" onClick={(e) => e.stopPropagation()}>
            <h2 className="font-mono font-bold text-xl text-text-primary mb-4">Add New Bet</h2>
            <div className="space-y-4">
              <div>
                <label className="text-text-muted text-sm">Event Name</label>
                <input
                  type="text"
                  value={newBet.event_name}
                  onChange={(e) => setNewBet({...newBet, event_name: e.target.value})}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-text-primary mt-1"
                  placeholder="Lakers vs Warriors"
                />
              </div>
              <div>
                <label className="text-text-muted text-sm">Selection</label>
                <input
                  type="text"
                  value={newBet.selection}
                  onChange={(e) => setNewBet({...newBet, selection: e.target.value})}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-text-primary mt-1"
                  placeholder="Lakers ML"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-text-muted text-sm">Stake ($)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={newBet.stake}
                    onChange={(e) => setNewBet({...newBet, stake: e.target.value})}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-text-primary mt-1"
                    placeholder="100"
                  />
                </div>
                <div>
                  <label className="text-text-muted text-sm">Odds</label>
                  <input
                    type="number"
                    step="0.01"
                    value={newBet.odds}
                    onChange={(e) => setNewBet({...newBet, odds: e.target.value})}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-text-primary mt-1"
                    placeholder="1.91"
                  />
                </div>
              </div>
              <div>
                <label className="text-text-muted text-sm">Result</label>
                <select
                  value={newBet.result}
                  onChange={(e) => setNewBet({...newBet, result: e.target.value})}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-text-primary mt-1"
                >
                  <option value="pending">Pending</option>
                  <option value="won">Won</option>
                  <option value="lost">Lost</option>
                </select>
              </div>
              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setShowAddBetModal(false)}
                  className="flex-1 px-4 py-2 bg-zinc-800 text-text-primary rounded-lg hover:bg-zinc-700"
                >
                  Cancel
                </button>
                <button
                  onClick={handleAddBet}
                  disabled={!newBet.event_name || !newBet.selection || !newBet.stake || !newBet.odds}
                  className="flex-1 btn-primary"
                >
                  Add Bet
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Performance;
