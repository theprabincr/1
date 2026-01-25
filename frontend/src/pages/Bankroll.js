import { useState, useEffect } from "react";
import axios from "axios";
import { 
  Wallet, Plus, Minus, TrendingUp, TrendingDown, 
  DollarSign, RefreshCw, Download, Target, Clock
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const Bankroll = () => {
  const [bankroll, setBankroll] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showDepositForm, setShowDepositForm] = useState(false);
  const [showWithdrawForm, setShowWithdrawForm] = useState(false);
  const [amount, setAmount] = useState("");
  const [description, setDescription] = useState("");
  const [pendingPredictions, setPendingPredictions] = useState([]);
  const [betAmount, setBetAmount] = useState({});

  const fetchData = async () => {
    setLoading(true);
    try {
      const [bankrollRes, predictionsRes] = await Promise.all([
        axios.get(`${API}/bankroll`),
        axios.get(`${API}/recommendations?limit=20`)
      ]);
      setBankroll(bankrollRes.data);
      setPendingPredictions(predictionsRes.data);
    } catch (error) {
      console.error("Error fetching bankroll data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleDeposit = async (e) => {
    e.preventDefault();
    try {
      await axios.post(`${API}/bankroll/deposit`, {
        amount: parseFloat(amount),
        description: description || "Deposit"
      });
      setAmount("");
      setDescription("");
      setShowDepositForm(false);
      fetchData();
    } catch (error) {
      console.error("Error depositing:", error);
      alert(error.response?.data?.detail || "Failed to deposit");
    }
  };

  const handleWithdraw = async (e) => {
    e.preventDefault();
    try {
      await axios.post(`${API}/bankroll/withdraw`, {
        amount: parseFloat(amount),
        description: description || "Withdrawal"
      });
      setAmount("");
      setDescription("");
      setShowWithdrawForm(false);
      fetchData();
    } catch (error) {
      console.error("Error withdrawing:", error);
      alert(error.response?.data?.detail || "Failed to withdraw");
    }
  };

  const handlePlaceBet = async (predictionId) => {
    const stake = betAmount[predictionId];
    if (!stake || stake <= 0) {
      alert("Please enter a valid stake amount");
      return;
    }
    
    try {
      await axios.post(`${API}/bankroll/place-bet`, {
        prediction_id: predictionId,
        stake: parseFloat(stake)
      });
      setBetAmount(prev => ({...prev, [predictionId]: ""}));
      fetchData();
    } catch (error) {
      console.error("Error placing bet:", error);
      alert(error.response?.data?.detail || "Failed to place bet");
    }
  };

  const handleExport = async () => {
    try {
      const response = await axios.get(`${API}/export/bankroll?format=csv`, {
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'bankroll_export.csv');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error("Error exporting:", error);
    }
  };

  // Generate chart data from transactions
  const generateChartData = () => {
    if (!bankroll?.recent_transactions) return [];
    
    const transactions = [...bankroll.recent_transactions].reverse();
    return transactions.slice(-30).map((t, i) => ({
      name: new Date(t.created_at).toLocaleDateString(),
      balance: t.balance_after
    }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
      </div>
    );
  }

  const chartData = generateChartData();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary flex items-center gap-2">
            <Wallet className="w-6 h-6 text-brand-primary" />
            Bankroll Management
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Track your betting bankroll and performance
          </p>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={handleExport}
            className="btn-outline flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
          <button 
            onClick={fetchData}
            className="btn-outline flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Balance Overview */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="w-5 h-5 text-brand-primary" />
            <p className="text-text-muted text-sm">Current Balance</p>
          </div>
          <p className="font-mono text-3xl font-bold text-text-primary">
            ${bankroll?.current_balance?.toFixed(2) || '0.00'}
          </p>
        </div>
        
        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-5 h-5 text-semantic-success" />
            <p className="text-text-muted text-sm">Total Won</p>
          </div>
          <p className="font-mono text-3xl font-bold text-semantic-success">
            ${bankroll?.total_won?.toFixed(2) || '0.00'}
          </p>
        </div>
        
        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-5 h-5 text-semantic-danger" />
            <p className="text-text-muted text-sm">Total Lost</p>
          </div>
          <p className="font-mono text-3xl font-bold text-semantic-danger">
            ${bankroll?.total_lost?.toFixed(2) || '0.00'}
          </p>
        </div>
        
        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-5 h-5 text-brand-secondary" />
            <p className="text-text-muted text-sm">ROI</p>
          </div>
          <p className={`font-mono text-3xl font-bold ${
            bankroll?.roi >= 0 ? 'text-semantic-success' : 'text-semantic-danger'
          }`}>
            {bankroll?.roi > 0 ? '+' : ''}{bankroll?.roi?.toFixed(2) || '0.00'}%
          </p>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3">
        <button 
          onClick={() => { setShowDepositForm(true); setShowWithdrawForm(false); }}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Deposit
        </button>
        <button 
          onClick={() => { setShowWithdrawForm(true); setShowDepositForm(false); }}
          className="btn-outline flex items-center gap-2"
        >
          <Minus className="w-4 h-4" />
          Withdraw
        </button>
      </div>

      {/* Deposit Form */}
      {showDepositForm && (
        <div className="stat-card border border-semantic-success">
          <h3 className="font-mono font-bold text-lg text-text-primary mb-4">Deposit Funds</h3>
          <form onSubmit={handleDeposit} className="space-y-4">
            <div>
              <label className="block text-text-muted text-sm mb-2">Amount ($)</label>
              <input
                type="number"
                step="0.01"
                min="0"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                placeholder="100.00"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 text-text-primary font-mono focus:outline-none focus:border-brand-primary"
                required
              />
            </div>
            <div>
              <label className="block text-text-muted text-sm mb-2">Description (optional)</label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Initial deposit"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 text-text-primary focus:outline-none focus:border-brand-primary"
              />
            </div>
            <div className="flex gap-3">
              <button type="submit" className="btn-primary">
                Deposit
              </button>
              <button 
                type="button" 
                onClick={() => setShowDepositForm(false)}
                className="btn-outline"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Withdraw Form */}
      {showWithdrawForm && (
        <div className="stat-card border border-semantic-warning">
          <h3 className="font-mono font-bold text-lg text-text-primary mb-4">Withdraw Funds</h3>
          <form onSubmit={handleWithdraw} className="space-y-4">
            <div>
              <label className="block text-text-muted text-sm mb-2">Amount ($)</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max={bankroll?.current_balance || 0}
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                placeholder="50.00"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 text-text-primary font-mono focus:outline-none focus:border-brand-primary"
                required
              />
            </div>
            <div>
              <label className="block text-text-muted text-sm mb-2">Description (optional)</label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Profit withdrawal"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 text-text-primary focus:outline-none focus:border-brand-primary"
              />
            </div>
            <div className="flex gap-3">
              <button type="submit" className="btn-primary">
                Withdraw
              </button>
              <button 
                type="button" 
                onClick={() => setShowWithdrawForm(false)}
                className="btn-outline"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Balance Chart */}
      {chartData.length > 1 && (
        <div className="stat-card">
          <h2 className="font-mono font-bold text-lg text-text-primary mb-4">Balance History</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="name" stroke="#666" fontSize={12} />
                <YAxis stroke="#666" fontSize={12} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                  labelStyle={{ color: '#bbb' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="balance" 
                  stroke="#ADFF2F" 
                  strokeWidth={2}
                  dot={{ fill: '#ADFF2F', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Place Bets Section */}
      {pendingPredictions.length > 0 && (
        <div className="stat-card">
          <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-brand-primary" />
            Place Bets on Pending Picks
          </h2>
          <div className="space-y-3">
            {pendingPredictions.filter(p => !p.stake).slice(0, 5).map((pred) => (
              <div key={pred.id} className="flex items-center justify-between p-4 bg-zinc-800 rounded-lg">
                <div>
                  <p className="text-text-primary font-semibold">
                    {pred.predicted_outcome}
                  </p>
                  <p className="text-text-muted text-sm">
                    {pred.home_team} vs {pred.away_team} @ {pred.odds_at_prediction?.toFixed(2)}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    step="1"
                    min="1"
                    max={bankroll?.current_balance || 0}
                    value={betAmount[pred.id] || ""}
                    onChange={(e) => setBetAmount(prev => ({...prev, [pred.id]: e.target.value}))}
                    placeholder="Stake"
                    className="w-24 bg-zinc-700 border border-zinc-600 rounded-lg px-3 py-2 text-text-primary font-mono text-sm focus:outline-none focus:border-brand-primary"
                  />
                  <button
                    onClick={() => handlePlaceBet(pred.id)}
                    className="btn-primary px-4 py-2"
                    disabled={!betAmount[pred.id] || parseFloat(betAmount[pred.id]) > (bankroll?.current_balance || 0)}
                  >
                    Bet
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Transactions */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
          <Clock className="w-5 h-5 text-text-muted" />
          Recent Transactions
        </h2>
        
        {bankroll?.recent_transactions?.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Type</th>
                  <th>Amount</th>
                  <th>Description</th>
                  <th>Balance</th>
                </tr>
              </thead>
              <tbody>
                {bankroll.recent_transactions.map((t) => (
                  <tr key={t.id}>
                    <td className="text-text-muted">
                      {new Date(t.created_at).toLocaleDateString()}
                    </td>
                    <td>
                      <span className={`px-2 py-1 rounded text-xs font-bold ${
                        t.type === 'deposit' || t.type === 'win' 
                          ? 'bg-semantic-success/20 text-semantic-success'
                          : t.type === 'withdrawal' || t.type === 'loss' || t.type === 'bet'
                          ? 'bg-semantic-danger/20 text-semantic-danger'
                          : 'bg-text-muted/20 text-text-muted'
                      }`}>
                        {t.type.toUpperCase()}
                      </span>
                    </td>
                    <td className={`font-mono font-bold ${
                      t.type === 'deposit' || t.type === 'win' 
                        ? 'text-semantic-success'
                        : 'text-semantic-danger'
                    }`}>
                      {t.type === 'deposit' || t.type === 'win' ? '+' : '-'}${t.amount.toFixed(2)}
                    </td>
                    <td className="text-text-secondary truncate max-w-xs">
                      {t.description}
                    </td>
                    <td className="font-mono text-text-primary">
                      ${t.balance_after.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8">
            <Wallet className="w-12 h-12 text-text-muted mx-auto mb-3" />
            <p className="text-text-muted">No transactions yet.</p>
            <p className="text-text-muted text-sm mt-1">
              Start by making a deposit.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Bankroll;
