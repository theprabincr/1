import { useState, useEffect } from "react";
import axios from "axios";
import { 
  Key, Plus, Trash2, RefreshCw, Check, AlertTriangle,
  Zap, RotateCcw, Shield, Eye, EyeOff
} from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const ApiKeys = () => {
  const [keys, setKeys] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyValue, setNewKeyValue] = useState("");
  const [apiUsage, setApiUsage] = useState({});
  const [showKey, setShowKey] = useState({});

  const fetchData = async () => {
    setLoading(true);
    try {
      const [keysRes, usageRes] = await Promise.all([
        axios.get(`${API}/api-keys`),
        axios.get(`${API}/api-usage`)
      ]);
      setKeys(keysRes.data);
      setApiUsage(usageRes.data);
    } catch (error) {
      console.error("Error fetching API keys:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleAddKey = async (e) => {
    e.preventDefault();
    try {
      await axios.post(`${API}/api-keys`, {
        key: newKeyValue,
        name: newKeyName
      });
      setNewKeyName("");
      setNewKeyValue("");
      setShowAddForm(false);
      fetchData();
    } catch (error) {
      console.error("Error adding API key:", error);
      alert(error.response?.data?.detail || "Failed to add API key");
    }
  };

  const handleDeleteKey = async (keyId) => {
    if (!window.confirm("Are you sure you want to delete this API key?")) return;
    try {
      await axios.delete(`${API}/api-keys/${keyId}`);
      fetchData();
    } catch (error) {
      console.error("Error deleting API key:", error);
    }
  };

  const handleActivateKey = async (keyId) => {
    try {
      await axios.put(`${API}/api-keys/${keyId}/activate`);
      fetchData();
    } catch (error) {
      console.error("Error activating API key:", error);
    }
  };

  const handleResetKey = async (keyId) => {
    if (!window.confirm("Reset usage for this key? (Use when new month starts)")) return;
    try {
      await axios.put(`${API}/api-keys/${keyId}/reset`);
      fetchData();
    } catch (error) {
      console.error("Error resetting API key:", error);
    }
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
            <Key className="w-6 h-6 text-brand-primary" />
            API Key Management
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Manage your Odds API keys for automatic rotation
          </p>
        </div>
        <button 
          onClick={() => setShowAddForm(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Add New Key
        </button>
      </div>

      {/* Current Usage Stats */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-brand-primary" />
          Current API Status
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-zinc-800 p-4 rounded-lg">
            <p className="text-text-muted text-sm">Active Key Remaining</p>
            <p className={`font-mono text-2xl font-bold ${
              apiUsage.requests_remaining > 100 ? 'text-semantic-success' :
              apiUsage.requests_remaining > 30 ? 'text-semantic-warning' : 'text-semantic-danger'
            }`}>
              {apiUsage.requests_remaining ?? '---'}
            </p>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <p className="text-text-muted text-sm">Total Across All Keys</p>
            <p className="font-mono text-2xl font-bold text-brand-primary">
              {apiUsage.total_remaining_all_keys ?? '---'}
            </p>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <p className="text-text-muted text-sm">Active Keys</p>
            <p className="font-mono text-2xl font-bold text-brand-secondary">
              {apiUsage.active_keys_count ?? 0}
            </p>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <p className="text-text-muted text-sm">Cache Duration</p>
            <p className="font-mono text-2xl font-bold text-text-primary">
              {apiUsage.cache_duration_minutes ?? 60} min
            </p>
          </div>
        </div>
      </div>

      {/* Add New Key Form */}
      {showAddForm && (
        <div className="stat-card border border-brand-primary">
          <h3 className="font-mono font-bold text-lg text-text-primary mb-4">Add New API Key</h3>
          <form onSubmit={handleAddKey} className="space-y-4">
            <div>
              <label className="block text-text-muted text-sm mb-2">Key Name (for reference)</label>
              <input
                type="text"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                placeholder="e.g., Account 1, Backup Key"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 text-text-primary focus:outline-none focus:border-brand-primary"
                required
              />
            </div>
            <div>
              <label className="block text-text-muted text-sm mb-2">API Key</label>
              <input
                type="text"
                value={newKeyValue}
                onChange={(e) => setNewKeyValue(e.target.value)}
                placeholder="Enter your Odds API key"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 text-text-primary font-mono focus:outline-none focus:border-brand-primary"
                required
              />
            </div>
            <div className="flex gap-3">
              <button type="submit" className="btn-primary">
                Add Key
              </button>
              <button 
                type="button" 
                onClick={() => setShowAddForm(false)}
                className="btn-outline"
              >
                Cancel
              </button>
            </div>
          </form>
          
          <div className="mt-4 p-3 bg-zinc-800 rounded-lg">
            <p className="text-text-muted text-sm flex items-start gap-2">
              <Shield className="w-4 h-4 mt-0.5 text-brand-primary" />
              <span>
                Get your free API key at{" "}
                <a 
                  href="https://the-odds-api.com" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-brand-primary hover:underline"
                >
                  the-odds-api.com
                </a>
                {" "}— 500 free calls per month per account.
              </span>
            </p>
          </div>
        </div>
      )}

      {/* Keys List */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4">
          Your API Keys ({keys.length})
        </h2>
        
        {keys.length === 0 ? (
          <div className="text-center py-8">
            <Key className="w-12 h-12 text-text-muted mx-auto mb-3" />
            <p className="text-text-muted">No API keys added yet.</p>
            <p className="text-text-muted text-sm mt-1">
              Using default key from environment.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {keys.map((key) => (
              <div 
                key={key.id}
                className={`p-4 rounded-lg border ${
                  key.is_active && !key.is_exhausted 
                    ? 'bg-brand-primary/10 border-brand-primary' 
                    : key.is_exhausted 
                    ? 'bg-semantic-danger/10 border-semantic-danger/30'
                    : 'bg-zinc-800 border-zinc-700'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="font-bold text-text-primary">{key.name}</span>
                    {key.is_active && !key.is_exhausted && (
                      <span className="px-2 py-0.5 rounded text-xs bg-semantic-success/20 text-semantic-success font-bold">
                        ACTIVE
                      </span>
                    )}
                    {key.is_exhausted && (
                      <span className="px-2 py-0.5 rounded text-xs bg-semantic-danger/20 text-semantic-danger font-bold">
                        EXHAUSTED
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {!key.is_active && !key.is_exhausted && (
                      <button
                        onClick={() => handleActivateKey(key.id)}
                        className="p-2 rounded-lg bg-brand-primary/20 text-brand-primary hover:bg-brand-primary/30"
                        title="Activate this key"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                    )}
                    <button
                      onClick={() => handleResetKey(key.id)}
                      className="p-2 rounded-lg bg-zinc-700 text-text-secondary hover:bg-zinc-600"
                      title="Reset usage (new month)"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDeleteKey(key.id)}
                      className="p-2 rounded-lg bg-semantic-danger/20 text-semantic-danger hover:bg-semantic-danger/30"
                      title="Delete key"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-text-muted">Key</p>
                    <p className="font-mono text-text-secondary flex items-center gap-2">
                      {showKey[key.id] ? key.key_masked : '••••••••••••'}
                      <button
                        onClick={() => setShowKey(prev => ({...prev, [key.id]: !prev[key.id]}))}
                        className="text-text-muted hover:text-text-primary"
                      >
                        {showKey[key.id] ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                      </button>
                    </p>
                  </div>
                  <div>
                    <p className="text-text-muted">Remaining</p>
                    <p className={`font-mono font-bold ${
                      key.requests_remaining > 100 ? 'text-semantic-success' :
                      key.requests_remaining > 30 ? 'text-semantic-warning' : 'text-semantic-danger'
                    }`}>
                      {key.requests_remaining ?? 500}
                    </p>
                  </div>
                  <div>
                    <p className="text-text-muted">Used</p>
                    <p className="font-mono text-text-secondary">{key.requests_used ?? 0}</p>
                  </div>
                  <div>
                    <p className="text-text-muted">Added</p>
                    <p className="text-text-secondary">
                      {new Date(key.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* How It Works */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-semantic-warning" />
          How API Key Rotation Works
        </h2>
        <div className="space-y-3 text-text-secondary text-sm">
          <div className="flex items-start gap-3">
            <span className="w-6 h-6 rounded-full bg-brand-primary/20 text-brand-primary flex items-center justify-center text-xs font-bold">1</span>
            <p>Add multiple API keys from different Odds API accounts.</p>
          </div>
          <div className="flex items-start gap-3">
            <span className="w-6 h-6 rounded-full bg-brand-primary/20 text-brand-primary flex items-center justify-center text-xs font-bold">2</span>
            <p>When the active key's calls drop to 5 or below, the system automatically switches to the next available key.</p>
          </div>
          <div className="flex items-start gap-3">
            <span className="w-6 h-6 rounded-full bg-brand-primary/20 text-brand-primary flex items-center justify-center text-xs font-bold">3</span>
            <p>You'll receive notifications when keys are running low or when rotation occurs.</p>
          </div>
          <div className="flex items-start gap-3">
            <span className="w-6 h-6 rounded-full bg-brand-primary/20 text-brand-primary flex items-center justify-center text-xs font-bold">4</span>
            <p>At the start of each month, use the reset button to restore 500 calls to your keys.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ApiKeys;
