import { useState, useEffect } from "react";
import axios from "axios";
import { 
  Settings as SettingsIcon, Bell, Clock, Save, RefreshCw,
  Zap, TrendingUp, AlertTriangle, Check, Globe
} from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const Settings = () => {
  const [settings, setSettings] = useState({
    cache_duration_minutes: 60,
    priority_sports: [],
    notification_preferences: {
      line_movement_alerts: true,
      line_movement_threshold: 5.0,
      result_alerts: true,
      daily_summary: true
    }
  });
  const [scraperStatus, setScraperStatus] = useState({ status: 'active' });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const sports = [
    { key: "basketball_nba", label: "NBA Basketball" },
    { key: "americanfootball_nfl", label: "NFL Football" },
    { key: "baseball_mlb", label: "MLB Baseball" },
    { key: "icehockey_nhl", label: "NHL Hockey" },
    { key: "soccer_epl", label: "EPL Soccer" },
    { key: "soccer_spain_la_liga", label: "La Liga Soccer" },
    { key: "mma_mixed_martial_arts", label: "MMA" },
  ];

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const [settingsRes, statusRes] = await Promise.all([
          axios.get(`${API}/settings`),
          axios.get(`${API}/scraper-status`)
        ]);
        setSettings(settingsRes.data);
        setScraperStatus(statusRes.data);
      } catch (error) {
        console.error("Error fetching settings:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchSettings();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      await axios.put(`${API}/settings`, settings);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (error) {
      console.error("Error saving settings:", error);
      alert("Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  const handleNotificationChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      notification_preferences: {
        ...prev.notification_preferences,
        [key]: value
      }
    }));
  };

  const handlePrioritySportToggle = (sportKey) => {
    setSettings(prev => ({
      ...prev,
      priority_sports: prev.priority_sports.includes(sportKey)
        ? prev.priority_sports.filter(s => s !== sportKey)
        : [...prev.priority_sports, sportKey]
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
            <SettingsIcon className="w-6 h-6 text-brand-primary" />
            Settings
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Configure app behavior and notifications
          </p>
        </div>
        <button 
          onClick={handleSave}
          disabled={saving}
          className="btn-primary flex items-center gap-2"
        >
          {saving ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : saved ? (
            <Check className="w-4 h-4" />
          ) : (
            <Save className="w-4 h-4" />
          )}
          {saved ? 'Saved!' : 'Save Settings'}
        </button>
      </div>

      {/* Data Source Settings */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-brand-primary" />
          Data Source
        </h2>
        <p className="text-text-muted text-sm mb-4">
          Choose where to get odds data from
        </p>
        
        <div className="grid md:grid-cols-2 gap-4">
          <button
            onClick={() => handleDataSourceChange('oddsportal')}
            className={`p-4 rounded-lg border text-left transition-all ${
              dataSource.current_source === 'oddsportal'
                ? 'bg-brand-primary/20 border-brand-primary'
                : 'bg-zinc-800 border-zinc-700 hover:border-zinc-600'
            }`}
          >
            <div className="flex items-center gap-3 mb-2">
              <Globe className={`w-5 h-5 ${dataSource.current_source === 'oddsportal' ? 'text-brand-primary' : 'text-text-muted'}`} />
              <span className="font-bold text-text-primary">OddsPortal</span>
              {dataSource.current_source === 'oddsportal' && (
                <span className="ml-auto px-2 py-0.5 rounded text-xs bg-semantic-success/20 text-semantic-success font-bold">ACTIVE</span>
              )}
            </div>
            <p className="text-text-muted text-sm mb-2">Free web scraping - No API limits!</p>
            <ul className="text-xs text-text-secondary space-y-1">
              <li>✓ Unlimited requests</li>
              <li>✓ Multiple bookmakers</li>
              <li>✓ Opening &amp; current odds</li>
              <li>✓ Hourly auto-updates</li>
            </ul>
          </button>
          
          <button
            onClick={() => handleDataSourceChange('oddsapi')}
            className={`p-4 rounded-lg border text-left transition-all ${
              dataSource.current_source === 'oddsapi'
                ? 'bg-brand-primary/20 border-brand-primary'
                : 'bg-zinc-800 border-zinc-700 hover:border-zinc-600'
            }`}
          >
            <div className="flex items-center gap-3 mb-2">
              <Zap className={`w-5 h-5 ${dataSource.current_source === 'oddsapi' ? 'text-brand-primary' : 'text-text-muted'}`} />
              <span className="font-bold text-text-primary">Odds API</span>
              {dataSource.current_source === 'oddsapi' && (
                <span className="ml-auto px-2 py-0.5 rounded text-xs bg-semantic-success/20 text-semantic-success font-bold">ACTIVE</span>
              )}
            </div>
            <p className="text-text-muted text-sm mb-2">Paid API - 500 free calls/month</p>
            <ul className="text-xs text-text-secondary space-y-1">
              <li>✓ Fast &amp; reliable</li>
              <li>✓ Official API</li>
              <li>⚠ Limited to 500 calls/month</li>
              <li>⚠ Requires API keys</li>
            </ul>
          </button>
        </div>
      </div>

      {/* API & Caching Settings */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-brand-primary" />
          API & Performance
        </h2>
        
        <div className="space-y-6">
          <div>
            <label className="flex items-center justify-between mb-2">
              <span className="text-text-primary">Cache Duration (minutes)</span>
              <span className="font-mono text-brand-primary">{settings.cache_duration_minutes} min</span>
            </label>
            <input
              type="range"
              min="15"
              max="120"
              step="15"
              value={settings.cache_duration_minutes}
              onChange={(e) => setSettings(prev => ({...prev, cache_duration_minutes: parseInt(e.target.value)}))}
              className="w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-brand-primary"
            />
            <div className="flex justify-between text-xs text-text-muted mt-1">
              <span>15 min</span>
              <span>60 min</span>
              <span>120 min</span>
            </div>
            <p className="text-text-muted text-xs mt-2">
              Higher values save more API calls but show less fresh data.
            </p>
          </div>

          <div className="flex items-center justify-between p-4 bg-zinc-800 rounded-lg">
            <div>
              <p className="text-text-primary font-semibold">Auto-Rotate API Keys</p>
              <p className="text-text-muted text-sm">Automatically switch to next key when current is exhausted</p>
            </div>
            <button
              onClick={() => setSettings(prev => ({...prev, auto_rotate_keys: !prev.auto_rotate_keys}))}
              className={`w-12 h-6 rounded-full transition-colors ${
                settings.auto_rotate_keys ? 'bg-brand-primary' : 'bg-zinc-600'
              }`}
            >
              <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                settings.auto_rotate_keys ? 'translate-x-6' : 'translate-x-0.5'
              }`} />
            </button>
          </div>
        </div>
      </div>

      {/* Notification Settings */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
          <Bell className="w-5 h-5 text-brand-primary" />
          Notifications
        </h2>
        
        <div className="space-y-4">
          {/* Line Movement Alerts */}
          <div className="p-4 bg-zinc-800 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-semantic-success" />
                <p className="text-text-primary font-semibold">Line Movement Alerts</p>
              </div>
              <button
                onClick={() => handleNotificationChange('line_movement_alerts', !settings.notification_preferences.line_movement_alerts)}
                className={`w-12 h-6 rounded-full transition-colors ${
                  settings.notification_preferences.line_movement_alerts ? 'bg-brand-primary' : 'bg-zinc-600'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                  settings.notification_preferences.line_movement_alerts ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </div>
            {settings.notification_preferences.line_movement_alerts && (
              <div className="mt-3">
                <label className="flex items-center justify-between mb-2">
                  <span className="text-text-muted text-sm">Alert threshold</span>
                  <span className="font-mono text-brand-primary">{settings.notification_preferences.line_movement_threshold}%</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="15"
                  step="0.5"
                  value={settings.notification_preferences.line_movement_threshold}
                  onChange={(e) => handleNotificationChange('line_movement_threshold', parseFloat(e.target.value))}
                  className="w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-brand-primary"
                />
                <p className="text-text-muted text-xs mt-1">
                  Get notified when odds move by this percentage
                </p>
              </div>
            )}
          </div>

          {/* Low API Alerts */}
          <div className="p-4 bg-zinc-800 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-semantic-warning" />
                <p className="text-text-primary font-semibold">Low API Calls Warning</p>
              </div>
              <button
                onClick={() => handleNotificationChange('low_api_alerts', !settings.notification_preferences.low_api_alerts)}
                className={`w-12 h-6 rounded-full transition-colors ${
                  settings.notification_preferences.low_api_alerts ? 'bg-brand-primary' : 'bg-zinc-600'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                  settings.notification_preferences.low_api_alerts ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </div>
            {settings.notification_preferences.low_api_alerts && (
              <div className="mt-3">
                <label className="flex items-center justify-between mb-2">
                  <span className="text-text-muted text-sm">Alert when below</span>
                  <span className="font-mono text-brand-primary">{settings.notification_preferences.low_api_threshold} calls</span>
                </label>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="10"
                  value={settings.notification_preferences.low_api_threshold}
                  onChange={(e) => handleNotificationChange('low_api_threshold', parseInt(e.target.value))}
                  className="w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-brand-primary"
                />
              </div>
            )}
          </div>

          {/* Result Alerts */}
          <div className="flex items-center justify-between p-4 bg-zinc-800 rounded-lg">
            <div className="flex items-center gap-2">
              <Check className="w-4 h-4 text-semantic-success" />
              <div>
                <p className="text-text-primary font-semibold">Result Notifications</p>
                <p className="text-text-muted text-sm">Get notified when your picks win or lose</p>
              </div>
            </div>
            <button
              onClick={() => handleNotificationChange('result_alerts', !settings.notification_preferences.result_alerts)}
              className={`w-12 h-6 rounded-full transition-colors ${
                settings.notification_preferences.result_alerts ? 'bg-brand-primary' : 'bg-zinc-600'
              }`}
            >
              <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                settings.notification_preferences.result_alerts ? 'translate-x-6' : 'translate-x-0.5'
              }`} />
            </button>
          </div>

          {/* Daily Summary */}
          <div className="flex items-center justify-between p-4 bg-zinc-800 rounded-lg">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-brand-secondary" />
              <div>
                <p className="text-text-primary font-semibold">Daily Summary</p>
                <p className="text-text-muted text-sm">Get a daily recap of your betting activity</p>
              </div>
            </div>
            <button
              onClick={() => handleNotificationChange('daily_summary', !settings.notification_preferences.daily_summary)}
              className={`w-12 h-6 rounded-full transition-colors ${
                settings.notification_preferences.daily_summary ? 'bg-brand-primary' : 'bg-zinc-600'
              }`}
            >
              <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                settings.notification_preferences.daily_summary ? 'translate-x-6' : 'translate-x-0.5'
              }`} />
            </button>
          </div>
        </div>
      </div>

      {/* Priority Sports */}
      <div className="stat-card">
        <h2 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-brand-primary" />
          Priority Sports
        </h2>
        <p className="text-text-muted text-sm mb-4">
          Select sports to prioritize for API calls when resources are limited.
        </p>
        
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {sports.map(sport => (
            <button
              key={sport.key}
              onClick={() => handlePrioritySportToggle(sport.key)}
              className={`p-3 rounded-lg border transition-all text-left ${
                settings.priority_sports.includes(sport.key)
                  ? 'bg-brand-primary/20 border-brand-primary text-brand-primary'
                  : 'bg-zinc-800 border-zinc-700 text-text-secondary hover:border-zinc-600'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{sport.label}</span>
                {settings.priority_sports.includes(sport.key) && (
                  <Check className="w-4 h-4" />
                )}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Settings;
