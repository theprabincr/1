import { useState, useEffect } from "react";
import axios from "axios";
import { 
  Settings as SettingsIcon, Bell, Clock, Save, RefreshCw,
  TrendingUp, Check, TestTube
} from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const Settings = () => {
  const [settings, setSettings] = useState({
    notification_preferences: {
      line_movement_alerts: true,
      line_movement_threshold: 5.0,
      result_alerts: true,
      daily_summary: true
    }
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [testingSent, setTestingSent] = useState(false);

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const settingsRes = await axios.get(`${API}/settings`);
        setSettings(settingsRes.data);
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

  const sendTestNotification = async () => {
    try {
      await axios.post(`${API}/notifications/test`);
      setTestingSent(true);
      setTimeout(() => setTestingSent(false), 3000);
    } catch (error) {
      console.error("Error sending test notification:", error);
      alert("Failed to send test notification");
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
    <div className="space-y-6" data-testid="settings-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary flex items-center gap-2">
            <SettingsIcon className="w-6 h-6 text-brand-primary" />
            Settings
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Configure notification preferences
          </p>
        </div>
        <button 
          onClick={handleSave}
          disabled={saving}
          className="btn-primary flex items-center gap-2"
          data-testid="save-settings-btn"
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

      {/* Notification Settings */}
      <div className="stat-card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-mono font-bold text-lg text-text-primary flex items-center gap-2">
            <Bell className="w-5 h-5 text-brand-primary" />
            Notifications
          </h2>
          <button
            onClick={sendTestNotification}
            className="btn-outline text-sm flex items-center gap-2"
            data-testid="test-notification-btn"
          >
            {testingSent ? (
              <>
                <Check className="w-4 h-4 text-semantic-success" />
                Sent!
              </>
            ) : (
              <>
                <TestTube className="w-4 h-4" />
                Test Notification
              </>
            )}
          </button>
        </div>
        
        <div className="space-y-4">
          {/* Line Movement Alerts */}
          <div className="p-4 bg-zinc-800 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-semantic-success" />
                <p className="text-text-primary font-semibold">Line Movement Alerts</p>
              </div>
              <button
                onClick={() => handleNotificationChange('line_movement_alerts', !settings.notification_preferences?.line_movement_alerts)}
                className={`w-12 h-6 rounded-full transition-colors ${
                  settings.notification_preferences?.line_movement_alerts ? 'bg-brand-primary' : 'bg-zinc-600'
                }`}
                data-testid="line-movement-toggle"
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                  settings.notification_preferences?.line_movement_alerts ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </div>
            <p className="text-text-muted text-sm mb-3">
              Get notified when odds move significantly on your active picks
            </p>
            {settings.notification_preferences?.line_movement_alerts && (
              <div className="mt-3 pt-3 border-t border-zinc-700">
                <label className="flex items-center justify-between mb-2">
                  <span className="text-text-muted text-sm">Alert threshold</span>
                  <span className="font-mono text-brand-primary">{settings.notification_preferences?.line_movement_threshold || 5}%</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="15"
                  step="0.5"
                  value={settings.notification_preferences?.line_movement_threshold || 5}
                  onChange={(e) => handleNotificationChange('line_movement_threshold', parseFloat(e.target.value))}
                  className="w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-brand-primary"
                  data-testid="threshold-slider"
                />
                <div className="flex justify-between text-xs text-text-muted mt-1">
                  <span>1%</span>
                  <span>5%</span>
                  <span>10%</span>
                  <span>15%</span>
                </div>
              </div>
            )}
          </div>

          {/* Result Alerts */}
          <div className="flex items-center justify-between p-4 bg-zinc-800 rounded-lg">
            <div className="flex items-center gap-2">
              <Check className="w-4 h-4 text-semantic-success" />
              <div>
                <p className="text-text-primary font-semibold">Result Notifications</p>
                <p className="text-text-muted text-sm">Get notified when picks win or lose</p>
              </div>
            </div>
            <button
              onClick={() => handleNotificationChange('result_alerts', !settings.notification_preferences?.result_alerts)}
              className={`w-12 h-6 rounded-full transition-colors ${
                settings.notification_preferences?.result_alerts ? 'bg-brand-primary' : 'bg-zinc-600'
              }`}
              data-testid="result-alerts-toggle"
            >
              <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                settings.notification_preferences?.result_alerts ? 'translate-x-6' : 'translate-x-0.5'
              }`} />
            </button>
          </div>

          {/* Daily Summary */}
          <div className="flex items-center justify-between p-4 bg-zinc-800 rounded-lg">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-brand-secondary" />
              <div>
                <p className="text-text-primary font-semibold">Daily Summary</p>
                <p className="text-text-muted text-sm">Get a daily recap of algorithm activity</p>
              </div>
            </div>
            <button
              onClick={() => handleNotificationChange('daily_summary', !settings.notification_preferences?.daily_summary)}
              className={`w-12 h-6 rounded-full transition-colors ${
                settings.notification_preferences?.daily_summary ? 'bg-brand-primary' : 'bg-zinc-600'
              }`}
              data-testid="daily-summary-toggle"
            >
              <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                settings.notification_preferences?.daily_summary ? 'translate-x-6' : 'translate-x-0.5'
              }`} />
            </button>
          </div>
        </div>
      </div>

      {/* Info Card */}
      <div className="stat-card bg-zinc-800/50">
        <h3 className="text-text-primary font-semibold mb-2">About Notifications</h3>
        <ul className="text-text-muted text-sm space-y-2">
          <li className="flex items-start gap-2">
            <TrendingUp className="w-4 h-4 mt-0.5 text-semantic-success flex-shrink-0" />
            <span><strong>Line Movement:</strong> Alerts when odds shift beyond your threshold on active picks</span>
          </li>
          <li className="flex items-start gap-2">
            <Check className="w-4 h-4 mt-0.5 text-brand-primary flex-shrink-0" />
            <span><strong>Results:</strong> Updates when games finish and picks are graded</span>
          </li>
          <li className="flex items-start gap-2">
            <Clock className="w-4 h-4 mt-0.5 text-brand-secondary flex-shrink-0" />
            <span><strong>Daily Summary:</strong> Overview of picks and performance each day</span>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default Settings;
