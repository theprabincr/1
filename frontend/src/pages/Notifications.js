import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { 
  Bell, Check, Trash2, RefreshCw, X,
  TrendingUp, AlertTriangle, Trophy, Zap, Key
} from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const NotificationIcon = ({ type }) => {
  const icons = {
    'line_movement': <TrendingUp className="w-5 h-5 text-semantic-success" />,
    'api_low': <AlertTriangle className="w-5 h-5 text-semantic-warning" />,
    'api_exhausted': <AlertTriangle className="w-5 h-5 text-semantic-danger" />,
    'api_rotation': <Key className="w-5 h-5 text-brand-primary" />,
    'api_key_added': <Key className="w-5 h-5 text-semantic-success" />,
    'result': <Trophy className="w-5 h-5 text-brand-primary" />,
    'recommendation': <Zap className="w-5 h-5 text-brand-primary" />,
  };
  return icons[type] || <Bell className="w-5 h-5 text-text-muted" />;
};

const Notifications = () => {
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showUnreadOnly, setShowUnreadOnly] = useState(false);

  const fetchNotifications = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/notifications?unread_only=${showUnreadOnly}`);
      setNotifications(response.data.notifications);
      setUnreadCount(response.data.unread_count);
    } catch (error) {
      console.error("Error fetching notifications:", error);
    } finally {
      setLoading(false);
    }
  }, [showUnreadOnly]);

  useEffect(() => {
    fetchNotifications();
    // Poll for new notifications every 30 seconds
    const interval = setInterval(fetchNotifications, 30000);
    return () => clearInterval(interval);
  }, [fetchNotifications]);

  const markAsRead = async (notifId) => {
    try {
      await axios.put(`${API}/notifications/${notifId}/read`);
      fetchNotifications();
    } catch (error) {
      console.error("Error marking notification as read:", error);
    }
  };

  const markAllAsRead = async () => {
    try {
      await axios.put(`${API}/notifications/read-all`);
      fetchNotifications();
    } catch (error) {
      console.error("Error marking all as read:", error);
    }
  };

  const deleteNotification = async (notifId) => {
    try {
      await axios.delete(`${API}/notifications/${notifId}`);
      fetchNotifications();
    } catch (error) {
      console.error("Error deleting notification:", error);
    }
  };

  const clearAllNotifications = async () => {
    if (window.confirm('Are you sure you want to delete all notifications?')) {
      try {
        await axios.delete(`${API}/notifications`);
        fetchNotifications();
      } catch (error) {
        console.error("Error clearing notifications:", error);
      }
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
            <Bell className="w-6 h-6 text-brand-primary" />
            Notifications
            {unreadCount > 0 && (
              <span className="ml-2 px-2 py-0.5 rounded-full text-xs bg-brand-primary text-zinc-950 font-bold">
                {unreadCount}
              </span>
            )}
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Stay updated on line movements, API status, and results
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowUnreadOnly(!showUnreadOnly)}
            className={`btn-outline ${showUnreadOnly ? 'border-brand-primary text-brand-primary' : ''}`}
          >
            {showUnreadOnly ? 'Show All' : 'Unread Only'}
          </button>
          {unreadCount > 0 && (
            <button
              onClick={markAllAsRead}
              className="btn-outline flex items-center gap-2"
            >
              <Check className="w-4 h-4" />
              Mark All Read
            </button>
          )}
        </div>
      </div>

      {/* Notifications List */}
      {notifications.length > 0 ? (
        <div className="space-y-3">
          {notifications.map((notif) => (
            <div
              key={notif.id}
              className={`p-4 rounded-lg border transition-all ${
                notif.read 
                  ? 'bg-zinc-900 border-zinc-800' 
                  : 'bg-zinc-800 border-brand-primary/30'
              }`}
            >
              <div className="flex items-start gap-4">
                <div className="mt-1">
                  <NotificationIcon type={notif.type} />
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className={`font-semibold ${notif.read ? 'text-text-secondary' : 'text-text-primary'}`}>
                      {notif.title}
                    </h3>
                    {!notif.read && (
                      <span className="w-2 h-2 rounded-full bg-brand-primary" />
                    )}
                  </div>
                  <p className="text-text-muted text-sm mb-2">
                    {notif.message}
                  </p>
                  <p className="text-text-muted text-xs">
                    {new Date(notif.created_at).toLocaleString()}
                  </p>
                </div>
                
                <div className="flex items-center gap-1">
                  {!notif.read && (
                    <button
                      onClick={() => markAsRead(notif.id)}
                      className="p-2 rounded-lg hover:bg-zinc-700 text-text-muted hover:text-text-primary"
                      title="Mark as read"
                    >
                      <Check className="w-4 h-4" />
                    </button>
                  )}
                  <button
                    onClick={() => deleteNotification(notif.id)}
                    className="p-2 rounded-lg hover:bg-semantic-danger/20 text-text-muted hover:text-semantic-danger"
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="stat-card text-center py-12">
          <Bell className="w-16 h-16 text-text-muted mx-auto mb-4" />
          <h3 className="text-text-primary font-bold text-lg mb-2">
            {showUnreadOnly ? 'No Unread Notifications' : 'No Notifications Yet'}
          </h3>
          <p className="text-text-muted">
            {showUnreadOnly 
              ? "You're all caught up!"
              : "You'll receive notifications for line movements, API alerts, and results."}
          </p>
        </div>
      )}
    </div>
  );
};

export default Notifications;
