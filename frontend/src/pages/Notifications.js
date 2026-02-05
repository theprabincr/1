import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { 
  Bell, Check, Trash2, RefreshCw, X,
  TrendingUp, Trophy, Zap, Target, BarChart3, Cpu, AlertTriangle
} from "lucide-react";
import { useNotifications } from "../App";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Confirmation Modal Component
const ConfirmModal = ({ isOpen, onClose, onConfirm, title, message }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-full bg-semantic-danger/20">
            <AlertTriangle className="w-6 h-6 text-semantic-danger" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-bold text-text-primary mb-2">{title}</h3>
            <p className="text-text-muted text-sm">{message}</p>
          </div>
        </div>
        
        <div className="flex gap-3 mt-6 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-zinc-800 text-text-secondary hover:bg-zinc-700 transition-colors font-medium"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 rounded-lg bg-semantic-danger text-white hover:bg-red-600 transition-colors font-medium flex items-center gap-2"
          >
            <Trash2 className="w-4 h-4" />
            Clear All
          </button>
        </div>
      </div>
    </div>
  );
};

const NotificationIcon = ({ type }) => {
  const icons = {
    'new_pick': <Target className="w-5 h-5 text-brand-primary" />,
    'result': <Trophy className="w-5 h-5 text-brand-primary" />,
    'line_movement': <TrendingUp className="w-5 h-5 text-semantic-success" />,
    'daily_summary': <BarChart3 className="w-5 h-5 text-blue-400" />,
    'ml_retrain': <Cpu className="w-5 h-5 text-purple-400" />,
    'recommendation': <Zap className="w-5 h-5 text-brand-primary" />,
  };
  return icons[type] || <Bell className="w-5 h-5 text-text-muted" />;
};

const Notifications = () => {
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showUnreadOnly, setShowUnreadOnly] = useState(false);
  const [showConfirmModal, setShowConfirmModal] = useState(false);

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

  const handleClearAllClick = () => {
    setShowConfirmModal(true);
  };

  const handleConfirmClearAll = async () => {
    try {
      await axios.delete(`${API}/notifications`);
      fetchNotifications();
    } catch (error) {
      console.error("Error clearing notifications:", error);
    } finally {
      setShowConfirmModal(false);
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
      {/* Confirmation Modal */}
      <ConfirmModal
        isOpen={showConfirmModal}
        onClose={() => setShowConfirmModal(false)}
        onConfirm={handleConfirmClearAll}
        title="Clear All Notifications"
        message="Are you sure you want to delete all notifications? This action cannot be undone."
      />

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
            New picks, results, line movements & daily summaries
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
          {notifications.length > 0 && (
            <button
              onClick={handleClearAllClick}
              className="btn-outline flex items-center gap-2 text-semantic-danger border-semantic-danger/50 hover:bg-semantic-danger/10"
            >
              <Trash2 className="w-4 h-4" />
              Clear All
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
              : "You'll receive notifications when new picks are available, game results come in, and more."}
          </p>
        </div>
      )}
    </div>
  );
};

export default Notifications;
