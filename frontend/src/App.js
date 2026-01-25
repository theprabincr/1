import { useState, useEffect, useCallback } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, NavLink, useNavigate } from "react-router-dom";
import axios from "axios";
import { 
  Home, TrendingUp, BarChart3, Trophy, 
  Activity, Settings, ChevronRight, RefreshCw,
  Calendar, Clock, Zap, Target, DollarSign, Wifi,
  Bell, Globe
} from "lucide-react";

// Pages
import Dashboard from "./pages/Dashboard";
import Events from "./pages/Events";
import LineMovement from "./pages/LineMovement";
import OddsComparison from "./pages/OddsComparison";
import Predictions from "./pages/Predictions";
import Performance from "./pages/Performance";
import SettingsPage from "./pages/Settings";
import Notifications from "./pages/Notifications";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND_URL}/api`;

// Navigation items
const navItems = [
  { path: "/", icon: Home, label: "Dashboard" },
  { path: "/events", icon: Calendar, label: "Events" },
  { path: "/line-movement", icon: TrendingUp, label: "Line Movement" },
  { path: "/odds-comparison", icon: BarChart3, label: "Odds Compare" },
  { path: "/predictions", icon: Target, label: "Predictions" },
  { path: "/performance", icon: Trophy, label: "Performance" },
  { path: "/settings", icon: Settings, label: "Settings" },
];

// Sidebar Component with API Usage and Notifications
const Sidebar = () => {
  const [apiUsage, setApiUsage] = useState({ requests_remaining: null, total_remaining_all_keys: null });
  const [unreadNotifications, setUnreadNotifications] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [usageRes, notifRes] = await Promise.all([
          axios.get(`${API}/api-usage`),
          axios.get(`${API}/notifications?unread_only=true&limit=1`)
        ]);
        setApiUsage(usageRes.data);
        setUnreadNotifications(notifRes.data.unread_count);
      } catch (error) {
        console.error("Error fetching sidebar data:", error);
      }
    };
    
    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  return (
    <aside className="sidebar" data-testid="sidebar">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-brand-primary flex items-center justify-center">
            <Zap className="w-6 h-6 text-zinc-950" />
          </div>
          <div>
            <h1 className="font-mono font-bold text-lg text-text-primary">BetPredictor</h1>
            <p className="text-xs text-text-muted">AI-Powered Picks</p>
          </div>
        </div>

        {/* Notifications Link */}
        <NavLink
          to="/notifications"
          className={({ isActive }) =>
            `flex items-center gap-3 p-3 rounded-lg mb-4 transition-all ${
              isActive 
                ? 'bg-brand-primary/20 text-brand-primary' 
                : 'bg-zinc-800 text-text-secondary hover:bg-zinc-700'
            }`
          }
        >
          <Bell className="w-5 h-5" />
          <span className="font-medium">Notifications</span>
          {unreadNotifications > 0 && (
            <span className="ml-auto px-2 py-0.5 rounded-full text-xs bg-brand-primary text-zinc-950 font-bold">
              {unreadNotifications}
            </span>
          )}
        </NavLink>
        
        <nav className="space-y-1 max-h-[calc(100vh-380px)] overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === "/"}
              className={({ isActive }) =>
                `nav-item ${isActive ? "active" : ""}`
              }
              data-testid={`nav-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
            >
              <item.icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </div>
      
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-zinc-800 space-y-2">
        <div className="flex items-center gap-3 text-sm">
          <Activity className="w-4 h-4 text-semantic-success" />
          <span className="text-text-muted">API: <span className="text-semantic-success">Online</span></span>
        </div>
        <div className="flex items-center gap-3 text-sm" data-testid="api-usage">
          <Wifi className="w-4 h-4 text-brand-primary" />
          <span className="text-text-muted">
            Active: <span className={`font-mono font-bold ${
              apiUsage.requests_remaining > 100 ? 'text-semantic-success' :
              apiUsage.requests_remaining > 20 ? 'text-semantic-warning' : 'text-semantic-danger'
            }`}>
              {apiUsage.requests_remaining !== null ? apiUsage.requests_remaining : '---'}
            </span>
          </span>
        </div>
        {apiUsage.total_remaining_all_keys > 0 && (
          <div className="flex items-center gap-3 text-sm">
            <Key className="w-4 h-4 text-brand-secondary" />
            <span className="text-text-muted">
              Total: <span className="font-mono text-brand-secondary">
                {apiUsage.total_remaining_all_keys}
              </span>
            </span>
          </div>
        )}
      </div>
    </aside>
  );
};

// Mobile Navigation
const MobileNav = () => {
  return (
    <nav className="mobile-nav md:hidden" data-testid="mobile-nav">
      {navItems.slice(0, 5).map((item) => (
        <NavLink
          key={item.path}
          to={item.path}
          end={item.path === "/"}
          className={({ isActive }) =>
            `flex-1 flex flex-col items-center py-3 ${
              isActive ? "text-brand-primary" : "text-text-muted"
            }`
          }
          data-testid={`mobile-nav-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
        >
          <item.icon className="w-5 h-5" />
          <span className="text-xs mt-1">{item.label.split(' ')[0]}</span>
        </NavLink>
      ))}
    </nav>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <div className="main-layout">
          <Sidebar />
          <main className="main-content pb-20 md:pb-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/events" element={<Events />} />
              <Route path="/line-movement" element={<LineMovement />} />
              <Route path="/odds-comparison" element={<OddsComparison />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/performance" element={<Performance />} />
              <Route path="/api-keys" element={<ApiKeys />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/notifications" element={<Notifications />} />
            </Routes>
          </main>
          <MobileNav />
        </div>
      </BrowserRouter>
    </div>
  );
}

export default App;
