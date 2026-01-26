import { useState, useEffect, useCallback } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, NavLink, useNavigate } from "react-router-dom";
import axios from "axios";
import { 
  Home, TrendingUp, Trophy, 
  Settings, ChevronRight, RefreshCw,
  Calendar, Clock, Zap, DollarSign, Wifi,
  Bell
} from "lucide-react";

// Pages
import Dashboard from "./pages/Dashboard";
import Events from "./pages/Events";
import LineMovement from "./pages/LineMovement";
import Performance from "./pages/Performance";
import SettingsPage from "./pages/Settings";
import Notifications from "./pages/Notifications";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND_URL}/api`;

// Navigation items - Removed Predictions tab (using algorithmic predictions only)
const navItems = [
  { path: "/", icon: Home, label: "Dashboard" },
  { path: "/events", icon: Calendar, label: "Events" },
  { path: "/line-movement", icon: TrendingUp, label: "Line Movement" },
  { path: "/performance", icon: Trophy, label: "Performance" },
  { path: "/settings", icon: Settings, label: "Settings" },
];

// Sidebar Component with Live Events
const Sidebar = () => {
  const [unreadNotifications, setUnreadNotifications] = useState(0);
  const [dataStatus, setDataStatus] = useState({ source: 'espn', lastUpdate: null });
  const [liveGames, setLiveGames] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [notifRes, statusRes, liveRes] = await Promise.all([
          axios.get(`${API}/notifications?unread_only=true&limit=1`),
          axios.get(`${API}/data-source-status`),
          axios.get(`${API}/live-scores`)
        ]);
        setUnreadNotifications(notifRes.data.unread_count);
        setDataStatus(statusRes.data);
        setLiveGames(liveRes.data.games || []);
      } catch (error) {
        console.error("Error fetching sidebar data:", error);
      }
    };
    
    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10 seconds
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
            <p className="text-xs text-text-muted">Algorithm-Powered</p>
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
        
        <nav className="space-y-1">
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

        {/* Live Games Section */}
        {liveGames.length > 0 && (
          <div className="mt-6 pt-4 border-t border-zinc-800">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-2 h-2 rounded-full bg-semantic-success animate-pulse"></div>
              <span className="text-xs font-mono text-text-muted uppercase">Live Games ({liveGames.length})</span>
            </div>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {liveGames.slice(0, 5).map((game, i) => (
                <div key={game.espn_id || i} className="p-2 bg-zinc-800/50 rounded-lg">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-text-muted truncate max-w-[80px]">{game.away_team?.split(' ').pop()}</span>
                    <span className="font-mono font-bold text-brand-primary">{game.away_score}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-text-muted truncate max-w-[80px]">{game.home_team?.split(' ').pop()}</span>
                    <span className="font-mono font-bold text-brand-primary">{game.home_score}</span>
                  </div>
                  <div className="text-[10px] text-text-muted text-center mt-1">
                    {game.clock || 'LIVE'} {game.period ? `Q${game.period}` : ''}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-zinc-800 space-y-2">
        <div className="flex items-center gap-3 text-sm">
          <Globe className="w-4 h-4 text-brand-primary" />
          <span className="text-text-muted">Source: <span className="text-brand-primary font-medium">ESPN</span></span>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <Activity className="w-4 h-4 text-semantic-success" />
          <span className="text-text-muted">Status: <span className="text-semantic-success">Live</span></span>
        </div>
        {dataStatus.lastUpdate && (
          <div className="flex items-center gap-3 text-sm">
            <Clock className="w-4 h-4 text-text-muted" />
            <span className="text-text-muted text-xs">
              Updated: {new Date(dataStatus.lastUpdate).toLocaleTimeString()}
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
              <Route path="/performance" element={<Performance />} />
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
