import { useState, useEffect, useCallback } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, NavLink, useNavigate } from "react-router-dom";
import axios from "axios";
import { 
  Home, TrendingUp, Trophy, 
  Settings, ChevronRight, RefreshCw,
  Calendar, Clock, Zap, DollarSign, Wifi,
  Bell, Activity, Circle
} from "lucide-react";

// Ballzy Logo Component - Modern minimalist design
const BallzyLogo = ({ className = "w-10 h-10" }) => (
  <svg className={className} viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    {/* Outer ring - represents precision/targeting */}
    <circle cx="24" cy="24" r="22" stroke="#CCFF00" strokeWidth="2" fill="none" opacity="0.3"/>
    {/* Main circle with gradient */}
    <circle cx="24" cy="24" r="18" fill="url(#ballzyGrad)"/>
    {/* Stylized "B" made of geometric shapes */}
    <path d="M18 12 L18 36" stroke="#09090B" strokeWidth="3" strokeLinecap="round"/>
    <path d="M18 12 L26 12 C30 12 32 14 32 17 C32 20 30 22 26 22 L18 22" stroke="#09090B" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
    <path d="M18 22 L28 22 C32 22 34 25 34 28 C34 32 32 36 28 36 L18 36" stroke="#09090B" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
    {/* Accent dot - represents the ball */}
    <circle cx="38" cy="10" r="4" fill="#CCFF00"/>
    <defs>
      <linearGradient id="ballzyGrad" x1="6" y1="6" x2="42" y2="42">
        <stop offset="0%" stopColor="#CCFF00"/>
        <stop offset="100%" stopColor="#99CC00"/>
      </linearGradient>
    </defs>
  </svg>
);

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

// Sidebar Component
const Sidebar = () => {
  const [unreadNotifications, setUnreadNotifications] = useState(0);
  const [dataStatus, setDataStatus] = useState({ source: 'live', lastUpdate: null });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [notifRes, statusRes] = await Promise.all([
          axios.get(`${API}/notifications?unread_only=true&limit=1`),
          axios.get(`${API}/data-source-status`)
        ]);
        setUnreadNotifications(notifRes.data.unread_count);
        setDataStatus(statusRes.data);
      } catch (error) {
        console.error("Error fetching sidebar data:", error);
      }
    };
    
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <aside className="sidebar" data-testid="sidebar">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-6">
          <BallzyLogo className="w-10 h-10" />
          <div>
            <h1 className="ballzy-logo-text text-xl text-text-primary">Ballzy</h1>
            <p className="text-xs text-text-muted font-medium">Smart Betting AI</p>
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
      </div>
      
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-zinc-800 space-y-2">
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
