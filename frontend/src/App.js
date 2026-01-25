import { useState, useEffect, useCallback } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, NavLink, useNavigate } from "react-router-dom";
import axios from "axios";
import { 
  Home, TrendingUp, BarChart3, Trophy, 
  Activity, Settings, ChevronRight, RefreshCw,
  Calendar, Clock, Zap, Target, DollarSign, Wifi
} from "lucide-react";

// Pages
import Dashboard from "./pages/Dashboard";
import Events from "./pages/Events";
import LineMovement from "./pages/LineMovement";
import OddsComparison from "./pages/OddsComparison";
import Predictions from "./pages/Predictions";
import Performance from "./pages/Performance";

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
];

// Sidebar Component with API Usage
const Sidebar = () => {
  const [apiUsage, setApiUsage] = useState({ requests_remaining: null });

  useEffect(() => {
    const fetchApiUsage = async () => {
      try {
        const response = await axios.get(`${API}/api-usage`);
        setApiUsage(response.data);
      } catch (error) {
        console.error("Error fetching API usage:", error);
      }
    };
    
    fetchApiUsage();
    // Refresh every 5 minutes
    const interval = setInterval(fetchApiUsage, 300000);
    return () => clearInterval(interval);
  }, []);

  return (
    <aside className="sidebar" data-testid="sidebar">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-lg bg-brand-primary flex items-center justify-center">
            <Zap className="w-6 h-6 text-zinc-950" />
          </div>
          <div>
            <h1 className="font-mono font-bold text-lg text-text-primary">BetPredictor</h1>
            <p className="text-xs text-text-muted">AI-Powered Picks</p>
          </div>
        </div>
        
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
      
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-zinc-800 space-y-3">
        <div className="flex items-center gap-3 text-sm">
          <Activity className="w-4 h-4 text-semantic-success" />
          <span className="text-text-muted">API Status: <span className="text-semantic-success">Online</span></span>
        </div>
        <div className="flex items-center gap-3 text-sm" data-testid="api-usage">
          <Wifi className="w-4 h-4 text-brand-primary" />
          <span className="text-text-muted">
            Calls Left: <span className={`font-mono font-bold ${
              apiUsage.requests_remaining > 100 ? 'text-semantic-success' :
              apiUsage.requests_remaining > 20 ? 'text-semantic-warning' : 'text-semantic-danger'
            }`}>
              {apiUsage.requests_remaining !== null ? apiUsage.requests_remaining : '---'}
            </span>
          </span>
        </div>
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
            </Routes>
          </main>
          <MobileNav />
        </div>
      </BrowserRouter>
    </div>
  );
}

export default App;
