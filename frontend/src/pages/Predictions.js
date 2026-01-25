import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import axios from "axios";
import { 
  Target, RefreshCw, Zap, Brain, AlertCircle,
  CheckCircle, XCircle, Clock, TrendingUp
} from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// AI Analysis Card
const AnalysisCard = ({ title, analysis, icon: Icon, color }) => {
  const colorClasses = {
    blue: "text-brand-secondary",
    purple: "text-purple-400",
    lime: "text-brand-primary"
  };

  return (
    <div className="stat-card" data-testid={`analysis-${title.toLowerCase()}`}>
      <div className="flex items-center gap-3 mb-4">
        <div className={`p-2 rounded-lg bg-zinc-800 ${colorClasses[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
        <h3 className="font-mono font-bold text-text-primary">{title}</h3>
      </div>
      
      <div className="prose prose-sm prose-invert max-w-none">
        <p className="text-text-secondary leading-relaxed whitespace-pre-wrap">
          {analysis || "Analysis not available"}
        </p>
      </div>
    </div>
  );
};

// Prediction Card
const PredictionCard = ({ prediction, onUpdateResult }) => {
  const confidence = (prediction.confidence * 100).toFixed(0);
  
  const statusColors = {
    pending: "badge-pending",
    win: "badge-win",
    loss: "badge-loss",
    push: "badge-push"
  };

  const statusIcons = {
    pending: Clock,
    win: CheckCircle,
    loss: XCircle,
    push: AlertCircle
  };

  const StatusIcon = statusIcons[prediction.result] || Clock;

  return (
    <div className="event-card" data-testid={`prediction-${prediction.id}`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <span className="text-xs font-mono text-text-muted uppercase">
            {prediction.sport_key?.replace(/_/g, ' ')}
          </span>
          <h4 className="text-text-primary font-semibold mt-1">
            {prediction.home_team} vs {prediction.away_team}
          </h4>
        </div>
        <span className={`px-2 py-1 rounded text-xs font-bold flex items-center gap-1 ${statusColors[prediction.result]}`}>
          <StatusIcon className="w-3 h-3" />
          {prediction.result?.toUpperCase()}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <p className="text-xs text-text-muted">Pick</p>
          <p className="font-mono font-bold text-brand-primary">{prediction.predicted_outcome}</p>
        </div>
        <div>
          <p className="text-xs text-text-muted">Odds</p>
          <p className={`font-mono font-bold ${
            prediction.odds_at_prediction > 0 ? 'text-semantic-success' : 'text-text-primary'
          }`}>
            {prediction.odds_at_prediction > 0 ? '+' : ''}{prediction.odds_at_prediction}
          </p>
        </div>
        <div>
          <p className="text-xs text-text-muted">Confidence</p>
          <p className={`font-mono font-bold ${
            confidence >= 70 ? 'text-semantic-success' :
            confidence >= 50 ? 'text-semantic-warning' : 'text-text-muted'
          }`}>
            {confidence}%
          </p>
        </div>
      </div>

      <div className="p-3 bg-zinc-800/50 rounded-lg mb-4">
        <p className="text-xs text-text-muted mb-1">AI Model: {prediction.ai_model}</p>
        <p className="text-sm text-text-secondary line-clamp-3">{prediction.analysis}</p>
      </div>

      {prediction.result === 'pending' && (
        <div className="flex gap-2">
          <button
            onClick={() => onUpdateResult(prediction.id, 'win')}
            className="flex-1 btn-outline text-semantic-success border-semantic-success/30 hover:bg-semantic-success/10 text-sm"
            data-testid={`mark-win-${prediction.id}`}
          >
            <CheckCircle className="w-4 h-4 inline mr-1" />
            Won
          </button>
          <button
            onClick={() => onUpdateResult(prediction.id, 'loss')}
            className="flex-1 btn-outline text-semantic-danger border-semantic-danger/30 hover:bg-semantic-danger/10 text-sm"
            data-testid={`mark-loss-${prediction.id}`}
          >
            <XCircle className="w-4 h-4 inline mr-1" />
            Lost
          </button>
          <button
            onClick={() => onUpdateResult(prediction.id, 'push')}
            className="flex-1 btn-outline text-text-muted text-sm"
            data-testid={`mark-push-${prediction.id}`}
          >
            Push
          </button>
        </div>
      )}
    </div>
  );
};

const Predictions = () => {
  const location = useLocation();
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [selectedSport, setSelectedSport] = useState("basketball_nba");
  const [events, setEvents] = useState([]);
  const [selectedEvent, setSelectedEvent] = useState(location.state?.analyzeEvent || null);

  const sports = [
    { key: "basketball_nba", label: "NBA" },
    { key: "americanfootball_nfl", label: "NFL" },
    { key: "baseball_mlb", label: "MLB" },
    { key: "icehockey_nhl", label: "NHL" },
    { key: "soccer_epl", label: "EPL" },
  ];

  const fetchPredictions = async () => {
    try {
      const response = await axios.get(`${API}/recommendations?limit=20`);
      setPredictions(response.data);
    } catch (error) {
      console.error("Error fetching predictions:", error);
    }
  };

  const fetchEvents = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/events/${selectedSport}`);
      setEvents(response.data);
    } catch (error) {
      console.error("Error fetching events:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
    fetchEvents();
  }, [selectedSport]);

  useEffect(() => {
    // Auto-analyze if coming from Events page
    if (location.state?.analyzeEvent) {
      setSelectedEvent(location.state.analyzeEvent);
      handleAnalyze(location.state.analyzeEvent);
    }
  }, [location.state]);

  const handleAnalyze = async (event) => {
    if (!event) return;
    
    setAnalyzing(true);
    setAnalysis(null);
    
    try {
      const response = await axios.post(`${API}/analyze`, {
        event_id: event.id,
        home_team: event.home_team,
        away_team: event.away_team,
        sport_key: selectedSport,
        odds_data: {
          bookmakers: event.bookmakers || []
        }
      });
      
      setAnalysis(response.data);
    } catch (error) {
      console.error("Error analyzing:", error);
      setAnalysis({
        error: "Analysis failed. Please try again.",
        gpt_analysis: "Unable to generate analysis at this time.",
        claude_analysis: "Unable to generate analysis at this time."
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const handleGeneratePicks = async () => {
    setAnalyzing(true);
    try {
      await axios.post(`${API}/generate-recommendations?sport_key=${selectedSport}`);
      await fetchPredictions();
    } catch (error) {
      console.error("Error generating picks:", error);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleUpdateResult = async (predictionId, result) => {
    try {
      await axios.put(`${API}/result`, {
        prediction_id: predictionId,
        result: result
      });
      await fetchPredictions();
    } catch (error) {
      console.error("Error updating result:", error);
    }
  };

  return (
    <div className="space-y-6" data-testid="predictions-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono font-bold text-2xl text-text-primary">AI Predictions</h1>
          <p className="text-text-muted text-sm mt-1">AI-powered bet recommendations</p>
        </div>
        <button 
          onClick={handleGeneratePicks}
          disabled={analyzing}
          className="btn-primary flex items-center gap-2"
          data-testid="generate-picks-btn"
        >
          {analyzing ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Zap className="w-4 h-4" />
          )}
          Generate Picks
        </button>
      </div>

      {/* Sport Filters */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {sports.map(sport => (
          <button
            key={sport.key}
            onClick={() => setSelectedSport(sport.key)}
            className={`px-4 py-2 rounded-lg font-mono text-sm whitespace-nowrap transition-all ${
              selectedSport === sport.key
                ? "bg-brand-primary text-zinc-950 font-bold"
                : "bg-zinc-800 text-text-secondary hover:bg-zinc-700"
            }`}
            data-testid={`pred-sport-${sport.key}`}
          >
            {sport.label}
          </button>
        ))}
      </div>

      {/* Event Analyzer */}
      <div className="stat-card">
        <h3 className="font-mono font-bold text-lg text-text-primary mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5 text-brand-secondary" />
          Analyze Event
        </h3>
        
        <div className="flex gap-4">
          <select
            value={selectedEvent?.id || ''}
            onChange={(e) => {
              const event = events.find(ev => ev.id === e.target.value);
              setSelectedEvent(event);
            }}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 text-text-primary font-mono focus:outline-none focus:ring-2 focus:ring-brand-primary"
            data-testid="analyze-event-selector"
          >
            <option value="">Select an event...</option>
            {events.map(event => (
              <option key={event.id} value={event.id}>
                {event.home_team} vs {event.away_team}
              </option>
            ))}
          </select>
          
          <button
            onClick={() => handleAnalyze(selectedEvent)}
            disabled={!selectedEvent || analyzing}
            className="btn-primary flex items-center gap-2"
            data-testid="run-analysis-btn"
          >
            {analyzing ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Zap className="w-4 h-4" />
            )}
            Analyze
          </button>
        </div>
      </div>

      {/* Analysis Results */}
      {analyzing && (
        <div className="stat-card text-center py-8">
          <RefreshCw className="w-12 h-12 text-brand-primary mx-auto mb-4 animate-spin" />
          <p className="text-text-secondary">Running AI analysis...</p>
          <p className="text-text-muted text-sm mt-2">This may take a moment</p>
        </div>
      )}

      {analysis && !analyzing && (
        <div className="space-y-4">
          <h3 className="font-mono font-bold text-lg text-text-primary">
            Analysis: {analysis.home_team} vs {analysis.away_team}
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <AnalysisCard 
              title="GPT-5.2 Analysis"
              analysis={analysis.gpt_analysis}
              icon={Brain}
              color="blue"
            />
            <AnalysisCard 
              title="Claude Analysis"
              analysis={analysis.claude_analysis}
              icon={Target}
              color="purple"
            />
          </div>
        </div>
      )}

      {/* Predictions List */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-mono font-bold text-lg text-text-primary flex items-center gap-2">
            <Target className="w-5 h-5 text-brand-primary" />
            Active Predictions
          </h3>
          <button 
            onClick={fetchPredictions}
            className="text-text-muted hover:text-text-primary"
            data-testid="refresh-predictions-btn"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        {predictions.length > 0 ? (
          <div className="grid md:grid-cols-2 gap-4">
            {predictions.map((prediction) => (
              <PredictionCard 
                key={prediction.id} 
                prediction={prediction}
                onUpdateResult={handleUpdateResult}
              />
            ))}
          </div>
        ) : (
          <div className="stat-card text-center py-12">
            <AlertCircle className="w-12 h-12 text-text-muted mx-auto mb-4" />
            <h3 className="text-text-primary font-semibold mb-2">No Predictions Yet</h3>
            <p className="text-text-muted text-sm mb-4">
              Generate AI-powered picks to get started
            </p>
            <button 
              onClick={handleGeneratePicks}
              className="btn-primary"
              data-testid="empty-generate-btn"
            >
              Generate Picks
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Predictions;
