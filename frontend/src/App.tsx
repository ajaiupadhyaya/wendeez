import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import axios from 'axios';
import Dashboard from './components/Dashboard';
import PlayerAnalysis from './components/PlayerAnalysis';
import MatchupAnalyzer from './components/MatchupAnalyzer';
import LineupOptimizer from './components/LineupOptimizer';
import Rankings from './components/Rankings';
import './index.css';

// Configure axios
axios.defaults.baseURL = 'http://localhost:8000';

function App() {
  const [healthStatus, setHealthStatus] = useState<any>(null);

  useEffect(() => {
    // Check API health on load
    axios.get('/api/health')
      .then(res => setHealthStatus(res.data))
      .catch(err => console.error('API health check failed:', err));
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <nav className="bg-white shadow-lg">
          <div className="max-w-7xl mx-auto px-4">
            <div className="flex justify-between h-16">
              <div className="flex space-x-8">
                <div className="flex items-center">
                  <span className="text-xl font-bold text-primary-600">
                    🏈 Elite Fantasy Predictor
                  </span>
                </div>
                <div className="hidden md:flex items-center space-x-4">
                  <Link to="/" className="text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
                    Dashboard
                  </Link>
                  <Link to="/players" className="text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
                    Player Analysis
                  </Link>
                  <Link to="/matchups" className="text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
                    Matchups
                  </Link>
                  <Link to="/lineup" className="text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
                    Lineup Optimizer
                  </Link>
                  <Link to="/rankings" className="text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
                    Rankings
                  </Link>
                </div>
              </div>
              <div className="flex items-center">
                {healthStatus && (
                  <span className={`text-sm ${healthStatus.status === 'healthy' ? 'text-green-600' : 'text-red-600'}`}>
                    API: {healthStatus.status}
                  </span>
                )}
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto py-6 px-4">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/players" element={<PlayerAnalysis />} />
            <Route path="/matchups" element={<MatchupAnalyzer />} />
            <Route path="/lineup" element={<LineupOptimizer />} />
            <Route path="/rankings" element={<Rankings />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;