import { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';

interface Player {
  player_id: string;
  name: string;
  position: string;
  team: string;
}

interface PlayerStats {
  player_id: string;
  name: string;
  position: string;
  team: string;
  games_played: number;
  avg_points: number;
  total_points: number;
  last_game_points: number;
  consistency_score: number;
  trend: string;
}

export default function PlayerAnalysis() {
  const [searchTerm, setSearchTerm] = useState('');
  const [players, setPlayers] = useState<Player[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
  const [playerStats, setPlayerStats] = useState<PlayerStats | null>(null);
  const [playerTrends, setPlayerTrends] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [week, setWeek] = useState(1);

  useEffect(() => {
    if (searchTerm.length >= 2) {
      searchPlayers();
    }
  }, [searchTerm]);

  const searchPlayers = async () => {
    try {
      const res = await axios.get(`/api/players?search=${searchTerm}`);
      setPlayers(res.data);
    } catch (error) {
      console.error('Error searching players:', error);
    }
  };

  const selectPlayer = async (player: Player) => {
    setSelectedPlayer(player);
    setLoading(true);

    try {
      // Get player stats
      const statsRes = await axios.get(`/api/players/${player.player_id}/stats`);
      setPlayerStats(statsRes.data);

      // Get player trends
      const trendsRes = await axios.get(`/api/trends/${player.player_id}`);
      setPlayerTrends(trendsRes.data);

      // Get prediction for current week
      const predRes = await axios.post('/api/predict', {
        player_id: player.player_id,
        week: week,
        season: 2025
      });
      setPrediction(predRes.data);
    } catch (error) {
      console.error('Error fetching player data:', error);
    } finally {
      setLoading(false);
    }
  };

  const updatePrediction = async () => {
    if (!selectedPlayer) return;
    
    setLoading(true);
    try {
      const predRes = await axios.post('/api/predict', {
        player_id: selectedPlayer.player_id,
        week: week,
        season: 2025
      });
      setPrediction(predRes.data);
    } catch (error) {
      console.error('Error updating prediction:', error);
    } finally {
      setLoading(false);
    }
  };

  const trendChartData = playerTrends ? {
    labels: playerTrends.data.map((d: any) => `W${d.week}`),
    datasets: [
      {
        label: 'Fantasy Points',
        data: playerTrends.data.map((d: any) => d.fantasy_points),
        borderColor: 'rgb(14, 165, 233)',
        backgroundColor: 'rgba(14, 165, 233, 0.1)',
        tension: 0.1,
      },
      {
        label: 'Average',
        data: playerTrends.data.map(() => playerTrends.avg),
        borderColor: 'rgb(156, 163, 175)',
        borderDash: [5, 5],
        pointRadius: 0,
      },
    ],
  } : null;

  return (
    <div className="space-y-6">
      {/* Search Section */}
      <div className="bg-white shadow rounded-lg p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Player Analysis</h1>
        <div className="max-w-xl">
          <label htmlFor="search" className="block text-sm font-medium text-gray-700">
            Search Players
          </label>
          <input
            type="text"
            id="search"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            placeholder="Enter player name..."
          />
          
          {/* Search Results */}
          {players.length > 0 && !selectedPlayer && (
            <div className="mt-2 bg-white border border-gray-200 rounded-md shadow-lg max-h-60 overflow-y-auto">
              {players.map((player) => (
                <button
                  key={player.player_id}
                  onClick={() => selectPlayer(player)}
                  className="w-full text-left px-4 py-2 hover:bg-gray-100 border-b border-gray-100"
                >
                  <div className="font-medium">{player.name}</div>
                  <div className="text-sm text-gray-500">
                    {player.position} - {player.team}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Player Details */}
      {selectedPlayer && (
        <>
          {/* Player Info Card */}
          <div className="bg-white shadow rounded-lg p-6">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">{selectedPlayer.name}</h2>
                <p className="text-gray-500">{selectedPlayer.position} - {selectedPlayer.team}</p>
              </div>
              <button
                onClick={() => {
                  setSelectedPlayer(null);
                  setPlayerStats(null);
                  setPlayerTrends(null);
                  setPrediction(null);
                  setSearchTerm('');
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>

            {loading ? (
              <div className="flex justify-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
              </div>
            ) : (
              <>
                {/* Stats Grid */}
                {playerStats && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div>
                      <p className="text-sm text-gray-500">Games Played</p>
                      <p className="text-xl font-semibold">{playerStats.games_played}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Avg Points</p>
                      <p className="text-xl font-semibold">{playerStats.avg_points.toFixed(1)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Last Game</p>
                      <p className="text-xl font-semibold">{playerStats.last_game_points.toFixed(1)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Consistency</p>
                      <p className="text-xl font-semibold">{playerStats.consistency_score.toFixed(2)}</p>
                    </div>
                  </div>
                )}

                {/* Trend Badge */}
                {playerStats && (
                  <div className="mb-6">
                    <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                      playerStats.trend === 'IMPROVING' ? 'bg-green-100 text-green-800' :
                      playerStats.trend === 'DECLINING' ? 'bg-red-100 text-red-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      Trend: {playerStats.trend}
                    </span>
                  </div>
                )}

                {/* Performance Chart */}
                {trendChartData && (
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold mb-2">Recent Performance</h3>
                    <Line 
                      data={trendChartData}
                      options={{
                        responsive: true,
                        plugins: {
                          legend: {
                            position: 'top' as const,
                          },
                        },
                        scales: {
                          y: {
                            beginAtZero: true,
                          },
                        },
                      }}
                    />
                  </div>
                )}
              </>
            )}
          </div>

          {/* Week Prediction */}
          <div className="bg-white shadow rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4">Week {week} Prediction</h3>
            
            <div className="flex items-center space-x-4 mb-4">
              <select
                value={week}
                onChange={(e) => setWeek(Number(e.target.value))}
                className="rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              >
                {[...Array(18)].map((_, i) => (
                  <option key={i + 1} value={i + 1}>Week {i + 1}</option>
                ))}
              </select>
              <button
                onClick={updatePrediction}
                className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
              >
                Update Prediction
              </button>
            </div>

            {prediction && (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <p className="text-sm text-gray-500">Projected</p>
                    <p className="text-3xl font-bold text-primary-600">
                      {prediction.predicted_points.toFixed(1)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-500">Floor</p>
                    <p className="text-2xl font-semibold text-red-600">
                      {prediction.floor.toFixed(1)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-500">Ceiling</p>
                    <p className="text-2xl font-semibold text-green-600">
                      {prediction.ceiling.toFixed(1)}
                    </p>
                  </div>
                </div>

                <div className="flex justify-center space-x-4">
                  <span className={`px-4 py-2 rounded-full text-sm font-semibold ${
                    prediction.recommendation === 'MUST START' ? 'bg-green-100 text-green-800' :
                    prediction.recommendation === 'START' ? 'bg-blue-100 text-blue-800' :
                    prediction.recommendation === 'FLEX' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {prediction.recommendation}
                  </span>
                  <span className={`px-4 py-2 rounded-full text-sm font-semibold ${
                    prediction.risk_level === 'LOW' ? 'bg-green-100 text-green-800' :
                    prediction.risk_level === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    Risk: {prediction.risk_level}
                  </span>
                </div>

                {/* Confidence Range Visualization */}
                <div className="mt-4">
                  <p className="text-sm text-gray-500 mb-2">80% Confidence Range</p>
                  <div className="relative h-8 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="absolute h-full bg-primary-400"
                      style={{
                        left: `${(prediction.confidence_80_lower / (prediction.ceiling * 1.2)) * 100}%`,
                        width: `${((prediction.confidence_80_upper - prediction.confidence_80_lower) / (prediction.ceiling * 1.2)) * 100}%`,
                      }}
                    />
                    <div 
                      className="absolute h-full w-1 bg-primary-800"
                      style={{
                        left: `${(prediction.predicted_points / (prediction.ceiling * 1.2)) * 100}%`,
                      }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>0</span>
                    <span>{(prediction.ceiling * 1.2).toFixed(0)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}