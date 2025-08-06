import { useState, useEffect } from 'react';
import axios from 'axios';

export default function MatchupAnalyzer() {
  const [week, setWeek] = useState(1);
  const [players, setPlayers] = useState<any[]>([]);
  const [selectedPlayers, setSelectedPlayers] = useState<string[]>([]);
  const [matchups, setMatchups] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchPlayers();
  }, []);

  const fetchPlayers = async () => {
    try {
      const res = await axios.get('/api/players?limit=200');
      setPlayers(res.data);
    } catch (error) {
      console.error('Error fetching players:', error);
    }
  };

  const analyzeMatchups = async () => {
    setLoading(true);
    const results = [];

    for (const playerId of selectedPlayers) {
      try {
        const res = await axios.get(`/api/players/${playerId}/matchup/${week}`);
        const player = players.find(p => p.player_id === playerId);
        results.push({
          ...res.data,
          player_name: player?.name,
          position: player?.position,
        });
      } catch (error) {
        console.error(`Error analyzing matchup for ${playerId}:`, error);
      }
    }

    setMatchups(results.sort((a, b) => b.matchup_score - a.matchup_score));
    setLoading(false);
  };

  return (
    <div className="space-y-6">
      <div className="bg-white shadow rounded-lg p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Matchup Analyzer</h1>
        <p className="text-gray-600">Analyze defensive matchups for Week {week}</p>
      </div>

      <div className="bg-white shadow rounded-lg p-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Week</label>
            <select
              value={week}
              onChange={(e) => setWeek(Number(e.target.value))}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            >
              {[...Array(18)].map((_, i) => (
                <option key={i + 1} value={i + 1}>Week {i + 1}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">Select Players</label>
            <select
              multiple
              value={selectedPlayers}
              onChange={(e) => {
                const values = Array.from(e.target.selectedOptions, option => option.value);
                setSelectedPlayers(values);
              }}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              size={10}
            >
              {players.map((player) => (
                <option key={player.player_id} value={player.player_id}>
                  {player.name} ({player.position} - {player.team})
                </option>
              ))}
            </select>
            <p className="mt-1 text-sm text-gray-500">Hold Ctrl/Cmd to select multiple players</p>
          </div>

          <button
            onClick={analyzeMatchups}
            disabled={selectedPlayers.length === 0}
            className="w-full px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:bg-gray-400"
          >
            Analyze Matchups
          </button>
        </div>
      </div>

      {loading && (
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      )}

      {matchups.length > 0 && !loading && (
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Matchup Analysis Results</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Player
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Position
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Opponent
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Matchup Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Difficulty
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Def Rank
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Recommendation
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {matchups.map((matchup, idx) => (
                  <tr key={idx}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {matchup.player_name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {matchup.position}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {matchup.opponent} {matchup.is_home ? '(H)' : '(A)'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              matchup.matchup_score >= 70 ? 'bg-green-500' :
                              matchup.matchup_score >= 40 ? 'bg-yellow-500' :
                              'bg-red-500'
                            }`}
                            style={{ width: `${matchup.matchup_score}%` }}
                          />
                        </div>
                        <span className="ml-2 text-sm">{matchup.matchup_score.toFixed(0)}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        matchup.difficulty === 'easy' ? 'bg-green-100 text-green-800' :
                        matchup.difficulty === 'favorable' ? 'bg-blue-100 text-blue-800' :
                        matchup.difficulty === 'average' ? 'bg-gray-100 text-gray-800' :
                        matchup.difficulty === 'tough' ? 'bg-orange-100 text-orange-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {matchup.difficulty}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {matchup.defense_rank}/32
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      {matchup.recommendation}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}