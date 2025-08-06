import { useState } from 'react';
import axios from 'axios';

export default function LineupOptimizer() {
  const [playerIds, setPlayerIds] = useState<string>('');
  const [week, setWeek] = useState(1);
  const [optimizedLineup, setOptimizedLineup] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const optimizeLineup = async () => {
    setLoading(true);
    try {
      const ids = playerIds.split(',').map(id => id.trim()).filter(id => id);
      const res = await axios.post('/api/optimize-lineup', {
        player_ids: ids,
        week: week,
      });
      setOptimizedLineup(res.data);
    } catch (error) {
      console.error('Error optimizing lineup:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white shadow rounded-lg p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Lineup Optimizer</h1>
        <p className="text-gray-600">Optimize your fantasy lineup for maximum points</p>
      </div>

      <div className="bg-white shadow rounded-lg p-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Week</label>
            <select
              value={week}
              onChange={(e) => setWeek(Number(e.target.value))}
              className="mt-1 block w-full rounded-md border-gray-300"
            >
              {[...Array(18)].map((_, i) => (
                <option key={i + 1} value={i + 1}>Week {i + 1}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Player IDs (comma-separated)
            </label>
            <textarea
              value={playerIds}
              onChange={(e) => setPlayerIds(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300"
              rows={4}
              placeholder="Enter player IDs separated by commas..."
            />
          </div>

          <button
            onClick={optimizeLineup}
            className="w-full px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
          >
            Optimize Lineup
          </button>
        </div>
      </div>

      {loading && (
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      )}

      {optimizedLineup && !loading && (
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Optimized Lineup</h2>
          <div className="mb-4">
            <p className="text-2xl font-bold text-primary-600">
              Total Projected: {optimizedLineup.total_projected_points.toFixed(1)} points
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold mb-2">Starting Lineup</h3>
              {Object.entries(optimizedLineup.lineup).map(([position, players]: [string, any]) => (
                <div key={position} className="mb-3">
                  <p className="font-medium text-gray-700">{position}</p>
                  {players.map((player: any, idx: number) => (
                    <div key={idx} className="ml-4 py-1">
                      <span className="font-medium">{player.name}</span>
                      <span className="ml-2 text-primary-600">
                        {player.predicted_points.toFixed(1)} pts
                      </span>
                    </div>
                  ))}
                </div>
              ))}
            </div>

            <div>
              <h3 className="font-semibold mb-2">Bench</h3>
              {optimizedLineup.bench.map((player: any, idx: number) => (
                <div key={idx} className="py-1">
                  <span className="font-medium">{player.name}</span>
                  <span className="ml-2 text-gray-500">({player.position})</span>
                  <span className="ml-2 text-gray-600">
                    {player.predicted_points.toFixed(1)} pts
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}