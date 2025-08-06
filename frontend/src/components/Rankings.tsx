import { useState, useEffect } from 'react';
import axios from 'axios';

export default function Rankings() {
  const [position, setPosition] = useState('QB');
  const [week, setWeek] = useState<number | null>(null);
  const [rankings, setRankings] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchRankings();
  }, [position, week]);

  const fetchRankings = async () => {
    setLoading(true);
    try {
      const url = week 
        ? `/api/rankings/${position}?week=${week}`
        : `/api/rankings/${position}`;
      const res = await axios.get(url);
      setRankings(res.data);
    } catch (error) {
      console.error('Error fetching rankings:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white shadow rounded-lg p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Player Rankings</h1>
        <p className="text-gray-600">
          {week ? `Week ${week} Projections` : 'Season Average Rankings'}
        </p>
      </div>

      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex space-x-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700">Position</label>
            <select
              value={position}
              onChange={(e) => setPosition(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300"
            >
              <option value="QB">Quarterback</option>
              <option value="RB">Running Back</option>
              <option value="WR">Wide Receiver</option>
              <option value="TE">Tight End</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">Week</label>
            <select
              value={week || ''}
              onChange={(e) => setWeek(e.target.value ? Number(e.target.value) : null)}
              className="mt-1 block w-full rounded-md border-gray-300"
            >
              <option value="">Season Average</option>
              {[...Array(18)].map((_, i) => (
                <option key={i + 1} value={i + 1}>Week {i + 1}</option>
              ))}
            </select>
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Rank
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Player
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Team
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {week ? 'Projected' : 'Avg Points'}
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Games
                  </th>
                  {week && (
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      vs Avg
                    </th>
                  )}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {rankings.map((player) => {
                  const diff = week ? player.projected_points - player.avg_points : 0;
                  return (
                    <tr key={player.player_id}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {player.rank}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {player.name}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {player.team}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-primary-600">
                        {week 
                          ? player.projected_points.toFixed(1)
                          : player.avg_points.toFixed(1)
                        }
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {player.games_played}
                      </td>
                      {week && (
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <span className={diff > 0 ? 'text-green-600' : 'text-red-600'}>
                            {diff > 0 ? '+' : ''}{diff.toFixed(1)}
                          </span>
                        </td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}