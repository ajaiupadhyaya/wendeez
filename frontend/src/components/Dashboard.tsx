import { useState, useEffect } from 'react';
import axios from 'axios';
import { ChartData, ChartOptions } from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface PlayerPrediction {
  player_id: string;
  player_name: string;
  position: string;
  team: string;
  predicted_points: number;
  floor: number;
  ceiling: number;
  recommendation: string;
}

export default function Dashboard() {
  const [week, setWeek] = useState(1);
  const [topPredictions, setTopPredictions] = useState<PlayerPrediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedPosition, setSelectedPosition] = useState('ALL');

  useEffect(() => {
    fetchTopPredictions();
  }, [week, selectedPosition]);

  const fetchTopPredictions = async () => {
    setLoading(true);
    try {
      // Fetch top players for the week
      const positions = selectedPosition === 'ALL' ? ['QB', 'RB', 'WR', 'TE'] : [selectedPosition];
      const predictions: PlayerPrediction[] = [];

      for (const pos of positions) {
        const rankingsRes = await axios.get(`/api/rankings/${pos}?week=${week}`);
        const topPlayers = rankingsRes.data.slice(0, 5);

        for (const player of topPlayers) {
          try {
            const predRes = await axios.post('/api/predict', {
              player_id: player.player_id,
              week: week,
              season: 2025
            });
            predictions.push(predRes.data);
          } catch (err) {
            console.error(`Failed to get prediction for ${player.name}`);
          }
        }
      }

      setTopPredictions(predictions.sort((a, b) => b.predicted_points - a.predicted_points).slice(0, 20));
    } catch (error) {
      console.error('Error fetching predictions:', error);
    } finally {
      setLoading(false);
    }
  };

  const chartData: ChartData<'bar'> = {
    labels: topPredictions.slice(0, 10).map(p => p.player_name),
    datasets: [
      {
        label: 'Predicted Points',
        data: topPredictions.slice(0, 10).map(p => p.predicted_points),
        backgroundColor: 'rgba(14, 165, 233, 0.5)',
        borderColor: 'rgb(14, 165, 233)',
        borderWidth: 1,
      },
      {
        label: 'Floor',
        data: topPredictions.slice(0, 10).map(p => p.floor),
        backgroundColor: 'rgba(239, 68, 68, 0.3)',
        borderColor: 'rgb(239, 68, 68)',
        borderWidth: 1,
      },
      {
        label: 'Ceiling',
        data: topPredictions.slice(0, 10).map(p => p.ceiling),
        backgroundColor: 'rgba(34, 197, 94, 0.3)',
        borderColor: 'rgb(34, 197, 94)',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions: ChartOptions<'bar'> = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: `Week ${week} Top Projections`,
      },
    },
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white shadow rounded-lg p-6">
        <h1 className="text-3xl font-bold text-gray-900">Fantasy Football Dashboard</h1>
        <p className="mt-2 text-gray-600">2025 Season Predictions & Analysis</p>
      </div>

      {/* Controls */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex space-x-4">
          <div>
            <label htmlFor="week" className="block text-sm font-medium text-gray-700">
              Week
            </label>
            <select
              id="week"
              value={week}
              onChange={(e) => setWeek(Number(e.target.value))}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            >
              {[...Array(18)].map((_, i) => (
                <option key={i + 1} value={i + 1}>
                  Week {i + 1}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="position" className="block text-sm font-medium text-gray-700">
              Position
            </label>
            <select
              id="position"
              value={selectedPosition}
              onChange={(e) => setSelectedPosition(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            >
              <option value="ALL">All Positions</option>
              <option value="QB">Quarterback</option>
              <option value="RB">Running Back</option>
              <option value="WR">Wide Receiver</option>
              <option value="TE">Tight End</option>
            </select>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Top Projections Chart</h2>
        {loading ? (
          <div className="flex justify-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
          </div>
        ) : (
          <Bar data={chartData} options={chartOptions} />
        )}
      </div>

      {/* Top Players Table */}
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Top Projected Players</h2>
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
                    Player
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Position
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Team
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Projected
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Floor
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Ceiling
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Recommendation
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {topPredictions.map((player) => (
                  <tr key={player.player_id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {player.player_name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {player.position}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {player.team}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-semibold">
                      {player.predicted_points.toFixed(1)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600">
                      {player.floor.toFixed(1)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600">
                      {player.ceiling.toFixed(1)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        player.recommendation === 'MUST START' ? 'bg-green-100 text-green-800' :
                        player.recommendation === 'START' ? 'bg-blue-100 text-blue-800' :
                        player.recommendation === 'FLEX' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {player.recommendation}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900">Top QB</h3>
          <p className="mt-2 text-3xl font-bold text-primary-600">
            {topPredictions.find(p => p.position === 'QB')?.player_name || '-'}
          </p>
          <p className="text-gray-500">
            {topPredictions.find(p => p.position === 'QB')?.predicted_points.toFixed(1) || '0'} pts
          </p>
        </div>
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900">Top RB</h3>
          <p className="mt-2 text-3xl font-bold text-primary-600">
            {topPredictions.find(p => p.position === 'RB')?.player_name || '-'}
          </p>
          <p className="text-gray-500">
            {topPredictions.find(p => p.position === 'RB')?.predicted_points.toFixed(1) || '0'} pts
          </p>
        </div>
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900">Top WR</h3>
          <p className="mt-2 text-3xl font-bold text-primary-600">
            {topPredictions.find(p => p.position === 'WR')?.player_name || '-'}
          </p>
          <p className="text-gray-500">
            {topPredictions.find(p => p.position === 'WR')?.predicted_points.toFixed(1) || '0'} pts
          </p>
        </div>
      </div>
    </div>
  );
}