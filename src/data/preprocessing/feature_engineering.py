"""
Feature Engineering Pipeline for Fantasy Football Predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger, log_execution_time
from src.utils.database import get_db_manager
from src.data.collectors.defensive_matchup_analyzer import DefensiveMatchupAnalyzer
from src.utils.config import get_config

logger = get_logger()


class FeatureEngineer:
    """Create advanced features for fantasy football predictions"""
    
    def __init__(self):
        """Initialize feature engineering pipeline"""
        self.db_manager = get_db_manager()
        self.config = get_config()
        self.matchup_analyzer = DefensiveMatchupAnalyzer()
        self.scaler = RobustScaler()  # Robust to outliers
        
        logger.info("Feature Engineering Pipeline initialized")
    
    @log_execution_time
    def create_player_features(self, player_id: str, target_week: int = None, 
                              target_season: int = 2025) -> Dict[str, float]:
        """
        Create comprehensive feature set for a player
        
        Args:
            player_id: Player ID
            target_week: Week to predict for
            target_season: Season to predict for
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        try:
            # Get player info
            player_info = self._get_player_info(player_id)
            if player_info.empty:
                logger.warning(f"No info found for player {player_id}")
                return features
            
            player = player_info.iloc[0]
            
            # 1. Historical performance features
            hist_features = self._create_historical_features(player_id, target_week, target_season)
            features.update(hist_features)
            
            # 2. Trend features
            trend_features = self._create_trend_features(player_id)
            features.update(trend_features)
            
            # 3. Matchup features
            if target_week:
                matchup_features = self._create_matchup_features(player_id, target_week, target_season)
                features.update(matchup_features)
            
            # 4. Team context features
            team_features = self._create_team_features(player['team'], target_season)
            features.update(team_features)
            
            # 5. Player profile features
            profile_features = self._create_profile_features(player)
            features.update(profile_features)
            
            # 6. Advanced statistical features
            advanced_features = self._create_advanced_features(player_id)
            features.update(advanced_features)
            
            # 7. Interaction features
            interaction_features = self._create_interaction_features(features)
            features.update(interaction_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to create features for player {player_id}: {e}")
            return features
    
    def _get_player_info(self, player_id: str) -> pd.DataFrame:
        """Get player information"""
        query = f"""
        SELECT * FROM players
        WHERE player_id = '{player_id}'
        """
        return self.db_manager.execute_query(query)
    
    def _create_historical_features(self, player_id: str, target_week: int = None,
                                   target_season: int = 2025) -> Dict[str, float]:
        """Create features based on historical performance"""
        features = {}
        
        # Get historical stats
        query = f"""
        SELECT * FROM game_stats
        WHERE player_id = '{player_id}'
        ORDER BY season DESC, week DESC
        """
        stats = self.db_manager.execute_query(query)
        
        if stats.empty:
            return features
        
        # Filter to games before target
        if target_week and target_season:
            stats = stats[
                (stats['season'] < target_season) | 
                ((stats['season'] == target_season) & (stats['week'] < target_week))
            ]
        
        # Recent performance (last 3, 6, 10 games)
        for window in [3, 6, 10]:
            recent = stats.head(window)
            if len(recent) > 0:
                features[f'avg_points_last_{window}'] = recent['fantasy_points'].mean()
                features[f'std_points_last_{window}'] = recent['fantasy_points'].std()
                features[f'max_points_last_{window}'] = recent['fantasy_points'].max()
                features[f'min_points_last_{window}'] = recent['fantasy_points'].min()
                
                # Position-specific stats
                if 'passing_yards' in recent.columns:
                    features[f'avg_pass_yards_last_{window}'] = recent['passing_yards'].mean()
                    features[f'avg_pass_tds_last_{window}'] = recent['passing_tds'].mean()
                
                if 'rushing_yards' in recent.columns:
                    features[f'avg_rush_yards_last_{window}'] = recent['rushing_yards'].mean()
                    features[f'avg_rush_tds_last_{window}'] = recent['rushing_tds'].mean()
                
                if 'receiving_yards' in recent.columns:
                    features[f'avg_rec_yards_last_{window}'] = recent['receiving_yards'].mean()
                    features[f'avg_receptions_last_{window}'] = recent['receptions'].mean()
                    features[f'avg_targets_last_{window}'] = recent['targets'].mean()
        
        # Season averages
        current_season = stats[stats['season'] == stats['season'].max()]
        if len(current_season) > 0:
            features['season_avg_points'] = current_season['fantasy_points'].mean()
            features['season_games_played'] = len(current_season)
        
        # Career averages
        features['career_avg_points'] = stats['fantasy_points'].mean()
        features['career_games'] = len(stats)
        features['career_max_points'] = stats['fantasy_points'].max()
        
        # Consistency metrics
        features['consistency_score'] = stats['fantasy_points'].mean() / (stats['fantasy_points'].std() + 1)
        features['floor'] = stats['fantasy_points'].quantile(0.25)
        features['ceiling'] = stats['fantasy_points'].quantile(0.75)
        
        return features
    
    def _create_trend_features(self, player_id: str) -> Dict[str, float]:
        """Create trend-based features"""
        features = {}
        
        query = f"""
        SELECT fantasy_points, week, season
        FROM game_stats
        WHERE player_id = '{player_id}'
        ORDER BY season DESC, week DESC
        LIMIT 10
        """
        stats = self.db_manager.execute_query(query)
        
        if len(stats) < 3:
            return features
        
        # Calculate trend using linear regression
        points = stats['fantasy_points'].values
        x = np.arange(len(points))
        
        # Reverse so trend is forward-looking
        points = points[::-1]
        
        if len(points) >= 3:
            # Linear trend
            coefficients = np.polyfit(x[-min(6, len(x)):], points[-min(6, len(x)):], 1)
            features['trend_coefficient'] = coefficients[0]
            features['trend_direction'] = 1 if coefficients[0] > 0 else -1
            
            # Moving averages
            features['ma_3_games'] = np.mean(points[-3:]) if len(points) >= 3 else points[-1]
            features['ma_5_games'] = np.mean(points[-5:]) if len(points) >= 5 else features['ma_3_games']
            
            # Momentum
            if len(points) >= 2:
                features['momentum'] = points[-1] - points[-2]
                features['momentum_pct'] = (points[-1] - points[-2]) / (points[-2] + 0.1) * 100
        
        return features
    
    def _create_matchup_features(self, player_id: str, target_week: int, 
                                target_season: int) -> Dict[str, float]:
        """Create matchup-specific features"""
        features = {}
        
        try:
            # Get player position and team
            player_info = self._get_player_info(player_id)
            if player_info.empty:
                return features
            
            player = player_info.iloc[0]
            
            # Get opponent for target week
            schedule = self.matchup_analyzer.get_2025_schedule()
            game = schedule[
                ((schedule['home_team'] == player['team']) | 
                 (schedule['away_team'] == player['team'])) &
                (schedule['week'] == target_week)
            ]
            
            if game.empty:
                return features
            
            game = game.iloc[0]
            opponent = game['away_team'] if game['home_team'] == player['team'] else game['home_team']
            is_home = game['home_team'] == player['team']
            
            # Calculate matchup score
            matchup_data = self.matchup_analyzer.calculate_matchup_score(
                player['position'], opponent, 2024  # Use 2024 data for 2025 predictions
            )
            
            features['matchup_score'] = matchup_data.get('matchup_score', 50)
            features['defense_rank'] = matchup_data.get('defense_rank', 16)
            features['avg_points_allowed'] = matchup_data.get('avg_points_allowed', 15)
            features['is_home'] = 1 if is_home else 0
            
            # Difficulty encoding
            difficulty = matchup_data.get('difficulty', 'average')
            features['matchup_easy'] = 1 if difficulty == 'easy' else 0
            features['matchup_favorable'] = 1 if difficulty == 'favorable' else 0
            features['matchup_tough'] = 1 if difficulty in ['tough', 'very_tough'] else 0
            
            # Get defense trends
            defense_trends = self.matchup_analyzer.get_defense_trends(opponent, weeks=6)
            features['defense_trend'] = defense_trends.get(f"{player['position']}_trend", 0)
            features['defense_recent_avg'] = defense_trends.get(f"{player['position']}_recent_avg", 15)
            
        except Exception as e:
            logger.error(f"Failed to create matchup features: {e}")
        
        return features
    
    def _create_team_features(self, team: str, season: int) -> Dict[str, float]:
        """Create team-level features"""
        features = {}
        
        try:
            # Get team stats (would need team_stats table populated)
            query = f"""
            SELECT * FROM team_stats
            WHERE team_code = '{team}'
            AND season >= {season - 1}
            ORDER BY season DESC, week DESC
            """
            team_stats = self.db_manager.execute_query(query)
            
            if not team_stats.empty:
                recent = team_stats.head(8)
                features['team_avg_points'] = recent['points_scored'].mean() if 'points_scored' in recent.columns else 20
                features['team_avg_yards'] = recent['total_yards'].mean() if 'total_yards' in recent.columns else 350
                features['team_offensive_rank'] = recent['offensive_rank'].mean() if 'offensive_rank' in recent.columns else 16
            
        except Exception as e:
            logger.warning(f"Could not create team features: {e}")
        
        return features
    
    def _create_profile_features(self, player: pd.Series) -> Dict[str, float]:
        """Create player profile features"""
        features = {}
        
        # Position encoding
        positions = ['QB', 'RB', 'WR', 'TE']
        for pos in positions:
            features[f'position_{pos}'] = 1 if player.get('position') == pos else 0
        
        # Experience and physical attributes
        if pd.notna(player.get('age')):
            features['age'] = player['age']
            features['age_squared'] = player['age'] ** 2
            features['is_prime_age'] = 1 if 24 <= player['age'] <= 29 else 0
        
        if pd.notna(player.get('draft_year')):
            features['years_experience'] = 2025 - player['draft_year']
            features['is_rookie'] = 1 if features['years_experience'] <= 1 else 0
            features['is_veteran'] = 1 if features['years_experience'] >= 5 else 0
        
        if pd.notna(player.get('weight')):
            features['weight'] = player['weight']
        
        if pd.notna(player.get('height_inches')):
            features['height'] = player['height_inches']
        
        return features
    
    def _create_advanced_features(self, player_id: str) -> Dict[str, float]:
        """Create advanced statistical features"""
        features = {}
        
        query = f"""
        SELECT * FROM game_stats
        WHERE player_id = '{player_id}'
        ORDER BY season DESC, week DESC
        LIMIT 16
        """
        stats = self.db_manager.execute_query(query)
        
        if len(stats) < 3:
            return features
        
        # Volatility metrics
        points = stats['fantasy_points'].values
        features['volatility'] = np.std(points)
        features['coefficient_variation'] = np.std(points) / (np.mean(points) + 0.1)
        
        # Boom/Bust metrics
        median = np.median(points)
        features['boom_rate'] = np.mean(points > median * 1.5)
        features['bust_rate'] = np.mean(points < median * 0.5)
        
        # Streak analysis
        current_streak = 1
        for i in range(1, min(len(points), 5)):
            if points[i] > median:
                if points[i-1] > median:
                    current_streak += 1
                else:
                    break
            else:
                if points[i-1] <= median:
                    current_streak -= 1
                else:
                    break
        features['current_streak'] = current_streak
        
        # Efficiency metrics for specific positions
        if 'targets' in stats.columns and stats['targets'].sum() > 0:
            features['target_share'] = stats['targets'].mean()
            features['reception_rate'] = stats['receptions'].sum() / stats['targets'].sum()
            features['yards_per_target'] = stats['receiving_yards'].sum() / stats['targets'].sum()
        
        if 'rushing_attempts' in stats.columns and stats['rushing_attempts'].sum() > 0:
            features['yards_per_carry'] = stats['rushing_yards'].sum() / stats['rushing_attempts'].sum()
            features['td_rate_rushing'] = stats['rushing_tds'].sum() / stats['rushing_attempts'].sum()
        
        if 'passing_attempts' in stats.columns and stats['passing_attempts'].sum() > 0:
            features['completion_rate'] = stats['passing_completions'].sum() / stats['passing_attempts'].sum()
            features['yards_per_attempt'] = stats['passing_yards'].sum() / stats['passing_attempts'].sum()
            features['td_rate_passing'] = stats['passing_tds'].sum() / stats['passing_attempts'].sum()
        
        return features
    
    def _create_interaction_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction features between existing features"""
        interaction_features = {}
        
        # Matchup × Form interactions
        if 'matchup_score' in features and 'avg_points_last_3' in features:
            interaction_features['matchup_form_interaction'] = (
                features['matchup_score'] * features['avg_points_last_3'] / 100
            )
        
        # Consistency × Matchup
        if 'consistency_score' in features and 'matchup_score' in features:
            interaction_features['consistency_matchup_interaction'] = (
                features['consistency_score'] * features['matchup_score']
            )
        
        # Home × Form
        if 'is_home' in features and 'trend_coefficient' in features:
            interaction_features['home_trend_interaction'] = (
                features['is_home'] * features['trend_coefficient']
            )
        
        # Age × Experience interaction
        if 'age' in features and 'years_experience' in features:
            interaction_features['age_experience_ratio'] = (
                features['age'] / (features['years_experience'] + 1)
            )
        
        return interaction_features
    
    @log_execution_time
    def create_training_dataset(self, positions: List[str] = ['QB', 'RB', 'WR', 'TE'],
                              min_games: int = 8) -> pd.DataFrame:
        """
        Create full training dataset with all features
        
        Args:
            positions: List of positions to include
            min_games: Minimum games played to include player
            
        Returns:
            DataFrame with features and targets
        """
        all_data = []
        
        # Get all players
        query = f"""
        SELECT DISTINCT p.player_id, p.name, p.position
        FROM players p
        JOIN game_stats gs ON p.player_id = gs.player_id
        WHERE p.position IN ({','.join([f"'{pos}'" for pos in positions])})
        GROUP BY p.player_id, p.name, p.position
        HAVING COUNT(*) >= {min_games}
        """
        players = self.db_manager.execute_query(query)
        
        logger.info(f"Creating training dataset for {len(players)} players")
        
        for _, player in players.iterrows():
            # Get all games for this player
            game_query = f"""
            SELECT season, week, fantasy_points
            FROM game_stats
            WHERE player_id = '{player['player_id']}'
            ORDER BY season, week
            """
            games = self.db_manager.execute_query(game_query)
            
            # Create features for each game (using data up to that point)
            for i in range(3, len(games)):  # Need at least 3 games of history
                target_game = games.iloc[i]
                
                # Create features
                features = self.create_player_features(
                    player['player_id'],
                    target_week=target_game['week'],
                    target_season=target_game['season']
                )
                
                if features:
                    features['player_id'] = player['player_id']
                    features['player_name'] = player['name']
                    features['target_week'] = target_game['week']
                    features['target_season'] = target_game['season']
                    features['target_points'] = target_game['fantasy_points']
                    
                    all_data.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        logger.info(f"Created training dataset with {len(df)} samples and {len(df.columns)} features")
        
        # Save to parquet for later use
        df.to_parquet(self.config.data.processed_dir / 'training_data.parquet')
        
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for modeling
        
        Args:
            df: Raw feature DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        # Separate features and target
        target = df['target_points']
        
        # Drop non-feature columns
        drop_cols = ['player_id', 'player_name', 'target_week', 'target_season', 'target_points']
        features = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Fill missing values
        features = features.fillna(features.median())
        
        # Scale features
        scaled_features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        return scaled_features, target


if __name__ == "__main__":
    # Test the feature engineering pipeline
    engineer = FeatureEngineer()
    
    # Test creating features for a specific player
    test_player_id = "00-0034796"  # Example player ID
    features = engineer.create_player_features(test_player_id, target_week=1, target_season=2025)
    
    if features:
        logger.info(f"Created {len(features)} features for player {test_player_id}")
        logger.info(f"Sample features: {list(features.keys())[:10]}")
    
    # Create training dataset (this will take a while)
    logger.info("Creating full training dataset...")
    training_data = engineer.create_training_dataset()
    logger.info(f"Training data shape: {training_data.shape}")
    
    # Save training data
    from src.utils.config import get_config
    config = get_config()
    training_data_path = config.data.processed_dir / 'training_data.parquet'
    training_data.to_parquet(training_data_path)
    logger.info(f"Training data saved to {training_data_path}")
