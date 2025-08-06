"""
Defensive Matchup Analyzer - Analyzes defensive strengths/weaknesses for matchup predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import nfl_data_py as nfl
from functools import lru_cache

from src.utils.logger import get_logger, log_execution_time
from src.utils.database import get_db_manager

logger = get_logger()


class DefensiveMatchupAnalyzer:
    """Analyze defensive matchups and calculate matchup scores"""
    
    def __init__(self):
        """Initialize the defensive matchup analyzer"""
        self.db_manager = get_db_manager()
        self.current_season = 2024  # 2024 season data for 2025 predictions
        logger.info("Defensive Matchup Analyzer initialized")
        
    @log_execution_time
    def calculate_defensive_rankings(self, season: int = 2024) -> pd.DataFrame:
        """
        Calculate defensive rankings by position
        
        Args:
            season: Season to analyze
            
        Returns:
            DataFrame with defensive rankings
        """
        try:
            # Get all game stats for the season
            query = f"""
            SELECT 
                gs.*,
                p.position
            FROM game_stats gs
            JOIN players p ON gs.player_id = p.player_id
            WHERE gs.season = {season}
            """
            
            df = self.db_manager.execute_query(query)
            
            if df.empty:
                logger.warning(f"No data available for season {season}")
                return pd.DataFrame()
            
            # Calculate points allowed by each defense by position
            defense_stats = df.groupby(['opponent', 'position']).agg({
                'fantasy_points': ['mean', 'sum', 'count'],
                'passing_yards': 'mean',
                'rushing_yards': 'mean',
                'passing_tds': 'mean',
                'rushing_tds': 'mean',
                'receiving_yards': 'mean',
                'receiving_tds': 'mean'
            }).reset_index()
            
            # Flatten column names
            defense_stats.columns = ['_'.join(col).strip('_') for col in defense_stats.columns.values]
            
            # Calculate rankings (lower rank = tougher defense)
            for pos in ['QB', 'RB', 'WR', 'TE']:
                pos_data = defense_stats[defense_stats['position'] == pos].copy()
                if not pos_data.empty:
                    pos_data[f'{pos}_defense_rank'] = pos_data['fantasy_points_mean'].rank(ascending=True)
                    defense_stats.loc[defense_stats['position'] == pos, f'{pos}_defense_rank'] = pos_data[f'{pos}_defense_rank']
            
            return defense_stats
            
        except Exception as e:
            logger.error(f"Failed to calculate defensive rankings: {e}")
            return pd.DataFrame()
    
    @log_execution_time
    def get_2025_schedule(self) -> pd.DataFrame:
        """
        Fetch the 2025 NFL schedule
        
        Returns:
            DataFrame with 2025 schedule
        """
        try:
            # Try to fetch 2025 schedule
            try:
                schedule = nfl.import_schedules([2025])
            except:
                # If 2025 not available, use 2024 as proxy
                logger.warning("2025 schedule not available, using 2024 as proxy")
                schedule = nfl.import_schedules([2024])
                schedule['season'] = 2025  # Adjust season
            
            # Process schedule data
            schedule['game_date'] = pd.to_datetime(schedule['gameday'])
            schedule['is_home'] = True  # Will be set based on home/away teams
            
            return schedule
            
        except Exception as e:
            logger.error(f"Failed to fetch schedule: {e}")
            return pd.DataFrame()
    
    @lru_cache(maxsize=256)
    def calculate_matchup_score(self, player_position: str, opponent_team: str, 
                               season: int = 2024) -> Dict[str, float]:
        """
        Calculate matchup score for a player against a specific defense
        
        Args:
            player_position: Player's position (QB, RB, WR, TE)
            opponent_team: Opponent team code
            season: Season for historical data
            
        Returns:
            Dictionary with matchup metrics
        """
        try:
            # Get defensive rankings
            defense_stats = self.calculate_defensive_rankings(season)
            
            if defense_stats.empty:
                return {'matchup_score': 50.0, 'difficulty': 'average'}
            
            # Filter for specific opponent and position
            opp_defense = defense_stats[
                (defense_stats['opponent'] == opponent_team) & 
                (defense_stats['position'] == player_position)
            ]
            
            if opp_defense.empty:
                return {'matchup_score': 50.0, 'difficulty': 'average'}
            
            # Get the defense ranking
            rank_col = f'{player_position}_defense_rank'
            if rank_col in opp_defense.columns:
                rank = opp_defense[rank_col].iloc[0]
                total_teams = 32
                
                # Calculate matchup score (0-100, higher is better for offense)
                matchup_score = (rank / total_teams) * 100
                
                # Determine difficulty category
                if matchup_score >= 75:
                    difficulty = 'easy'
                elif matchup_score >= 60:
                    difficulty = 'favorable'
                elif matchup_score >= 40:
                    difficulty = 'average'
                elif matchup_score >= 25:
                    difficulty = 'tough'
                else:
                    difficulty = 'very_tough'
                
                # Get additional stats
                avg_points_allowed = opp_defense['fantasy_points_mean'].iloc[0]
                games_played = opp_defense['fantasy_points_count'].iloc[0]
                
                return {
                    'matchup_score': matchup_score,
                    'difficulty': difficulty,
                    'defense_rank': rank,
                    'avg_points_allowed': avg_points_allowed,
                    'games_analyzed': games_played
                }
            
            return {'matchup_score': 50.0, 'difficulty': 'average'}
            
        except Exception as e:
            logger.error(f"Failed to calculate matchup score: {e}")
            return {'matchup_score': 50.0, 'difficulty': 'average'}
    
    def get_upcoming_matchups(self, player_id: str, weeks_ahead: int = 4) -> pd.DataFrame:
        """
        Get upcoming matchups for a player
        
        Args:
            player_id: Player ID
            weeks_ahead: Number of weeks to look ahead
            
        Returns:
            DataFrame with upcoming matchups and scores
        """
        try:
            # Get player info
            query = f"""
            SELECT player_id, name, position, team
            FROM players
            WHERE player_id = '{player_id}'
            """
            player_info = self.db_manager.execute_query(query)
            
            if player_info.empty:
                logger.warning(f"Player {player_id} not found")
                return pd.DataFrame()
            
            player = player_info.iloc[0]
            
            # Get schedule
            schedule = self.get_2025_schedule()
            
            if schedule.empty:
                return pd.DataFrame()
            
            # Get current week (estimate based on date)
            current_week = self._estimate_current_week()
            
            # Filter for player's team upcoming games
            upcoming = schedule[
                ((schedule['home_team'] == player['team']) | 
                 (schedule['away_team'] == player['team'])) &
                (schedule['week'] >= current_week) &
                (schedule['week'] < current_week + weeks_ahead)
            ].copy()
            
            # Determine opponent for each game
            upcoming['opponent'] = upcoming.apply(
                lambda x: x['away_team'] if x['home_team'] == player['team'] else x['home_team'],
                axis=1
            )
            
            # Calculate matchup scores
            matchup_scores = []
            for _, game in upcoming.iterrows():
                score_data = self.calculate_matchup_score(
                    player['position'], 
                    game['opponent']
                )
                matchup_scores.append({
                    'week': game['week'],
                    'opponent': game['opponent'],
                    'is_home': game['home_team'] == player['team'],
                    **score_data
                })
            
            return pd.DataFrame(matchup_scores)
            
        except Exception as e:
            logger.error(f"Failed to get upcoming matchups: {e}")
            return pd.DataFrame()
    
    def _estimate_current_week(self) -> int:
        """Estimate current NFL week based on date"""
        now = datetime.now()
        
        # NFL season typically starts in early September
        season_start = datetime(2025, 9, 7)  # Approximate 2025 season start
        
        if now < season_start:
            return 1  # Pre-season
        
        weeks_since_start = (now - season_start).days // 7
        return min(max(1, weeks_since_start + 1), 18)  # Cap at week 18
    
    def analyze_season_matchups(self, player_id: str) -> Dict[str, any]:
        """
        Analyze full season matchup difficulty for a player
        
        Args:
            player_id: Player ID
            
        Returns:
            Dictionary with season matchup analysis
        """
        try:
            # Get all matchups
            matchups = self.get_upcoming_matchups(player_id, weeks_ahead=18)
            
            if matchups.empty:
                return {}
            
            # Calculate statistics
            analysis = {
                'avg_matchup_score': matchups['matchup_score'].mean(),
                'easiest_matchup': matchups.loc[matchups['matchup_score'].idxmax()].to_dict(),
                'toughest_matchup': matchups.loc[matchups['matchup_score'].idxmin()].to_dict(),
                'easy_games': len(matchups[matchups['difficulty'] == 'easy']),
                'tough_games': len(matchups[matchups['difficulty'].isin(['tough', 'very_tough'])]),
                'home_games': len(matchups[matchups['is_home'] == True]),
                'away_games': len(matchups[matchups['is_home'] == False]),
                'strength_of_schedule': 100 - matchups['matchup_score'].mean()  # Higher = tougher
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze season matchups: {e}")
            return {}
    
    def get_defense_trends(self, team: str, weeks: int = 6) -> Dict[str, float]:
        """
        Get recent defensive performance trends
        
        Args:
            team: Team code
            weeks: Number of recent weeks to analyze
            
        Returns:
            Dictionary with trend metrics
        """
        try:
            # Get recent games where this team was the opponent
            query = f"""
            SELECT 
                gs.fantasy_points,
                gs.week,
                gs.season,
                p.position
            FROM game_stats gs
            JOIN players p ON gs.player_id = p.player_id
            WHERE gs.opponent = '{team}'
            AND gs.season >= 2023
            ORDER BY gs.season DESC, gs.week DESC
            LIMIT {weeks * 50}  -- Approximate for all positions
            """
            
            df = self.db_manager.execute_query(query)
            
            if df.empty:
                return {}
            
            # Calculate trends by position
            trends = {}
            for pos in ['QB', 'RB', 'WR', 'TE']:
                pos_data = df[df['position'] == pos]['fantasy_points']
                if len(pos_data) > 0:
                    # Calculate trend (positive = getting worse for defense)
                    recent = pos_data.head(len(pos_data)//2).mean()
                    older = pos_data.tail(len(pos_data)//2).mean()
                    trend = ((recent - older) / older * 100) if older > 0 else 0
                    
                    trends[f'{pos}_trend'] = trend
                    trends[f'{pos}_recent_avg'] = recent
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to get defense trends: {e}")
            return {}


if __name__ == "__main__":
    # Test the analyzer
    analyzer = DefensiveMatchupAnalyzer()
    
    # Calculate defensive rankings
    rankings = analyzer.calculate_defensive_rankings(2023)
    if not rankings.empty:
        logger.info(f"Calculated rankings for {len(rankings)} defense-position combinations")
        
        # Show top 5 toughest defenses against QBs
        qb_defenses = rankings[rankings['position'] == 'QB'].sort_values('fantasy_points_mean')
        logger.info(f"Top 5 toughest defenses against QBs:\n{qb_defenses.head()}")
    
    # Test matchup score calculation
    score = analyzer.calculate_matchup_score('RB', 'BUF', 2023)
    logger.info(f"RB vs BUF matchup score: {score}")