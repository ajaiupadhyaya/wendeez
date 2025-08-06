"""
NFL Data Collector - Fetches player stats, game data, and team information
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import nfl_data_py as nfl
import time
from functools import lru_cache
import json

from src.utils.logger import get_logger, log_execution_time
from src.utils.config import get_config
from src.utils.database import get_db_manager, Player, GameStats, TeamStats

logger = get_logger()


class NFLDataCollector:
    """Collect and process NFL data from various sources"""
    
    def __init__(self):
        """Initialize the NFL data collector"""
        self.config = get_config()
        self.db_manager = get_db_manager()
        self.cache_dir = self.config.data.cache_dir
        self.raw_dir = self.config.data.raw_dir
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("NFL Data Collector initialized")
        
    @log_execution_time
    def fetch_player_stats(self, years: List[int], positions: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch player statistics for specified years
        
        Args:
            years: List of years to fetch data for
            positions: Optional list of positions to filter (QB, RB, WR, TE)
            
        Returns:
            DataFrame with player statistics
        """
        try:
            # Validate years - NFL data typically available from 1999 to current year - 1
            current_year = datetime.now().year
            valid_years = [y for y in years if 1999 <= y < current_year]
            
            if not valid_years:
                logger.warning(f"No valid years in {years}. Using default years.")
                valid_years = [current_year - 2, current_year - 1]  # Use last two complete seasons
            
            if valid_years != years:
                logger.warning(f"Adjusted years from {years} to {valid_years}")
            
            logger.info(f"Fetching player stats for years: {valid_years}")
            
            # Use nfl_data_py to get weekly player data
            df = nfl.import_weekly_data(valid_years)
            
            # Filter by positions if specified
            if positions:
                df = df[df['position'].isin(positions)]
                
            # Calculate fantasy points
            df = self._calculate_fantasy_points(df)
            
            # Save raw data
            filename = f"player_stats_{min(valid_years)}_{max(valid_years)}.parquet"
            df.to_parquet(self.raw_dir / filename)
            
            logger.info(f"Fetched {len(df)} player stat records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch player stats: {str(e)}")
            raise
            
    @log_execution_time
    def fetch_team_stats(self, years: List[int]) -> pd.DataFrame:
        """
        Fetch team-level statistics by aggregating weekly data
        
        Args:
            years: List of years to fetch
            
        Returns:
            DataFrame with team statistics
        """
        try:
            # Validate years
            current_year = datetime.now().year
            valid_years = [y for y in years if 1999 <= y < current_year]
            
            if not valid_years:
                valid_years = [current_year - 2, current_year - 1]
            
            logger.info(f"Fetching team stats for years: {valid_years}")
            
            # Fetch seasonal data which includes team stats
            df = nfl.import_seasonal_data(valid_years)
            
            # Filter to team-level stats (remove individual player stats)
            if not df.empty:
                # Aggregate by team if needed
                team_cols = ['season', 'team']
                if all(col in df.columns for col in team_cols):
                    # Group by team and season
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    agg_dict = {col: 'sum' for col in numeric_cols if col not in team_cols}
                    
                    if agg_dict:
                        team_df = df.groupby(team_cols).agg(agg_dict).reset_index()
                        
                        # Calculate per-game metrics if games column exists
                        if 'games' in team_df.columns:
                            games = team_df['games'].replace(0, np.nan)
                            for col in ['points', 'total_yards', 'passing_yards', 'rushing_yards']:
                                if col in team_df.columns:
                                    team_df[f'{col}_per_game'] = team_df[col] / games
                        
                        df = team_df
            
            # Save raw data
            filename = f"team_stats_{min(valid_years)}_{max(valid_years)}.parquet"
            df.to_parquet(self.raw_dir / filename)
            
            logger.info(f"Fetched {len(df)} team stat records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch team stats: {str(e)}")
            # Return empty DataFrame instead of raising to allow other operations to continue
            logger.warning("Returning empty DataFrame for team stats")
            return pd.DataFrame()
            
    @log_execution_time
    def fetch_roster_data(self, years: List[int]) -> pd.DataFrame:
        """
        Fetch NFL roster data
        
        Args:
            years: List of years to fetch
            
        Returns:
            DataFrame with roster information
        """
        try:
            # Validate years
            current_year = datetime.now().year
            valid_years = [y for y in years if 1999 <= y < current_year]
            
            if not valid_years:
                valid_years = [current_year - 2, current_year - 1]
            
            logger.info(f"Fetching roster data for years: {valid_years}")
            
            # Use seasonal_rosters which is the correct function name
            df = nfl.import_seasonal_rosters(valid_years)
            
            # Clean and process roster data
            if not df.empty:
                df = self._process_roster_data(df)
            
            # Save raw data
            if not df.empty:
                filename = f"rosters_{min(valid_years)}_{max(valid_years)}.parquet"
                df.to_parquet(self.raw_dir / filename)
            
            logger.info(f"Fetched {len(df)} roster records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch roster data: {str(e)}")
            logger.warning("Returning empty DataFrame for roster data")
            return pd.DataFrame()
            
    @log_execution_time
    def fetch_schedule_data(self, years: List[int]) -> pd.DataFrame:
        """
        Fetch NFL schedule and game results
        
        Args:
            years: List of years to fetch
            
        Returns:
            DataFrame with schedule information
        """
        try:
            # Validate years
            current_year = datetime.now().year
            valid_years = [y for y in years if 1999 <= y < current_year]
            
            if not valid_years:
                valid_years = [current_year - 2, current_year - 1]
            
            logger.info(f"Fetching schedule data for years: {valid_years}")
            
            # Fetch schedule data
            df = nfl.import_schedules(valid_years)
            
            # Process schedule data
            if not df.empty:
                if 'result' in df.columns:
                    df['is_completed'] = df['result'].notna()
                if 'gameday' in df.columns:
                    df['game_date'] = pd.to_datetime(df['gameday'], errors='coerce')
            
            # Save raw data
            filename = f"schedules_{min(valid_years)}_{max(valid_years)}.parquet"
            df.to_parquet(self.raw_dir / filename)
            
            logger.info(f"Fetched {len(df)} schedule records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch schedule data: {str(e)}")
            raise
            
    @log_execution_time
    def fetch_advanced_stats(self, years: List[int]) -> pd.DataFrame:
        """
        Fetch advanced player statistics
        
        Args:
            years: List of years to fetch
            
        Returns:
            DataFrame with advanced statistics
        """
        try:
            # Validate years
            current_year = datetime.now().year
            valid_years = [y for y in years if 2006 <= y < current_year]  # QBR data starts from 2006
            
            if not valid_years:
                valid_years = [current_year - 2, current_year - 1]
            
            logger.info(f"Fetching advanced stats for years: {valid_years}")
            
            dfs = []
            
            # Try to fetch QBR data (available from 2006)
            try:
                qbr = nfl.import_qbr(valid_years)
                if not qbr.empty:
                    dfs.append(qbr)
                    logger.info(f"Fetched {len(qbr)} QBR records")
            except Exception as e:
                logger.warning(f"Could not fetch QBR data: {e}")
            
            # Try to fetch snap counts
            try:
                snap_counts = nfl.import_snap_counts(valid_years)
                if not snap_counts.empty:
                    dfs.append(snap_counts)
                    logger.info(f"Fetched {len(snap_counts)} snap count records")
            except Exception as e:
                logger.warning(f"Could not fetch snap counts: {e}")
            
            # Merge available data
            if len(dfs) > 1:
                # Find common columns for merging
                common_cols = set(dfs[0].columns)
                for df in dfs[1:]:
                    common_cols = common_cols.intersection(set(df.columns))
                
                merge_cols = [col for col in ['player', 'season', 'week'] if col in common_cols]
                
                if merge_cols:
                    df = dfs[0]
                    for next_df in dfs[1:]:
                        df = pd.merge(df, next_df, on=merge_cols, how='outer')
                else:
                    df = pd.concat(dfs, axis=0, ignore_index=True)
            elif len(dfs) == 1:
                df = dfs[0]
            else:
                df = pd.DataFrame()
            
            if not df.empty:
                # Save raw data
                filename = f"advanced_stats_{min(valid_years)}_{max(valid_years)}.parquet"
                df.to_parquet(self.raw_dir / filename)
                logger.info(f"Fetched advanced stats for {len(df)} records")
            else:
                logger.warning("No advanced stats data available")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch advanced stats: {str(e)}")
            raise
            
    def _calculate_fantasy_points(self, df: pd.DataFrame, 
                                 scoring_system: str = "standard") -> pd.DataFrame:
        """
        Calculate fantasy points based on scoring system
        
        Args:
            df: DataFrame with player stats
            scoring_system: Type of scoring (standard, ppr, half_ppr)
            
        Returns:
            DataFrame with fantasy points added
        """
        # Standard scoring
        points = 0
        
        # Passing
        if 'passing_yards' in df.columns:
            points += df['passing_yards'].fillna(0) / 25  # 1 point per 25 yards
            points += df['passing_tds'].fillna(0) * 4      # 4 points per TD
            points -= df['interceptions'].fillna(0) * 2    # -2 points per INT
            
        # Rushing
        if 'rushing_yards' in df.columns:
            points += df['rushing_yards'].fillna(0) / 10   # 1 point per 10 yards
            points += df['rushing_tds'].fillna(0) * 6      # 6 points per TD
            
        # Receiving
        if 'receiving_yards' in df.columns:
            points += df['receiving_yards'].fillna(0) / 10 # 1 point per 10 yards
            points += df['receiving_tds'].fillna(0) * 6    # 6 points per TD
            
            # PPR scoring adjustments
            if scoring_system == "ppr":
                points += df['receptions'].fillna(0)        # 1 point per reception
            elif scoring_system == "half_ppr":
                points += df['receptions'].fillna(0) * 0.5  # 0.5 points per reception
                
        # Fumbles
        if 'fumbles_lost' in df.columns:
            points -= df['fumbles_lost'].fillna(0) * 2     # -2 points per fumble lost
            
        df['fantasy_points'] = points
        return df
        
    def _process_roster_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean roster data
        
        Args:
            df: Raw roster DataFrame
            
        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df
            
        # Clean player names - check for different possible column names
        name_cols = ['player_name', 'player', 'full_name']
        for col in name_cols:
            if col in df.columns:
                df['player_name'] = df[col].astype(str).str.strip()
                break
        
        # Convert height to inches
        if 'height' in df.columns:
            df['height_inches'] = df['height'].apply(self._height_to_inches)
            
        # Calculate age
        if 'birth_date' in df.columns:
            df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
            df['age'] = (datetime.now() - df['birth_date']).dt.days / 365.25
        elif 'age' not in df.columns and 'birthdate' in df.columns:
            df['birth_date'] = pd.to_datetime(df['birthdate'], errors='coerce')
            df['age'] = (datetime.now() - df['birth_date']).dt.days / 365.25
            
        # Clean position groups
        if 'position' in df.columns:
            df['position_group'] = df['position'].apply(self._get_position_group)
        
        return df
        
    @staticmethod
    def _height_to_inches(height_str: str) -> Optional[int]:
        """Convert height string to inches"""
        if pd.isna(height_str):
            return None
        try:
            feet, inches = height_str.split('-')
            return int(feet) * 12 + int(inches)
        except:
            return None
            
    @staticmethod
    def _get_position_group(position: str) -> str:
        """Get position group from specific position"""
        position_groups = {
            'QB': 'QB',
            'RB': 'RB', 'FB': 'RB',
            'WR': 'WR',
            'TE': 'TE',
            'K': 'K',
            'DEF': 'DEF',
            'DST': 'DEF'
        }
        return position_groups.get(position, 'OTHER')
        
    def update_database(self, years: List[int], positions: List[str] = None):
        """
        Update database with latest data
        
        Args:
            years: Years to update
            positions: Positions to update
        """
        try:
            logger.info(f"Updating database for years {years}")
            
            # Fetch all data
            player_stats = self.fetch_player_stats(years, positions)
            team_stats = self.fetch_team_stats(years)
            rosters = self.fetch_roster_data(years)
            schedules = self.fetch_schedule_data(years)
            
            # Process and store in database
            self._store_player_data(rosters)
            self._store_game_stats(player_stats)
            self._store_team_stats(team_stats)
            
            logger.info("Database update completed successfully")
            
        except Exception as e:
            logger.error(f"Database update failed: {str(e)}")
            raise
            
    def _store_player_data(self, df: pd.DataFrame):
        """Store player data in database"""
        if df.empty:
            logger.warning("No player data to store")
            return
            
        players = []
        for _, row in df.iterrows():
            # Try different ID columns
            player_id = row.get('player_id') or row.get('gsis_id') or row.get('player')
            
            if not player_id:
                continue  # Skip if no ID
                
            player_data = {
                'player_id': str(player_id),
                'name': row.get('player_name') or row.get('player') or row.get('full_name'),
                'position': row.get('position'),
                'team': row.get('team'),
                'age': row.get('age') if pd.notna(row.get('age')) else None,
                'height': row.get('height'),
                'weight': row.get('weight') if pd.notna(row.get('weight')) else None,
                'college': row.get('college'),
                'draft_year': row.get('draft_year') if pd.notna(row.get('draft_year')) else None,
                'draft_round': row.get('draft_round') if pd.notna(row.get('draft_round')) else None,
                'draft_pick': row.get('draft_pick') if pd.notna(row.get('draft_pick')) else None,
                'status': row.get('status', 'active')
            }
            players.append(player_data)
            
        if players:
            try:
                self.db_manager.bulk_insert(Player, players)
                logger.info(f"Stored {len(players)} player records")
            except Exception as e:
                logger.error(f"Failed to store player data: {e}")
            
    def _store_game_stats(self, df: pd.DataFrame):
        """Store game statistics in database"""
        if df.empty:
            logger.warning("No game stats to store")
            return
            
        stats = []
        for _, row in df.iterrows():
            # Get player ID
            player_id = row.get('player_id') or row.get('player')
            if not player_id:
                continue
                
            stat_data = {
                'player_id': str(player_id),
                'game_id': row.get('game_id'),
                'week': int(row.get('week')) if pd.notna(row.get('week')) else None,
                'season': int(row.get('season')) if pd.notna(row.get('season')) else None,
                'opponent': row.get('opponent_team') or row.get('opponent'),
                'passing_attempts': int(row.get('attempts', 0)) if pd.notna(row.get('attempts')) else 0,
                'passing_completions': int(row.get('completions', 0)) if pd.notna(row.get('completions')) else 0,
                'passing_yards': int(row.get('passing_yards', 0)) if pd.notna(row.get('passing_yards')) else 0,
                'passing_tds': int(row.get('passing_tds', 0)) if pd.notna(row.get('passing_tds')) else 0,
                'passing_ints': int(row.get('interceptions', 0)) if pd.notna(row.get('interceptions')) else 0,
                'rushing_attempts': int(row.get('carries', 0)) if pd.notna(row.get('carries')) else 0,
                'rushing_yards': int(row.get('rushing_yards', 0)) if pd.notna(row.get('rushing_yards')) else 0,
                'rushing_tds': int(row.get('rushing_tds', 0)) if pd.notna(row.get('rushing_tds')) else 0,
                'targets': int(row.get('targets', 0)) if pd.notna(row.get('targets')) else 0,
                'receptions': int(row.get('receptions', 0)) if pd.notna(row.get('receptions')) else 0,
                'receiving_yards': int(row.get('receiving_yards', 0)) if pd.notna(row.get('receiving_yards')) else 0,
                'receiving_tds': int(row.get('receiving_tds', 0)) if pd.notna(row.get('receiving_tds')) else 0,
                'fantasy_points': float(row.get('fantasy_points', 0)) if pd.notna(row.get('fantasy_points')) else 0
            }
            stats.append(stat_data)
            
        if stats:
            try:
                self.db_manager.bulk_insert(GameStats, stats)
                logger.info(f"Stored {len(stats)} game stat records")
            except Exception as e:
                logger.error(f"Failed to store game stats: {e}")
            
    def _store_team_stats(self, df: pd.DataFrame):
        """Store team statistics in database"""
        if df.empty:
            logger.warning("No team stats to store")
            return
            
        stats = []
        for _, row in df.iterrows():
            team_code = row.get('team')
            if not team_code:
                continue
                
            stat_data = {
                'team_code': str(team_code),
                'week': int(row.get('week')) if pd.notna(row.get('week')) else None,
                'season': int(row.get('season')) if pd.notna(row.get('season')) else None,
                'offensive_rank': int(row.get('offense_rank')) if pd.notna(row.get('offense_rank')) else None,
                'points_scored': float(row.get('points', 0)) if pd.notna(row.get('points')) else 0,
                'total_yards': float(row.get('total_yards', 0)) if pd.notna(row.get('total_yards')) else 0,
                'passing_yards': float(row.get('passing_yards', 0)) if pd.notna(row.get('passing_yards')) else 0,
                'rushing_yards': float(row.get('rushing_yards', 0)) if pd.notna(row.get('rushing_yards')) else 0,
                'turnovers': int(row.get('turnovers', 0)) if pd.notna(row.get('turnovers')) else 0,
                'defensive_rank': int(row.get('defense_rank')) if pd.notna(row.get('defense_rank')) else None,
                'points_allowed': float(row.get('points_allowed', 0)) if pd.notna(row.get('points_allowed')) else 0,
                'yards_allowed': float(row.get('yards_allowed', 0)) if pd.notna(row.get('yards_allowed')) else 0
            }
            stats.append(stat_data)
            
        if stats:
            try:
                self.db_manager.bulk_insert(TeamStats, stats)
                logger.info(f"Stored {len(stats)} team stat records")
            except Exception as e:
                logger.error(f"Failed to store team stats: {e}")
            
    @lru_cache(maxsize=128)
    def get_player_recent_form(self, player_id: str, weeks: int = 5) -> Dict[str, Any]:
        """
        Get player's recent form and statistics
        
        Args:
            player_id: Player ID
            weeks: Number of recent weeks to analyze
            
        Returns:
            Dictionary with recent form metrics
        """
        stats = self.db_manager.get_player_stats(player_id, weeks)
        
        if stats.empty:
            return {}
            
        return {
            'avg_points': stats['fantasy_points'].mean(),
            'std_points': stats['fantasy_points'].std(),
            'trend': self._calculate_trend(stats['fantasy_points'].tolist()),
            'consistency': 1 - (stats['fantasy_points'].std() / stats['fantasy_points'].mean()),
            'ceiling': stats['fantasy_points'].quantile(0.9),
            'floor': stats['fantasy_points'].quantile(0.1)
        }
        
    @staticmethod
    def _calculate_trend(values: List[float]) -> float:
        """Calculate trend coefficient"""
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        return coefficients[0]


if __name__ == "__main__":
    # Test the data collector
    collector = NFLDataCollector()
    
    # Use known valid years (2022 and 2023 should have complete data)
    years = [2022, 2023]
    
    try:
        # Fetch player stats for QBs and RBs
        logger.info("Testing player stats fetch...")
        stats = collector.fetch_player_stats(years, positions=['QB', 'RB'])
        logger.info(f"✓ Successfully fetched {len(stats)} player records")
        
        # Fetch team stats
        logger.info("Testing team stats fetch...")
        team_stats = collector.fetch_team_stats(years)
        logger.info(f"✓ Successfully fetched {len(team_stats)} team records")
        
        # Fetch roster data
        logger.info("Testing roster data fetch...")
        rosters = collector.fetch_roster_data(years)
        logger.info(f"✓ Successfully fetched {len(rosters)} roster records")
        
        logger.success("All data collector tests passed!")
        
    except Exception as e:
        logger.error(f"Data collector test failed: {str(e)}")
        import traceback
        traceback.print_exc()
