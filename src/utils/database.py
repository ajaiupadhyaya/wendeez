"""
Database connection and management for the Elite Fantasy Football Predictor
"""

from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import NullPool, QueuePool
from datetime import datetime
import pandas as pd
from pathlib import Path

from src.utils.config import get_database_config
from src.utils.logger import get_logger

logger = get_logger()

# Create base class for ORM models
Base = declarative_base()


class Player(Base):
    """Player information table"""
    __tablename__ = 'players'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    position = Column(String(10), nullable=False, index=True)
    team = Column(String(10), index=True)
    age = Column(Integer)
    height = Column(String(10))
    weight = Column(Integer)
    college = Column(String(100))
    draft_year = Column(Integer)
    draft_round = Column(Integer)
    draft_pick = Column(Integer)
    status = Column(String(20))  # active, injured, suspended, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(JSON)  # Store additional flexible data


class GameStats(Base):
    """Game-level statistics for players"""
    __tablename__ = 'game_stats'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(String(50), nullable=False, index=True)
    game_id = Column(String(50), nullable=True, index=True)  # Made nullable
    week = Column(Integer, nullable=False, index=True)
    season = Column(Integer, nullable=False, index=True)
    opponent = Column(String(10))
    is_home = Column(Boolean)
    
    # Passing stats
    passing_attempts = Column(Integer, default=0)
    passing_completions = Column(Integer, default=0)
    passing_yards = Column(Integer, default=0)
    passing_tds = Column(Integer, default=0)
    passing_ints = Column(Integer, default=0)
    passing_rating = Column(Float)
    
    # Rushing stats
    rushing_attempts = Column(Integer, default=0)
    rushing_yards = Column(Integer, default=0)
    rushing_tds = Column(Integer, default=0)
    rushing_long = Column(Integer, default=0)
    
    # Receiving stats
    targets = Column(Integer, default=0)
    receptions = Column(Integer, default=0)
    receiving_yards = Column(Integer, default=0)
    receiving_tds = Column(Integer, default=0)
    receiving_long = Column(Integer, default=0)
    
    # General stats
    fumbles = Column(Integer, default=0)
    fumbles_lost = Column(Integer, default=0)
    snap_percentage = Column(Float)
    fantasy_points = Column(Float)
    
    # Advanced metrics
    target_share = Column(Float)
    air_yards = Column(Float)
    red_zone_targets = Column(Integer, default=0)
    red_zone_touches = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON)


class Predictions(Base):
    """Model predictions table"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    prediction_id = Column(String(50), unique=True, nullable=False)
    player_id = Column(String(50), nullable=False, index=True)
    week = Column(Integer, nullable=False, index=True)
    season = Column(Integer, nullable=False, index=True)
    model_name = Column(String(50), nullable=False, index=True)
    model_version = Column(String(20))
    
    # Predictions
    predicted_points = Column(Float, nullable=False)
    confidence_lower = Column(Float)  # Lower bound of confidence interval
    confidence_upper = Column(Float)  # Upper bound of confidence interval
    prediction_std = Column(Float)    # Standard deviation
    
    # Actual results (filled after game)
    actual_points = Column(Float)
    error = Column(Float)
    absolute_error = Column(Float)
    
    # Additional predictions
    ceiling = Column(Float)  # Best case scenario
    floor = Column(Float)    # Worst case scenario
    boom_probability = Column(Float)  # Probability of exceeding expectations
    bust_probability = Column(Float)  # Probability of underperforming
    
    # Metadata
    features_used = Column(JSON)
    model_parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    
class TeamStats(Base):
    """Team-level statistics"""
    __tablename__ = 'team_stats'
    
    id = Column(Integer, primary_key=True)
    team_code = Column(String(10), nullable=False, index=True)
    week = Column(Integer, nullable=False, index=True)
    season = Column(Integer, nullable=False, index=True)
    
    # Offensive stats
    offensive_rank = Column(Integer)
    points_scored = Column(Float)
    total_yards = Column(Float)
    passing_yards = Column(Float)
    rushing_yards = Column(Float)
    turnovers = Column(Integer)
    
    # Defensive stats
    defensive_rank = Column(Integer)
    points_allowed = Column(Float)
    yards_allowed = Column(Float)
    passing_yards_allowed = Column(Float)
    rushing_yards_allowed = Column(Float)
    takeaways = Column(Integer)
    
    # Advanced metrics
    pace_of_play = Column(Float)
    red_zone_efficiency = Column(Float)
    third_down_conversion = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON)


class ModelPerformance(Base):
    """Track model performance over time"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50), nullable=False, index=True)
    model_version = Column(String(20), nullable=False)
    evaluation_date = Column(DateTime, nullable=False, index=True)
    week = Column(Integer, index=True)
    season = Column(Integer, index=True)
    
    # Performance metrics
    mae = Column(Float)  # Mean Absolute Error
    rmse = Column(Float)  # Root Mean Square Error
    mape = Column(Float)  # Mean Absolute Percentage Error
    r2_score = Column(Float)
    accuracy = Column(Float)  # Directional accuracy
    
    # Additional metrics
    total_predictions = Column(Integer)
    confidence_calibration = Column(Float)  # How well calibrated are confidence intervals
    
    # Detailed metrics by position
    metrics_by_position = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Manage database connections and operations"""
    
    def __init__(self, config=None):
        """
        Initialize database manager
        
        Args:
            config: Database configuration object
        """
        self.config = config or get_database_config()
        self.engine = None
        self.session_factory = None
        self.scoped_session = None
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialize SQLAlchemy engine"""
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.config.connection_string,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,  # Verify connections before using
                echo=False  # Set to True for SQL query logging
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
            
            # Create scoped session for thread safety
            self.scoped_session = scoped_session(self.session_factory)
            
            logger.info(f"Database engine initialized: {self.config.driver}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
            
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
            
    def drop_tables(self):
        """Drop all database tables"""
        try:
            Base.metadata.drop_all(self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {str(e)}")
            raise
            
    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session with automatic cleanup
        
        Yields:
            SQLAlchemy Session
        """
        session = self.scoped_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()
            
    def bulk_insert(self, model_class, data: List[Dict[str, Any]]):
        """
        Bulk insert data into database with duplicate handling
        
        Args:
            model_class: SQLAlchemy model class
            data: List of dictionaries to insert
        """
        with self.get_session() as session:
            try:
                # For SQLite, we need to handle duplicates differently
                # Try inserting records one by one, skipping duplicates
                inserted = 0
                skipped = 0
                
                for record in data:
                    try:
                        obj = model_class(**record)
                        session.add(obj)
                        session.flush()  # Flush to catch duplicates early
                        inserted += 1
                    except Exception as e:
                        session.rollback()
                        if "UNIQUE constraint failed" in str(e) or "IntegrityError" in str(e):
                            skipped += 1
                            continue  # Skip duplicates
                        else:
                            # Re-raise other errors
                            raise
                
                session.commit()
                logger.info(f"Bulk insert to {model_class.__tablename__}: {inserted} inserted, {skipped} skipped (duplicates)")
                
            except Exception as e:
                logger.error(f"Bulk insert failed: {str(e)}")
                raise
                
    def upsert_dataframe(self, df: pd.DataFrame, table_name: str):
        """
        Upsert a pandas DataFrame to database table
        
        Args:
            df: DataFrame to upsert
            table_name: Target table name
        """
        try:
            df.to_sql(
                table_name,
                self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            logger.info(f"Upserted {len(df)} records to {table_name}")
        except Exception as e:
            logger.error(f"DataFrame upsert failed: {str(e)}")
            raise
            
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as pandas DataFrame
        """
        try:
            df = pd.read_sql_query(query, self.engine)
            logger.debug(f"Query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
            
    def get_latest_predictions(self, player_id: str, week: int, season: int) -> pd.DataFrame:
        """
        Get latest predictions for a player
        
        Args:
            player_id: Player ID
            week: Week number
            season: Season year
            
        Returns:
            DataFrame with predictions
        """
        query = f"""
        SELECT * FROM predictions
        WHERE player_id = '{player_id}'
        AND week = {week}
        AND season = {season}
        ORDER BY created_at DESC
        """
        return self.execute_query(query)
        
    def get_player_stats(self, player_id: str, last_n_games: int = 10) -> pd.DataFrame:
        """
        Get recent player statistics
        
        Args:
            player_id: Player ID
            last_n_games: Number of recent games to fetch
            
        Returns:
            DataFrame with player stats
        """
        query = f"""
        SELECT * FROM game_stats
        WHERE player_id = '{player_id}'
        ORDER BY season DESC, week DESC
        LIMIT {last_n_games}
        """
        return self.execute_query(query)
        
    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Singleton database manager
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create database manager singleton"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


if __name__ == "__main__":
    # Test database setup
    db = get_db_manager()
    
    # Create tables
    db.create_tables()
    
    # Test connection
    with db.get_session() as session:
        # Add a test player
        test_player = Player(
            player_id="TEST001",
            name="Test Player",
            position="QB",
            team="TST",
            age=25
        )
        session.add(test_player)
        
    logger.info("Database test completed successfully")
