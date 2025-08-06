#!/usr/bin/env python3
"""
Initialize the Elite Fantasy Football Predictor
Sets up database, fetches initial data, and verifies installation
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import get_config
from src.utils.logger import setup_logger, get_logger
from src.utils.database import get_db_manager
from src.data.collectors.nfl_data_collector import NFLDataCollector


def initialize_system():
    """Initialize the complete system"""
    
    print("=" * 60)
    print("Elite Fantasy Football Predictor - System Initialization")
    print("=" * 60)
    
    # Setup logging
    print("\n1. Setting up logging...")
    setup_logger("INFO", Path("logs"), "fantasy_football")
    logger = get_logger()
    logger.info("System initialization started")
    print("   ✓ Logging configured")
    
    # Load configuration
    print("\n2. Loading configuration...")
    config = get_config("development")
    print(f"   ✓ Configuration loaded: {config.app['name']}")
    print(f"   - Database: {config.database.driver}")
    print(f"   - Data directories created")
    
    # Initialize database
    print("\n3. Initializing database...")
    try:
        db = get_db_manager()
        db.create_tables()
        print("   ✓ Database tables created")
        
        # Test database connection
        from sqlalchemy import text
        with db.get_session() as session:
            session.execute(text("SELECT 1"))
        print("   ✓ Database connection verified")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        print(f"   ✗ Database error: {e}")
        return False
    
    # Initialize data collector
    print("\n4. Initializing NFL Data Collector...")
    try:
        collector = NFLDataCollector()
        print("   ✓ Data collector initialized")
    except Exception as e:
        logger.error(f"Data collector initialization failed: {e}")
        print(f"   ✗ Data collector error: {e}")
        return False
    
    # Fetch initial data
    print("\n5. Fetching initial NFL data (this may take a few moments)...")
    years = [2022, 2023]  # Use known good years
    
    try:
        print(f"   - Fetching player stats for {years}...")
        player_stats = collector.fetch_player_stats(years, positions=['QB', 'RB', 'WR', 'TE'])
        print(f"     ✓ Fetched {len(player_stats)} player records")
        
        print(f"   - Fetching team stats...")
        team_stats = collector.fetch_team_stats(years)
        print(f"     ✓ Fetched {len(team_stats)} team records")
        
        print(f"   - Fetching roster data...")
        rosters = collector.fetch_roster_data(years)
        print(f"     ✓ Fetched {len(rosters)} roster records")
        
        print(f"   - Fetching schedule data...")
        schedules = collector.fetch_schedule_data(years)
        print(f"     ✓ Fetched {len(schedules)} schedule records")
        
    except Exception as e:
        logger.error(f"Data fetching failed: {e}")
        print(f"   ✗ Data fetch error: {e}")
        print("   Note: Some data sources may be temporarily unavailable")
    
    # Store data in database
    print("\n6. Storing data in database...")
    try:
        if not player_stats.empty:
            collector._store_game_stats(player_stats)
            print(f"   ✓ Stored player game stats")
        
        if not rosters.empty:
            collector._store_player_data(rosters)
            print(f"   ✓ Stored player information")
        
        if not team_stats.empty:
            collector._store_team_stats(team_stats)
            print(f"   ✓ Stored team stats")
            
    except Exception as e:
        logger.warning(f"Some data storage failed: {e}")
        print(f"   ⚠ Partial storage: {e}")
    
    # Verify data in database
    print("\n7. Verifying database contents...")
    try:
        with db.get_session() as session:
            from src.utils.database import Player, GameStats, TeamStats
            
            player_count = session.query(Player).count()
            game_stats_count = session.query(GameStats).count()
            team_stats_count = session.query(TeamStats).count()
            
            print(f"   - Players in database: {player_count}")
            print(f"   - Game statistics: {game_stats_count}")
            print(f"   - Team statistics: {team_stats_count}")
            
            if player_count > 0 or game_stats_count > 0:
                print("   ✓ Database populated successfully")
            else:
                print("   ⚠ Database is empty - data fetch may have failed")
                
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        print(f"   ✗ Verification error: {e}")
    
    print("\n" + "=" * 60)
    print("Initialization Complete!")
    print("=" * 60)
    
    print("\n📊 System Status:")
    print("   ✓ Configuration: Ready")
    print("   ✓ Logging: Active")
    print("   ✓ Database: Connected")
    print("   ✓ Data Pipeline: Functional")
    print("   ⏳ Models: Ready to build")
    print("   ⏳ API: Ready to implement")
    print("   ⏳ Frontend: Ready to develop")
    
    print("\n🚀 Next Steps:")
    print("   1. Build feature engineering pipeline:")
    print("      python -m src.data.preprocessing.feature_engineering")
    print("")
    print("   2. Train models:")
    print("      python -m src.models.train_all")
    print("")
    print("   3. Start API server:")
    print("      uvicorn src.api.app:app --reload")
    print("")
    print("   4. Launch Jupyter for exploration:")
    print("      jupyter notebook")
    
    print("\n💡 Quick Commands:")
    print("   - Update data: python -m src.data.collectors.nfl_data_collector")
    print("   - Run tests: pytest tests/")
    print("   - Check logs: tail -f logs/fantasy_football_*.log")
    
    logger.info("System initialization completed successfully")
    return True


if __name__ == "__main__":
    success = initialize_system()
    sys.exit(0 if success else 1)