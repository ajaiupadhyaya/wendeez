"""
FastAPI Backend for Elite Fantasy Football Predictor
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.database import get_db_manager
from src.data.collectors.nfl_data_collector import NFLDataCollector
from src.data.collectors.defensive_matchup_analyzer import DefensiveMatchupAnalyzer
from src.data.preprocessing.feature_engineering import FeatureEngineer
from src.models.ensemble_predictor import EnsemblePredictor

# Initialize
config = get_config()
logger = get_logger()
db_manager = get_db_manager()
nfl_collector = NFLDataCollector()
matchup_analyzer = DefensiveMatchupAnalyzer()
feature_engineer = FeatureEngineer()
predictor = EnsemblePredictor()

# Load models on startup
try:
    predictor.load_models()
    logger.info("Models loaded successfully")
except:
    logger.warning("No pre-trained models found")

# Create FastAPI app
app = FastAPI(
    title="Elite Fantasy Football Predictor API",
    description="Advanced NFL fantasy football predictions using ML/DL",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class PlayerInfo(BaseModel):
    player_id: str
    name: str
    position: str
    team: str
    age: Optional[float]
    height: Optional[float]
    weight: Optional[float]


class PredictionRequest(BaseModel):
    player_id: str
    week: int = Field(..., ge=1, le=18)
    season: int = Field(default=2025, ge=2020, le=2030)


class PredictionResponse(BaseModel):
    player_id: str
    player_name: str
    position: str
    team: str
    week: int
    season: int
    predicted_points: float
    floor: float
    ceiling: float
    confidence_80_lower: float
    confidence_80_upper: float
    uncertainty: float
    matchup_score: float
    opponent: str
    recommendation: str
    risk_level: str


class PlayerStats(BaseModel):
    player_id: str
    name: str
    position: str
    team: str
    games_played: int
    avg_points: float
    total_points: float
    last_game_points: float
    consistency_score: float
    trend: str


class MatchupAnalysis(BaseModel):
    player_id: str
    week: int
    opponent: str
    matchup_score: float
    difficulty: str
    defense_rank: int
    avg_points_allowed: float
    is_home: bool
    recommendation: str


class LineupOptimizationRequest(BaseModel):
    player_ids: List[str]
    week: int
    budget: Optional[float] = None
    constraints: Optional[Dict] = None


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Elite Fantasy Football Predictor API",
        "status": "operational",
        "version": "1.0.0"
    }


@app.get("/api/players", response_model=List[PlayerInfo])
async def get_players(
    position: Optional[str] = Query(None, description="Filter by position"),
    team: Optional[str] = Query(None, description="Filter by team"),
    search: Optional[str] = Query(None, description="Search by name"),
    limit: int = Query(100, le=500)
):
    """Get list of players"""
    try:
        query = "SELECT * FROM players WHERE 1=1"
        
        if position:
            query += f" AND position = '{position}'"
        if team:
            query += f" AND team = '{team}'"
        if search:
            query += f" AND name LIKE '%{search}%'"
        
        query += f" LIMIT {limit}"
        
        df = db_manager.execute_query(query)
        
        players = []
        for _, row in df.iterrows():
            players.append(PlayerInfo(
                player_id=row['player_id'],
                name=row['name'],
                position=row['position'],
                team=row['team'],
                age=row.get('age'),
                height=row.get('height'),
                weight=row.get('weight')
            ))
        
        return players
        
    except Exception as e:
        logger.error(f"Error fetching players: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_player_performance(request: PredictionRequest):
    """Predict player fantasy points for a specific week"""
    try:
        # Get player info
        player_query = f"""
        SELECT * FROM players 
        WHERE player_id = '{request.player_id}'
        """
        player_df = db_manager.execute_query(player_query)
        
        if player_df.empty:
            raise HTTPException(status_code=404, detail="Player not found")
        
        player = player_df.iloc[0]
        
        # Get prediction
        prediction = predictor.predict_player(
            request.player_id,
            request.week,
            request.season
        )
        
        if not prediction:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Get matchup info
        matchup_data = matchup_analyzer.calculate_matchup_score(
            player['position'],
            "",  # Would need to get opponent from schedule
            request.season - 1
        )
        
        # Determine recommendation
        if prediction['predicted_points'] > 15:
            recommendation = "MUST START"
        elif prediction['predicted_points'] > 10:
            recommendation = "START"
        elif prediction['predicted_points'] > 7:
            recommendation = "FLEX"
        else:
            recommendation = "BENCH"
        
        # Determine risk level
        if prediction['uncertainty'] > 5:
            risk_level = "HIGH"
        elif prediction['uncertainty'] > 3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return PredictionResponse(
            player_id=request.player_id,
            player_name=player['name'],
            position=player['position'],
            team=player['team'],
            week=request.week,
            season=request.season,
            predicted_points=prediction['predicted_points'],
            floor=prediction['lower_bound'],
            ceiling=prediction['upper_bound'],
            confidence_80_lower=prediction['confidence_80_lower'],
            confidence_80_upper=prediction['confidence_80_upper'],
            uncertainty=prediction['uncertainty'],
            matchup_score=matchup_data.get('matchup_score', 50),
            opponent="TBD",  # Would get from schedule
            recommendation=recommendation,
            risk_level=risk_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/players/{player_id}/stats", response_model=PlayerStats)
async def get_player_stats(player_id: str):
    """Get player statistics"""
    try:
        # Get player info
        player_query = f"""
        SELECT * FROM players 
        WHERE player_id = '{player_id}'
        """
        player_df = db_manager.execute_query(player_query)
        
        if player_df.empty:
            raise HTTPException(status_code=404, detail="Player not found")
        
        player = player_df.iloc[0]
        
        # Get stats
        stats_query = f"""
        SELECT * FROM game_stats
        WHERE player_id = '{player_id}'
        ORDER BY season DESC, week DESC
        """
        stats_df = db_manager.execute_query(stats_query)
        
        if stats_df.empty:
            games_played = 0
            avg_points = 0
            total_points = 0
            last_game_points = 0
            consistency = 0
            trend = "STABLE"
        else:
            games_played = len(stats_df)
            avg_points = stats_df['fantasy_points'].mean()
            total_points = stats_df['fantasy_points'].sum()
            last_game_points = stats_df.iloc[0]['fantasy_points'] if len(stats_df) > 0 else 0
            consistency = avg_points / (stats_df['fantasy_points'].std() + 1)
            
            # Calculate trend
            if len(stats_df) >= 3:
                recent = stats_df.head(3)['fantasy_points'].mean()
                older = stats_df.tail(3)['fantasy_points'].mean()
                if recent > older * 1.1:
                    trend = "IMPROVING"
                elif recent < older * 0.9:
                    trend = "DECLINING"
                else:
                    trend = "STABLE"
            else:
                trend = "INSUFFICIENT_DATA"
        
        return PlayerStats(
            player_id=player_id,
            name=player['name'],
            position=player['position'],
            team=player['team'],
            games_played=games_played,
            avg_points=avg_points,
            total_points=total_points,
            last_game_points=last_game_points,
            consistency_score=consistency,
            trend=trend
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching player stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/players/{player_id}/matchup/{week}", response_model=MatchupAnalysis)
async def get_matchup_analysis(player_id: str, week: int):
    """Get matchup analysis for a player"""
    try:
        # Get player info
        player_query = f"""
        SELECT * FROM players 
        WHERE player_id = '{player_id}'
        """
        player_df = db_manager.execute_query(player_query)
        
        if player_df.empty:
            raise HTTPException(status_code=404, detail="Player not found")
        
        player = player_df.iloc[0]
        
        # Get schedule for the week
        schedule = matchup_analyzer.get_2025_schedule()
        game = schedule[
            ((schedule['home_team'] == player['team']) | 
             (schedule['away_team'] == player['team'])) &
            (schedule['week'] == week)
        ]
        
        if game.empty:
            opponent = "BYE"
            is_home = False
            matchup_score = 0
            difficulty = "bye_week"
            defense_rank = 0
            avg_points_allowed = 0
            recommendation = "BYE WEEK"
        else:
            game = game.iloc[0]
            opponent = game['away_team'] if game['home_team'] == player['team'] else game['home_team']
            is_home = game['home_team'] == player['team']
            
            # Get matchup data
            matchup_data = matchup_analyzer.calculate_matchup_score(
                player['position'],
                opponent,
                2024
            )
            
            matchup_score = matchup_data.get('matchup_score', 50)
            difficulty = matchup_data.get('difficulty', 'average')
            defense_rank = matchup_data.get('defense_rank', 16)
            avg_points_allowed = matchup_data.get('avg_points_allowed', 15)
            
            # Recommendation based on matchup
            if matchup_score >= 70:
                recommendation = "EXCELLENT MATCHUP - START"
            elif matchup_score >= 50:
                recommendation = "GOOD MATCHUP - START"
            elif matchup_score >= 30:
                recommendation = "TOUGH MATCHUP - CONSIDER ALTERNATIVES"
            else:
                recommendation = "VERY TOUGH MATCHUP - BENCH IF POSSIBLE"
        
        return MatchupAnalysis(
            player_id=player_id,
            week=week,
            opponent=opponent,
            matchup_score=matchup_score,
            difficulty=difficulty,
            defense_rank=defense_rank,
            avg_points_allowed=avg_points_allowed,
            is_home=is_home,
            recommendation=recommendation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing matchup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rankings/{position}")
async def get_position_rankings(
    position: str,
    week: Optional[int] = Query(None, description="Week number"),
    scoring: str = Query("standard", description="Scoring system")
):
    """Get position rankings"""
    try:
        # Get all players at position
        query = f"""
        SELECT p.*, 
               AVG(gs.fantasy_points) as avg_points,
               COUNT(gs.id) as games_played
        FROM players p
        LEFT JOIN game_stats gs ON p.player_id = gs.player_id
        WHERE p.position = '{position}'
        GROUP BY p.player_id
        HAVING games_played > 0
        ORDER BY avg_points DESC
        LIMIT 50
        """
        
        df = db_manager.execute_query(query)
        
        rankings = []
        for rank, (_, player) in enumerate(df.iterrows(), 1):
            # Get prediction if week specified
            if week:
                try:
                    prediction = predictor.predict_player(
                        player['player_id'],
                        week,
                        2025
                    )
                    projected_points = prediction.get('predicted_points', player['avg_points'])
                except:
                    projected_points = player['avg_points']
            else:
                projected_points = player['avg_points']
            
            rankings.append({
                'rank': rank,
                'player_id': player['player_id'],
                'name': player['name'],
                'team': player['team'],
                'avg_points': player['avg_points'],
                'projected_points': projected_points,
                'games_played': player['games_played']
            })
        
        return rankings
        
    except Exception as e:
        logger.error(f"Error getting rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize-lineup")
async def optimize_lineup(request: LineupOptimizationRequest):
    """Optimize fantasy lineup"""
    try:
        predictions = []
        
        # Get predictions for all players
        for player_id in request.player_ids:
            try:
                pred = predictor.predict_player(player_id, request.week, 2025)
                
                # Get player info
                player_query = f"SELECT * FROM players WHERE player_id = '{player_id}'"
                player_df = db_manager.execute_query(player_query)
                
                if not player_df.empty:
                    player = player_df.iloc[0]
                    predictions.append({
                        'player_id': player_id,
                        'name': player['name'],
                        'position': player['position'],
                        'predicted_points': pred.get('predicted_points', 0),
                        'uncertainty': pred.get('uncertainty', 0)
                    })
            except:
                continue
        
        # Sort by predicted points
        predictions.sort(key=lambda x: x['predicted_points'], reverse=True)
        
        # Build optimal lineup based on position constraints
        lineup_constraints = {
            'QB': 1,
            'RB': 2,
            'WR': 2,
            'TE': 1,
            'FLEX': 1  # RB/WR/TE
        }
        
        lineup = {pos: [] for pos in lineup_constraints.keys()}
        bench = []
        
        for player in predictions:
            pos = player['position']
            
            # Check if can be added to main position
            if pos in lineup and len(lineup[pos]) < lineup_constraints.get(pos, 0):
                lineup[pos].append(player)
            # Check FLEX eligibility
            elif pos in ['RB', 'WR', 'TE'] and len(lineup['FLEX']) < 1:
                lineup['FLEX'].append(player)
            else:
                bench.append(player)
        
        # Calculate total projected points
        total_points = sum(
            player['predicted_points'] 
            for position_players in lineup.values() 
            for player in position_players
        )
        
        return {
            'lineup': lineup,
            'bench': bench,
            'total_projected_points': total_points,
            'week': request.week
        }
        
    except Exception as e:
        logger.error(f"Error optimizing lineup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trends/{player_id}")
async def get_player_trends(player_id: str, weeks: int = Query(10, le=20)):
    """Get player performance trends"""
    try:
        query = f"""
        SELECT week, season, fantasy_points
        FROM game_stats
        WHERE player_id = '{player_id}'
        ORDER BY season DESC, week DESC
        LIMIT {weeks}
        """
        
        df = db_manager.execute_query(query)
        
        if df.empty:
            return {"error": "No data found"}
        
        # Reverse for chronological order
        df = df.iloc[::-1]
        
        return {
            'player_id': player_id,
            'data': df.to_dict('records'),
            'avg': df['fantasy_points'].mean(),
            'trend': 'up' if len(df) > 1 and df.iloc[-1]['fantasy_points'] > df.iloc[0]['fantasy_points'] else 'down'
        }
        
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        db_manager.execute_query("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "models_loaded": predictor.is_trained,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
