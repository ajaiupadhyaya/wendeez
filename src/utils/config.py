"""
Configuration management for the Elite Fantasy Football Predictor
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from functools import lru_cache
import logging

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class DatabaseConfig(BaseModel):
    """Database configuration"""
    driver: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    name: str = "fantasy_football"
    user: str = "postgres"
    password: str = "postgres"
    pool_size: int = 10
    max_overflow: int = 20
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string"""
        if self.driver == "sqlite":
            # SQLite uses a different connection string format
            from pathlib import Path
            db_path = Path("data") / self.name
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_path}"
        else:
            return f"{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class APIConfig(BaseModel):
    """API configuration"""
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    workers: int = 4
    cors_origins: list = Field(default_factory=list)
    rate_limit: Dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    """Models configuration"""
    training: Dict[str, Any] = Field(default_factory=dict)
    hyperparameter_tuning: Dict[str, Any] = Field(default_factory=dict)
    ensemble: Dict[str, Any] = Field(default_factory=dict)
    deep_learning: Dict[str, Any] = Field(default_factory=dict)
    performance_thresholds: Dict[str, float] = Field(default_factory=dict)


class DataConfig(BaseModel):
    """Data configuration"""
    cache_dir: Path = Path("data/cache")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("data/models")
    update_frequency: str = "daily"
    
    @validator("cache_dir", "raw_dir", "processed_dir", "models_dir", pre=True)
    def resolve_path(cls, v):
        """Resolve relative paths to absolute paths"""
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute():
            v = PROJECT_ROOT / v
        return v


class CacheConfig(BaseModel):
    """Cache configuration"""
    redis: Dict[str, Any] = Field(default_factory=dict)
    disk: Dict[str, Any] = Field(default_factory=dict)


class Config(BaseModel):
    """Main configuration class"""
    app: Dict[str, Any] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    nfl_api: Dict[str, Any] = Field(default_factory=dict)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    features: Dict[str, Any] = Field(default_factory=dict)
    monitoring: Dict[str, Any] = Field(default_factory=dict)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse nested configs
        if 'database' in config_dict:
            config_dict['database'] = DatabaseConfig(**config_dict['database'])
        if 'api' in config_dict:
            config_dict['api'] = APIConfig(**config_dict['api'])
        if 'data' in config_dict:
            config_dict['data'] = DataConfig(**config_dict['data'])
        if 'models' in config_dict:
            config_dict['models'] = ModelsConfig(**config_dict['models'])
        if 'cache' in config_dict:
            config_dict['cache'] = CacheConfig(**config_dict['cache'])
            
        return cls(**config_dict)
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.data.cache_dir,
            self.data.raw_dir,
            self.data.processed_dir,
            self.data.models_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {directory}")


@lru_cache()
def get_config(environment: Optional[str] = None) -> Config:
    """
    Get configuration singleton
    
    Args:
        environment: Environment name (development, production, testing)
        
    Returns:
        Config object
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    config_path = PROJECT_ROOT / "config" / f"{environment}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = Config.from_yaml(config_path)
    config.create_directories()
    
    return config


# Convenience function to get specific config sections
def get_database_config(environment: Optional[str] = None) -> DatabaseConfig:
    """Get database configuration"""
    return get_config(environment).database


def get_api_config(environment: Optional[str] = None) -> APIConfig:
    """Get API configuration"""
    return get_config(environment).api


def get_models_config(environment: Optional[str] = None) -> ModelsConfig:
    """Get models configuration"""
    return get_config(environment).models


def get_data_config(environment: Optional[str] = None) -> DataConfig:
    """Get data configuration"""
    return get_config(environment).data


if __name__ == "__main__":
    # Test configuration loading
    config = get_config("development")
    print(f"App Name: {config.app['name']}")
    print(f"Database: {config.database.connection_string}")
    print(f"API Port: {config.api.port}")
    print(f"Data Cache Dir: {config.data.cache_dir}")
