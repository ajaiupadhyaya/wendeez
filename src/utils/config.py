"""
Enterprise Configuration Management for Elite Fantasy Football Predictor

This module provides comprehensive configuration management with:
- Type-safe configuration models using Pydantic
- Environment-based configuration loading
- Validation and error handling
- Secret management integration
- Configuration caching and optimization
"""

import os
import sys
import yaml
import json
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, validator, SecretStr
from functools import lru_cache
import logging
from enum import Enum
from datetime import datetime, timedelta

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Setup logging for configuration module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Supported environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Supported log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SecurityConfig(BaseModel):
    """Security configuration with encryption and authentication"""
    secret_key: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(32)))
    api_key_header: str = "X-API-Key"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"])
    cors_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    rate_limit_per_minute: int = 100
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    
    @validator("cors_origins")
    def validate_cors_origins(cls, v):
        """Validate CORS origins are valid URLs"""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        for origin in v:
            if not url_pattern.match(origin):
                raise ValueError(f"Invalid CORS origin: {origin}")
        return v


class DatabaseConfig(BaseModel):
    """Enhanced database configuration with connection pooling and SSL"""
    driver: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    name: str = "fantasy_football"
    user: str = "postgres"
    password: SecretStr = SecretStr("postgres")
    
    # Connection pool settings
    pool_size: int = 20
    max_overflow: int = 30
    pool_pre_ping: bool = True
    pool_recycle: int = 3600  # 1 hour
    
    # SSL and security
    ssl_mode: str = "prefer"
    ssl_cert_path: Optional[Path] = None
    ssl_key_path: Optional[Path] = None
    ssl_ca_path: Optional[Path] = None
    
    # Performance settings
    echo: bool = False
    query_timeout: int = 30
    connection_timeout: int = 10
    
    # Backup and maintenance
    backup_retention_days: int = 30
    auto_vacuum: bool = True
    
    @property
    def connection_string(self) -> str:
        """Generate secure database connection string"""
        if self.driver == "sqlite":
            db_path = PROJECT_ROOT / "data" / f"{self.name}.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_path}"
        
        password = self.password.get_secret_value()
        base_url = f"{self.driver}://{self.user}:{password}@{self.host}:{self.port}/{self.name}"
        
        # Add SSL parameters if configured
        ssl_params = []
        if self.ssl_mode != "disable":
            ssl_params.append(f"sslmode={self.ssl_mode}")
        if self.ssl_cert_path:
            ssl_params.append(f"sslcert={self.ssl_cert_path}")
        if self.ssl_key_path:
            ssl_params.append(f"sslkey={self.ssl_key_path}")
        if self.ssl_ca_path:
            ssl_params.append(f"sslrootcert={self.ssl_ca_path}")
        
        if ssl_params:
            base_url += "?" + "&".join(ssl_params)
        
        return base_url
    
    @property
    def async_connection_string(self) -> str:
        """Generate async database connection string"""
        return self.connection_string.replace("postgresql://", "postgresql+asyncpg://")


class RedisConfig(BaseModel):
    """Redis configuration for caching and session management"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[SecretStr] = None
    ssl: bool = False
    
    # Connection settings
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    max_connections: int = 50
    
    # Cache settings
    default_ttl: int = 3600  # 1 hour
    key_prefix: str = "ff_predictor:"
    
    @property
    def connection_string(self) -> str:
        """Generate Redis connection string"""
        auth = ""
        if self.password:
            auth = f":{self.password.get_secret_value()}@"
        
        protocol = "rediss" if self.ssl else "redis"
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


class APIConfig(BaseModel):
    """Enhanced API configuration with security and performance settings"""
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    workers: int = 4
    worker_class: str = "uvicorn.workers.UvicornWorker"
    
    # Security settings
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"])
    cors_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    cors_headers: List[str] = Field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Request settings
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 30
    keep_alive_timeout: int = 5
    
    # SSL/TLS
    ssl_keyfile: Optional[Path] = None
    ssl_certfile: Optional[Path] = None
    ssl_ca_certs: Optional[Path] = None
    
    # Logging
    access_log: bool = True
    error_log: bool = True
    log_level: LogLevel = LogLevel.INFO


class MLConfig(BaseModel):
    """Machine Learning and Model Configuration"""
    
    # Model training settings
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    cross_validation_folds: int = 5
    
    # Feature engineering
    feature_selection_method: str = "mutual_info_regression"
    feature_selection_k: int = 50
    polynomial_degree: int = 2
    interaction_features: bool = True
    
    # Ensemble settings
    ensemble_methods: List[str] = Field(default_factory=lambda: [
        "random_forest", "gradient_boosting", "xgboost", "lightgbm", "neural_network"
    ])
    ensemble_voting: str = "soft"  # soft or hard
    ensemble_weights: Optional[List[float]] = None
    
    # Hyperparameter optimization
    optimization_trials: int = 100
    optimization_timeout: int = 3600  # 1 hour
    optimization_sampler: str = "tpe"  # tpe, random, grid
    
    # Model persistence
    model_save_format: str = "joblib"  # joblib, pickle, onnx
    model_compression: bool = True
    model_versioning: bool = True
    
    # Performance thresholds
    min_accuracy: float = 0.7
    min_precision: float = 0.65
    min_recall: float = 0.65
    min_f1_score: float = 0.65
    
    # Deep learning specific
    dl_batch_size: int = 32
    dl_epochs: int = 100
    dl_learning_rate: float = 0.001
    dl_early_stopping_patience: int = 10
    dl_dropout_rate: float = 0.2
    dl_l2_regularization: float = 0.01


class DataConfig(BaseModel):
    """Enhanced data configuration with validation and processing settings"""
    cache_dir: Path = Path("data/cache")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("data/models")
    logs_dir: Path = Path("logs")
    backups_dir: Path = Path("data/backups")
    
    # Data collection settings
    update_frequency: str = "daily"
    data_retention_days: int = 365
    max_concurrent_requests: int = 10
    request_delay: float = 1.0  # seconds between requests
    
    # Data validation
    validate_data_quality: bool = True
    data_quality_threshold: float = 0.95
    missing_data_threshold: float = 0.1
    outlier_detection_method: str = "isolation_forest"
    
    # Data processing
    chunk_size: int = 10000
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    
    @validator("cache_dir", "raw_dir", "processed_dir", "models_dir", "logs_dir", "backups_dir", pre=True)
    def resolve_path(cls, v):
        """Resolve relative paths to absolute paths"""
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute():
            v = PROJECT_ROOT / v
        v.mkdir(parents=True, exist_ok=True)
        return v


class MonitoringConfig(BaseModel):
    """System monitoring and observability configuration"""
    
    # Metrics collection
    collect_metrics: bool = True
    metrics_interval: int = 60  # seconds
    metrics_retention_days: int = 30
    
    # Performance monitoring
    track_prediction_latency: bool = True
    track_model_accuracy: bool = True
    track_data_drift: bool = True
    track_system_resources: bool = True
    
    # Alerting
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "prediction_latency_ms": 1000,
        "model_accuracy_drop": 0.05,
        "memory_usage_percent": 85,
        "cpu_usage_percent": 90,
        "disk_usage_percent": 90
    })
    
    # External monitoring
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    grafana_enabled: bool = False
    sentry_enabled: bool = False
    sentry_dsn: Optional[SecretStr] = None
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_rotation: str = "midnight"
    log_backup_count: int = 7


class ExternalAPIConfig(BaseModel):
    """Configuration for external API integrations"""
    
    # NFL Data APIs
    nfl_api_key: Optional[SecretStr] = None
    nfl_api_base_url: str = "https://api.nfl.com"
    nfl_api_timeout: int = 30
    nfl_api_retries: int = 3
    
    # Sports data providers
    sportradar_api_key: Optional[SecretStr] = None
    espn_api_key: Optional[SecretStr] = None
    fantasypros_api_key: Optional[SecretStr] = None
    
    # Weather API
    weather_api_key: Optional[SecretStr] = None
    weather_api_base_url: str = "https://api.openweathermap.org"
    
    # Rate limiting for external APIs
    api_rate_limits: Dict[str, int] = Field(default_factory=lambda: {
        "nfl_api": 100,  # requests per minute
        "weather_api": 60,
        "sportradar_api": 120
    })


class Config(BaseModel):
    """Main enterprise configuration class"""
    
    # Environment and application info
    environment: Environment = Environment.DEVELOPMENT
    app_name: str = "Elite Fantasy Football Predictor"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Core configurations
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    external_apis: ExternalAPIConfig = Field(default_factory=ExternalAPIConfig)
    
    # Additional settings
    timezone: str = "UTC"
    locale: str = "en_US"
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load configuration from YAML file with environment variable override"""
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {yaml_path}, using defaults")
            config_dict = {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        
        # Override with environment variables
        config_dict = cls._override_with_env_vars(config_dict)
        
        return cls(**config_dict)
    
    @classmethod
    def _override_with_env_vars(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables"""
        
        # Environment
        if env := os.getenv("ENVIRONMENT"):
            config_dict["environment"] = env
        
        # Database overrides
        db_config = config_dict.get("database", {})
        if db_host := os.getenv("DB_HOST"):
            db_config["host"] = db_host
        if db_port := os.getenv("DB_PORT"):
            try:
                db_config["port"] = int(db_port)
            except ValueError:
                logger.warning(f"Invalid DB_PORT value: {db_port}")
        if db_name := os.getenv("DB_NAME"):
            db_config["name"] = db_name
        if db_user := os.getenv("DB_USER"):
            db_config["user"] = db_user
        if db_password := os.getenv("DB_PASSWORD"):
            db_config["password"] = SecretStr(db_password)
        config_dict["database"] = db_config
        
        # Redis overrides
        redis_config = config_dict.get("redis", {})
        if redis_host := os.getenv("REDIS_HOST"):
            redis_config["host"] = redis_host
        if redis_port := os.getenv("REDIS_PORT"):
            try:
                redis_config["port"] = int(redis_port)
            except ValueError:
                logger.warning(f"Invalid REDIS_PORT value: {redis_port}")
        if redis_password := os.getenv("REDIS_PASSWORD"):
            redis_config["password"] = SecretStr(redis_password)
        config_dict["redis"] = redis_config
        
        # API overrides
        api_config = config_dict.get("api", {})
        if api_host := os.getenv("API_HOST"):
            api_config["host"] = api_host
        if api_port := os.getenv("API_PORT"):
            try:
                api_config["port"] = int(api_port)
            except ValueError:
                logger.warning(f"Invalid API_PORT value: {api_port}")
        config_dict["api"] = api_config
        
        return config_dict
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.data.cache_dir,
            self.data.raw_dir,
            self.data.processed_dir,
            self.data.models_dir,
            self.data.logs_dir,
            self.data.backups_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def validate_configuration(self) -> bool:
        """Validate the complete configuration"""
        try:
            # Test database connection string
            _ = self.database.connection_string
            
            # Validate directories
            self.create_directories()
            
            # Validate external API configurations
            if self.external_apis.nfl_api_key:
                logger.info("NFL API key configured")
            
            # Validate ML configuration
            if not 0 < self.ml.test_size < 1:
                raise ValueError("ML test_size must be between 0 and 1")
            
            if not 0 < self.ml.validation_size < 1:
                raise ValueError("ML validation_size must be between 0 and 1")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


@lru_cache()
def get_config(environment: Optional[str] = None) -> Config:
    """
    Get configuration singleton with caching
    
    Args:
        environment: Environment name (development, production, testing, staging)
        
    Returns:
        Config object
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    config_path = PROJECT_ROOT / "config" / f"{environment}.yaml"
    
    try:
        config = Config.from_yaml(config_path)
        
        # Validate configuration
        if not config.validate_configuration():
            logger.warning("Configuration validation failed, but proceeding with defaults")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        return Config()


# Convenience functions for accessing specific configurations
def get_database_config(environment: Optional[str] = None) -> DatabaseConfig:
    """Get database configuration"""
    return get_config(environment).database


def get_api_config(environment: Optional[str] = None) -> APIConfig:
    """Get API configuration"""
    return get_config(environment).api


def get_ml_config(environment: Optional[str] = None) -> MLConfig:
    """Get ML configuration"""
    return get_config(environment).ml


def get_data_config(environment: Optional[str] = None) -> DataConfig:
    """Get data configuration"""
    return get_config(environment).data


def get_monitoring_config(environment: Optional[str] = None) -> MonitoringConfig:
    """Get monitoring configuration"""
    return get_config(environment).monitoring


def get_security_config(environment: Optional[str] = None) -> SecurityConfig:
    """Get security configuration"""
    return get_config(environment).security


if __name__ == "__main__":
    # Test configuration loading and validation
    import sys
    
    print("=" * 50)
    print("Configuration Test Suite")
    print("=" * 50)
    
    try:
        config = get_config("development")
        print(f"✓ App Name: {config.app_name}")
        print(f"✓ Environment: {config.environment}")
        print(f"✓ Database: {config.database.connection_string}")
        print(f"✓ API Host:Port: {config.api.host}:{config.api.port}")
        print(f"✓ Cache Dir: {config.data.cache_dir}")
        print(f"✓ ML Random State: {config.ml.random_state}")
        print(f"✓ Security Secret Key: {'***' if config.security.secret_key else 'None'}")
        
        print("\n✓ All configuration tests passed!")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        sys.exit(1)
