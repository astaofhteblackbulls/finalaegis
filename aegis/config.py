import os

class Config:
    """Configuration settings for the AEGIS application."""
    
    # Database configuration
    # Use DATABASE_URL environment variable, or default to SQLite for development
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///aegis.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    
    # Flask configuration
    DEBUG = os.environ.get('FLASK_DEBUG', True)
    
    # AEGIS specific configuration
    ANOMALY_THRESHOLD = 0.6  # Threshold for anomaly detection (lowered from 0.75)
