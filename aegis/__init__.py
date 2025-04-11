import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from flask_cors import CORS
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a handler for the alerts.log file
alerts_handler = logging.FileHandler('logs/alerts.log')
alerts_handler.setLevel(logging.WARNING)
alerts_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
alerts_handler.setFormatter(alerts_formatter)
logger.addHandler(alerts_handler)

# Setup SQLAlchemy
class Base(DeclarativeBase):
    pass

# Initialize extensions
db = SQLAlchemy(model_class=Base)
socketio = SocketIO()

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # Enable CORS
    CORS(app)
    
    # Load configuration
    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_object('aegis.config.Config')
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)
    
    # Set the secret key
    app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
    
    # Initialize the database
    db.init_app(app)
    
    # Initialize SocketIO
    socketio.init_app(app, cors_allowed_origins="*")
    
    # Create logs directory if it doesn't exist
    try:
        os.makedirs('logs', exist_ok=True)
    except OSError:
        logger.error("Error creating logs directory")
    
    with app.app_context():
        # Import models to ensure they're registered with SQLAlchemy
        from aegis import models
        
        # Create all database tables
        db.create_all()
        
        # Register blueprints
        from aegis.routes import api_bp
        app.register_blueprint(api_bp)
        
        # Register web routes
        from aegis.web_routes import register_web_routes
        register_web_routes(app)
        
        return app
