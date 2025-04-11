import json
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from aegis import db, socketio
from aegis.models import SensorData, AnomalyAlert
from aegis.ml_model import predict_anomaly
from aegis.utils import log_sensor_data, validate_sensor_data

# Create blueprint for API routes
api_bp = Blueprint('api', __name__)

# Setup logger
logger = logging.getLogger(__name__)

@api_bp.route('/api/sensor-input', methods=['POST'])
def receive_sensor_data():
    """
    Endpoint for receiving sensor data from IoT devices.
    
    Expected JSON format:
    {
        "timestamp": "2025-04-11T13:45:00Z",
        "device_id": "iot-001",
        "cmos_data": [134, 110, 98],
        "ir_temp": 36.5,
        "motion_detected": true,
        "pressure": 1013.25,
        "rpm": 1220
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate the input data
        validation_result = validate_sensor_data(data)
        if not validation_result['valid']:
            return jsonify({
                'success': False,
                'error': validation_result['error']
            }), 400
        
        # Parse timestamp if provided, or use current time
        try:
            if 'timestamp' in data and data['timestamp']:
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            else:
                timestamp = datetime.utcnow()
        except ValueError:
            timestamp = datetime.utcnow()
            logger.warning(f"Invalid timestamp format. Using current time: {timestamp}")
        
        # Prepare for ML model prediction
        anomaly_score = predict_anomaly(data)
        
        # Determine if it's a potential malware based on threshold
        threshold = current_app.config.get('ANOMALY_THRESHOLD', 0.75)
        is_potential_malware = anomaly_score > threshold
        
        # Store CMOS data as a JSON string
        cmos_data_json = json.dumps(data.get('cmos_data', []))
        
        # Create a new SensorData instance
        # Convert NumPy types to Python native types to avoid DB issues
        sensor_data = SensorData(
            timestamp=timestamp,
            device_id=data.get('device_id', 'unknown'),
            cmos_data=cmos_data_json,
            ir_temp=float(data.get('ir_temp')) if data.get('ir_temp') is not None else None,
            motion_detected=bool(data.get('motion_detected', False)),
            pressure=float(data.get('pressure')) if data.get('pressure') is not None else None,
            rpm=float(data.get('rpm')) if data.get('rpm') is not None else None,
            anomaly_score=float(anomaly_score),  # Convert NumPy float64 to Python float
            is_potential_malware=bool(is_potential_malware)
        )
        
        # Save to database
        db.session.add(sensor_data)
        db.session.commit()
        
        # Log the sensor data to CSV
        log_sensor_data(data, anomaly_score, is_potential_malware)
        
        # If it's an anomaly, create an alert and emit a WebSocket event
        if is_potential_malware:
            # Create and save anomaly alert
            alert = AnomalyAlert(
                sensor_data_id=sensor_data.id,
                timestamp=timestamp,
                score=float(anomaly_score),  # Convert NumPy float64 to Python float
                description=f"Anomaly detected for device {data.get('device_id', 'unknown')} with score {float(anomaly_score):.2f}"
            )
            db.session.add(alert)
            db.session.commit()
            
            # Log to alerts.log
            logger.warning(f"ANOMALY DETECTED: Device {data.get('device_id', 'unknown')} - Score: {anomaly_score:.2f}")
            
            # Emit WebSocket event
            alert_data = {
                "type": "anomaly_alert",
                "device_id": data.get('device_id', 'unknown'),
                "timestamp": timestamp.isoformat() + 'Z',
                "score": float(anomaly_score),  # Convert NumPy float to Python float
                "sensor_data": data,
                "is_potential_malware": True
            }
            socketio.emit('aegis_alert', alert_data)
        
        # Return success response
        return jsonify({
            'success': True,
            'data': {
                'id': sensor_data.id,
                'anomaly_score': float(anomaly_score),  # Convert NumPy float to Python float
                'is_potential_malware': bool(is_potential_malware)  # Ensure it's a Python bool
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing sensor data: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}"
        }), 500

@api_bp.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Endpoint to retrieve anomaly alerts."""
    try:
        # Get query parameters
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        # Query the database
        alerts = AnomalyAlert.query.order_by(
            AnomalyAlert.timestamp.desc()
        ).limit(limit).offset(offset).all()
        
        # Convert to list of dictionaries
        result = [alert.to_dict() for alert in alerts]
        
        # Return the results
        return jsonify({
            'success': True,
            'data': result,
            'count': len(result),
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        logger.error(f"Error retrieving alerts: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}"
        }), 500

@api_bp.route('/api/devices', methods=['GET'])
def get_devices():
    """Endpoint to get a list of devices with their latest status."""
    try:
        # Subquery to get the latest record for each device
        from sqlalchemy import func
        subq = db.session.query(
            SensorData.device_id, 
            func.max(SensorData.timestamp).label('max_timestamp')
        ).group_by(SensorData.device_id).subquery()
        
        # Join with the main table to get the full records
        latest_data = db.session.query(SensorData).join(
            subq, 
            db.and_(
                SensorData.device_id == subq.c.device_id,
                SensorData.timestamp == subq.c.max_timestamp
            )
        ).all()
        
        # Convert to list of dictionaries
        result = [data.to_dict() for data in latest_data]
        
        # Return the results
        return jsonify({
            'success': True,
            'data': result,
            'count': len(result)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving devices: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}"
        }), 500

@api_bp.route('/api/device/<device_id>/history', methods=['GET'])
def get_device_history(device_id):
    """Endpoint to get historical data for a specific device."""
    try:
        # Get query parameters
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        # Query the database
        history = SensorData.query.filter_by(
            device_id=device_id
        ).order_by(
            SensorData.timestamp.desc()
        ).limit(limit).offset(offset).all()
        
        # Convert to list of dictionaries
        result = [data.to_dict() for data in history]
        
        # Return the results
        return jsonify({
            'success': True,
            'device_id': device_id,
            'data': result,
            'count': len(result),
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        logger.error(f"Error retrieving device history: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}"
        }), 500

@api_bp.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })

@api_bp.route('/api/train-model', methods=['POST'])
def train_model():
    """Endpoint to manually trigger model training."""
    try:
        from aegis.ml_model import anomaly_detector, save_trained_model
        
        # Retrain the model using all available data
        # Set is_fitted to False to force retraining
        anomaly_detector.is_fitted = False
        
        # Trigger training with a dummy prediction
        dummy_data = {
            'ir_temp': 25.0,
            'pressure': 1013.0,
            'rpm': 1000,
            'motion_detected': False,
            'cmos_data': [100, 100, 100]
        }
        score = anomaly_detector.predict_anomaly(dummy_data)
        
        # Save the trained model
        saved = save_trained_model()
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'model_saved': saved,
            'test_score': float(score)
        })
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error training model: {str(e)}"
        }), 500
        
@api_bp.route('/api/export-model', methods=['GET'])
def export_model_endpoint():
    """
    Endpoint to export the trained model for use on other platforms.
    
    Returns a download link to a zip file containing the model, metadata,
    and sample code for inference.
    """
    try:
        from aegis.ml_model import export_model
        import os
        from flask import send_file, url_for
        
        # Export the model
        export_path = export_model()
        
        if not export_path or not os.path.exists(export_path):
            return jsonify({
                'success': False,
                'error': 'Failed to export model'
            }), 500
        
        # Return the exported model file
        return send_file(export_path, as_attachment=True, download_name='aegis_model.zip')
    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error exporting model: {str(e)}"
        }), 500
        
@api_bp.route('/api/import-model', methods=['POST'])
def import_model_endpoint():
    """
    Endpoint to import a model trained on another platform.
    
    Accepts multipart form data with:
    - model_file: The model file to import
    - metadata_file: Optional metadata file
    - format: The format of the model file (joblib, pickle, onnx)
    """
    try:
        from aegis.ml_model import import_model
        import os
        from werkzeug.utils import secure_filename
        
        # Check if files are in the request
        if 'model_file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No model file provided'
            }), 400
            
        # Get the model file
        model_file = request.files['model_file']
        if model_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No model file selected'
            }), 400
            
        # Get the metadata file (optional)
        metadata_file = None
        if 'metadata_file' in request.files:
            metadata_file = request.files['metadata_file']
            if metadata_file.filename != '':
                metadata_path = os.path.join('models', 'imported', secure_filename(metadata_file.filename))
                os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
                metadata_file.save(metadata_path)
            else:
                metadata_file = None
                metadata_path = None
        else:
            metadata_path = None
            
        # Get the format
        model_format = request.form.get('format', 'joblib')
        
        # Save the model file
        model_path = os.path.join('models', 'imported', secure_filename(model_file.filename))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_file.save(model_path)
        
        # Import the model
        success = import_model(model_path, metadata_path, model_format)
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Failed to import model'
            }), 500
            
        return jsonify({
            'success': True,
            'message': 'Model imported successfully',
            'model_path': model_path,
            'metadata_path': metadata_path,
            'format': model_format
        })
    except Exception as e:
        logger.error(f"Error importing model: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error importing model: {str(e)}"
        }), 500
