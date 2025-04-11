import csv
import json
import os
import logging
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

def validate_sensor_data(data):
    """
    Validate the incoming sensor data.
    
    Args:
        data: Dictionary containing the sensor data
        
    Returns:
        Dictionary with 'valid' boolean and optional 'error' message
    """
    # Check if data is a dictionary
    if not isinstance(data, dict):
        return {'valid': False, 'error': 'Data must be a JSON object'}
    
    # Check required fields
    if 'device_id' not in data:
        return {'valid': False, 'error': 'Missing required field: device_id'}
    
    # Check data types
    if 'ir_temp' in data and not isinstance(data['ir_temp'], (int, float)):
        return {'valid': False, 'error': 'ir_temp must be a number'}
    
    if 'pressure' in data and not isinstance(data['pressure'], (int, float)):
        return {'valid': False, 'error': 'pressure must be a number'}
    
    if 'rpm' in data and not isinstance(data['rpm'], (int, float)):
        return {'valid': False, 'error': 'rpm must be a number'}
    
    if 'motion_detected' in data and not isinstance(data['motion_detected'], bool):
        return {'valid': False, 'error': 'motion_detected must be a boolean'}
    
    if 'cmos_data' in data and not isinstance(data['cmos_data'], list):
        return {'valid': False, 'error': 'cmos_data must be an array'}
    
    # All checks passed
    return {'valid': True}

def log_sensor_data(data, anomaly_score, is_potential_malware):
    """
    Log sensor data to a CSV file.
    
    Args:
        data: Dictionary containing the sensor data
        anomaly_score: The calculated anomaly score
        is_potential_malware: Boolean indicating if this is potential malware
    """
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Path to the log file
        log_file = 'logs/data_log.csv'
        
        # Check if file exists to determine if we need to write the header
        file_exists = os.path.isfile(log_file)
        
        # Get current timestamp
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Prepare data for CSV
        csv_data = {
            'timestamp': data.get('timestamp', timestamp),
            'device_id': data.get('device_id', 'unknown'),
            'ir_temp': data.get('ir_temp', ''),
            'pressure': data.get('pressure', ''),
            'rpm': data.get('rpm', ''),
            'motion_detected': '1' if data.get('motion_detected', False) else '0',
            'cmos_data': json.dumps(data.get('cmos_data', [])),
            'anomaly_score': f'{anomaly_score:.4f}',
            'is_potential_malware': '1' if is_potential_malware else '0'
        }
        
        # Open file in append mode
        with open(log_file, 'a', newline='') as f:
            # Define headers based on csv_data keys
            headers = list(csv_data.keys())
            
            # Create CSV writer
            writer = csv.DictWriter(f, fieldnames=headers)
            
            # Write header if file didn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write data row
            writer.writerow(csv_data)
            
        logger.debug(f"Logged sensor data for device {data.get('device_id', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Error logging sensor data: {str(e)}")

def format_log_message(data, anomaly_score, is_potential_malware):
    """
    Format a log message for the alerts log.
    
    Args:
        data: Dictionary containing the sensor data
        anomaly_score: The calculated anomaly score
        is_potential_malware: Boolean indicating if this is potential malware
        
    Returns:
        Formatted log message string
    """
    timestamp = data.get('timestamp', datetime.utcnow().isoformat() + 'Z')
    device_id = data.get('device_id', 'unknown')
    
    if is_potential_malware:
        return f"ALERT: {timestamp} - Device {device_id} - Anomaly score: {anomaly_score:.4f} - POTENTIAL MALWARE DETECTED"
    else:
        return f"INFO: {timestamp} - Device {device_id} - Anomaly score: {anomaly_score:.4f} - Normal operation"
