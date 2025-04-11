from datetime import datetime
from aegis import db

class SensorData(db.Model):
    """Model for storing IoT sensor data and anomaly detection results."""
    
    __tablename__ = 'sensor_data'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    device_id = db.Column(db.String(50), nullable=False)
    
    # Sensor readings
    cmos_data = db.Column(db.String(200))  # Stored as JSON string
    ir_temp = db.Column(db.Float)
    motion_detected = db.Column(db.Boolean)
    pressure = db.Column(db.Float)
    rpm = db.Column(db.Float)
    
    # Analysis results
    anomaly_score = db.Column(db.Float)
    is_potential_malware = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<SensorData id={self.id} device={self.device_id} score={self.anomaly_score}>'
    
    def to_dict(self):
        """Convert model instance to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() + 'Z',
            'device_id': self.device_id,
            'cmos_data': self.cmos_data,
            'ir_temp': self.ir_temp,
            'motion_detected': self.motion_detected,
            'pressure': self.pressure,
            'rpm': self.rpm,
            'anomaly_score': self.anomaly_score,
            'is_potential_malware': self.is_potential_malware
        }

class AnomalyAlert(db.Model):
    """Model for storing detailed anomaly alerts."""
    
    __tablename__ = 'anomaly_alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    sensor_data_id = db.Column(db.Integer, db.ForeignKey('sensor_data.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    alert_type = db.Column(db.String(50), default='anomaly_alert')
    score = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text)
    
    # Relationship with SensorData
    sensor_data = db.relationship('SensorData', backref=db.backref('alerts', lazy=True))
    
    def __repr__(self):
        return f'<AnomalyAlert id={self.id} score={self.score}>'
    
    def to_dict(self):
        """Convert model instance to dictionary."""
        return {
            'id': self.id,
            'type': self.alert_type,
            'device_id': self.sensor_data.device_id,
            'timestamp': self.timestamp.isoformat() + 'Z',
            'score': self.score,
            'sensor_data': self.sensor_data.to_dict(),
            'description': self.description,
            'is_potential_malware': self.sensor_data.is_potential_malware
        }
