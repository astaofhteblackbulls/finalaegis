import os
import json
import numpy as np
import logging
import joblib
from sklearn.ensemble import IsolationForest
import pickle
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Class responsible for anomaly detection using a machine learning model.
    If a pre-trained model is not available, it will create a simple 
    Isolation Forest model as a placeholder.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the anomaly detector.
        
        Args:
            model_path: Path to a pre-trained model file (optional)
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.warning("Pre-trained model not found. Creating a default Isolation Forest model.")
            self._create_default_model()
    
    def _load_model(self, model_path):
        """
        Load a pre-trained model from a file.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            # Try to load with joblib first (recommended for scikit-learn models)
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path} using joblib")
        except:
            try:
                # Fall back to pickle if joblib fails
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Model loaded successfully from {model_path} using pickle")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                self._create_default_model()
    
    def _create_default_model(self):
        """Create a default Isolation Forest model for anomaly detection."""
        # Increased contamination parameter to detect more anomalies
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.2,  # Increased from 0.1 to 0.2 to be more sensitive
            random_state=42,
            n_jobs=-1
        )
        # Since we don't have data to fit the model, 
        # we'll fit it on first prediction with a "warm-up" flag
        self.is_fitted = False
        
        # We'll define normal ranges for each sensor type to use in prediction
        self.normal_ranges = {
            'ir_temp': (20.0, 45.0),       # Normal temperature range in Celsius
            'pressure': (950.0, 1050.0),   # Normal atmospheric pressure range (hPa)
            'rpm': (0, 2500),              # Normal RPM range for most devices
            'cmos_mean': (50, 200)         # Normal CMOS mean value range
        }
    
    def _preprocess_data(self, data_dict):
        """
        Extract and normalize features from the input data.
        
        Args:
            data_dict: Dictionary containing sensor data
            
        Returns:
            Numpy array of normalized features
        """
        features = []
        
        # Extract numeric features
        ir_temp = data_dict.get('ir_temp', 0)
        pressure = data_dict.get('pressure', 0)
        rpm = data_dict.get('rpm', 0)
        motion = 1 if data_dict.get('motion_detected', False) else 0
        
        # Handle CMOS data
        cmos_data = data_dict.get('cmos_data', [])
        cmos_mean = np.mean(cmos_data) if cmos_data else 0
        cmos_std = np.std(cmos_data) if cmos_data and len(cmos_data) > 1 else 0
        
        # Combine features
        features = [ir_temp, pressure, rpm, motion, cmos_mean, cmos_std]
        return np.array(features).reshape(1, -1)
    
    def predict_anomaly(self, data_dict):
        """
        Predict anomaly score for the given sensor data.
        
        Args:
            data_dict: Dictionary containing the sensor data
            
        Returns:
            Anomaly score between 0 and 1, where higher values indicate more anomalous data
        """
        features = self._preprocess_data(data_dict)
        
        # Check if the model needs to be fitted (only for default model)
        if not self.is_fitted and isinstance(self.model, IsolationForest):
            logger.info("Fetching historical data to train the model")
            # Use real historical data to train the model if available
            try:
                from aegis.models import SensorData, db
                from sqlalchemy import func
                import json
                
                # Query for at least 50 data points, or what's available
                training_data = []
                
                # Check if we have a database connection and data
                if db.session:
                    # Get a sample of data from each device for better representation
                    device_counts = db.session.query(
                        SensorData.device_id, 
                        func.count(SensorData.id).label('count')
                    ).group_by(SensorData.device_id).all()
                    
                    if device_counts:
                        for device_id, count in device_counts:
                            # Calculate how many samples to take from each device
                            # with a minimum of 10 and maximum of 100
                            sample_size = min(max(10, int(count * 0.2)), 100)
                            
                            # Get the data
                            device_data = SensorData.query.filter_by(
                                device_id=device_id
                            ).order_by(func.random()).limit(sample_size).all()
                            
                            for record in device_data:
                                # Extract features
                                data_dict = {
                                    'ir_temp': record.ir_temp,
                                    'pressure': record.pressure,
                                    'rpm': record.rpm,
                                    'motion_detected': record.motion_detected,
                                    'cmos_data': json.loads(record.cmos_data) if record.cmos_data else []
                                }
                                # Preprocess and add to training data
                                processed = self._preprocess_data(data_dict)
                                training_data.append(processed.reshape(-1))
                
                # If we have enough data, use it; otherwise, create some reasonable values
                if len(training_data) >= 10:
                    logger.info(f"Training model with {len(training_data)} historical data points")
                    train_array = np.array(training_data)
                    self.model.fit(train_array)
                else:
                    logger.info("Not enough historical data, using default training approach")
                    # Generate data within normal ranges as a fallback
                    normal_data = []
                    for _ in range(100):
                        sample = {
                            'ir_temp': np.random.uniform(20, 45),  # Normal temperature range
                            'pressure': np.random.uniform(950, 1050),  # Normal pressure range
                            'rpm': np.random.uniform(100, 2000),  # Normal RPM range
                            'motion_detected': np.random.choice([True, False]),
                            'cmos_data': [
                                np.random.randint(50, 200) for _ in range(3)
                            ]
                        }
                        processed = self._preprocess_data(sample)
                        normal_data.append(processed.reshape(-1))
                        
                    # Add current features
                    normal_data.append(features.reshape(-1))
                    self.model.fit(np.array(normal_data))
                    
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                # Fallback to simple approach
                synth_data = np.random.normal(size=(100, features.shape[1]))
                synth_data = np.vstack([synth_data, features])
                self.model.fit(synth_data)
            
            self.is_fitted = True
        
        try:
            # First check if any values are outside normal ranges
            # This is a rule-based approach to complement the ML model
            rule_based_score = self._rule_based_anomaly_score(data_dict)
            
            # For Isolation Forest, decision_function gives anomaly score
            # Convert to a 0-1 scale where 1 is most anomalous
            if hasattr(self.model, 'decision_function'):
                # Decision function gives negative scores for anomalies
                raw_score = self.model.decision_function(features)[0]
                # Convert to 0-1 scale where 1 is most anomalous
                ml_score = 1 - (1 / (1 + np.exp(-raw_score)))
            elif hasattr(self.model, 'predict_proba'):
                # Some models provide probability directly
                ml_score = self.model.predict_proba(features)[0][1]  # Assume binary classification
            else:
                # Fall back to predict, which gives -1 for anomalies, 1 for normal
                raw_prediction = self.model.predict(features)[0]
                ml_score = 0.8 if raw_prediction == -1 else 0.2
            
            # Combine both scores, giving more weight to rule-based detection for extreme values
            final_score = max(ml_score, rule_based_score)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            # Return a moderate score in case of errors
            return 0.5
            
    def _rule_based_anomaly_score(self, data_dict):
        """
        Calculate anomaly score based on predefined rules and normal ranges.
        
        Args:
            data_dict: Dictionary containing the sensor data
            
        Returns:
            Anomaly score between 0 and 1 based on rules
        """
        # Extract values
        ir_temp = data_dict.get('ir_temp', 0)
        pressure = data_dict.get('pressure', 0)
        rpm = data_dict.get('rpm', 0)
        
        # Handle CMOS data
        cmos_data = data_dict.get('cmos_data', [])
        cmos_mean = np.mean(cmos_data) if cmos_data else 0
        
        # Initialize anomaly scores for each parameter
        scores = []
        
        # Check temperature
        if ir_temp is not None:
            min_temp, max_temp = self.normal_ranges['ir_temp']
            if ir_temp < min_temp or ir_temp > max_temp:
                # Calculate how far it is outside the normal range
                if ir_temp < min_temp:
                    deviation = (min_temp - ir_temp) / min_temp
                else:
                    deviation = (ir_temp - max_temp) / max_temp
                # Limit the score to 0.95 to allow ML model to have final say in extreme cases
                temp_score = min(0.95, 0.6 + deviation * 0.3)
                scores.append(temp_score)
        
        # Check pressure
        if pressure is not None:
            min_press, max_press = self.normal_ranges['pressure']
            if pressure < min_press or pressure > max_press:
                if pressure < min_press:
                    deviation = (min_press - pressure) / min_press
                else:
                    deviation = (pressure - max_press) / max_press
                press_score = min(0.95, 0.6 + deviation * 0.3)
                scores.append(press_score)
        
        # Check RPM
        if rpm is not None:
            min_rpm, max_rpm = self.normal_ranges['rpm']
            if rpm < min_rpm or rpm > max_rpm:
                if rpm < min_rpm:
                    deviation = (min_rpm - rpm) / max(1, min_rpm)  # Avoid division by zero
                else:
                    deviation = (rpm - max_rpm) / max_rpm
                rpm_score = min(0.95, 0.6 + deviation * 0.3)
                scores.append(rpm_score)
        
        # Check CMOS mean
        if cmos_mean > 0:
            min_cmos, max_cmos = self.normal_ranges['cmos_mean']
            if cmos_mean < min_cmos or cmos_mean > max_cmos:
                if cmos_mean < min_cmos:
                    deviation = (min_cmos - cmos_mean) / min_cmos
                else:
                    deviation = (cmos_mean - max_cmos) / max_cmos
                cmos_score = min(0.95, 0.6 + deviation * 0.3)
                scores.append(cmos_score)
        
        # Return the maximum score if any were calculated, otherwise return 0
        return max(scores) if scores else 0.0

    def save_model(self, model_path='models/isolation_forest.joblib', format='joblib'):
        """
        Save the trained model to disk in various formats.
        
        Args:
            model_path: Path where to save the model
            format: Format to save the model in ('joblib', 'pickle', 'pmml', 'onnx')
            
        Returns:
            Boolean indicating whether the model was saved successfully
        """
        if not self.is_fitted:
            logger.warning("Cannot save model that hasn't been fitted yet")
            return False
            
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save metadata about when the model was saved and normal ranges
            metadata = {
                'timestamp': datetime.utcnow().isoformat(),
                'model_type': type(self.model).__name__,
                'normal_ranges': self.normal_ranges,
                'format': format,
                'version': '1.0',
                'features': ['ir_temp', 'pressure', 'rpm', 'motion_detected', 'cmos_data_0', 'cmos_data_1', 'cmos_data_2']
            }
            
            # Save in the specified format
            if format.lower() == 'joblib':
                joblib.dump(self.model, model_path)
                logger.info(f"Model saved successfully to {model_path} in joblib format")
            elif format.lower() == 'pickle':
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                logger.info(f"Model saved successfully to {model_path} in pickle format")
            elif format.lower() == 'pmml':
                try:
                    from sklearn2pmml import sklearn2pmml
                    from sklearn2pmml.pipeline import PMMLPipeline
                    
                    # Create a pipeline with the model
                    pipeline = PMMLPipeline([("model", self.model)])
                    
                    # Save as PMML
                    sklearn2pmml(pipeline, model_path, with_repr=True)
                    logger.info(f"Model saved successfully to {model_path} in PMML format")
                except ImportError:
                    logger.error("sklearn2pmml not installed. Please install it to use PMML format.")
                    return False
            elif format.lower() == 'onnx':
                try:
                    import skl2onnx
                    from skl2onnx import convert_sklearn
                    from skl2onnx.common.data_types import FloatTensorType
                    
                    # Define input features
                    initial_type = [('features', FloatTensorType([None, 7]))]
                    
                    # Convert to ONNX
                    onnx_model = convert_sklearn(self.model, initial_types=initial_type)
                    
                    # Save the model
                    with open(model_path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                    logger.info(f"Model saved successfully to {model_path} in ONNX format")
                except ImportError:
                    logger.error("skl2onnx not installed. Please install it to use ONNX format.")
                    return False
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            # Save metadata to a JSON file
            metadata_path = model_path + '.meta.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
            
    def export_model_for_transfer(self, export_path='models/export'):
        """
        Export the model and all necessary data for transfer to another platform.
        Creates a zip file with the model, metadata, and feature scaling information.
        
        Args:
            export_path: Base path where to save the export files
            
        Returns:
            Path to the generated zip file or None if export failed
        """
        if not self.is_fitted:
            logger.warning("Cannot export model that hasn't been fitted yet")
            return None
        
        try:
            # Create the export directory
            os.makedirs(export_path, exist_ok=True)
            
            # Save the model in multiple formats
            self.save_model(os.path.join(export_path, 'model.joblib'), format='joblib')
            self.save_model(os.path.join(export_path, 'model.pkl'), format='pickle')
            
            # Export feature info and preprocessing data
            feature_info = {
                'normal_ranges': self.normal_ranges,
                'preprocessing_info': {
                    'cmos_data_length': 3,
                    'normalizes_features': True
                }
            }
            
            with open(os.path.join(export_path, 'feature_info.json'), 'w') as f:
                json.dump(feature_info, f, indent=2)
                
            # Export sample code for inference
            sample_code = """
# Sample Python code for making predictions with this model
import joblib
import numpy as np

# Load the model
model = joblib.load('model.joblib')

# Function to preprocess data
def preprocess_data(data_dict):
    # Extract features from the input data dictionary
    features = []
    
    # Add temperature, pressure, rpm
    features.append(float(data_dict.get('ir_temp', 25.0)))
    features.append(float(data_dict.get('pressure', 1013.0)))
    features.append(float(data_dict.get('rpm', 1500.0)))
    
    # Add motion as 0 or 1
    features.append(1.0 if data_dict.get('motion_detected', False) else 0.0)
    
    # Add CMOS data (if available)
    cmos_data = data_dict.get('cmos_data', [100, 100, 100])
    if isinstance(cmos_data, list):
        # Ensure we have exactly 3 values
        if len(cmos_data) < 3:
            cmos_data = cmos_data + [100] * (3 - len(cmos_data))
        elif len(cmos_data) > 3:
            cmos_data = cmos_data[:3]
    else:
        cmos_data = [100, 100, 100]
    
    # Add CMOS data to features
    features.extend([float(val) for val in cmos_data])
    
    return np.array(features).reshape(1, -1)

# Example usage
data = {
    'ir_temp': 36.5,
    'pressure': 1013.25,
    'rpm': 1500,
    'motion_detected': True,
    'cmos_data': [120, 130, 115]
}

# Preprocess the data
X = preprocess_data(data)

# Make a prediction (get anomaly score)
anomaly_score = model.decision_function(X)
# Convert to a 0-1 scale where higher means more anomalous
normalized_score = 1.0 - 1.0 / (1.0 + np.exp(-anomaly_score))

print(f"Anomaly score: {normalized_score}")
"""
            
            with open(os.path.join(export_path, 'sample_inference.py'), 'w') as f:
                f.write(sample_code)
                
            # Create a zip file of all exported files
            import shutil
            
            zip_path = f"{export_path}.zip"
            shutil.make_archive(export_path, 'zip', export_path)
            
            logger.info(f"Model successfully exported to {zip_path}")
            return zip_path
            
        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            return None

# Create a singleton instance of the anomaly detector
anomaly_detector = AnomalyDetector()

def predict_anomaly(data_dict):
    """
    Wrapper function to call the anomaly detector's predict_anomaly method.
    
    Args:
        data_dict: Dictionary containing the sensor data
        
    Returns:
        Anomaly score between 0 and 1
    """
    return anomaly_detector.predict_anomaly(data_dict)

def save_trained_model(path=None, format='joblib'):
    """
    Save the currently trained model.
    
    Args:
        path: Optional custom path to save the model
        format: Format to save the model in ('joblib', 'pickle', 'pmml', 'onnx')
    
    Returns:
        Boolean indicating success or failure
    """
    if path:
        return anomaly_detector.save_model(path, format=format)
    else:
        return anomaly_detector.save_model(format=format)

def export_model(export_path=None):
    """
    Export the currently trained model for transfer to another platform.
    
    Args:
        export_path: Optional base path where to save the export files
        
    Returns:
        Path to the generated zip file or None if export failed
    """
    if export_path:
        return anomaly_detector.export_model_for_transfer(export_path)
    else:
        return anomaly_detector.export_model_for_transfer()
        
def import_model(model_path, metadata_path=None, model_format='joblib'):
    """
    Import a model trained on another platform.
    
    Args:
        model_path: Path to the model file
        metadata_path: Optional path to a metadata JSON file
        model_format: Format of the model file ('joblib', 'pickle', 'onnx')
        
    Returns:
        Boolean indicating whether the model was imported successfully
    """
    try:
        global anomaly_detector
        
        # Load the model based on the format
        if model_format.lower() == 'joblib':
            model = joblib.load(model_path)
        elif model_format.lower() == 'pickle':
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_format.lower() == 'onnx':
            try:
                import onnxruntime as ort
                
                # Create an ONNX runtime session
                anomaly_detector.onnx_session = ort.InferenceSession(model_path)
                anomaly_detector.using_onnx = True
                
                # If we have metadata, load it
                if metadata_path:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Update the normal ranges if present
                    if 'normal_ranges' in metadata:
                        anomaly_detector.normal_ranges = metadata['normal_ranges']
                
                # Update the anomaly detector's state
                anomaly_detector.is_fitted = True
                logger.info(f"ONNX model loaded successfully from {model_path}")
                
                # Monkey patch the predict_anomaly method to use ONNX
                original_preprocess = anomaly_detector._preprocess_data
                
                def onnx_predict_anomaly(self, data_dict):
                    # Preprocess the data
                    features = original_preprocess(data_dict)
                    
                    # Make prediction using ONNX runtime
                    input_name = self.onnx_session.get_inputs()[0].name
                    output_name = self.onnx_session.get_outputs()[0].name
                    
                    # Run inference
                    result = self.onnx_session.run([output_name], {input_name: features.astype(np.float32)})[0]
                    
                    # Calculate anomaly score (higher means more anomalous)
                    anomaly_score = 1.0 - 1.0 / (1.0 + np.exp(-result[0]))
                    
                    # Apply rule-based scoring
                    rule_score = self._rule_based_anomaly_score(data_dict)
                    
                    # Return max of ML score and rule score
                    return max(anomaly_score, rule_score)
                
                # Replace the predict_anomaly method
                anomaly_detector.original_predict_anomaly = anomaly_detector.predict_anomaly
                anomaly_detector.predict_anomaly = onnx_predict_anomaly.__get__(anomaly_detector, AnomalyDetector)
                
                return True
            except ImportError:
                logger.error("onnxruntime not installed. Please install it to use ONNX models.")
                return False
        else:
            logger.error(f"Unsupported model format: {model_format}")
            return False
            
        # For scikit-learn models (joblib, pickle)
        if model_format.lower() in ['joblib', 'pickle']:
            # Check if it's a valid scikit-learn model with decision_function
            if not hasattr(model, 'decision_function'):
                logger.error("The imported model does not have a decision_function method")
                return False
                
            # Set the model in the anomaly detector
            anomaly_detector.model = model
            anomaly_detector.is_fitted = True
            
            # Load metadata if available
            if metadata_path:
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Update the normal ranges if present
                    if 'normal_ranges' in metadata:
                        anomaly_detector.normal_ranges = metadata['normal_ranges']
                except Exception as e:
                    logger.warning(f"Error loading metadata: {str(e)}")
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        return False
    except Exception as e:
        logger.error(f"Error importing model: {str(e)}")
        return False
