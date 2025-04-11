import json
from flask import Blueprint, render_template, request, redirect, url_for, flash
from sqlalchemy import func, desc
from flask_paginate import Pagination

from .models import db, SensorData, AnomalyAlert
from .config import Config

# Create a blueprint for web routes
web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def dashboard():
    """Main dashboard view."""
    # Get statistics
    device_count = db.session.query(func.count(func.distinct(SensorData.device_id))).scalar()
    data_count = db.session.query(func.count(SensorData.id)).scalar()
    alert_count = db.session.query(func.count(AnomalyAlert.id)).scalar()
    
    # Get latest data for each device
    subq = db.session.query(
        SensorData.device_id, 
        func.max(SensorData.timestamp).label('max_timestamp')
    ).group_by(SensorData.device_id).subquery()
    
    devices = db.session.query(SensorData).join(
        subq, 
        db.and_(
            SensorData.device_id == subq.c.device_id,
            SensorData.timestamp == subq.c.max_timestamp
        )
    ).order_by(desc(SensorData.timestamp)).all()
    
    # Get recent alerts
    recent_alerts = AnomalyAlert.query.order_by(
        AnomalyAlert.timestamp.desc()
    ).limit(5).all()
    
    return render_template(
        'dashboard.html',
        device_count=device_count,
        data_count=data_count,
        alert_count=alert_count,
        devices=devices,
        recent_alerts=recent_alerts
    )

@web_bp.route('/devices')
def devices():
    """List all devices."""
    # Get latest data for each device
    subq = db.session.query(
        SensorData.device_id, 
        func.max(SensorData.timestamp).label('max_timestamp')
    ).group_by(SensorData.device_id).subquery()
    
    latest_data = db.session.query(SensorData).join(
        subq, 
        db.and_(
            SensorData.device_id == subq.c.device_id,
            SensorData.timestamp == subq.c.max_timestamp
        )
    ).all()
    
    # Add data point count for each device
    for device in latest_data:
        device.data_count = db.session.query(func.count(SensorData.id)).filter(
            SensorData.device_id == device.device_id
        ).scalar()
    
    # Get alert count for navbar
    alert_count = db.session.query(func.count(AnomalyAlert.id)).scalar()
    
    return render_template('devices.html', devices=latest_data, alert_count=alert_count)

@web_bp.route('/device/<device_id>')
def device_detail(device_id):
    """Show device details and history."""
    # Get the latest data for this device
    device = SensorData.query.filter_by(device_id=device_id).order_by(
        SensorData.timestamp.desc()
    ).first_or_404()
    
    # Get paginated history
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page
    
    history_query = SensorData.query.filter_by(device_id=device_id).order_by(
        SensorData.timestamp.desc()
    )
    
    total = history_query.count()
    history = history_query.limit(per_page).offset(offset).all()
    
    pagination = Pagination(
        page=page, 
        total=total, 
        per_page=per_page, 
        css_framework='bootstrap5',
        record_name='entries'
    )
    
    # Prepare JSON data for charts
    history_for_chart = history_query.limit(50).all()  # Limit to 50 for the chart
    history_json = json.dumps([item.to_dict() for item in history_for_chart])
    
    # Get alert count for navbar
    alert_count = db.session.query(func.count(AnomalyAlert.id)).scalar()
    
    return render_template(
        'device_detail.html',
        device=device,
        history=history,
        pagination=pagination,
        history_json=history_json,
        alert_count=alert_count
    )

@web_bp.route('/alerts')
def alerts():
    """Show all anomaly alerts."""
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page
    
    alerts_query = AnomalyAlert.query.order_by(AnomalyAlert.timestamp.desc())
    total = alerts_query.count()
    alerts_list = alerts_query.limit(per_page).offset(offset).all()
    
    pagination = Pagination(
        page=page, 
        total=total, 
        per_page=per_page, 
        css_framework='bootstrap5',
        record_name='alerts'
    )
    
    return render_template(
        'alerts.html',
        alerts=alerts_list,
        pagination=pagination,
        alert_count=total
    )

@web_bp.route('/models')
def model_management():
    """Model management interface."""
    # Get alert count for navbar
    alert_count = db.session.query(func.count(AnomalyAlert.id)).scalar()
    
    return render_template('model_management.html', alert_count=alert_count)

def register_web_routes(app):
    """Register web routes with the Flask app."""
    app.register_blueprint(web_bp)