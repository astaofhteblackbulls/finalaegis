import os
from aegis import create_app, socketio

app = create_app()

if __name__ == "__main__":
    # Run the app using Flask-SocketIO
    # Bind to 0.0.0.0 to make the app accessible from outside the container
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)
