from flask import Flask
from routes.maternal import maternal_bp
from routes.cardiovascular import cardiovascular_bp
from routes.glucose import glucose_bp
from swagger_ui import create_swagger_ui
import os

def create_app():
    app = Flask(__name__)

    # Register maternal routes
    app.register_blueprint(maternal_bp, url_prefix="/api/maternal")
    
    # Register cardiovascular routes
    app.register_blueprint(cardiovascular_bp, url_prefix="/api/cardiovascular")
    
    # Register glucose routes
    app.register_blueprint(glucose_bp, url_prefix="/api/glucose")

    # Add Swagger UI documentation
    create_swagger_ui(app)

    @app.route("/", methods=["GET"])
    def home():
        """
        Root endpoint to verify API is running.
        """
        return {
            "message": "Health Models API is running!",
            "documentation": "/docs/",
            "endpoints": {
                "maternal": "/api/maternal",
                "cardiovascular": "/api/cardiovascular", 
                "glucose": "/api/glucose"
            }
        }

    return app


# 👇 expose the app at module level for Gunicorn
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets $PORT dynamically
    app.run(host="0.0.0.0", port=port, debug=True)
