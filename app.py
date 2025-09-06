from flask import Flask
from routes.maternal import maternal_bp
from routes.cardiovascular import cardiovascular_bp
import os

def create_app():
    app = Flask(__name__)

    # Register maternal routes
    app.register_blueprint(maternal_bp, url_prefix="/api/maternal")
    
    # Register cardiovascular routes
    app.register_blueprint(cardiovascular_bp, url_prefix="/api/cardiovascular")

    @app.route("/", methods=["GET"])
    def home():
        """
        Root endpoint to verify API is running.
        """
        return {"message": "Health Models API is running!"}

    return app


# 👇 expose the app at module level for Gunicorn
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets $PORT dynamically
    app.run(host="0.0.0.0", port=port, debug=True)
