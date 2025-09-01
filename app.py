from flask import Flask
from routes.maternal import maternal_bp

def create_app():
    app = Flask(__name__)

    # Register maternal routes
    app.register_blueprint(maternal_bp, url_prefix="/api/maternal")

    @app.route("/", methods=["GET"])
    def home():
        """
        Root endpoint to verify API is running.
        """
        return {"message": "Health Models API is running!"}

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
