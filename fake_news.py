"""
SemanticTruth AI FactGuard Intelligence
Entrypoint wrapper for running the Flask application.
"""
from app import app

if __name__ == "__main__":
    print("=" * 60)
    print("Starting SemanticTruth AI FactGuard Intelligence Web Platform")
    print("Server running at: http://127.0.0.1:5000/")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=5000)
