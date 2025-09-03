import os
from flask import Flask, request, jsonify


def create_app():
    app = Flask(__name__)

    @app.post("/save")
    def save():
        try:
            # id may be sent as form field; not strictly required for saving
            req_id = request.form.get("id")

            if "file" not in request.files:
                return jsonify({"error": "no_file"}), 400

            f = request.files["file"]
            if f.filename == "":
                return jsonify({"error": "empty_filename"}), 400

            out_path = os.path.abspath(os.path.join(os.getcwd(), "received.zip"))
            f.save(out_path)

            return jsonify({"status": "ok", "saved": out_path, "id": req_id}), 200
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    # Listen on port 5001 as requested
    app.run(host="0.0.0.0", port=5001)


