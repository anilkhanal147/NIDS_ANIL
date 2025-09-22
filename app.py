import os
import uuid
import joblib
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from threading import Thread
from flask import jsonify
from real_time_sniffer import start_sniff

# === Flask Config ===
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
socketio = SocketIO(app)

# === Load Model ===
model, scaler, selector = joblib.load("model.pkl")

# === File Validation ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("home.html")

# === CSV Anomaly Detection ===
def predict_anomalies(csv_path):
    df = pd.read_csv(csv_path)

    # Drop label column if it exists
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    X = df.select_dtypes(include=['int64', 'float64'])

    # Apply selector and scaler before prediction
    X_selected = selector.transform(X)
    X_scaled = scaler.transform(X_selected)
    preds = model.predict(X_scaled)

    df['anomaly'] = [1 if p == 1 else 0 for p in preds]  # 1 = anomaly for RF
    df['Predicted'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == 1 else 'Not Anomaly')

    return df, df[df['anomaly'] == 1]


# === Routes ===

@app.route("/upload", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            df_all, anomalies = predict_anomalies(file_path)

            summary = {
                "total_records": len(df_all),
                "anomalies_detected": len(anomalies),
                "anomaly_rate": round(100 * len(anomalies) / len(df_all), 2)
            }

            result_file = os.path.join(app.config['UPLOAD_FOLDER'], f"results_{unique_filename}")
            df_all.to_csv(result_file, index=False)

            # Handle missing 'proto' column gracefully
            if 'proto' in df_all.columns:
                proto_counts = df_all.groupby("proto")["anomaly"].sum().to_dict()
            else:
                proto_counts = {}

            return render_template("index.html",
                       summary=summary,
                       tables=[df_all.head(1000).style.apply(
                           lambda row: ['background-color: #ffcccc' if row['anomaly'] == 1 else '' for _ in row], axis=1
                       ).to_html(classes='table table-striped', index=False)],
                       csv_file=result_file,
                       anomaly_count=int(summary["anomalies_detected"]),
                       normal_count=int(summary["total_records"]) - int(summary["anomalies_detected"]),
                       protocol_data=proto_counts)

    return render_template("index.html")

@app.route("/download")
def download():
    file = request.args.get("file")
    if os.path.exists(file):
        return send_file(file, as_attachment=True)
    return "File not found."

@app.route("/realtime")
def realtime():
    return render_template("realtime.html")

@app.route("/download_filtered", methods=["POST"])
def download_filtered():
    import io
    import csv

    data = request.get_json()
    if not data or "rows" not in data:
        return jsonify({"error": "No data received"}), 400

    # Create CSV in-memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["time", "src_ip", "dst_ip", "protocol", "length", "status", "reason"])
    writer.writeheader()
    for row in data["rows"]:
        writer.writerow({
            "time": row["time"],
            "src_ip": row["src_ip"],
            "dst_ip": row["dst_ip"],
            "protocol": row["protocol"],
            "length": row["length"],
            "status": "Anomaly" if row["anomaly"] else "Normal",
            "reason": row["reason"]
        })

    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    output.close()

    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name="filtered_results.csv")


# === Run the App ===
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Start sniffing in the background
    Thread(target=lambda: start_sniff(interface="vmnet1", sockio=socketio)).start()

    # Start Flask + SocketIO
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

