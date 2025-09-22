import joblib
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP
from flask_socketio import SocketIO



PROTO_MAP = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    58: "ICMPv6"
}

# Flask-SocketIO instance will be injected later
socketio = None

# ✅ Fix: Declare last_results before using it
last_results = []
# Load model
model, scaler, proto_mapping = joblib.load("realtime_model.pkl")


def extract_features(pkt):
    if IP in pkt:
        proto_num = pkt[IP].proto
        proto_name = PROTO_MAP.get(proto_num, f"Proto-{proto_num}")

        return {
            'src_ip': pkt[IP].src,
            'dst_ip': pkt[IP].dst,
            'src_port': pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0),
            'dst_port': pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0),
            'proto_num': proto_num,
            'protocol': proto_name,
            'length': len(pkt)
        }
    return None


def packet_handler(pkt):
    features = extract_features(pkt)
    if not features:
        return

    df = pd.DataFrame([features])
    X = df[['src_port', 'dst_port', 'proto_num', 'length']]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    is_anomaly = int(prediction[0] == -1)

    # Basic explanation heuristics
    reason = "Length unusually small" if df['length'].values[0] < 60 else \
             "Uncommon port/protocol pattern" if df['src_port'].values[0] == 0 or df['dst_port'].values[0] == 0 else \
             "Deviation from normal traffic profile" if is_anomaly else \
             "Normal"

    result = {
        **features,
        'anomaly': is_anomaly,
        'reason': reason
    }

    socketio.emit("packet_result", result)
    last_results.append(result)
    if len(last_results) > 1000:
        last_results.pop(0)


# Start sniffing
def start_sniff(interface="vmnet1", sockio=None):
    global socketio
    socketio = sockio  # Inject the socketio object from app.py
    print(f"[INFO] Starting real-time sniffing on {interface}...")
    sniff(iface=interface, prn=packet_handler, store=False)
