import os
import joblib
import pandas as pd
from scapy.all import rdpcap, TCP, UDP, IP
from sklearn.preprocessing import StandardScaler

# === Step 1: Extract features from packets ===
def extract_features_from_pcap(pcap_path):
    packets = rdpcap(pcap_path)
    features = []

    for pkt in packets:
        if IP in pkt:
            proto = pkt[IP].proto
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            packet_len = len(pkt)
            src_port = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
            dst_port = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)

            features.append({
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': proto,
                'length': packet_len
            })

    return pd.DataFrame(features)


# === Step 2: Preprocess features for model input ===
def preprocess_features(df):
    df_numeric = df[['src_port', 'dst_port', 'protocol', 'length']]
    scaler = StandardScaler()
    return scaler.fit_transform(df_numeric)


# === Step 3: Load ML model and predict anomalies ===
def detect_anomalies(X, model_path="model.pkl", threshold=0.5):
    model = joblib.load(model_path)
    preds = model.predict(X)
    
    # If model returns probabilities/anomaly scores, use threshold
    # For example, if using IsolationForest:
    # preds = model.decision_function(X)
    # anomalies = (preds < threshold).astype(int)

    return preds  # Either binary or score


# === Step 4: Full processing pipeline ===
def analyze_pcap(pcap_file, model_file="model.pkl"):
    print("[INFO] Extracting features...")
    df = extract_features_from_pcap(pcap_file)

    print("[INFO] Preprocessing data...")
    X = preprocess_features(df)

    print("[INFO] Running anomaly detection...")
    preds = detect_anomalies(X, model_file)

    df['anomaly'] = preds
    anomalies = df[df['anomaly'] == 1]

    summary = {
        "total_packets": len(df),
        "anomalies_detected": len(anomalies),
        "anomaly_percentage": round(100 * len(anomalies) / len(df), 2)
    }

    return summary, anomalies


# === CLI Example Usage ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PCAP Analyzer with ML Anomaly Detection")
    parser.add_argument("pcap", help="Path to PCAP file")
    parser.add_argument("--model", default="model.pkl", help="Path to trained ML model")
    args = parser.parse_args()

    summary, anomalies = analyze_pcap(args.pcap, args.model)
    
    print("\n==== Summary ====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    
    print("\nTop 5 Anomalous Packets:")
    print(anomalies.head())
