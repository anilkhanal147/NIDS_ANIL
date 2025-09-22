import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# === Load and preprocess TON dataset ===
df = pd.read_csv("dataset/TON_Dataset.csv")

# Replace 'proto' with numeric mapping for real-time compatibility
proto_mapping = {proto: idx for idx, proto in enumerate(df['proto'].unique())}
df['proto_num'] = df['proto'].map(proto_mapping)

# Calculate 'length' as src_bytes + dst_bytes (to simulate packet length)
df['length'] = df['src_bytes'] + df['dst_bytes']

# Select real-time compatible features
realtime_features = ['src_port', 'dst_port', 'proto_num', 'length']
df_realtime = df[realtime_features].fillna(0)

# Scale and train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_realtime)

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_scaled)

# Save model and scaler for real-time use
joblib.dump((model, scaler, proto_mapping), "realtime_model.pkl")
print("[✅] Real-time model and scaler saved as 'realtime_model.pkl'")
