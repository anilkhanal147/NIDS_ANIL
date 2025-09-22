<<<<<<< HEAD
=======
You'll need:

A .pcap file (e.g., test.pcap)

>>>>>>> 5327519 (Initial commit)
A trained ML model (model.pkl — use IsolationForest, Autoencoder, etc.)

joblib, scikit-learn, pandas, scapy installed:

pip install scapy pandas scikit-learn joblib


train:
Example Usage
1. Train without labels (unsupervised):

python train_model.py dataset.csv
2. Train with evaluation (if label column exists):

python train_model.py dataset.csv --label label


Run:
pip install flask pandas joblib scikit-learn
python app.py


Realtime:

pip install flask-socketio scapy joblib pandas scikit-learn eventlet

