# 🚀 Network Intrusion Detection System (NIDS) – Machine Learning Based

 📖 Project Overview

This project implements a web-based real-time Network Intrusion Detection System (NIDS) that leverages machine learning to detect anomalies in network traffic. The system supports both CSV-based traffic log uploads and live packet sniffing via Scapy, with results visualized through an interactive dashboard built using Flask and Chart.js.

Our system demonstrates how lightweight ML algorithms can achieve near-perfect detection while remaining deployable in real-world, resource-constrained environments.


 🎯 Aims & Objectives

* Detect anomalies in network traffic using machine learning models.
* Provide both offline (CSV upload) and real-time (packet sniffing) detection.
* Develop an interactive, analyst-friendly dashboard for results visualization.
* Ensure the system is lightweight, modular, and extensible for future upgrades.

 🛠️ Features

✅ CSV Upload for offline anomaly detection
✅ Real-time packet sniffing with Scapy
✅ ML models trained on the TON IoT Network Dataset
✅ Random Forest Classifier (best-performing model, F1-score = 1.00)
✅ Interactive dashboard with Chart.js visualizations
✅ Results downloadable as CSV with anomaly marking
✅ Modular design for easy integration of new ML models


 📊 Results

| Model               | Precision | Recall | F1-Score | Accuracy |
| ------------------- | --------- | ------ | -------- | -------- |
| Random Forest   | 1.00      | 1.00   | 1.00     | 1.00     |
| Logistic Regression | 0.80      | 0.77   | 0.77     | 0.77     |
| Isolation Forest    | Poor      | Poor   | Poor     | Poor     |

Random Forest was chosen as the final model for deployment.
* Real-time detection achieved sub-second latency.
* Dashboard enhanced usability with anomaly summaries and protocol breakdowns.

 🚀 Getting Started

# 🔧 Installation

1. Clone the repository:

   
   git clone https://github.com/anilkhanal147/NIDS_ANIL.git
   cd NIDS_ANIL
   

2. Create and activate a virtual environment:

   
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   

3. Install dependencies:

   
   pip install -r requirements.txt
   


▶️ Usage

 Train Model


python train_model.py dataset/TON_Dataset.csv --label label --output model.pkl


 Run Web App


python app.py


Open in your browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

 Real-Time Sniffing

Make sure you have the right permissions:


sudo python app.py



 📈 Visualizations

Anomaly Distribution Pie Chart
Anomalies by Protocol (Bar Chart)
Detection Summary (Total Records, Anomalies, Anomaly Rate)


 🔮 Future Work

* Extend models with Deep Learning (LSTM, Autoencoders)
* Integration with ELK Stack (Elasticsearch, Logstash, Kibana)
* Support for Deep Packet Inspection (DPI)
* Automated alerting systems and traffic shaping
* Improved real-time dashboard interactivity (IP tracing, drill-downs)

 📚 References

* Scikit-learn
* Flask
* Chart.js
* Scapy
* TON IoT Dataset – [Kaggle Link](https://www.kaggle.com/datasets/zoya77/multi-type-network-attack-detection-dataset)


💡 Developed by: Anil Khanal
📘 MSc Dissertation Project
