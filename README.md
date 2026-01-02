# 🛡️ AI-Powered Network Intrusion Detection System (IDS)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

> Advanced network intrusion detection system combining Deep Learning and Machine Learning for real-time anomaly detection without predefined attack signatures.

![IDS Demo](docs/demo.gif)  <!-- Ajoutez un GIF/screenshot -->

---

## 🎯 Overview

This project implements a **hybrid multi-level anomaly detection system** that learns the "normal" behavior of network traffic and automatically detects suspicious deviations in real-time.

### Key Features

- 🧠 **Multi-Model Architecture**: LSTM, CNN, Dense Autoencoders + Isolation Forest + LOF
- ⚡ **Real-Time Detection**: <100ms latency per packet analysis
- 🔄 **Multi-Threaded Capture**: Simultaneous capture on multiple network interfaces
- 🎛️ **Auto-Optimization**: Hyperparameter tuning with Optuna
- 📊 **Dual-Level Analysis**: Packet-level + Sequence-level detection
- 🏭 **Production-Ready**: Signal handling, logging, fault tolerance

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│              Network Traffic Capture                     │
│           (Multi-threaded, Multi-interface)              │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              Feature Extraction Engine                   │
│   • TCP Flags  • Packet Size  • IP Addresses            │
│   • Protocols  • Ports  • Temporal Statistics           │
└──────────────┬──────────────────────┬───────────────────┘
               ↓                      ↓
    ┌──────────────────┐   ┌──────────────────┐
    │  Individual       │   │   Sequence       │
    │  Packets         │   │   Analysis       │
    └────────┬─────────┘   └────────┬─────────┘
             ↓                      ↓
    ┌────────────────┐      ┌────────────────┐
    │ Dense AE       │      │ LSTM AE        │
    │ + IF + LOF     │      │ CNN AE         │
    │                │      │ + IF + LOF     │
    └────────┬───────┘      └────────┬───────┘
             └───────────┬────────────┘
                         ↓
              ┌──────────────────┐
              │ Ensemble Voting  │
              │   (Weighted)     │
              └─────────┬────────┘
                        ↓
                  Alert / Normal
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Linux/Unix system (for packet capture)
Root/sudo privileges (for network capture)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/VOTRE_USERNAME/AI-IDS.git
cd AI-IDS

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Basic Training & Detection
```bash
# Run with default parameters (20s capture, then real-time detection)
sudo python main.py
```

#### 2. Custom Configuration

Edit `config.py`:
```python
CONFIG = {
    "cap_interval": 60,      # Training capture duration (seconds)
    "save_interval": 20,     # Save interval (seconds)
    "epochs": 32,            # Training epochs
    "batch_size": 4,         # Batch size
    "prop_ano_threshold": 0.3,  # Anomaly proportion threshold
}
```

#### 3. Specify Network Interface
```bash
sudo python main.py --interface eth0
```

---

## 📊 How It Works

### 1. Feature Extraction

For each packet, 30+ features are extracted:
- **Network**: Source/Destination IPs (IPv4/IPv6), MAC addresses
- **Transport**: TCP/UDP ports, flags (SYN, ACK, FIN, RST, PSH, URG)
- **Payload**: Length, entropy
- **Temporal**: Timestamps, inter-arrival times

For sequences (sliding window of N packets):
- Statistical aggregations (mean, max, counts)
- Temporal patterns

### 2. Model Training

**Phase 1: Capture Normal Behavior**
- Capture network traffic for specified duration
- Extract features from all packets
- Build sequences with sliding window

**Phase 2: Train Models**

*Individual Packet Models:*
- Dense Autoencoder: Learns to reconstruct normal packet features
- Isolation Forest: Detects outliers in feature space
- Local Outlier Factor: Identifies local anomalies

*Sequence Models:*
- LSTM Autoencoder: Captures temporal dependencies
- CNN Autoencoder: Detects spatial patterns
- Isolation Forest + LOF on sequences

**Phase 3: Hyperparameter Optimization**
- Optuna automatically tunes IF and LOF parameters
- 2-10 trials per model (configurable)

### 3. Real-Time Detection
```python
For each incoming packet:
    1. Extract features
    2. Normalize using trained scaler
    3. Pass through Dense AE + IF + LOF
    4. Get prediction (-1: anomaly, 1: normal)
    
For each complete sequence:
    1. Extract sequence features
    2. Pass through LSTM + CNN + IF + LOF
    3. Ensemble voting (weighted)
    4. Final decision
```

---

## 🧪 Testing

### Simulate Normal Traffic
```bash
# Browse websites, download files, etc.
# Let the system learn for 1-2 minutes
```

### Simulate Attacks (for testing)
```bash
# Port scan
nmap -sS TARGET_IP

# Slow HTTP attack
slowhttptest -c 1000 -H -g -o output -i 10 -r 200 -t GET -u http://TARGET

# Note: Only test on networks you own or have permission
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Detection Latency | <100ms |
| Packets/sec | 500-1000 |
| False Positive Rate | ~5% (adjustable) |
| Training Time | 2-5 minutes (depends on capture duration) |

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow/Keras
- **Machine Learning**: scikit-learn
- **Optimization**: Optuna
- **Packet Capture**: dpkt, pcap
- **Concurrency**: Python threading
- **Data Processing**: NumPy, Pandas

---

## 📁 Project Structure
```
AI-IDS/
├── main.py              # Entry point
├── core.py              # Packet capture & preprocessing
├── models.py            # ML/DL models
├── features.py          # Feature extraction
├── config.py            # Configuration
├── requirements.txt     # Dependencies
├── data/
│   ├── datasets/        # Captured packets
│   └── models/          # Trained models
└── docs/                # Documentation
```

---

## ⚙️ Configuration Options

### `config.py`
```python
CONFIG = {
    # Training
    "lr": 1e-5,                    # Learning rate
    "epochs": 32,                  # Training epochs
    "batch_size": 4,               # Batch size
    "n_trials": 10,                # Optuna trials
    
    # Detection
    "c": 0.05,                     # Contamination (expected anomaly proportion)
    "prop_ano_threshold": 0.3,     # Sequence anomaly threshold
    "detection_mode": "all",       # "all" (strict) or "any" (lenient)
    
    # Capture
    "cap_interval": 60,            # Training capture duration (s)
    "save_interval": 20,           # Checkpoint interval (s)
    "seq_length": 10,              # Packets per sequence
}
```

---

## 🔮 Future Improvements

- [ ] Web dashboard (real-time visualization)
- [ ] REST API for integration
- [ ] Support for more protocols (DNS, HTTPS deep inspection)
- [ ] Pre-trained models on public datasets
- [ ] Docker containerization
- [ ] Distributed deployment support
- [ ] Integration with SIEM systems

---

## 📝 Known Limitations

- Requires root/sudo for packet capture
- Works best with consistent network behavior
- Initial training period needed (~1-2 minutes minimum)
- False positive rate depends on network variability

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by research in network anomaly detection
- Built with modern ML/DL frameworks
- Thanks to the open-source community

---

## 📧 Contact

**Hounso Samuel**

- LinkedIn: [votre-profil](https://linkedin.com/in/votre-profil)
- Email: votre.email@example.com
- Portfolio: [votre-site.com](https://votre-site.com)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=VOTRE_USERNAME/AI-IDS&type=Date)](https://star-history.com/#VOTRE_USERNAME/AI-IDS&Date)

---

**Made with ❤️ and lots of ☕**
