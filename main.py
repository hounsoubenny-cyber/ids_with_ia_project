#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 11:37:12 2026

@author: hounsousamuel
"""

import os, sys
import dill
import threading
import pcap, dpkt
from queue import Queue
from collections import deque
import signal, time
import numpy as np
from core import core, get_ifaces
from models import Models
from config import CONFIG_MODELS as config_model, CONFIG_MAIN as conf_main, SEQ_lENGTH, SEUIL
from features import extract_pack_features as features_extractor, extract_seq_features as seq_extractor
import logging
from datetime import datetime

_dir_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data','models')
os.makedirs(_dir_, exist_ok=True)



class IDSLogger1:
    def __init__(self, log_file="ids_logs.txt"):
        self.logger = logging.getLogger("IDS")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        # Format
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_packet_alert(self, packet_info, scores=None):
        msg = f"🚨 PACKET ANOMALY - Time: {datetime.now()}"
        if scores:
            msg += f" | Scores: {scores}"
        self.logger.warning(msg)
    
    def log_sequence_alert(self, seq_info, decision):
        msg = f"🔴 SEQUENCE ANOMALY - Decision: {decision}"
        self.logger.critical(msg)
    
    def log_normal(self, info):
        self.logger.info(f"✅ Normal - {info}")

class IDSLogger:
    def __init__(self, log_file="ids_alerts.log"):
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        
        self.logger = logging.getLogger("IDS")
        self.logger.setLevel(logging.INFO)
        
        # Éviter doublons
        if self.logger.handlers:
            return
        
        # File handler (tous les logs)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        
        # Console handler (seulement warnings et errors)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        # Format
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.packet_count = 0
        self.anomaly_count = 0
        self.sequence_count = 0
        self.seq_anomaly_count = 0
    
    def log_packet_alert(self, score, packet_info=None):
        self.packet_count += 1
        self.anomaly_count += 1
        msg = f"🚨 PACKET ANOMALY #{self.anomaly_count} - Score: {score:.3f}"
        if packet_info:
            msg += f" | {packet_info}"
        self.logger.warning(msg)
    
    def log_sequence_alert(self, decision, info=None):
        self.sequence_count += 1
        self.seq_anomaly_count += 1
        msg = f"🔴 SEQUENCE ANOMALY - Decision: {decision}"
        if info:
            msg += f" | {info}"
        self.logger.critical(msg)
    
    def log_sequence_normal(self):
        self.sequence_count += 1
        self.logger.info("✅ Sequence normal")
    
    def log_stats(self):
        if self.packet_count > 0:
            anomaly_rate = (self.anomaly_count / self.packet_count) * 100
            msg = f"\n{'='*60}\n"
            msg += "📊 STATISTICS\n"
            msg += f"{'='*60}\n"
            msg += f"Total packets analyzed: {self.packet_count}\n"
            msg += f"Packet anomalies detected: {self.anomaly_count} ({anomaly_rate:.2f}%)\n"
            msg += f"Sequences analyzed: {self.sequence_count}\n"
            msg += f"Sequence anomalies: {self.seq_anomaly_count}\n"
            msg += f"{'='*60}\n"
            print(msg)
            self.logger.info(msg)
        
class IDS:
    def __init__(self):
        self.cap_interval = conf_main.get("cap_interval", 100)
        self.save_interval = conf_main.get("save_interval", self.cap_interval // 2)
        self.filename = conf_main.get("filename", "packets.pkl")
        model_file = self.filename = conf_main.get("model_file", "model.pkl")
        self.model_file = os.path.join(_dir_, model_file)
        self.ifaces = conf_main.get('ifaces', None)
        self.model = Models(**config_model)
        self.e = conf_main.get('epochs', 64)
        self.b = conf_main.get('batch_size', 16)
        self.prop_ano = conf_main.get('prop_anomalie', 0.4)
        self.mode = conf_main.get("mode", "all")
        self.v = conf_main.get('verbose', 0)
        self.logger = IDSLogger()
        self.q = Queue()
        self.event = threading.Event()
        self.last_save = time.time()
        self.stats_interval = conf_main.get('stats_interval', 60)
    
    def save(self, filename, value):
        try:
            with open(filename, "wb") as f:
                dill.dump(value, f, protocol=4, recurse=True) 
                print("Sauvegarde réussie dans ", filename)
        except Exception as e:
            print('Erreur dans la sauvegarde dans ', filename, ': ', str(e))
    
    def capture_pcap(self, event:threading.Event, ifaces:list = None):
        """
        

        Parameters
        ----------
        event : threading.Event
            Event pour controller la capture et arrêter les threads
            
        ifaces : list, optional
            La liste des interfaces de captures. The default is None.

        Returns
        -------
        Liste de threads deja demarré qui gèrent la capture chacun sur une interface.

        """
        if isinstance(ifaces, str):
            ifaces = [ifaces]
        if ifaces is None:
            ifaces = get_ifaces()
        lock = threading.Lock()
        def _cap(iface:str):
            try: # Au cas où le name ne marcherait pas
                p = pcap.pcap(
                    name=iface,
                    snaplen=65535,
                    timeout_ms=30,   # Pour temps réel
                    immediate=True,
                    promisc=True,
                    buffer_size=16*1024*1024
                    )
            except:
                p = pcap.pcap(
                    name=None,
                    snaplen=65535,
                    timeout_ms=30,   # Pour temps réel
                    immediate=True,
                    promisc=True,
                    buffer_size=16*1024*1024
                    )
                p.setfilter('tcp or icmp or udp')
            try:
                while not event.is_set():
                    for ts, pkt in p:
                        if event.is_set():
                            break
                        try:
                            eth = dpkt.ethernet.Ethernet(pkt)
                            eth.ts = ts  # Ajouter le timestamp
                            with lock:
                                self.q.put(eth)
                                # print(self.q.qsize(), " items ! ")
                        except:
                            pass
                p.close()
            except Exception as e:
                print('Erreur au niveau de la capture : ', str(e))
                p.close()
        threads = []
        for iface in ifaces:
            th = threading.Thread(target=_cap, args=(iface,), daemon=True, name=f'{iface}_thread')
            th.start()
            threads.append(th)
        return threads

    def signal_manager(self, threads):
        
        def signal_handler(sig, frame):
            for th in threads:
                if th.is_alive():
                    th.join(1)
            sys.exit(0)
                    
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGQUIT, signal_handler)
        
    def main(self):
        X_seq, scaler_seq, X_pkt, scaler_pkt = core(
            ifaces=self.ifaces,
            cap_interval=self.cap_interval,
            save_interval=self.save_interval,
            filename=self.filename,
            )
        if all(c is not None for c in (X_pkt, X_seq, scaler_pkt, scaler_seq)):
            ae_seq, cnn_seq, iso_f, lof_seq, ae_pkt, iso_f_pkt, lof_pkt = self.model.fit(
                X_pkt=X_pkt,
                X_sequences=X_seq,
                epochs=self.e,
                batch=self.b,
                verbose=self.v
                )
            
            obj = {
                "ae_seq": ae_seq,
                "cnn_seq": cnn_seq,
                "iso_f": iso_f,
                "lof_seq": lof_seq,
                "ae_pkt": ae_pkt,
                "iso_f_pkt": iso_f_pkt,
                "lof_pkt": lof_pkt,
                "scaler_seq": scaler_seq,
                "scaler_pkt": scaler_pkt
                }
            
            self.save(
                value=obj,
                filename=self.model_file
                )
            threads = self.capture_pcap(self.event, self.ifaces)
            self.signal_manager(threads)
            print('Lancement de la détetion !')
            
            while True:
                deq = deque(maxlen=SEQ_lENGTH)
                deq_pred = deque(maxlen=SEQ_lENGTH)
                try:
                    pkt = self.q.get_nowait()
                except:
                    continue
                
                if not pkt :
                    continue
                pkt_fea = features_extractor(pkt)
                deq.append(pkt_fea)
                pred = self.model.predict_pkt(list(pkt_fea.values()), scaler_pkt, ae_pkt, iso_f_pkt, lof_pkt, method='predict', mode=self.mode)
                score = self.model.predict_pkt(list(pkt_fea.values()), scaler_pkt, ae_pkt, iso_f_pkt, lof_pkt, method='decision', mode=self.mode)
                # print('Heure : ', time.ctime())
                if score < SEUIL:
                    info = {
                    'timestamp': time.ctime(),
                    "features": pkt_fea,
                    "pred": pred,
                    "confiance": score
                    }
                    self.logger.log_packet_alert(
                        score,
                        info
                    )
                    print('Alerte , packet anormale ({time.ctime()}) ! ')
                else:
                    print(f"Packet normal, ({time.ctime()})")
                    
                if len(deq) == SEQ_lENGTH:
                    seq_feat = seq_extractor(list(deq))
                    key = seq_feat[0].keys()
                    seq_feat = np.array([[s[k] for k in key] for s in seq_feat])
                    prop = sum(p == -1 for p in deq_pred) / SEQ_lENGTH 
                    r = -1 if prop > self.prop_ano else 1
                    pred = self.model.predic_seq(seq_feat, scaler_seq, ae_seq, cnn_seq, iso_f, lof_seq, method="predict", mode=self.mode)
                    score = self.model.predic_seq(seq_feat, scaler_seq, ae_seq, cnn_seq, iso_f, lof_seq, method="decision", mode=self.mode)
                    if self.mode == "all":
                        dec = -1 if np.all(np.array([r, pred]) == -1) else 1
                    else:
                        dec = -1 if np.any(np.array([r, pred]) == -1) else 1
                    if score < SEUIL:
                        print('Alerte sur un séquence, anomalie détectée !')
                        info = {
                            'timestamp': time.ctime(),
                             "features": seq_feat,
                             "pred": dec,
                             "confiance": score
                        }
                        self.logger.log_sequence_alert(
                            score,
                            info,
                        )
                    else:
                        self.logger.log_sequence_normal()
                        print('Séquence normale !')
                        
                if time.time() - self.last_save > self.stats_interval:
                    self.logger.log_stats()
                    self.last_save = time.time()
            else:
                raise ValueError('core a echoué ! Echec du lancement de l\'ids')
                
            
if __name__ == '__main__':
    try:
        ids = IDS()
        ids.main()
    except KeyboardInterrupt:
        print('Interruption détecté !')
    except Exception as e:
        print('Erreur dans ids.main : ', str(e))
        
#https://github.com/hounsoubenny-cyber/ids_with_ia_project