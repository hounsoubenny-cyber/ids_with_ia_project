#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 09:55:32 2026

@author: hounsousamuel
"""

import os, sys
import pcap, dpkt
import threading
import netifaces
import time, joblib
import numpy as np
from config import SEQ_lENGTH
from sklearn.preprocessing import StandardScaler
from features import extract_pack_features as features_extractor, extract_seq_features as seq_extractor

QUEUE = []

_dir_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data','datasets')
os.makedirs(_dir_, exist_ok=True)

def get_ifaces():
    """
    Fontions pour obtenir les interfaces systemes.

    Returns
    -------
    Liste des interfaces.
    """
    ifaces = list(netifaces.interfaces()) + list(pcap.findalldevs())
    to_return = set()
    PREFIXES = ['lo', 'any', "usb", 'bluetooth', 'nf']
    for iface in ifaces:
        iface = str(iface)
        if any(iface.startswith(p) for p in PREFIXES):
            continue
        to_return.add(iface)
    return list(to_return)
        
def capture_pcap(event:threading.Event, ifaces:list = None):
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
                            QUEUE.append(eth)
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


def _save(filename, value):
    try:
        joblib.dump(value, filename)
    except Exception as e:
        print(f'Erreur de sauvegarde dans {filename} : ', str(e))
        
def save(save_interval:float, event:threading.Event, filename:str):
    while not event.is_set():
        time.sleep(save_interval)
        try:
            _save(filename, QUEUE)
        except:
            pass
        

def capture_manager(ifaces:str = None, filename:str = "packets.pkl", save_interval:float = 100, cap_interval:float = 200):
    event = threading.Event()
    threads = capture_pcap(event, ifaces)
    filename = os.path.join(_dir_, filename)
    save_thread = threading.Thread(target=save, args=(save_interval, event, filename, ), daemon=True)
    save_thread.start()
    start = time.time()
    time.sleep(2) # Attendre que les threads soit complètement lancés
    try:
        while time.time() - start <= cap_interval:
            time.sleep(1)
            elapsed = time.time() - start
            print(f"Capture lancé pour {cap_interval} seconde(s) il y a {elapsed:.2f} {'seconde' if elapsed < 2 else 'secondes'}({elapsed / 60 :.2f} {'minute' if (elapsed / 60) < 2 else 'minutes'})", end="\r")  # Ecrire sur la meme ligne a chaque fois
            if elapsed > cap_interval:
                break
    except KeyboardInterrupt:
        event.set()
    except Exception as e:
        print('Erreur dans capture_manager : ', str(e))
    finally:
        event.set()
        save_thread.join(1)
        for th in threads:
            try:
                if th.is_alive():
                    th.join(1)
            except Exception as e:
                print('Erreur lors de l\arrêt d\'un thread : ', str(e))
                
    return QUEUE, filename

def core(ifaces:str = None, filename:str = 'packets.pkl', save_interval:float = 100, cap_interval:float = 200):
    try:
        LIST, filename = capture_manager(ifaces, filename, save_interval, cap_interval)
        if not LIST:
            try:
                LIST = joblib.load(filename)
            except:
                raise
        if not LIST:
            raise ValueError('LIST vide !')
        print()
        print("\n", len(LIST), "packets enrégistrés durant la durée !")
        n_seq = len(LIST) - SEQ_lENGTH + 1  # Decalage
        if n_seq < 0:
            raise ValueError(f'Pas assez de pakcet pour un séquence ({len(LIST)})')
    
        scaler = StandardScaler()
        seq_scaler = StandardScaler()
        features = [features_extractor(pkt) for pkt in LIST]
        keys = features[0].keys()
        array_fea = np.array([[f[k] for k in keys] for f in features]) # Arranger selon l'ordre du premier feature
        sequences = [features[i:SEQ_lENGTH + i] for i in range(n_seq)]
        sequences_array = []
        for seq in sequences:
            _seq = list(seq_extractor(seq))
            _seq = [[s[k] for  k in keys] for s in _seq]
            sequences_array.append(np.array(_seq))
            
        sequences_array = np.array(sequences_array)
        flat = sequences_array.reshape(-1, sequences_array.shape[-1])
        seq_scaler.fit(flat)
        print("[DEBUG] Avant nettoyage:")
        print(f"  NaN dans les packets: {np.isnan(array_fea).sum()}")
        print(f"  Inf dans packets: {np.isinf(array_fea).sum()}")
        print(f"  Min/Max: {array_fea.min():.2f} / {array_fea.max():.2f}")
        
        print(f"  NaN dans séquences: {np.isnan(sequences_array).sum()}")
        print(f"  Inf dans séquences: {np.isinf(sequences_array).sum()}")
        print(f"  Min/Max: {sequences_array.min():.2f} / {sequences_array.max():.2f}")
        
        array_fea = np.nan_to_num(array_fea, posinf=1.0, neginf=-1.0, nan=0.0)
        sequences_array = np.nan_to_num(sequences_array, posinf=1.0, neginf=-1.0, nan=0.0)
        
        print("[DEBUG] Après nettoyage:")
        print(f"  NaN dans les packets: {np.isnan(array_fea).sum()}")
        print(f"  Inf dans packets: {np.isinf(array_fea).sum()}")
        print(f"  Min/Max: {array_fea.min():.2f} / {array_fea.max():.2f}")
        
        print(f"  NaN dans séquences: {np.isnan(sequences_array).sum()}")
        print(f"  Inf dans séquences: {np.isinf(sequences_array).sum()}")
        print(f"  Min/Max: {sequences_array.min():.2f} / {sequences_array.max():.2f}")
    
        sequences_array = np.array([seq_scaler.transform(x) for x in sequences_array])
        array_fea = scaler.fit_transform(array_fea)
        return sequences_array, seq_scaler, array_fea, scaler
    except Exception as e:
        print('Erreur dans core : ', str(e))
        import traceback
        traceback.print_exc()
        return None, None, None, None
    
if __name__ == "__main__":
    cap = 100
    sa = 50
    s, ss, p, sp = core(cap_interval=cap, save_interval=sa)
            
             
        
                    
                    
                
            