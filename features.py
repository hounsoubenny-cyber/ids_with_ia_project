#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 20:18:22 2025

@author: hounsousamuel
"""

import numpy as np
import dpkt
import time
import socket
import hashlib

def extract_pack_features(eth:dpkt.ethernet.Ethernet):
    try:
        features = {
        'length': 0,
        'time': 0.0,
        'src_mac': 0,
        'dst_mac': 0,
        'ttl': 0,
        'protocol': 0,
        'src_ip0': 0, 'src_ip1': 0, 'src_ip2': 0, 'src_ip3': 0,
        'dst_ip0': 0, 'dst_ip1': 0, 'dst_ip2': 0, 'dst_ip3': 0,
        'sport': 0, 'dport': 0,
        'SYN': 0, 'ACK': 0, 'FIN': 0, 'RST': 0, 'PSH': 0, 'URG': 0,
        'icmp_type': 0, 'icmp_code': 0,
        'payload_len': 0
    }
        if not eth:
            return features
        src_ip_bytes = []
        dst_ip_bytes = []
        features['time'] = getattr(eth, 'ts', 0.0)
        features['length'] = len(eth)
        max_mac = 281474976710655  # 2^48 - 1
        src = eth.src
        dst = eth.dst
        if isinstance(src, bytes):
            features['src_mac'] = int.from_bytes(eth.src, 'big') / max_mac
        if isinstance(dst, bytes):
            features['dst_mac'] = int.from_bytes(eth.dst, 'big') / max_mac
        
        ip = eth.data
        ipv6 = False
        if isinstance(ip, dpkt.ip.IP):
            features['ttl'] = ip.ttl
            features['protocol'] = ip.p
            src = ip.src
            dst = ip.dst
            if isinstance(src, str):
                src = bytes(src.encode())
            if isinstance(dst, str):
                dst = bytes(dst.encode())
            src_ip_bytes = str(socket.inet_ntoa(src) or b'0.0.0.0').split('.')
            dst_ip_bytes = str(socket.inet_ntoa(dst) or b'0.0.0.0').split('.')
            
        
        elif isinstance(ip, dpkt.ip6.IP6):
            ipv6 = True
            features['ttl'] = ip.hlim
            features['protocol'] = ip.nxt
            src = ip.src
            dst = ip.dst
            if isinstance(src, str):
                src = bytes(src.encode())
            if isinstance(dst, str):
                dst = bytes(dst.encode())
            src_ip_bytes = str(socket.inet_ntop(socket.AF_INET6, src) or b'::::')
            dst_ip_bytes = str(socket.inet_ntop(socket.AF_INET6, src) or b'::::')
            
        if not ipv6:
            for i in range(4):
                if i < len(src_ip_bytes):
                    if src_ip_bytes[i]:
                        try:
                            features[f'src_ip{i}'] = int(src_ip_bytes[i])
                        except:
                            features[f'src_ip{i}'] = src_ip_bytes[i]
                            
                if i < len(dst_ip_bytes):
                    if dst_ip_bytes[i]:
                        try:
                            features[f'dst_ip{i}'] = int(dst_ip_bytes[i])
                        except:
                            features[f'dst_ip{i}'] = dst_ip_bytes[i]
        else:
            hash_obj_src = hashlib.md5(src_ip_bytes.encode())
            hash_obj_dst = hashlib.md5(dst_ip_bytes.encode())
            hash_bytes_src = hash_obj_src.digest()
            hash_bytes_dst = hash_obj_dst.digest()
            hash_bytes_src = [p for p in hash_bytes_src if p]
            hash_bytes_dst = [p for p in hash_bytes_dst if p]
            for i in range(4):
                try:
                    if hash_obj_src[i]:
                        try:
                            features[f'src_ip{i}'] = int(hash_obj_src[i])
                        except:
                            features[f'src_ip{i}'] = hash_obj_src[i]
                except:
                    pass
                
                try:
                    if hash_bytes_dst[i]:
                        try:
                            features[f'src_ip{i}'] = int(hash_bytes_dst[i])
                        except:
                            features[f'src_ip{i}'] = hash_bytes_dst[i]
                except:
                    pass
        
        transport = ip.data

        if isinstance(transport, dpkt.tcp.TCP):
            features['sport'] = transport.sport
            features['dport'] = transport.dport
            features['payload_len'] = len(transport.data)
        
            flags = transport.flags
            features['SYN'] = 1 if (flags & 0x02) else 0   # SYN
            features['ACK'] = 1 if (flags & 0x10) else 0   # ACK
            features['FIN'] = 1 if (flags & 0x01) else 0   # FIN
            features['RST'] = 1 if (flags & 0x04) else 0   # RST
            features['PSH'] = 1 if (flags & 0x08) else 0   # PSH
            features['URG'] = 1 if (flags & 0x20) else 0   # URG
        
        elif isinstance(transport, dpkt.udp.UDP):
            features['sport'] = transport.sport
            features['dport'] = transport.dport
            features['payload_len'] = len(transport.data)
        
        elif isinstance(transport, dpkt.icmp.ICMP):
            features['icmp_type'] = transport.type
            features['icmp_code'] = transport.code 
        
        return features
    
    except Exception as e:
        print(f"Erreur extraction features: {e}")
        # Retourner des features par défaut en cas d'erreur
        return features

def extract_seq_features(seq_dicts):
    """
    Prend une liste de dictionnaires (une séquence de paquets) et
    ajoute à chaque paquet les features calculées sur toute la séquence.

    Args:
        seq_dicts : liste de dicts, chaque dict = features d'un paquet

    Returns:
        nouvelle liste de dicts enrichie avec les features de la séquence
    """
    # Récupérer les clés du dict pour accès cohérent
    # print(seq_dicts[0])
    # # print(buffer_fea)
    # input()
    keys = list(seq_dicts[0].keys())

    # Convertir en array pour faciliter les calculs
    data = np.array([[pkt[k] for k in keys] for pkt in seq_dicts])
    # print(data.dtype)
    # input()

    # Calcul des features par séquence
    seq_features = {
        "seq_length_mean": np.mean(data[:, keys.index("length")], dtype=np.int64),
        "seq_length_max": np.max(data[:, keys.index("length")]),
        "seq_payload_mean": np.mean(data[:, keys.index("payload_len")], dtype=np.int64),
        "seq_SYN_count": np.sum(data[:, keys.index("SYN")], dtype=np.int64),
        "seq_ACK_count": np.sum(data[:, keys.index("ACK")], dtype=np.int64),
        "seq_FIN_count": np.sum(data[:, keys.index("FIN")], dtype=np.int64),
        "seq_RST_count": np.sum(data[:, keys.index("RST")], dtype=np.int64),
        "seq_PSH_count": np.sum(data[:, keys.index("PSH")], dtype=np.int64),
        "seq_URG_count": np.sum(data[:, keys.index("URG")], dtype=np.int64),
        "seq_ICMP_count": np.sum(data[:, keys.index("icmp_type")] > 0, dtype=np.int64)
    }

    # Ajouter ces features à chaque paquet
    enriched_seq = []
    for pkt in seq_dicts:
        pkt_copy = pkt.copy()
        pkt_copy.update(seq_features)
        enriched_seq.append(pkt_copy)

    return enriched_seq
