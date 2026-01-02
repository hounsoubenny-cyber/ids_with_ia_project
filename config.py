#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 15:21:26 2026

@author: hounsousamuel
"""

import time

SEQ_lENGTH = 60
SEUIL = -0.6

CONFIG_MODELS = {
    "lr": 1e-5, 
    "inputs":  128, 
    "dropout": 0.2, 
    "max_": 128, 
    "n_estimator": 500,
    "c":  0.08,
    "n_trial": 20
    }

CONFIG_MAIN = {
    "cap_interval": 100,
    "save_interval": 50,
    "prop_anomalie": 0.4,
    "model_file": f"model_{time.time()}.pkl",
    "filename": "packets.pkl",
    "ifaces": None,
    "mode": "all",
    "epochs": 64,
    "batch_size": 16,
    "verbose": 1,
    "stats_interval": 60
    }
