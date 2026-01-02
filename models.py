#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 09:55:51 2026

@author: hounsousamuel
"""

import os, sys
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, TimeDistributed, Dropout, LayerNormalization, Attention,
    Conv1D, MaxPooling1D, UpSampling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna, time

class Models:
    """
    Gestionnaire des modèles.
    """
    def __init__(self, lr:float = 1e-5, inputs:int = 128, dropout:float = 0.2, max_:int = 128, n_estimator:int = 500, c:float = 0.01, n_trial:int = 20):
        self.lr = lr
        self.inputs = inputs
        self.dropout = dropout
        self.max = max_
        self.n_estimator = n_estimator
        self.c = c
        self.n_trial = n_trial
        self.v = 0
    
    def _get_cnn(self, n_pkt, n_seq_fea):
        """
        
        Parameters
        ----------
        n_pkt : int
            Nombre de packets dans la séquence.
        n_seq_fea : int
            Nombre de features dans un packet.

        Returns
        -------
        Le models compilé.

        """
        input_ = Input(shape=(n_pkt, n_seq_fea))
        x = Conv1D(filters=max(self.max, self.inputs), kernel_size=3, padding='same', activation='swish')(input_)
        # x = MaxPooling1D(pool_size=2)(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        x = Conv1D(filters=max(64, self.inputs // 2), kernel_size=5, padding='same', activation='swish')(x)
        # x = MaxPooling1D(pool_size=2)(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        x = Conv1D(filters=max(32, self.inputs // 4), kernel_size=6, padding='same', activation='swish')(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        x = Conv1D(filters=max(16, self.inputs // 8), kernel_size=7, padding='same', activation='swish')(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        dec = Conv1D(filters=max(16, self.inputs // 8), kernel_size=7, padding='same', activation='swish')(x)
        dec = LayerNormalization()(dec)
        dec = Dropout(self.dropout)(dec)
        
        dec = Conv1D(filters=max(32, self.inputs // 4), kernel_size=6, padding='same', activation='swish')(dec)
        dec = LayerNormalization()(dec)
        dec = Dropout(self.dropout)(dec)
        
        dec = Conv1D(filters=max(64, self.inputs // 2), kernel_size=5, padding='same', activation='swish')(dec)
        # dec = UpSampling1D(size=2)(dec)
        dec = LayerNormalization()(dec)
        dec = Dropout(self.dropout)(dec)
        
        dec = Conv1D(filters=max(self.max,  self.inputs), kernel_size=3, padding='same', activation='swish')(dec)
        dec = LayerNormalization()(dec)
        # dec = UpSampling1D(size=2)(dec)
        
        out = Conv1D(filters=n_seq_fea, kernel_size=3, padding='same', activation='linear')(dec)
        model = Model(input_, out)
        
        model.compile(loss="mse", optimizer=Adam(self.lr), metrics=self._get_metrics())
        return model
        
    def _get_ltsm(self, n_pkt:int, n_seq_fea:int):
        """

        Parameters
        ----------
        n_pkt : int
            Nombre de packets dans la séquence.
        n_seq_fea : int
            Nombre de features dans un packet.

        Returns
        -------
        Le models compilé.

        """
        input_ = Input(shape=(n_pkt, n_seq_fea))
        x = TimeDistributed(Dense(max(self.max, self.inputs), activation='swish'))(input_)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        attn = Attention()([x, x])
        x = TimeDistributed(Dense(max(64 , self.inputs // 2), activation='swish'))(attn)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        x = LSTM(max(32, self.inputs // 4), return_sequences=True)(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        attn = Attention()([x, x])
        x = LSTM(max(16, self.inputs // 8), return_sequences=True)(attn)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        attn = Attention()([x, x])
        dec = LSTM(max(16, self.inputs // 8), return_sequences=True)(attn)
        dec = LayerNormalization()(dec)
        dec = Dropout(self.dropout)(dec)
        
        dec = LSTM(max(32, self.inputs // 4), return_sequences=True)(x)
        dec = LayerNormalization()(dec)
        dec = Dropout(self.dropout)(dec)
        
        attn = Attention()([x, x])
        dec = TimeDistributed(Dense(max(64 , self.inputs // 2), activation='swish'))(attn)
        dec = LayerNormalization()(dec)
        dec = Dropout(self.dropout)(dec)
        
        dec = TimeDistributed(Dense(max(self.max , self.inputs), activation='swish'))(dec)
        dec = LayerNormalization()(dec)
        
        out = TimeDistributed(Dense(n_seq_fea, activation="linear"))(dec)
        
        model = Model(input_, out)
        model.compile(loss="mse", optimizer=Adam(self.lr), metrics=self._get_metrics())
        return model
        
        
        
    def _get_callbacks(self):
        return [
            EarlyStopping(patience=15, monitor="loss", restore_best_weights=True)
            ]
    
    def _get_dense(self, n_pkt_fea:int):
        """
        
        Returns
        -------
        None.

        """
        input_ = Input(shape=(n_pkt_fea,))
        x = Dense(max(self.max, self.inputs), activation="swish")(input_)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        x = Dense(max(64, self.inputs // 2), activation="swish")(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        x = Dense(max(32, self.inputs // 4), activation="swish")(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        x = Dense(max(16, self.inputs // 8), activation="swish")(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        dec = Dense(max(16, self.inputs // 8), activation="swish")(x)
        dec = LayerNormalization()(dec)
        dec = Dropout(self.dropout)(dec)
        
        dec = Dense(max(32, self.inputs // 4), activation="swish")(dec)
        dec = LayerNormalization()(dec)
        dec = Dropout(self.dropout)(dec)
        
        dec = Dense(max(64, self.inputs // 2), activation="swish")(dec)
        dec = LayerNormalization()(dec)
        dec = Dropout(self.dropout)(dec)
        
        dec = Dense(max(self.max, self.inputs), activation="swish")(dec)
        dec = LayerNormalization()(dec)
        dec = Dropout(self.dropout)(dec)
        
        out = Dense(n_pkt_fea, activation='linear')(dec)
        model = Model(input_, out)
        model.compile(loss="mse", metrics=self._get_metrics(), optimizer=Adam(self.lr))
        return model
    
    def _get_metrics(self):
        """
        Donne les métrics de fitting.

        Returns
        -------
        None.

        """
        return [
            tf.keras.metrics.MeanSquaredError(name='mse'),      # Gros erreurs pénalisées
            tf.keras.metrics.MeanAbsoluteError(name='mae'),     # Erreurs moyennes
            tf.keras.metrics.CosineSimilarity(name='cosine_sim'),
            #tf.keras.metrics.MeanAbsolutePercentageError(name='mape')  # Pourcentage d'erreur
        ]
    
    def build_model(self, n_pkt, n_pkt_fea, n_seq_fea):
        """

        Parameters
        ----------
        n_pkt : TYPE
            DESCRIPTION.
        n_pkt_fea : TYPE
            DESCRIPTION.
        n_seq_fea : TYPE
            DESCRIPTION.

        Returns
        -------
        ae_seq : TYPE
            DESCRIPTION.
        cnn_seq : TYPE
            DESCRIPTION.
        iso_f : TYPE
            DESCRIPTION.
        lof_seq : TYPE
            DESCRIPTION.
        ae_pkt : TYPE
            DESCRIPTION.
        iso_f_pkt : TYPE
            DESCRIPTION.
        lof_pkt : TYPE
            DESCRIPTION.

        """
        lof_seq = LocalOutlierFactor(
            n_jobs=-1,
            novelty=True,
            n_neighbors=self.n_estimator,
            contamination=self.c
            )
        
        lof_pkt = LocalOutlierFactor(
            n_jobs=-1,
            novelty=True,
            n_neighbors=self.n_estimator,
            contamination=self.c
            )
        
        iso_f = IsolationForest(
            n_estimators=self.n_estimator,
            contamination=self.c,
            n_jobs=-1,
            )
        
        iso_f_pkt = IsolationForest(
            n_estimators=self.n_estimator,
            contamination=self.c,
            n_jobs=-1,
            )
        
        ae_seq = self._get_ltsm(n_pkt, n_seq_fea)
        ae_pkt = self._get_dense(n_pkt_fea)
        cnn_seq = self._get_cnn(n_pkt, n_seq_fea)
        
        return ae_seq, cnn_seq, iso_f, lof_seq, ae_pkt, iso_f_pkt, lof_pkt
    
    def get_opt_params_lof(self, trial):
        params_opt_one = {
                    'n_neighbors': trial.suggest_int('n_neighbors',15, 50),
                    'contamination': trial.suggest_float('contamination', 0.05,0.2),
                    'leaf_size': trial.suggest_int('leaf_size',20, 40),
                    'metric': trial.suggest_categorical('metric', ["euclidean", "minkowski", "manhattan"]),
                }

        return params_opt_one

    def get_opt_params_if(self, trial):
        params_opt_if = {
                'contamination': trial.suggest_float('contamination', 0.05, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                "max_samples":  trial.suggest_float('max_samples', 0.1, 1.0),
                "max_features":  trial.suggest_float('max_features', 0.5, 1.0),
            }
        return params_opt_if

    def optimize(self, X,  classe, name, n_trial=50, timeout=None, rs=None):
        """
        

        Parameters
        ----------
        X : np.array
            Donné de fitting.
        classe : class
            La classe du modèle.
        name : str
            Le nom du modèle.
        n_trial : int, optional
            Nombre d'itérations. The default is 50.
        timeout : float, optional
            Durée d'optimisation. The default is None.
        rs : int, optional
            random_state. The default is None.

        Returns
        -------
        dict
            Dictinnaire complet contenat le modèles les stats et les meilleurs params.

        """
        def _optimize(trial):
                params = self.get_opt_params_if(trial) if name == "IsolationForest" else self.get_opt_params_lof(trial)
                if name == "Local Outlier Factor":
                    model =  classe(**params, novelty=True, n_jobs=-1)
                    model.fit(X)
                else:
                    model = classe(**params) if not rs else classe(**params, random_state=rs)
                    model.fit(X, X)
                return np.mean(model.score_samples(X))
            
        s = time.time()
        study = optuna.create_study(direction='maximize')
        study.optimize(_optimize, n_trials=n_trial, timeout=timeout, n_jobs=-1, show_progress_bar=bool(self.v))
        t = time.time() - s
        print('Optimisation terminé en ', t,' secondes')
        if name == 'Local Outlier Factor':
            best_model = classe(**study.best_params, novelty=True, n_jobs=-1)
        else:
            best_model = classe(**study.best_params) if not rs else classe(**study.best_params, random_state=rs)
            
        return {
            'best_model': best_model,
            'best_params' : study.best_params,
            'best_score': study.best_value,
            'df': study.trials_dataframe()
        }
    
    def fit(self, X_sequences, X_pkt, epochs, batch, verbose=0):
        self.v = verbose
        X_sequences, X_pkt = np.asarray(X_sequences), np.asarray(X_pkt)
        n_pkt, n_pkt_fea, n_seq_fea = X_sequences.shape[1], X_pkt.shape[-1], X_sequences.shape[2]
        ae_seq, cnn_seq, iso_f, lof_seq, ae_pkt, iso_f_pkt, lof_pkt = self.build_model(n_pkt, n_pkt_fea, n_seq_fea)
        
        ae_seq.fit(
            X_sequences,
            X_sequences,
            callbacks=self._get_callbacks(),
            verbose=self.v,
            epochs=epochs,
            batch_size=batch,
            shuffle=True,
            validation_split=0.1
            )
        
        cnn_seq.fit(
            X_sequences,
            X_sequences,
            callbacks=self._get_callbacks(),
            verbose=self.v,
            epochs=epochs,
            batch_size=batch,
            shuffle=True,
            validation_split=0.1
            )
        
        ae_pkt.fit(
            X_pkt,
            X_pkt,
            callbacks=self._get_callbacks(),
            verbose=self.v,
            epochs=epochs,
            batch_size=batch,
            shuffle=True,
            validation_split=0.1
            )
        
        X_seq_pred_cnn, X_seq_pred_ae = np.asarray(cnn_seq.predict(X_sequences)), np.asarray(ae_seq.predict(X_sequences))
        flat_cnn, flat_ae = X_seq_pred_cnn.reshape(X_seq_pred_cnn.shape[0],-1), X_seq_pred_ae.reshape(X_seq_pred_ae.shape[0], -1)  # Maintenir le nombre d'element, la premiere dim et transformer en 2D
        diff_cnn, diff_ae = X_sequences - X_seq_pred_cnn, X_sequences - X_seq_pred_ae
        mse_cnn = np.mean(diff_cnn ** 2, axis=(1, 2)).reshape(-1, 1)
        mae_cnn = np.mean(np.abs(diff_cnn), axis=(1, 2)).reshape(-1, 1)
        
        mse_ae = np.mean(diff_ae ** 2, axis=(1, 2)).reshape(-1, 1)
        mae_ae = np.mean(np.abs(diff_ae), axis=(1, 2)).reshape(-1, 1)
        
        X_seq_sklearn = np.concatenate((flat_cnn, flat_ae, mae_cnn, mse_cnn, mae_ae, mse_ae), axis=1)  # Ajouter des colonnes / features
        dic_iso_f = self.optimize(X_seq_sklearn, IsolationForest, "IsolationForest", rs=42, n_trial=self.n_trial)
        iso_f = dic_iso_f["best_model"]
        iso_f.fit(X_seq_sklearn)
        print('Meilleur score : ', dic_iso_f['best_score'])
        print('Meilleur params : ', dic_iso_f['best_params'])
        print('DataFrame Trial : \n', dic_iso_f['df'])
        
        dic_iso_f = self.optimize(X_seq_sklearn, IsolationForest, "IsolationForest", rs=42, n_trial=self.n_trial)
        iso_f = dic_iso_f["best_model"]
        iso_f.fit(X_seq_sklearn)
        print('Meilleur score : ', dic_iso_f['best_score'])
        print('Meilleur params : ', dic_iso_f['best_params'])
        print('DataFrame Trial : \n', dic_iso_f['df'])
        
        dic_lof = self.optimize(X_seq_sklearn, LocalOutlierFactor, "Local Outlier Factor", n_trial=self.n_trial)
        lof_seq = dic_lof["best_model"]
        lof_seq.fit(X_seq_sklearn)
        print('Meilleur score : ', dic_lof['best_score'])
        print('Meilleur params : ', dic_lof['best_params'])
        print('DataFrame Trial : \n', dic_lof['df'])
        
        # Maintenant les packets
        ae_pkt_pred = np.asarray(ae_pkt.predict(X_pkt))
        diff = X_pkt - ae_pkt_pred
        mae = np.mean(np.abs(diff), axis=1).reshape(-1, 1)
        mse = np.mean(diff ** 2, axis=1).reshape(-1, 1)
        X_pkt_sklearn = np.concatenate((ae_pkt_pred, mae, mse), axis=1)
        
        dic_iso_f_pkt = self.optimize(X_pkt_sklearn, IsolationForest, name="IsolationForest", n_trial=self.n_trial, rs=42)
        iso_f_pkt = dic_iso_f_pkt["best_model"]
        iso_f_pkt.fit(X_pkt_sklearn)
        print('Meilleur score : ', dic_iso_f_pkt['best_score'])
        print('Meilleur params : ', dic_iso_f_pkt['best_params'])
        print('DataFrame Trial : \n', dic_iso_f_pkt['df'])
        
        dic_lof_pkt = self.optimize(X_pkt_sklearn, LocalOutlierFactor, name="Local Outlier Factor", n_trial=self.n_trial)
        lof_pkt = dic_lof_pkt["best_model"]
        lof_pkt.fit(X_pkt_sklearn)
        print('Meilleur score : ', dic_lof_pkt['best_score'])
        print('Meilleur params : ', dic_lof_pkt['best_params'])
        print('DataFrame Trial : \n', dic_lof_pkt['df'])
        
        return ae_seq, cnn_seq, iso_f, lof_seq, ae_pkt, iso_f_pkt, lof_pkt
    
    def predic_seq(self, X, scaler, ae_seq, cnn_seq, iso_f, lof_seq, mode="and",  method='predict'):
        scaled = scaler.transform(X)
        new = scaled[np.newaxis, :, :]   # Ajouter une dimension, car (n_sequences(ici 1), n_pkt, n_features)
        cnn_pred, ae_pred = np.asarray(cnn_seq.predict(new, verbose=self.v)), np.asarray(ae_seq.predict(new, verbose=self.v))
        diff_cnn, diff_ae = new - cnn_pred, new - ae_pred
        mse_cnn = np.mean(diff_cnn ** 2, axis=(1, 2)).reshape(-1, 1)
        mae_cnn = np.mean(np.abs(diff_cnn), axis=(1, 2)).reshape(-1, 1)
        
        mse_ae = np.mean(diff_ae ** 2, axis=(1, 2)).reshape(-1, 1)
        mae_ae = np.mean(np.abs(diff_ae), axis=(1, 2)).reshape(-1, 1
                                                               )
        flat_cnn, flat_ae = cnn_pred.reshape(cnn_pred.shape[0], -1), ae_pred.repeat(ae_pred.shape[0], -1)
        X_seq_sklearn = np.concatenate((flat_cnn, flat_ae, mae_cnn, mse_cnn, mae_ae, mse_ae), axis=1)  # Ajouter des colonnes / features
        if "decision" in method.lower() :
            preds = [c.decision_function(X_seq_sklearn) for c in (iso_f, lof_seq)]
            return float(np.clip(np.mean(np.array(preds)), -1, 1))
        else:
            preds = [c.predict(X_seq_sklearn) for c in (iso_f, lof_seq)]
            if mode == "any":
                r = -1 if np.any(np.array(preds) == -1) else 1
            else:
                r = -1 if np.all(np.array(preds) == -1) else 1
            return r
        
    def predict_pkt(self, X, scaler, ae_pkt, iso_f_pkt, lof_pkt, mode="and", method='predict'):
        if isinstance(X, dict):
            X = [list(X.values())]
        scaled = scaler.transform([X])
        pred_ae = np.asarray(ae_pkt.predict(scaled, verbose=self.v))
        diff = scaled - pred_ae
        mae = np.mean(np.abs(diff), axis=1).reshape(-1, 1)
        mse = np.mean(diff ** 2, axis=1).reshape(-1, 1)
        X_pkt_sklearn = np.concatenate((pred_ae, mae, mse), axis=1)
        if "decision" in method.lower() :
            preds = [c.decision_function(X_pkt_sklearn) for c in (iso_f_pkt, lof_pkt)]
            return float(np.clip(np.mean(np.array(preds)), -1, 1))
        else:
            preds = [c.predict(X_pkt_sklearn) for c in (iso_f_pkt, lof_pkt)]
            if mode == "any":
                r = -1 if np.any(np.array(preds) == -1) else 1
            else:
                r = -1 if np.all(np.array(preds) == -1) else 1
            return r
       
        