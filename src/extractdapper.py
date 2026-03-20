import os
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt, find_peaks

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from scipy.integrate import trapezoid
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
import scipy.signal as scisig
from scipy.signal import resample
import json
import ast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import onnx

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    matthews_corrcoef, roc_auc_score
)

import warnings
warnings.filterwarnings(
    "ignore",
    module="neurokit2"
)

np.set_printoptions(precision=3, suppress=True)

# ============================ Data Feature Parameters =========================================
USE_FREQ_HRV = False

def dom_nonzero_freq(signal, fs_hz):
    """
    Returns most dominant non-zero frequency via fft
    """
    T, K, C = signal.shape
    y = np.fft.rfft(signal, axis = 0)
    yf = np.fft.rfftfreq(T, 1/fs_hz)
    mag = np.abs(y)
    mag[0, :, :] = 0
    idx = np.argmax(mag, axis = 0)
    return yf[idx]

def eda_features(eda, fs, corr_method="pearson"):
    """
    Compute SCR/SCL features from an EDA signal.

    Parameters
    ----------
    eda : array-like
        Raw EDA signal (skin conductance).
    fs : float
        Sampling frequency in Hz.
    corr_method : str
        "pearson" or "spearman" for SCL-time correlation.

    Returns
    -------
    features : dict
        Dictionary of computed features.
    signals : pd.DataFrame
        Processed signal dataframe from neurokit2.
    info : dict
        Event metadata returned by neurokit2.
    """
    eda = np.asarray(eda, dtype=float)

    # 1) Clean + decompose + detect SCR peaks
    # signals columns typically include:
    # EDA_Clean, EDA_Tonic (SCL), EDA_Phasic (SCR), SCR_Onsets, SCR_Peaks, SCR_Recovery, ...
    signals, info = nk.eda_process(eda, sampling_rate=fs)

    scl = signals["EDA_Tonic"].to_numpy()   # tonic = SCL
    scr = signals["EDA_Phasic"].to_numpy()  # phasic = SCR

    # Time vector
    t = np.arange(len(eda)) / fs

    # 2) Correlation between SCL and time
    if corr_method.lower() == "pearson":
        scl_time_corr, scl_time_corr_p = pearsonr(t, scl)
    elif corr_method.lower() == "spearman":
        scl_time_corr, scl_time_corr_p = spearmanr(t, scl)
    else:
        raise ValueError("corr_method must be 'pearson' or 'spearman'")

    # 3) Get SCR event indices
    # NeuroKit2 stores indices in info, but keys may vary slightly depending on version.
    onsets = np.array(info.get("SCR_Onsets", []), dtype=float)
    peaks = np.array(info.get("SCR_Peaks", []), dtype=float)
    recovery = np.array(info.get("SCR_Recovery", []), dtype=float)

    # Remove NaNs / invalid values and cast to int
    onsets = onsets[np.isfinite(onsets)].astype(int)
    peaks = peaks[np.isfinite(peaks)].astype(int)
    recovery = recovery[np.isfinite(recovery)].astype(int)

    # Align events robustly (onset -> peak -> recovery)
    # We'll create matched segments where onset < peak < recovery if possible.
    segments = []
    used_recovery = set()

    for peak in peaks:
        onset_candidates = onsets[onsets < peak]
        if len(onset_candidates) == 0:
            continue
        onset = onset_candidates[-1]  # nearest onset before peak

        rec_candidates = recovery[(recovery > peak)]
        rec_candidates = [r for r in rec_candidates if r not in used_recovery]
        if len(rec_candidates) == 0:
            # If no recovery marker, estimate end at next time SCR returns near local baseline
            # Simple fallback: use peak + 4s (capped to signal length)
            rec = min(len(scr) - 1, int(peak + 4 * fs))
        else:
            rec = rec_candidates[0]
            used_recovery.add(rec)

        if onset < peak < rec:
            segments.append((onset, peak, rec))

    # 4) Number of SCR segments
    n_scr_segments = len(segments)

    # 5) SCR startle magnitudes and response durations
    # Startle magnitude here = peak amplitude relative to onset baseline (phasic component)
    # Duration = recovery - onset (seconds)
    startle_magnitudes = []
    response_durations = []
    scr_auc_list = []

    for onset, peak, rec in segments:
        magnitude = scr[peak] - scr[onset]
        duration = (rec - onset) / fs

        # Area under identified SCR segment (above onset baseline)
        segment_y = scr[onset:rec + 1] - scr[onset]
        # Optional: clip negative values so only positive response contributes
        segment_y = np.clip(segment_y, 0, None)
        segment_t = t[onset:rec + 1]
        auc = trapezoid(segment_y, segment_t)

        startle_magnitudes.append(magnitude)
        response_durations.append(duration)
        scr_auc_list.append(auc)

    startle_magnitudes = np.array(startle_magnitudes, dtype=float)
    response_durations = np.array(response_durations, dtype=float)
    scr_auc_list = np.array(scr_auc_list, dtype=float)

    sum_scr_startle_magnitudes = float(np.nansum(startle_magnitudes)) if len(startle_magnitudes) else 0.0
    sum_response_durations = float(np.nansum(response_durations)) if len(response_durations) else 0.0
    area_under_identified_scr = float(np.nansum(scr_auc_list)) if len(scr_auc_list) else 0.0

    features = {
        "EDA_mean": np.mean(eda),
        "EDA_std": np.std(eda),
        "EDA_max": max(eda),
        "EDA_min": min(eda),
        "EDA_slope": np.polyfit(np.arange(len(eda)), eda, 1)[0],
        "EDA_range": max(eda) - min(eda),

        # Whole-signal tonic/phasic summaries
        "SCL_mean": float(np.nanmean(scl)),
        "SCL_std": float(np.nanstd(scl)),
        "SCR_mean": float(np.nanmean(scr)),
        "SCR_std": float(np.nanstd(scr)),

        # Correlation
        "SCL_time_corr": float(scl_time_corr),
        "SCL_time_corr_pvalue": float(scl_time_corr_p),

        # Event-based SCR features
        "num_SCR_segments": int(n_scr_segments),
        "sum_SCR_startle_magnitudes": sum_scr_startle_magnitudes,
        "sum_response_durations_sec": sum_response_durations,
        "area_under_identified_SCR": area_under_identified_scr,

        # Optional per-event stats
        "mean_SCR_startle_magnitude": float(np.nanmean(startle_magnitudes)) if len(startle_magnitudes) else 0.0,
        "mean_response_duration_sec": float(np.nanmean(response_durations)) if len(response_durations) else 0.0,
        "mean_SCR_auc": float(np.nanmean(scr_auc_list)) if len(scr_auc_list) else 0.0,
    }

    return features, signals, info

# =========================== Data processing: Dapper ===================================================
SAMPLE_FS_Dapper = {'ACC': 20, 'PPG': 20, 'GSR': 40}
INPUT_WINDOW = 60
STEP_TIME = 6

dapper_path = "/home/general/Documents/Eric/Autonomous-Bio-Kinetic-Event-Detection/data/Dapper/"
pids = os.listdir(dapper_path + 'Physiol_Rec')

label_file = pd.read_excel(dapper_path + 'Psychol_Rec/ESM.xlsx')
label_file[" StartTime "] = pd.to_datetime(label_file[" StartTime "])

csv_data = pd.DataFrame() #pd.read_csv("output.csv")

for pid in pids:
    if pid == "README.txt":
        continue
    print(pid)
    # if int(pid) in csv_data["PID"].values:
    #     continue
    folder = dapper_path + f'Physiol_Rec/{pid}/'
    unique_dates = set()

    pid_label = label_file[label_file["Participant ID"] == int(pid)]

    for file in os.listdir(folder):
        file_arr = file.split('_')
        file_arr[-1] = file_arr[-1][:-4]
        if file_arr[0] in unique_dates:
            continue
        unique_dates.add(file_arr[0])

        if int(pid) == 3024 and file_arr[0] == "20191210220625":
            continue

        acc = pd.read_csv(folder + file_arr[0] + '_' + file_arr[1] + '_ACC.csv')
        acc = np.array(acc.drop(columns=['csv_time_motion']))
        
        gsr = pd.read_csv(folder + file_arr[0] + '_' + file_arr[1] + '_GSR.csv')
        gsr = np.array(gsr["GSR"])
        
        ppg = pd.read_csv(folder + file_arr[0] + '_' + file_arr[1] + '_PPG.csv')
        time = pd.to_datetime(ppg['csv_time_PPG'], format='mixed')
        ppg = np.array(ppg["PPG"])
        ppg_signals, ppg_info = nk.ppg_process(ppg, sampling_rate=SAMPLE_FS_Dapper['PPG'])

        for start in range(int((len(ppg) / SAMPLE_FS_Dapper['PPG'] - INPUT_WINDOW + STEP_TIME) // STEP_TIME)):
            ppg_s = ppg[start * STEP_TIME * SAMPLE_FS_Dapper['PPG']: start * STEP_TIME * SAMPLE_FS_Dapper['PPG'] + INPUT_WINDOW * SAMPLE_FS_Dapper['PPG']]
            peaks_s = np.array(ppg_signals['PPG_Peaks'][start * STEP_TIME * SAMPLE_FS_Dapper['PPG']: start * STEP_TIME * SAMPLE_FS_Dapper['PPG'] + INPUT_WINDOW * SAMPLE_FS_Dapper['PPG']])
            if np.count_nonzero(peaks_s) <= 2 or len(ppg_s) != INPUT_WINDOW * SAMPLE_FS_Dapper['PPG']:
                continue
            acc_s = acc[start * STEP_TIME * SAMPLE_FS_Dapper['ACC']: start * STEP_TIME * SAMPLE_FS_Dapper['ACC'] + INPUT_WINDOW * SAMPLE_FS_Dapper['ACC']]
            if acc_s.shape[0] != INPUT_WINDOW * SAMPLE_FS_Dapper['ACC']:
                continue
            acc_s = acc_s.reshape(100, 12, 3)

            gsr_s = gsr[start * STEP_TIME * SAMPLE_FS_Dapper['GSR']: start * STEP_TIME * SAMPLE_FS_Dapper['GSR'] + INPUT_WINDOW * SAMPLE_FS_Dapper['GSR']]
            if gsr_s.shape[0] != INPUT_WINDOW * SAMPLE_FS_Dapper['GSR']:
                continue
            acc_mean = np.mean(acc_s, axis = 0)
            acc_std = np.std(acc_s, axis = 0)
            acc_sum = np.trapezoid(np.abs(acc_s), axis = 0)
            acc_peak_freq = dom_nonzero_freq(acc_s, SAMPLE_FS_Dapper['ACC'])
            acc_data = np.concatenate([acc_mean.flatten(), acc_std.flatten(), acc_sum.flatten(), acc_peak_freq.flatten()])

            hrv = nk.hrv_time(peaks_s, sampling_rate=SAMPLE_FS_Dapper['PPG'], show=False) if not USE_FREQ_HRV else nk.hrv(peaks_s, sampling_rate=SAMPLE_FS_Dapper['PPG'], show=False)
            hr_mean = hrv['HRV_MeanNN']
            hr_std = hrv['HRV_SDNN']
            hrv_rmssd = hrv['HRV_RMSSD']
            hrv_pNN50 = hrv['HRV_pNN50']
            hrv_tinn = hrv['HRV_TINN']
            hrv_data = np.array([hr_mean, hr_std, hrv_rmssd, hrv_pNN50, hrv_tinn]).squeeze(1)

            if USE_FREQ_HRV:
                hrv_data.append(hrv['HRV_ULF'])
                hrv_data.append(hrv['HRV_LF'])
                hrv_data.append(hrv['HRV_HF'])
                hrv_data.append(hrv['HRV_VHF'])
                hrv_data.append(hrv['HRV_LFHF'])
                hrv_data.append(hrv['HRV_LFn'])
                hrv_data.append(hrv['HRV_HFn'])
            
            eda_feature, _, _ = eda_features(gsr_s, SAMPLE_FS_Dapper['GSR'])
            eda_data = [eda_feature["EDA_mean"], eda_feature["EDA_std"], eda_feature["EDA_max"], eda_feature["EDA_min"], eda_feature["EDA_slope"], eda_feature["EDA_range"], 
                        eda_feature["SCL_mean"], eda_feature["SCL_std"], eda_feature["SCR_mean"], eda_feature["SCR_std"], eda_feature["SCL_time_corr"], eda_feature["num_SCR_segments"],
                        eda_feature["sum_SCR_startle_magnitudes"], eda_feature["sum_response_durations_sec"], eda_feature["area_under_identified_SCR"]]

            merged = np.array(np.concatenate([acc_data, hrv_data, eda_data]))

            time_s = time.iloc[start * STEP_TIME * SAMPLE_FS_Dapper['PPG']]
            time_label = pid_label[time_s.floor('min') == pid_label[" StartTime "].dt.floor('min')]
            
            stress_label = 0
            if len(time_label) > 0 and ((time_label["PANAS_1"] >= 3).any() or (time_label["PANAS_6"] >= 3).any() or (time_label["PANAS_2"] >= 3).any() or (time_label["PANAS_4"] >= 3).any() or (time_label["PANAS_9"] >= 3).any()):
                stress_label = 1

            merged = {"PID": pid, "Features": merged, "Time": time_s, "Label": stress_label}

            csv_data = pd.concat([csv_data, pd.DataFrame([merged])], ignore_index=True)

    csv_data.to_parquet("Dapper_unnormed_6s.parquet", index=False)

