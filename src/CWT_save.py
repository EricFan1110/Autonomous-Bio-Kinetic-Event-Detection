import os
import pickle
import numpy as np
import pandas as pd
# import neurokit2 as nk

from scipy.stats import pearsonr, spearmanr
from scipy.integrate import trapezoid
from scipy import signal

import pyarrow as pa
import pyarrow.parquet as pq

import torch
from tqdm import tqdm


from NormWear.modules.normwear import *

import warnings
warnings.filterwarnings(
    "ignore",
    module="neurokit2"
)

np.set_printoptions(precision=3, suppress=True)

# def dom_nonzero_freq(signal, fs_hz):
#     """
#     Returns most dominant non-zero frequency via fft
#     """
#     T, K, C = signal.shape
#     y = np.fft.rfft(signal, axis = 0)
#     yf = np.fft.rfftfreq(T, 1/fs_hz)
#     mag = np.abs(y)
#     mag[0, :, :] = 0
#     idx = np.argmax(mag, axis = 0)
#     return yf[idx]

# def eda_features(eda, fs, corr_method="pearson"):
#     """
#     Compute SCR/SCL features from an EDA signal.

#     Parameters
#     ----------
#     eda : array-like
#         Raw EDA signal (skin conductance).
#     fs : float
#         Sampling frequency in Hz.
#     corr_method : str
#         "pearson" or "spearman" for SCL-time correlation.

#     Returns
#     -------
#     features : dict
#         Dictionary of computed features.
#     signals : pd.DataFrame
#         Processed signal dataframe from neurokit2.
#     info : dict
#         Event metadata returned by neurokit2.
#     """
#     eda = np.asarray(eda, dtype=float)

#     # 1) Clean + decompose + detect SCR peaks
#     # signals columns typically include:
#     # EDA_Clean, EDA_Tonic (SCL), EDA_Phasic (SCR), SCR_Onsets, SCR_Peaks, SCR_Recovery, ...
#     signals, info = nk.eda_process(eda, sampling_rate=fs)

#     scl = signals["EDA_Tonic"].to_numpy()   # tonic = SCL
#     scr = signals["EDA_Phasic"].to_numpy()  # phasic = SCR

#     # Time vector
#     t = np.arange(len(eda)) / fs

#     # 2) Correlation between SCL and time
#     if corr_method.lower() == "pearson":
#         scl_time_corr, scl_time_corr_p = pearsonr(t, scl)
#     elif corr_method.lower() == "spearman":
#         scl_time_corr, scl_time_corr_p = spearmanr(t, scl)
#     else:
#         raise ValueError("corr_method must be 'pearson' or 'spearman'")

#     # 3) Get SCR event indices
#     # NeuroKit2 stores indices in info, but keys may vary slightly depending on version.
#     onsets = np.array(info.get("SCR_Onsets", []), dtype=float)
#     peaks = np.array(info.get("SCR_Peaks", []), dtype=float)
#     recovery = np.array(info.get("SCR_Recovery", []), dtype=float)

#     # Remove NaNs / invalid values and cast to int
#     onsets = onsets[np.isfinite(onsets)].astype(int)
#     peaks = peaks[np.isfinite(peaks)].astype(int)
#     recovery = recovery[np.isfinite(recovery)].astype(int)

#     # Align events robustly (onset -> peak -> recovery)
#     # We'll create matched segments where onset < peak < recovery if possible.
#     segments = []
#     used_recovery = set()

#     for peak in peaks:
#         onset_candidates = onsets[onsets < peak]
#         if len(onset_candidates) == 0:
#             continue
#         onset = onset_candidates[-1]  # nearest onset before peak

#         rec_candidates = recovery[(recovery > peak)]
#         rec_candidates = [r for r in rec_candidates if r not in used_recovery]
#         if len(rec_candidates) == 0:
#             # If no recovery marker, estimate end at next time SCR returns near local baseline
#             # Simple fallback: use peak + 4s (capped to signal length)
#             rec = min(len(scr) - 1, int(peak + 4 * fs))
#         else:
#             rec = rec_candidates[0]
#             used_recovery.add(rec)

#         if onset < peak < rec:
#             segments.append((onset, peak, rec))

#     # 4) Number of SCR segments
#     n_scr_segments = len(segments)

#     # 5) SCR startle magnitudes and response durations
#     # Startle magnitude here = peak amplitude relative to onset baseline (phasic component)
#     # Duration = recovery - onset (seconds)
#     startle_magnitudes = []
#     response_durations = []
#     scr_auc_list = []

#     for onset, peak, rec in segments:
#         magnitude = scr[peak] - scr[onset]
#         duration = (rec - onset) / fs

#         # Area under identified SCR segment (above onset baseline)
#         segment_y = scr[onset:rec + 1] - scr[onset]
#         # Optional: clip negative values so only positive response contributes
#         segment_y = np.clip(segment_y, 0, None)
#         segment_t = t[onset:rec + 1]
#         auc = trapezoid(segment_y, segment_t)

#         startle_magnitudes.append(magnitude)
#         response_durations.append(duration)
#         scr_auc_list.append(auc)

#     startle_magnitudes = np.array(startle_magnitudes, dtype=float)
#     response_durations = np.array(response_durations, dtype=float)
#     scr_auc_list = np.array(scr_auc_list, dtype=float)

#     sum_scr_startle_magnitudes = float(np.nansum(startle_magnitudes)) if len(startle_magnitudes) else 0.0
#     sum_response_durations = float(np.nansum(response_durations)) if len(response_durations) else 0.0
#     area_under_identified_scr = float(np.nansum(scr_auc_list)) if len(scr_auc_list) else 0.0

#     features = {
#         "EDA_mean": np.mean(eda),
#         "EDA_std": np.std(eda),
#         "EDA_max": max(eda),
#         "EDA_min": min(eda),
#         "EDA_slope": np.polyfit(np.arange(len(eda)), eda, 1)[0],
#         "EDA_range": max(eda) - min(eda),

#         # Whole-signal tonic/phasic summaries
#         "SCL_mean": float(np.nanmean(scl)),
#         "SCL_std": float(np.nanstd(scl)),
#         "SCR_mean": float(np.nanmean(scr)),
#         "SCR_std": float(np.nanstd(scr)),

#         # Correlation
#         "SCL_time_corr": float(scl_time_corr),
#         "SCL_time_corr_pvalue": float(scl_time_corr_p),

#         # Event-based SCR features
#         "num_SCR_segments": int(n_scr_segments),
#         "sum_SCR_startle_magnitudes": sum_scr_startle_magnitudes,
#         "sum_response_durations_sec": sum_response_durations,
#         "area_under_identified_SCR": area_under_identified_scr,

#         # Optional per-event stats
#         "mean_SCR_startle_magnitude": float(np.nanmean(startle_magnitudes)) if len(startle_magnitudes) else 0.0,
#         "mean_response_duration_sec": float(np.nanmean(response_durations)) if len(response_durations) else 0.0,
#         "mean_SCR_auc": float(np.nanmean(scr_auc_list)) if len(scr_auc_list) else 0.0,
#     }

#     return features, signals, info
# =========================== Data processing: WESAD Time Data for Normwear ===================================================
import torchaudio.transforms as T
from NormWear.main_model import NormWearModel

LABEL_WINDOW = 6
INPUT_WINDOW = 6
SAMPLE_FS = {'ACC': 32, 'BVP': 64, 'TEMP': 4, 'LABEL': 700, 'EDA': 4}
WESAD_path = "/diniuvol/jonah/Eric/data/WESAD"
USE_FREQ_HRV = False
SUBJECT_IDS = (
    [f"S{i}" for i in range(2, 12)] +
    [f"S{i}" for i in range(13, 18)]
)
optimized_cwt = True
saved_file_name = "/diniuvol/jonah/Eric/data/WESAD/WESAD_CWT_NormWear_Data_6s_6s.parquet"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def load_subject(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")
    
def load_survey(path):
    with open(path,"r") as f:
        return f.readlines()

def wt(ts, lf=0.1, hf=65):
    # in: L
    # out: FxL
    cwtmatr = signal.cwt(ts, signal.ricker, np.arange(lf, hf))
    return cwtmatr #[F, L]

def spec_cwt(audio_data): # [nvar, L]
    x1 = audio_data[:, 1:] - audio_data[:, :-1]
    x2 = x1[:, 1:] - x1[:, :-1]

    all_specs = list()
    for c_i in range(audio_data.shape[0]):
        all_specs.append(torch.stack([
            torch.from_numpy(wt(audio_data[c_i, 2:])).permute(1, 0), # [L, n_mels]
            torch.from_numpy(wt(x1[c_i, 1:])).permute(1, 0), 
            torch.from_numpy(wt(x2[c_i])).permute(1, 0)
        ])) # [3, L, n_mels]

    all_specs = torch.stack(all_specs) # [nvar, 3, L, n_mels]

    return all_specs

def calc_cwt(x):
    # x: [bn, nvar, L]
    # return: # bn, nvar, 3, L, n_scales
    bn, nvar, L = x.shape

    if optimized_cwt: # use the version implemented with pytorch
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(device)
        # calculate cwt
        cwt_res = cwt_wrap(x.view(bn*nvar, L), 0.1, 64) # bn*nvar, 3, L, n_scales
        _, n_, new_L, n_scale = cwt_res.shape
        cwt_res = cwt_res.view(bn, nvar, n_, new_L, n_scale) # bn, nvar, 3, L, n_scales
    else: # vanilla CWT
        cwt_res = torch.stack([spec_cwt(sample) for sample in x]) # bn, nvar, 3, L, n_scales
    return cwt_res

temp_resampler = T.Resample(SAMPLE_FS['TEMP'], 65)
bvp_resampler = T.Resample(SAMPLE_FS['BVP'], 65)
acc_resampler = T.Resample(SAMPLE_FS['ACC'], 65)
eda_resampler = T.Resample(SAMPLE_FS['EDA'], 65)
# model = NormWearModel(weight_path="./NormWear/normwear_pretrain_ckpt.pth", optimized_cwt=True).to(device)
# model.eval()

schema = pa.schema(
    [
        ("PID", pa.string()),
        ("Normwear_CWT", pa.list_(pa.float64())),
        ("Time", pa.int64()),
        ("Label", pa.int64()),
        ("Index", pa.int64())
    ]
)
writer = pq.ParquetWriter(saved_file_name, schema)

row_idx = 0
for sid in SUBJECT_IDS:
    print(sid)
    subject = load_subject(f"{WESAD_path}/{sid}/{sid}.pkl")

    labels = np.array(subject['label'])
    labels = np.where((labels >= 1) & (labels <= 4), labels - 1, 0)
    labels = labels[:len(labels) - len(labels) % (SAMPLE_FS['LABEL'] * LABEL_WINDOW)]
    labels = labels.reshape(len(labels)//(SAMPLE_FS['LABEL'] * LABEL_WINDOW), SAMPLE_FS['LABEL'] * LABEL_WINDOW)
    labels=labels.max(axis=1)

    temp = np.array(subject['signal']['wrist']['TEMP'])
    bvp = np.array(subject['signal']['wrist']['BVP'])
    acc = np.array(subject['signal']['wrist']['ACC'])
    eda = np.array(subject['signal']['wrist']['EDA']).squeeze(1)

    buffer = np.zeros((10, 455130))
    
    chunk = []
    for start in tqdm(range(0, len(labels))):
        temp_s = temp[start * LABEL_WINDOW * SAMPLE_FS['TEMP']: start * LABEL_WINDOW * SAMPLE_FS['TEMP'] + INPUT_WINDOW * SAMPLE_FS['TEMP']]
        bvp_s = bvp[start * LABEL_WINDOW * SAMPLE_FS['BVP']: start * LABEL_WINDOW * SAMPLE_FS['BVP'] + INPUT_WINDOW * SAMPLE_FS['BVP']]
        acc_s = acc[start * LABEL_WINDOW * SAMPLE_FS['ACC']: start * LABEL_WINDOW * SAMPLE_FS['ACC'] + INPUT_WINDOW * SAMPLE_FS['ACC']]
        eda_s = eda[start * LABEL_WINDOW * SAMPLE_FS['EDA']: start * LABEL_WINDOW * SAMPLE_FS['EDA'] + INPUT_WINDOW * SAMPLE_FS['EDA']]
        
        temp_s_resampled = temp_resampler(torch.tensor(temp_s, dtype=torch.float32).squeeze(1))
        bvp_s_resampled = bvp_resampler(torch.tensor(bvp_s, dtype=torch.float32).squeeze(1))
        acc_s_resampled = acc_resampler(torch.tensor(acc_s, dtype=torch.float32).T)
        eda_s_resampled = eda_resampler(torch.tensor(eda_s, dtype=torch.float32))

        x_s = torch.concatenate([temp_s_resampled.unsqueeze(0), bvp_s_resampled.unsqueeze(0), 
                                acc_s_resampled, eda_s_resampled.unsqueeze(0)]).unsqueeze(0)
        if x_s.shape != (1, 6, 390):
            print(f"Normwear data shape mismatch {x_s.shape}")
            continue
        x_s_cwt = calc_cwt(x_s.clone())
        x_s = x_s.cpu().numpy()
        x_s_cwt = x_s_cwt.cpu().numpy()

        x_s_cwt_flat = x_s_cwt.flatten() # To revert it, use x_s_cwt_flat.reshape(1,6,3,389,65)

        # if start < 9:
        #     buffer[start] = x_s_cwt_flat
        #     continue
        
        # buffer[1:] = buffer[:-1]
        # buffer[0] = x_s_cwt_flat
        # x_s_cwt_flat = buffer.mean(axis=0)

        if acc_s.shape[0] < SAMPLE_FS['ACC'] * INPUT_WINDOW:
            acc_s = np.pad(acc_s, ((0, SAMPLE_FS['ACC'] * INPUT_WINDOW - acc_s.shape[0]), (0, 0)), mode='constant')

        merged = {"PID": sid, "Normwear_CWT": x_s_cwt_flat.tolist(), "Time": start * LABEL_WINDOW, "Label": labels[start], "Index": row_idx}
        chunk.append(merged)
        row_idx += 1

        if len(chunk) >= 200 or start == len(labels) - 1:
            add_data = pd.DataFrame(chunk)
            chunk = []

            table = pa.Table.from_pandas(add_data, schema=schema)
            writer.write_table(table)

writer.close()
