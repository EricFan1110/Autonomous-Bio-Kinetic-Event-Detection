import os
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk

from sklearn.preprocessing import StandardScaler
from scipy import signal

import torch
from tqdm import tqdm

from NormWear.modules.normwear import *

import warnings
warnings.filterwarnings(
    "ignore",
    module="neurokit2"
)

np.set_printoptions(precision=3, suppress=True)

# =========================== Data processing: WESAD Time Data for Normwear ===================================================
import torchaudio.transforms as T
# from NormWear.main_model import NormWearModel

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
saved_file_name = f"/diniuvol/jonah/Eric/data/WESAD/WESAD_Data_{INPUT_WINDOW}s_{LABEL_WINDOW}s"

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

csv_data = pd.DataFrame()
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
    bvp_signals, bvp_info = nk.ppg_process(bvp, sampling_rate=SAMPLE_FS['BVP'])
    acc = np.array(subject['signal']['wrist']['ACC'])
    eda = np.array(subject['signal']['wrist']['EDA']).squeeze(1)

    for start in tqdm(range(0, len(labels))):
        temp_s = temp[start * LABEL_WINDOW * SAMPLE_FS['TEMP']: start * LABEL_WINDOW * SAMPLE_FS['TEMP'] + INPUT_WINDOW * SAMPLE_FS['TEMP']]
        bvp_s = bvp[start * LABEL_WINDOW * SAMPLE_FS['BVP']: start * LABEL_WINDOW * SAMPLE_FS['BVP'] + INPUT_WINDOW * SAMPLE_FS['BVP']]
        peaks_s = np.array(bvp_signals['PPG_Peaks'][start * LABEL_WINDOW * SAMPLE_FS['BVP']: start * LABEL_WINDOW * SAMPLE_FS['BVP'] + INPUT_WINDOW * SAMPLE_FS['BVP']])
        if np.count_nonzero(peaks_s) <= 3 or len(bvp_s) != INPUT_WINDOW * SAMPLE_FS['BVP']:
            continue
        acc_s = acc[start * LABEL_WINDOW * SAMPLE_FS['ACC']: start * LABEL_WINDOW * SAMPLE_FS['ACC'] + INPUT_WINDOW * SAMPLE_FS['ACC']]
        eda_s = eda[start * LABEL_WINDOW * SAMPLE_FS['EDA']: start * LABEL_WINDOW * SAMPLE_FS['EDA'] + INPUT_WINDOW * SAMPLE_FS['EDA']]
        
        temp_s_resampled = temp_resampler(torch.tensor(temp_s, dtype=torch.float32).squeeze(1))
        bvp_s_resampled = bvp_resampler(torch.tensor(bvp_s, dtype=torch.float32).squeeze(1))
        acc_s_resampled = acc_resampler(torch.tensor(acc_s, dtype=torch.float32).T)
        eda_s_resampled = eda_resampler(torch.tensor(eda_s, dtype=torch.float32))

        x_s = torch.concatenate([temp_s_resampled.unsqueeze(0), bvp_s_resampled.unsqueeze(0), 
                                acc_s_resampled, eda_s_resampled.unsqueeze(0)]).flatten().numpy()
        if x_s.shape != (2340,):
            print(f"Normwear data shape mismatch {x_s.shape}")
            continue
        # x_s_cwt = calc_cwt(x_s.clone())
        # x_s = x_s.cpu().numpy()
        # x_s_cwt = x_s_cwt.cpu().numpy()

        # if acc_s.shape[0] < SAMPLE_FS['ACC'] * INPUT_WINDOW:
        #     acc_s = np.pad(acc_s, ((0, SAMPLE_FS['ACC'] * INPUT_WINDOW - acc_s.shape[0]), (0, 0)), mode='constant')
        # acc_s = acc_s.reshape(160, 12, 3)
        # acc_mean = np.mean(acc_s, axis = 0)
        # acc_std = np.std(acc_s, axis = 0)
        # acc_sum = np.trapezoid(np.abs(acc_s), axis = 0)
        # acc_peak_freq = dom_nonzero_freq(acc_s, SAMPLE_FS['ACC'])
        # acc_data = np.concatenate([acc_mean.flatten(), acc_std.flatten(), acc_sum.flatten(), acc_peak_freq.flatten()])

        # temp_mean = np.mean(temp_s)
        # temp_std = np.std(temp_s)
        # temp_min = np.min(temp_s)
        # temp_max = np.max(temp_s)
        # temp_range = temp_max - temp_min
        # temp_slope = np.polyfit(np.arange(len(temp_s)), temp_s, 1)[0][0]
        # temp_data = [temp_mean, temp_std, temp_min, temp_max, temp_range, temp_slope]
        
        # hrv = nk.hrv_time(peaks_s, sampling_rate=SAMPLE_FS['BVP'], show=False)
        # hr_mean = hrv['HRV_MeanNN']
        # hr_std = hrv['HRV_SDNN']
        # hrv_rmssd = hrv['HRV_RMSSD']
        # hrv_pNN50 = hrv['HRV_pNN50']
        # hrv_tinn = hrv['HRV_TINN']
        # hrv_data = [hr_mean, hr_std, hrv_rmssd, hrv_pNN50, hrv_tinn]

        # eda_feature, _, _ = eda_features(eda_s, SAMPLE_FS['EDA'])
        # eda_data = [eda_feature["EDA_mean"], eda_feature["EDA_std"], eda_feature["EDA_max"], eda_feature["EDA_min"], eda_feature["EDA_slope"], eda_feature["EDA_range"], 
        #             eda_feature["SCL_mean"], eda_feature["SCL_std"], eda_feature["SCR_mean"], eda_feature["SCR_std"], eda_feature["SCL_time_corr"], eda_feature["num_SCR_segments"],
        #             eda_feature["sum_SCR_startle_magnitudes"], eda_feature["sum_response_durations_sec"], eda_feature["area_under_identified_SCR"]]
        # handcrafted_features = np.array(np.concatenate([acc_data, temp_data, np.array(hrv_data).squeeze(1), eda_data]))

        # with torch.inference_mode():
        #     out = model.get_embedding(x_s, sampling_rate=65, device=device)
        #     out = out.mean(dim=2).reshape(out.shape[0], -1).squeeze(0)
        # out = np.array(out.cpu())
        
        # merged = {"PID": sid, "Normwear_Input": x_s.tolist(), "Normwewar_Embed": out.tolist(), "Handcraft": handcrafted_features, "Time": start * LABEL_WINDOW, "Label": labels[start], "Index": row_idx}
        merged = {"PID": sid, "Normwear_Input": x_s.tolist(), "Time": start * LABEL_WINDOW, "Label": labels[start], "Index": row_idx}
        row_idx += 1
        csv_data = pd.concat([csv_data, pd.DataFrame([merged])], ignore_index=True)

    csv_data.to_parquet(saved_file_name + ".parquet", index=False)

extracted_data = pd.read_parquet(saved_file_name + ".parquet")
all_features = np.stack(extracted_data["Handcraft"].values)

scaler = StandardScaler()
z_scores = scaler.fit_transform(all_features)

s = []
for id in extracted_data["PID"].unique():
    subject_data = extracted_data[extracted_data['PID'] == id]['Handcraft']
    subject_data = np.stack(subject_data.values)
    subject_norm = scaler.fit_transform(subject_data)
    s.append(subject_norm)

print(z_scores.shape)
subject_combined = np.concatenate(s, axis = 0)
norm_data = np.concatenate([z_scores, subject_combined], axis = 1)
print(norm_data.shape)

extracted_data["Handcraft"] = list(norm_data)
extracted_data.to_parquet(saved_file_name + "_norm.parquet", index=False)
