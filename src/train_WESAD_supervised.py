import os
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt, find_peaks

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    matthews_corrcoef, roc_auc_score
)
from datetime import datetime
from tqdm import tqdm


def per_class_precision_recall_from_cm(cm):
    cm = np.asarray(cm)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    return precision, recall

class MLP(nn.Module):
    def __init__(self, input_shape, num_class):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_class), 
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    epochs = 300
    batch_size=64
    lr=0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeding_length = 4608
    feature_length = 340

    feature_df = pd.read_csv("./data/WESAD/WESAD_table_norm.csv")
    feature_df["Features"] = feature_df['Features'].str.strip('[]').str.split().apply(lambda x: [float(i) for i in x])
    embed_df = pd.read_parquet("./data/WESAD/WESAD_table_Normwear_Encoding_60s.parquet")

    SUBJECT_IDS = (
        [f"S{i}" for i in range(2, 12)] +
        [f"S{i}" for i in range(13, 18)]
    )
    
    for opt in ["Both"]:
        for mode in ["Bin_stress"]: #, "Tri_stress_amuse", "Fourclass"]:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = f'./run/{timestamp}_{opt}_{mode}'
            writer = SummaryWriter(log_dir=log_dir)

            avg_loso_f1 = 0
            avg_loso_acc = 0
            for sid in SUBJECT_IDS:
                tr_data = embed_df
                val_data = embed_df

                input_dim = embeding_length + feature_length
                num_class = 2

                if opt == "NormWear_Embed":
                    tr_data = embed_df[embed_df["PID"] != sid]
                    val_data = embed_df[embed_df["PID"] == sid]
                elif opt == "Hand_Craft":
                    input_dim = feature_length
                    tr_data = feature_df[feature_df["PID"] != sid]
                    val_data = feature_df[feature_df["PID"] == sid]
                elif opt == "Both":
                    combine_data = pd.merge(embed_df, feature_df, on=['Time', "PID"], how='inner')
                    combine_data["Features"] = [np.concatenate([a, b]) for a, b in zip(combine_data['Features_x'], combine_data['Features_y'])]
                    combine_data["Baseline"] = combine_data["Baseline_x"]
                    combine_data["Stress"] = combine_data["Stress_x"]
                    combine_data["Amusement"] = combine_data["Amusement_x"]
                    combine_data["Meditation"] = combine_data["Meditation_x"]
                    combine_data["Valence"] = combine_data["Valence_x"]
                    combine_data["Arousal"] = combine_data["Arousal_x"]
                    tr_data = combine_data[combine_data["PID"] != sid]
                    val_data = combine_data[combine_data["PID"] == sid]
                
                tr_X_tensor = torch.tensor(np.vstack(tr_data["Features"].to_numpy()), dtype=torch.float32)
                val_X_tensor = torch.tensor(np.vstack(val_data["Features"].to_numpy()), dtype=torch.float32)

                tr_y_tensor = torch.empty(0)
                val_y_tensor = torch.empty(0)
                if mode == "Bin_stress":
                    tr_y_tensor = torch.tensor(np.where(tr_data[["Baseline", "Amusement", "Meditation"]].sum(axis=1) > tr_data['Stress'].to_numpy(), 0, 1))
                    val_y_tensor = torch.tensor(np.where(val_data[["Baseline", "Amusement", "Meditation"]].sum(axis=1) > val_data['Stress'].to_numpy(), 0, 1))

                elif mode == "Tri_stress_amuse":
                    tr_sum_vec = np.column_stack((
                        tr_data['Baseline'] + tr_data['Meditation'],
                        tr_data['Stress'],
                        tr_data['Amusement']
                    ))
                    tr_y_tensor = torch.tensor(np.argmax(tr_sum_vec, axis=1))

                    val_sum_vec = np.column_stack((
                        val_data['Baseline'] + val_data['Meditation'],
                        val_data['Stress'],
                        val_data['Amusement']
                    ))
                    val_y_tensor = torch.tensor(np.argmax(val_sum_vec, axis=1))
                    num_class = 3
                
                elif mode == "Fourclass":
                    tr_y_tensor = torch.tensor((tr_data["Valence"] * 2 + tr_data["Arousal"]).to_numpy())
                    val_y_tensor = torch.tensor((val_data["Valence"] * 2 + val_data["Arousal"]).to_numpy())
                    num_class = 4

                train_dataset = TensorDataset(tr_X_tensor, tr_y_tensor)
                val_dataset = TensorDataset(val_X_tensor, val_y_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                model = MLP(input_shape=input_dim, num_class=num_class).to(device)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(model.parameters(), lr=lr)

                best_val_loss = float('inf')
                best_val_acc = 0
                best_val_f1 = 0
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0.0
                    
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        
                        logits = model(batch_X)
                        loss = criterion(logits, batch_y)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * batch_X.size(0)
                        
                    train_loss /= len(train_loader.dataset)
                    
                    # 4. Validation & Metrics Phase
                    model.eval()
                    val_loss = 0.0
                    all_preds = []
                    all_targets = []
                    
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            
                            logits = model(batch_X)
                            loss = criterion(logits, batch_y)
                            val_loss += loss.item() * batch_X.size(0)
                            
                            # Convert logits to binary predictions
                            preds = logits.argmax(dim=1)
                            
                            all_preds.append(preds.cpu().numpy())
                            all_targets.append(batch_y.cpu().numpy())
                            
                    val_loss /= len(val_loader.dataset)

                    all_preds = np.concatenate(all_preds)
                    all_targets = np.concatenate(all_targets)

                    val_f1 = f1_score(all_targets, all_preds, zero_division=0, average = "micro")
                    val_weighted_f1 = f1_score(all_targets, all_preds, zero_division=0, average = "weighted")
                    val_acc = accuracy_score(all_targets, all_preds)
                    
                    # Log metrics to TensorBoard
                    writer.add_scalar(f'{sid}/Loss/Train', train_loss, epoch)
                    writer.add_scalar(f'{sid}/Loss/Validation', val_loss, epoch)
                    writer.add_scalar(f'{sid}/F1', val_f1, epoch)
                    writer.add_scalar(f'{sid}/WeightedF1', val_weighted_f1, epoch)
                    writer.add_scalar(f'{sid}/ACC', val_acc, epoch)
                    
                    torch.save(model.state_dict(), os.path.join(log_dir, f'model_latest_{sid}.pth'))
                    
                    # Save Best Loss Model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_loss_{sid}.pth'))
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_acc_{sid}.pth'))
                    
                    if val_weighted_f1 > best_val_f1:
                        best_val_f1 = val_weighted_f1
                        torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_f1_{sid}.pth'))
                    
                avg_loso_f1 += best_val_f1
                avg_loso_acc += best_val_acc
            avg_loso_f1 /= len(SUBJECT_IDS)
            avg_loso_acc /= len(SUBJECT_IDS)
            print(f"Average f1 for LOSO: {avg_loso_f1} | ACC: {avg_loso_acc}")

            writer.close()