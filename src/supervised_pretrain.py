import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

validation_subjects = [1005, 1014, 1019, 1021, 1028, 1030, 2001, 2006, 2010, 2017, 2021, 2025, 3003, 3008, 3011, 3022, 3023, 3117]

class EmotionDataset(Dataset):
    def __init__(self, df):
        df_filtered = df.groupby('Label Index', group_keys=False).apply(lambda x: x.iloc[::6]).reset_index(drop=True)
        
        self.X = []
        self.y = []
        
        for label_idx, group_df in df_filtered.groupby('Label Index'):
            group_df = group_df.reset_index(drop=True)

            # for i in range(0, len(group_df) - 59, 12):
            #     block = group_df.iloc[i : i + 60]
            for i in range(30, len(group_df) - 9, 1):
                block = group_df.iloc[i : i + 10]
                
                features = np.concatenate(block['Features'].values)

                valence = block['High Valence'].iloc[-1]
                arousal = block['High Arousal'].iloc[-1]
                # valence = block['Valence'].iloc[-1]
                # valence = 1 if valence > 3 else 0
                # arousal = block['Arousal'].iloc[-1]
                #arousal = 1 if arousal > 3 else 0
                
                self.X.append(features)
                self.y.append([valence, arousal])
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        # self.net = nn.Sequential(
        #     nn.Linear(input_shape, 2048),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(2048, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 2),
        # )

        self.net = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )


    def forward(self, x):
        return self.net(x)
    
def create_cm_figure(y_true, y_pred, title):
    """Generates a matplotlib figure for the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    epochs = 200
    batch_size=64
    lr=0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet("./data/Dapper/dapper_dqn_10s_step_62min_norm_w_subject_mean_arousal_valence.parquet")
    
    all_pids = list(df["PID"].unique())
    validation_subjects = random.sample(all_pids, 9)

    train_df = df[~df["PID"].isin(validation_subjects)]
    val_df = df[df["PID"].isin(validation_subjects)]

    train_dataset = EmotionDataset(train_df)
    val_dataset = EmotionDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup Model
    # input_dim = 19680
    input_dim = 3280
    model = MLP(input_shape=input_dim).to(device)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'./run/{timestamp}_supervised'
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float('inf')

    # Training Loop
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
                preds = (torch.sigmoid(logits) > 0.5).float()
                # preds = logits.argmax(dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        # Aggregate epoch predictions to calculate F1 correctly
        # all_preds = np.concatenate(all_preds)
        # all_targets = np.concatenate(all_targets)
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Target 0 is Valence, Target 1 is Arousal
        # val_f1_valence = f1_score(all_targets, all_preds, zero_division=0)
        # val_f1_arousal = f1_score(all_targets, all_preds, zero_division=0)
        val_f1_valence = f1_score(all_targets[:, 0], all_preds[:, 0], zero_division=0)
        val_f1_arousal = f1_score(all_targets[:, 1], all_preds[:, 1], zero_division=0)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('F1/Val_Valence', val_f1_valence, epoch)
        writer.add_scalar('F1/Val_Arousal', val_f1_arousal, epoch)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1 (Valence): {val_f1_valence:.4f} | Val F1 (Arousal): {val_f1_arousal:.4f}")
        
        torch.save(model.state_dict(), os.path.join(log_dir, 'model_latest.pth'))
        
        # Save Best Loss Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'model_best_loss.pth'))

        # Log Confusion Matrices as figures (e.g., every 5 epochs and on the last epoch)
        # if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
        #     fig_valence = create_cm_figure(all_targets, all_preds, f"Valence CM - Epoch {epoch+1}")
        #     fig_arousal = create_cm_figure(all_targets, all_preds, f"Arousal CM - Epoch {epoch+1}")
        #     # fig_valence = create_cm_figure(all_targets[:, 0], all_preds[:, 0], f"Valence CM - Epoch {epoch+1}")
        #     # fig_arousal = create_cm_figure(all_targets[:, 1], all_preds[:, 1], f"Arousal CM - Epoch {epoch+1}")
            
        #     writer.add_figure('Confusion Matrix/Valence', fig_valence, epoch)
        #     writer.add_figure('Confusion Matrix/Arousal', fig_arousal, epoch)
            
    writer.close()