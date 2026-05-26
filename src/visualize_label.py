import os
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch

SUBJECT_IDS = (
    [f"S{i}" for i in range(2, 12)] +
    [f"S{i}" for i in range(13, 18)]
)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 40
step_size = 5
df = pd.read_parquet("./data/WESAD/WESAD_Data_60s_6s_norm.parquet")
saved_model_dir = "/home/general/Documents/Eric/Autonomous-Bio-Kinetic-Event-Detection/run/20260519-153218_Handcraft_Bin_stress"
for sid in SUBJECT_IDS:
    print(sid)
    feature_length = ((340 - window_size) // step_size + 1) * window_size
    model = MLP(input_shape=feature_length, num_class=2)
    model.load_state_dict(torch.load(f"{saved_model_dir}/model_best_f1_{sid}.pth"))
    model.to(device=device)
    model.eval()

    subject_data = df[df["PID"] == sid]
    subject_data = subject_data.sort_values("Time").reset_index(drop=True)
    
    subject_data_tensor = torch.tensor(np.vstack(subject_data["Handcraft"].to_numpy()), dtype=torch.float32)
    subject_data_tensor = subject_data_tensor.unfold(1, window_size, step_size).flatten(1)

    labels = []
    predictions = []
    
    for index, row in subject_data.iterrows():
        feature = subject_data_tensor[index]
        model_input = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(model_input)
            probabilities = torch.softmax(output, dim=-1)
            pred_label = probabilities[0, 1].item()

            labels.append(row['Label'])
            predictions.append(pred_label)


    x = np.arange(len(labels))
    plt.plot(x, labels, label="True Labels")
    plt.plot(x, predictions, label="Predictions")
    plt.legend()
    plt.savefig(f"{sid}.png")
    plt.clf()
