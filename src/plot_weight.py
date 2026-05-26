import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the saved state dictionary
# Use map_location='cpu' in case the model was saved on a GPU and you are plotting on a CPU
state_dict = torch.load('./run/20260409-161754_Both_Tri_stress_amuse/model_best_acc.pth', map_location='cpu')

# 2. Extract the first layer's weight matrix
first_layer_name = None
first_layer_weights = None

for name, param in state_dict.items():
    if 'weight' in name:
        first_layer_name = name
        first_layer_weights = param.numpy()
        break # Stop after finding the first weight matrix

if first_layer_weights is not None:
    # Ensure it's a 2D matrix (standard for MLP Linear layers)
    if len(first_layer_weights.shape) == 2:
        plt.figure(figsize=(10, 8))
        
        # Find the absolute maximum value to center the colormap exactly at 0
        max_val = np.max(np.abs(first_layer_weights))
        
        # 3. Plot the matrix as an image
        # cmap='coolwarm': Blue is negative, Red is positive, White is zero
        # aspect='auto': Allows the image to stretch to fit the figure size
        plt.imshow(first_layer_weights, cmap='coolwarm', aspect='auto', 
                   vmin=-max_val, vmax=max_val)
        
        # Add a colorbar to the side to show the scale
        plt.colorbar(label='Weight Value')
        
        plt.title(f'Weight Matrix Heatmap: {first_layer_name}')
        plt.xlabel('Input Neurons')
        plt.ylabel('Output Neurons')
        
        # 4. Save the image to a file
        plt.savefig('first_layer_matrix.png', dpi=300, bbox_inches='tight')
        print(f"Saved {first_layer_name} weight matrix to 'first_layer_matrix.png'")
        
        plt.close()
    else:
        print(f"The first weight tensor ({first_layer_name}) is not 2D. Shape: {first_layer_weights.shape}")
else:
    print("No weights found in the state dictionary.")