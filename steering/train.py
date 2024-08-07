import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import torch_geometric

import sys
from tqdm import tqdm
import json
from models import *
import os
from utils import *

# Constants
PROPAGATION_DISTANCE = 2.0  # meters
MAX_PROP_ANGLE = np.pi / 4.0

DETECTION_PROB = 0.9
MAX_FALSE_POSITIVES = 20
MIN_PERCEPTION_RANGE = 15.0
MAX_PERCEPTION_RANGE = 15.0

model = PPGNN()

# print number of parameters
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

model_name = f"gnn.pth"

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# criterion = SignSensitiveMSELoss(sign_penalty=0)
criterion = nn.MSELoss()

print("Loading training data...")
X, Y = prepare_data(num_tracks=200000, 
                    prop_dist=PROPAGATION_DISTANCE, 
                    detection_prob=DETECTION_PROB, 
                    max_false_positives=MAX_FALSE_POSITIVES,
                    max_prop_angle=MAX_PROP_ANGLE,
                    min_perception_range=MIN_PERCEPTION_RANGE,
                    max_perception_range=MAX_PERCEPTION_RANGE)

# Convert labels to a tensor
Y_tensor = torch.tensor(Y, dtype=torch.float32)
dataset = list(zip(X, Y_tensor))
train_loader = torch_geometric.loader.DataLoader(dataset, batch_size=16, shuffle=True)

print("Loading validation data...")
X_val, Y_val = prepare_data(num_tracks=100000,
                            prop_dist=PROPAGATION_DISTANCE, 
                            detection_prob=DETECTION_PROB, 
                            max_false_positives=MAX_FALSE_POSITIVES,
                            max_prop_angle=MAX_PROP_ANGLE,
                            min_perception_range=MIN_PERCEPTION_RANGE,
                            max_perception_range=MAX_PERCEPTION_RANGE)


# Convert labels to a tensor
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
dataset_val = list(zip(X_val, Y_val_tensor))
val_loader = torch_geometric.loader.DataLoader(dataset_val, batch_size=32, shuffle=False)


print("Starting training...")
# Training loop
for i in range(1000):
    try:
        for x_batch, y_batch in tqdm(train_loader):
            optimizer.zero_grad()
            x = x_batch.x
            batch = x_batch.batch
            outputs = model(x, batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            avg_loss = 0
            for x_batch, y_batch in tqdm(val_loader):
                x = x_batch.x
                batch = x_batch.batch
                outputs = model(x, batch).squeeze()
                loss = criterion(outputs, y_batch)
                avg_loss += loss.item()
            avg_loss /= len(val_loader)

        print(f"Epoch {i} - Validation loss: {avg_loss}")


        # Save model
        torch.save(model.state_dict(), model_name)

    except KeyboardInterrupt:
        print("Training stopped, saving model...")
        torch.save(model.state_dict(), model_name)
        break
    