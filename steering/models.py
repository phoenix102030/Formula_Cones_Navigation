import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, BatchNorm, global_max_pool
from torch_geometric.data import Data
import torch_geometric

class SignSensitiveMSELoss(torch.nn.Module):
    def __init__(self, sign_penalty):
        super(SignSensitiveMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.sign_penalty = sign_penalty

    def forward(self, y_pred, y_true):
        # squared errors
        sqe = (y_pred - y_true) ** 2

        mse_loss = sqe.mean()

        # Penalty for sign mismatch
        sign_mismatch = (y_pred * y_true < 0).type(torch.float32)  # 1 for sign mismatch, 0 otherwise
        sign_loss = sign_mismatch * self.sign_penalty

        # Total loss
        total_loss = mse_loss + sign_loss.mean()  
        return total_loss

class PPGNN(torch.nn.Module):
    def __init__(self):
        super(PPGNN , self).__init__()
        self.fc1 = torch.nn.Linear(2, 16)  # 2 features per node
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 64)

        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5 = torch.nn.Linear(32, 16)
        self.fc6 = torch.nn.Linear(16, 1)

    def forward(self, x, batch):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = global_mean_pool(x, batch)
        
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.tanh(self.fc6(x))  # Output between -1 and 1
        return x.squeeze()