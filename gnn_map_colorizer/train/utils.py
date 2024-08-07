import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import torch_geometric
from models import PPGNN, SignSensitiveMSELoss
import os
import sys
from tqdm import tqdm

sys.path.append('/ws/build')
import fsgenerator as fsg

def cones_to_graph(cones):
    cones = np.array(cones)
    cones = torch.tensor(cones, dtype=torch.float32)

    origin = torch.tensor([0, 0], dtype=torch.float32)

    cones = torch.cat([origin.unsqueeze(0), cones], dim=0)

    # connect the cones to the origin
    edge_index = torch.tensor([[0, i] for i in range(1, len(cones))]).T

    # Create the graph
    graph = Data(x=cones, edge_index=edge_index)
    return graph

def prepare_data(num_tracks, prop_dist, detection_prob, max_false_positives, max_prop_angle, min_perception_range, max_perception_range):
    X = []
    Y = []
    
    for _ in tqdm(range(num_tracks)):
        cones, angle = fsg.get_track_cones(propagation_dist=prop_dist,                                        
                                           detection_prob=detection_prob, 
                                           max_false_positives=max_false_positives, 
                                           max_prop_angle=max_prop_angle)
                                           
        #percep_range = np.random.uniform(min_perception_range, max_perception_range)
        cones = [cone for cone in cones if np.linalg.norm(cone) < max_perception_range]
        if len(cones) < 3:
            continue

        cones = np.array(cones)
        graph = cones_to_graph(cones)
        
    




        X.append(graph)
        Y.append(angle / max_prop_angle)
    return X, Y