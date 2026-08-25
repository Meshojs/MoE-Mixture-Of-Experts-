"""
    (TOP K)
    By Mashary.
    Creating Mixture of Experts
        --  The goal of this code (like numpy's one) is training a MoE model
            that understands the pattern of (n*2) 2n
            
                        [Expert 1]
    input -> Router ->  [Expert 2] -> output
                        [Expert N]
                   
                        
"""

import torch as t
from torch import nn
from torch import TensorType
#import torch.functional as F

class config :
    pass


class Experts(nn.Module):
    def __init__(self,infeatures,hidden_size,outfeatures):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(infeatures,hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size , hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size , outfeatures)
            )
        
    def forward(self,x)-> TensorType:
        return self.network(x)
    
    
class Router(nn.Module):
    def __init__(self,infeatures,n_experts):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(infeatures, n_experts)
            )
        
    def forward(self , x)-> TensorType: 
        softmax = nn.Softmax(dim=-1)
        return softmax(self.network(x))
        
        

class MoE(nn.Module):
    def __init__(self, infeatures, hidden_size , outfeatures , n_experts ,topk):
        super().__init__()
        self.topk = topk
        self.outfeatures = outfeatures
        self.n_experts = n_experts
        self.router = Router(infeatures,n_experts)
        self.experts = nn.ModuleList([Experts(infeatures,hidden_size,outfeatures)
                        for _ in range(n_experts)])
        
        # new 

        
    def forward(self, x):
        rou_output = self.router(x)
        values, indices = t.topk(rou_output, k=self.topk, dim=-1)
        values = nn.Softmax(dim=-1)(values)
        
        output = t.zeros(x.shape[0], self.outfeatures, device=x.device)
        self.last_indices = indices  
        for k in range(self.topk):
            expert_idx = indices[:, k]
            expert_weight = values[:, k].unsqueeze(-1)
            
            for i in range(self.n_experts):
                mask = (expert_idx == i)
                if mask.any():
                    output[mask] += self.experts[i](x[mask]) * expert_weight[mask]
                    
        return output
            
# Data :>

x = t.arange(0, 100).float().reshape(-1, 1)  
y = x * 2  


model = MoE(1, 8, 1, 4 , 2)


def train(lr, epochs, x, y):
    criterion = nn.MSELoss()
    optim = t.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        optim.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optim.step()
        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss.item():.6f}")

train(1e-2, 5000, x, y)

""" 
YAY 🥀
Predictions: [20.0, 40.0, 60.0, 80.0, 100.0]
Expected: [20.0, 40.0, 60.0, 80.0, 100.0]
Used : Experts([[3, 0],
        [3, 1],
        [3, 1],
        [3, 1],
        [3, 1]])
"""
    

xtest = t.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]])
ytest = t.tensor([[20.0], [40.0], [60.0], [80.0], [100.0]])

model.eval()
with t.no_grad():
    pred = model(xtest)
    print("Predictions:", t.round(pred.flatten()).tolist())
    print("Expected:", ytest.flatten().tolist())

print(model.last_indices)
    
    
    
    
    
    
    
    
    
    

