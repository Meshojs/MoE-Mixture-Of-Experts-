"""
    By Mashary.
    Creating Mixture of Experts
        --  The goal of this code is training a MoE model
            that understands the pattern of (n*2) 2n
            
                        [Expert 1]
    input -> Router ->  [Expert 2] -> output
                        [Expert N]
                   
                        
"""

import numpy as np
import torch as t 
from numpy import ndarray


def ReLU(z):
    return np.maximum(0,z)

def sigmoid(z): 
    return 1 / ( 1 + np.exp(-z))


def softmax(r_logits):
    soft_logits = []

    for i in range(len(r_logits)):
        z = r_logits[i]

        z = z - np.max(z)

        exp_z = np.exp(z)
        result = exp_z / np.sum(exp_z)

        soft_logits.append(result)

    return np.array(soft_logits)


class Expert:
    def __init__(self, infeatures, outfeatures):
        self.w = np.random.randn(infeatures, outfeatures) * 0.01
        self.b = np.zeros(outfeatures)

    def forward(self, x):
        z = x @ self.w + self.b
        return ReLU(z), z
    
    
    
class Router:
    def __init__(self, infeatures,n_expert):
        self.wr = np.random.randn(infeatures, n_expert)* 0.01
        self.br = np.zeros(n_expert)
        
    def forward(self, x): 
        z = x @ self.wr + self.br
        return softmax(z), z
    
    

class MoE:
    def __init__(self,infeature,outfeature,n_experts):
        self.infeature = infeature 
        self.outfeature = outfeature
        self.n_experts = n_experts
        
        self.router = Router(
            infeature,
            n_experts
            )
        
        self.experts = [Expert(infeature , 
                               outfeature)
                        for _ in range(n_experts)]
        
    def forward(self, x)->list[ndarray]:

        routing_probs, router_logits = self.router.forward(x)
    
        routing_probs = np.array(routing_probs)
        routing_probs = t.tensor(routing_probs)
    
        values, expert_ids = t.topk(routing_probs, k=2, dim=1)
        values = values / values.sum(dim=1, keepdim=True)

        final = []
        all_expert_z = []
        all_expert_ids = []
        all_weights = []
    
        for inpt, ids, weights in zip(x, expert_ids, values):
    
            temp = []
            temp_z = []
    
            ids = ids.numpy()
            weights = weights.numpy()
    
            for expert_id in ids:
    
                output, z = self.experts[int(expert_id)].forward(inpt)
    
                temp.append(output)
                temp_z.append(z)
    
            final.append(temp)
            all_expert_z.append(temp_z)
            all_expert_ids.append(ids)
            all_weights.append(weights)
    
        outputs = []
    
        for expert_outputs, weights in zip(final, all_weights):
    
            expert_outputs = np.array(expert_outputs)
    
            result = expert_outputs * weights
    
            outputs.append(np.sum(result, axis=0))
    
        return (
            np.array(outputs),
            np.array(all_weights),
            np.array(all_expert_ids),
            np.array(all_expert_z)
            ,np.array(final),
            np.array(routing_probs),
            np.array(router_logits)
        )


def MSE(y_hat , y):
    return np.mean((y_hat - y)  ** 2)

def MSE_derivative(y_hat , y , n):
    return 2/n * (y_hat - y )


def get_routing_grad(logits_np, ids, d_output, expert_outputs):

    logits = t.tensor(logits_np, requires_grad=True, dtype=t.float64)
    p = t.softmax(logits, dim=0)

    values, _ = t.topk(p, k=2)
    weights = values / values.sum()  


    d_output_t = t.tensor(d_output, dtype=t.float64)
    e0 = t.tensor(expert_outputs[0], dtype=t.float64)
    e1 = t.tensor(expert_outputs[1], dtype=t.float64)

    out = weights[0]*e0 + weights[1]*e1
    surrogate_loss = t.sum(d_output_t * out) 

    surrogate_loss.backward()

    return logits.grad.numpy() 

class Pipeline:
    def __init__(self, infeature,outfeature , n_experts):
        self.infeature = infeature
        self.outfeature = outfeature
        self.n_experts = n_experts
        
        self.x = np.arange(100).reshape(50,self.infeature) / 100
        self.y = self.x * 2
        self.moe = MoE(infeature, outfeature, n_experts)

    def loop(self):
        lr=0.1
        moe = self.moe
        y_hat, weights, expert_ids, z, expert_outputs, routing_probs, router_logits = moe.forward(self.x)
        loss = MSE(y_hat , self.y)
        dp = MSE_derivative(y_hat , self.y ,  y_hat.size) #dy

        dW_experts = [np.zeros_like(expert.w) for expert in moe.experts]
        db_experts = [np.zeros_like(expert.b) for expert in moe.experts]
        
        dW_router = np.zeros_like(moe.router.wr)
        db_router = np.zeros_like(moe.router.br)
        
        for inpt, d_output, weights_i, ids, z_i, expert_outputs_i, p, router_logits in zip(
            self.x,
            dp,
            weights,
            expert_ids,
            z,
            expert_outputs,
            routing_probs,
            router_logits
        ):
        
            # EXPERT BACKWARD
            dE0 = d_output * weights_i[0]
            dE1 = d_output * weights_i[1]
        
            #dLdz0 = dE0 * (z_i[0] > 0)
            #dLdz1 = dE1 * (z_i[1] > 0)
                    
            dLdz0 = dE0
            dLdz1 = dE1
        
            dLdw0 = np.outer(inpt, dLdz0)
            dLdw1 = np.outer(inpt, dLdz1)
        
            # ROUTER BACKWARD
            dLdp = np.zeros(self.n_experts)
        
            dLdp[ids[0]] = np.sum(
                d_output * expert_outputs_i[0]
            )
        
            dLdp[ids[1]] = np.sum(
                d_output * expert_outputs_i[1]
            )
        

            dLdzr = get_routing_grad(router_logits, ids, d_output, expert_outputs_i)
        
            dLdw_r = np.outer(inpt, dLdzr)
            dLdb_r = dLdzr
        
            id0 = int(ids[0])
            id1 = int(ids[1])
            
            dW_experts[id0] += dLdw0
            db_experts[id0] += dLdz0
            
            dW_experts[id1] += dLdw1
            db_experts[id1] += dLdz1
            
            dW_router += dLdw_r
            db_router += dLdb_r
            


            
        for i, expert in enumerate(moe.experts):
            expert.w -= lr * dW_experts[i]
            expert.b -= lr * db_experts[i]
            
        moe.router.wr -= lr * dW_router
        moe.router.br -= lr * db_router
            

        return loss 
    








