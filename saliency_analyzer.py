import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class IntegratedGradientsExplainer:
    def __init__(self, model, state_dim, action_dim, single_obs_dim):
        
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.single_obs_dim = single_obs_dim
    
    def compute_integrated_gradients(self, x, p=25):
        """
        Compute_integrated_gradients
        input: x [B, T, N] (Batch, time_step, single_obs_dim)
        output: ig [T, N]  
        """
        batch_size, time_steps, _ = x.shape
        baseline = torch.zeros_like(x)
        total_grads = torch.zeros((p, batch_size, time_steps, self.single_obs_dim), device=x.device)
        total_G = torch.zeros((p, batch_size, time_steps, self.single_obs_dim), device=x.device)
        
        with torch.enable_grad():
            for k in range(p):
                interpolated_x = baseline + ((k+1) / p) * (x - baseline)
                
                flat_x = interpolated_x.detach().squeeze(0).requires_grad_(True)
                flat_actions = self.model(flat_x)
                actions = flat_actions.unsqueeze(0)
                actions.sum().backward(retain_graph=True)
                print(actions[..., 0],actions[..., 0].shape,"action")
                grad = flat_x.grad.view(batch_size, time_steps, self.state_dim)
                total_grads[k] = torch.abs(grad[...,-self.single_obs_dim:])
                total_G[k] = torch.abs(((x - baseline)/p )* grad)[...,-self.single_obs_dim:]

        avg_grad = torch.mean(total_grads, dim=0)[0]
        avg_G = torch.mean(total_G, dim=0)[0]

        avg_G = torch.cat((avg_G[...,4:28],avg_G[...,40:46]),dim=-1)
        print(total_G.shape)
        print(avg_G.shape)
        # 4:28 40-46
        return avg_G  # 5, 57
    
    def compute_raw_saliency(self, ig):

        epsilon = ig.mean()
        return torch.where(ig > epsilon, ig - epsilon, torch.zeros_like(ig))
    
    def normalize_saliency(self, rsv):

        max_val = rsv.max()
        return rsv / max_val if max_val != 0 else torch.zeros_like(rsv)

class SaliencyVisualizer:
    def __init__(self):
        pass
    
    def plot_saliency_map(self, saliency, labels, time_steps, title="Feedback Saliency Over Time"):
        plt.figure(figsize=(18, 9))
        colors = [(0,  '#f8f8c9'),
                  (0.2,'#32bfc3'),
                  (0.4,'#135f83'),
                  (0.6,'#122255'),
                  (0.8,'#011240'),
                  (1. ,'#000000')
                  ]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        im = plt.imshow(
            saliency.T,
            cmap=cmap,
            aspect='equal',
            extent=[0, time_steps, 0, len(labels)],
            vmin=0, vmax=1,
            origin='lower'
        )
        plt.yticks(np.arange(len(labels)) + 0.5, labels, fontsize=10)
        plt.xticks(np.arange(0, time_steps+1, 10), fontsize=10)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Feedback State', fontsize=12)
        plt.title(title, fontsize=14, pad=20)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Saliency', rotation=270, labelpad=15, fontsize=12)
        # plt.grid(visible=True, axis='x', color='white', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

class QuadrupedPolicy(nn.Module):
    def __init__(self, state_dim=64, action_dim=12):

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)
    

if __name__ == '__main__':
    state_dim = 64
    action_dim= 12
    policy = QuadrupedPolicy()
    single_obs_dim = 64
    explainer = IntegratedGradientsExplainer(policy, state_dim, action_dim, single_obs_dim)
    visualizer = SaliencyVisualizer()
    obs_lst = []
    total_steps = 100
    for i in range (total_steps):
        obs = torch.rand((state_dim)) #([64])
        obs_lst.append(obs)


    obs_tensor = torch.stack(obs_lst).unsqueeze(0)  #([1,5, 64])

    with torch.no_grad():
        ig = explainer.compute_integrated_gradients(obs_tensor, p=25)

    rsv = explainer.compute_raw_saliency(ig)
    nsv = explainer.normalize_saliency(rsv)

    feedback_labels = [
        *[f' {i} ' for i in range(state_dim)],
    ]
    feedback_labels = feedback_labels[:state_dim]
    # 4. 可视化结果
    visualizer.plot_saliency_map(
        nsv, 
        feedback_labels,
        time_steps=total_steps, 
        title="Saliency Map of Feedback States "
    )