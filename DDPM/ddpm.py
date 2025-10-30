import torch
import torch.nn as nn

class DDPM(nn.Module):
    def __init__(
            self,
            input_dims: int,
            denoising_steps: int,
            beta_schedule,
    ):
        self.input_dims = input_dims
        self.T = denoising_steps
        self.beta_schedule = beta_schedule
        assert self.T == len(self.beta_schedule), f"Beta schedule should be len {self.T}"

        self.alpha = torch.tensor(1 - self.beta_schedule, dtype=torch.float32)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.t_dims = 16
        self.temporal_embeddings = self.embed_timesteps(self.t_dims)

        self.denoiser = nn.Sequential(
            nn.Linear(self.input_dims + self.t_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dims)
        )
    
    def embed_timesteps(self, embedding_dim):
        e = torch.zeros(size=(self.T, embedding_dim))
        ts = torch.arange(0, self.T).float().unsqueeze(1)
        i = torch.arange(0, embedding_dim, 2).float().unsqueeze(0)
        d = ts / (10000 ** (i / embedding_dim))

        e[:, 0::2] = torch.sin(d)
        e[:, 1::2] = torch.cos(d)

        return e
    
    def forward(self, x_0):
        t = torch.randint(
            low=0,
            high=self.T,
            size=(x_0.shape[0],),
            dtype=torch.long
        )

        eps = torch.randn_like(x_0)
        a_t = self.alpha[t]
        ah_t = self.alpha_hat[t].view(-1, 1)

        x_t = (torch.sqrt(ah_t) * x_0) + (torch.sqrt(1 - ah_t) * eps)

        h_t = self.temporal_embeddings[t]
        input = torch.cat([x_t, h_t], dim=1)

        eps_theta = self.denoiser(input)

        return eps, eps_theta
    
    @staticmethod
    def compute_loss(eps, eps_theta):
        return ((eps - eps_theta) ** 2).mean()
    
    @torch.no_grad()
    def sample(self, n_samples=100):
        x_t = torch.randn(n_samples, self.input_dims)

        for t in reversed(range(self.T)):
            ts = torch.full((n_samples,), t, dtype=torch.long)
            a_t = self.alpha[t]
            ah_t = self.alpha_hat[t]
            ah_t_prev = self.alpha_hat[t-1]
            b_t = self.beta_schedule[t]

            eps = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

            h_t = self.temporal_embeddings[ts]
            input = torch.cat([x_t, h_t], dim=1)

            eps_theta = self.denoiser(input)

            b_tilde = ((1 - ah_t_prev) / (1 - ah_t)) * b_t

            c1 = (1 / torch.sqrt(a_t)) * (x_t - ((1 - a_t) / torch.sqrt(1 - ah_t) * eps_theta))
            c2 = torch.sqrt(b_tilde) * eps

            x_t = c1 + c2

        return x_t

        