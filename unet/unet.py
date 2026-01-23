import torch
import torch.nn as nn

def sinusoidal_time_embedding(t, embedding_dim):
    device = t.device
    B = t.shape[0]

    e = torch.zeros((B, embedding_dim), device=device)

    ts = t.float().unsqueeze(1)  # [B, 1]
    i = torch.arange(0, embedding_dim, 2, device=device).float().unsqueeze(0)  # [1, D/2]

    d = ts / (10000 ** (i / embedding_dim))  # [B, D/2]

    e[:, 0::2] = torch.sin(d)
    e[:, 1::2] = torch.cos(d)

    return e

def group_norm(num_groups, channels):
    groups = min(num_groups, channels)

    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)

class Downsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.downsample = nn.Conv2d(
            in_ch,
            in_ch, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )

    def forward(self, x):
        return self.downsample(x)

class Upsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_ch, 
            in_ch, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )

    def forward(self, x):
        return self.upsample(x)

class ResBlock(nn.Module):
    def __init__(self, num_groups, in_ch, out_ch, t_emb_dims, dropout: float=0.0):
        super().__init__()

        self.norm1 = group_norm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = group_norm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.act = nn.SiLU()
        
        self.t_proj = nn.Linear(t_emb_dims, out_ch)

        self.dropout = nn.Dropout(dropout)

        self.skip = (
            nn.Identity() if in_ch == out_ch else
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x, t_embedding):
        h = self.conv1(self.act(self.norm1(x)))

        # Each feature channel gets a learned scalar time conditioning
        h = h + self.t_proj(self.act(t_embedding))[:, :, None, None] # (B, C) -> (B, C, x, y)

        h = self.conv2(self.dropout(self.act(self.norm2(h))))

        return h + self.skip(x)
    
class TimeMLP(nn.Module):
    def __init__(self, base_ch):
        super().__init__()
        self.base_ch = base_ch
        self.nn = nn.Sequential(
            nn.Linear(base_ch, 4 * base_ch),
            nn.SiLU(),
            nn.Linear(4 * base_ch, 4 * base_ch)
        )
    
    def forward(self, t):
        embs = sinusoidal_time_embedding(t, self.base_ch)
        return self.nn(embs)

class DenoisingUNet(nn.Module):
    def __init__(self, base_ch: int=64, num_groups: int=32):
        super().__init__()

        self.time = TimeMLP(base_ch)
        t_emb_dims = 4 * base_ch

        self.in_conv = nn.Conv2d(3, base_ch, kernel_size=3, padding=1)

        # Encoder
        self.rb32 = ResBlock(num_groups, base_ch, base_ch, t_emb_dims)
        self.down1 = Downsample(base_ch)

        self.rb16 = ResBlock(num_groups, base_ch, base_ch * 2, t_emb_dims)
        self.down2 = Downsample(base_ch * 2)

        self.rb8 = ResBlock(num_groups, base_ch * 2, base_ch * 4, t_emb_dims)
        
        # Bottleneck
        self.mid = ResBlock(num_groups, base_ch * 4, base_ch * 4, t_emb_dims)

        # Decoder
        self.up1 = Upsample(base_ch * 4)
        self.rb8up = ResBlock(num_groups, base_ch * 4 + base_ch * 2, base_ch * 2, t_emb_dims)

        self.up2 = Upsample(base_ch * 2)
        self.rb16up = ResBlock(num_groups, base_ch * 2 + base_ch, base_ch, t_emb_dims)

        # Out
        self.out_norm = group_norm(num_groups, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)

        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.time(t)

        h = self.in_conv(x)

        h32 = self.rb32(h, t_emb)

        h = self.down1(h32)
        h16 = self.rb16(h, t_emb)

        h = self.down2(h16)
        h8 = self.rb8(h, t_emb)

        h = self.mid(h8, t_emb)

        h = self.up1(h)
        h = torch.cat([h, h16], dim=1)
        h = self.rb8up(h, t_emb)

        h = self.up2(h)
        h = torch.cat([h, h32], dim=1)
        h = self.rb16up(h, t_emb)

        h = self.out_conv(self.act(self.out_norm(h)))
        return h
    
    @torch.no_grad()
    def sample(self, device, beta_schedule, n_samples: int = 24, T: int = 1000):
        xt = torch.randn(size=(n_samples, 3, 32, 32), device=device)
        alphas = 1 - beta_schedule
        alpha_hats = torch.cumprod(alphas, dim=0).to(device)

        for t in reversed(range(T)):
            t_batch = torch.full((n_samples,), t, device=device)

            alpha_hat_t = alpha_hats[t_batch].view(n_samples, 1, 1, 1)
            alpha_hat_t_prev = alpha_hats[t_batch - 1].view(n_samples, 1, 1, 1)

            coef1 = 1 / torch.sqrt(alphas[t])
            coef2 = (1 - alphas[t]) / torch.sqrt(1 - alpha_hat_t)

            eps_theta = self.forward(xt, t_batch)

            mu = coef1 * (xt - (coef2 * eps_theta))

            if t == 0: 
                xt = mu
            else:
                beta_tilde = ((1 - alpha_hat_t_prev) / (1 - alpha_hat_t)) * beta_schedule[t]
                sigma = torch.sqrt(beta_tilde)
                z = torch.randn_like(xt)
                xt = mu + (sigma * z)

        return xt