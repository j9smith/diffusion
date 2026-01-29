import torch
import torch.nn as nn

def sinusoidal_time_embedding(t, embedding_dim):
    assert embedding_dim % 2 == 0
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
    
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_ch, num_groups, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads

        self.norm = group_norm(num_groups, in_ch)
        self.qkv = nn.Conv1d(in_ch, in_ch * 3, kernel_size=1)
        self.proj = nn.Conv1d(in_ch, in_ch, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        HW = H*W
        
        x_flat = self.norm(x).reshape(B, C, HW)

        qkv = self.qkv(x_flat) # C -> C*3

        q, k, v = torch.chunk(qkv, 3, dim=1) # Chunk along C dim

        assert C % self.num_heads == 0
        d = C // self.num_heads

        q = q.view(B, self.num_heads, d, HW)
        k = k.view(B, self.num_heads, d, HW)
        v = v.view(B, self.num_heads, d, HW)

        attn = q.transpose(-2, -1) @ k # -> B, heads, HW, HW

        scaled_attn = attn * (d ** -0.5)

        attn = torch.softmax(scaled_attn, dim=-1)

        out = attn @ v.transpose(-2, -1) # -> B, heads, HW, d
        out = out.transpose(-1, -2).reshape(B, C, HW)

        out = self.proj(out).view(B, C, H, W)

        return x + out
    
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
    def __init__(self, base_ch: int=64, num_groups: int=32, use_attn: bool = True):
        super().__init__()

        self.time = TimeMLP(base_ch)
        t_emb_dims = 4 * base_ch

        self.in_conv = nn.Conv2d(3, base_ch, kernel_size=3, padding=1)

        self.attn1 = SpatialSelfAttention(base_ch * 2, num_groups) if use_attn else nn.Identity()
        self.attn2 = SpatialSelfAttention(base_ch * 2, num_groups) if use_attn else nn.Identity()

        # Encoder
        self.rb32 = ResBlock(num_groups, base_ch, base_ch, t_emb_dims)
        self.down1 = Downsample(base_ch) # 32x32 -> 16x16

        self.rb16 = ResBlock(num_groups, base_ch, base_ch * 2, t_emb_dims)
        self.down2 = Downsample(base_ch * 2) # 16x16 -> 8x8

        self.rb8 = ResBlock(num_groups, base_ch * 2, base_ch * 4, t_emb_dims)
        self.down3 = Downsample(base_ch * 4)

        self.rb4 = ResBlock(num_groups, base_ch * 4, base_ch * 8, t_emb_dims)
        # Bottleneck
        self.mid = ResBlock(num_groups, base_ch * 8, base_ch * 8, t_emb_dims)

        # Decoder
        self.up1 = Upsample(base_ch * 8)
        self.rb8up = ResBlock(num_groups, base_ch * 8 + base_ch * 4, base_ch * 4, t_emb_dims)
        
        self.up2 = Upsample(base_ch * 4) # 8x8 -> 16x16
        self.rb16up = ResBlock(num_groups, base_ch * 4 + base_ch * 2, base_ch * 2, t_emb_dims)

        self.up3 = Upsample(base_ch * 2) # 16x16 -> 32x32
        self.rb32up = ResBlock(num_groups, base_ch * 2 + base_ch, base_ch, t_emb_dims)

        # Out
        self.out_norm = group_norm(num_groups, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)

        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.time(t)

        h = self.in_conv(x)

        h32 = self.rb32(h, t_emb)

        h = self.down1(h32)
        h16 = self.attn1(self.rb16(h, t_emb))

        h = self.down2(h16)
        h8 = self.rb8(h, t_emb)

        h = self.down3(h8)
        h4 = self.rb4(h, t_emb)

        h = self.mid(h4, t_emb)

        h = self.up1(h)
        h = torch.cat([h, h8], dim=1)
        h = self.rb8up(h, t_emb)

        h = self.up2(h)
        h = torch.cat([h, h16], dim=1)
        h = self.attn2(self.rb16up(h, t_emb))

        h = self.up3(h)
        h = torch.cat([h, h32], dim=1)
        h = self.rb32up(h, t_emb)

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

            coef1 = 1 / torch.sqrt(alphas[t])
            coef2 = (1 - alphas[t]) / torch.sqrt(1 - alpha_hat_t)

            eps_theta = self.forward(xt, t_batch)

            mu = coef1 * (xt - (coef2 * eps_theta))

            if t == 0: 
                xt = mu
            else:
                alpha_hat_t_prev = alpha_hats[t_batch - 1].view(n_samples, 1, 1, 1)
                beta_tilde = ((1 - alpha_hat_t_prev) / (1 - alpha_hat_t)) * beta_schedule[t]
                sigma = torch.sqrt(beta_tilde)
                z = torch.randn_like(xt)
                xt = mu + (sigma * z)

        return xt