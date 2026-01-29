import torch
import torch.nn as nn
from unet import DenoisingUNet
import time
from torch.utils.tensorboard import SummaryWriter
import os

def step(model, x0, alpha_hats, T, device):
    batch_size = x0.shape[0]
    t = torch.randint(low=0, high=T, size=(batch_size,), device=device)
    alpha_hats = alpha_hats[t].view(batch_size, 1, 1, 1)
    eps = torch.randn_like(x0)

    xt = torch.sqrt(alpha_hats) * x0 + torch.sqrt(1 - alpha_hats) * eps

    return eps, model(xt, t)

def train(dataloader, epochs, T, beta_schedule, device):
    writer = SummaryWriter(log_dir='runs/')
    os.makedirs("weights/2", exist_ok=True)
    model = DenoisingUNet(use_attn=True).to(device)
    model.load_state_dict(torch.load('weights/1/ddpm_unet_final.pt', map_location=device))

    model.train()

    alphas = 1 - beta_schedule
    alpha_hats = torch.cumprod(alphas, dim=0).to(device)

    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for i in range(epochs):
        start = time.time()

        epoch_loss = 0
        batch_count = 0

        for x0, _ in dataloader:
            x0 = x0.to(device)
            optim.zero_grad()
            batch_count += 1

            eps, eps_theta = step(model, x0, alpha_hats, T, device)
            
            loss = loss_fn(eps, eps_theta)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
 
        duration = time.time() - start
        avg_loss = epoch_loss/batch_count
        if i % 10 == 0:
            print(
                f"Epoch: {i+1}/{epochs} | "
                f"Avg. batch loss: {avg_loss:.5f} | "
                f"Time taken: {duration:.2f}s"
            )
        
        if i % 100 == 0:
            torch.save(model.state_dict(), f'weights/2/ddpm_unet_{i}.pt')

        writer.add_scalar('train/loss_epoch', avg_loss, i+5000)

    writer.close()
    torch.save(model.state_dict(), 'weights/2/ddpm_unet_final.pt')