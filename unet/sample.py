from unet import DenoisingUNet
import torch
from torchvision.utils import make_grid, save_image

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenoisingUNet(use_attn=True).to(device)
    model.eval()

    model.load_state_dict(torch.load('weights/2/ddpm_unet_final.pt', map_location=device))

    T = 1000
    beta_schedule = torch.linspace(start=1e-4, end=0.02, steps=T, device=device)

    samples = model.sample(device, beta_schedule)

    #samples = samples * 0.5 + 0.5
    grid = make_grid(samples, nrow=6, normalize=True, value_range=(-1, 1))
    save_image(grid, 'samples.png')


