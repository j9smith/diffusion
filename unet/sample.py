from unet import DenoisingUNet
import torch
from torchvision.utils import make_grid, save_image

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenoisingUNet().to(device)
    model.eval()

    model.load_state_dict(torch.load('ddpm_unet.pt', map_location=device))

    T = 1000
    beta_schedule = torch.linspace(start=0.0001, end=0.01, steps=T)

    samples = model.sample(device, beta_schedule)
    print(samples.min().item(), samples.max().item(), samples.mean().item(), samples.std().item())

    grid = make_grid(samples, nrow=6, normalize=True)
    save_image(grid, 'samples.png')


