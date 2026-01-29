import train
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

if __name__ == '__main__':
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_ds = datasets.CIFAR10(
        './data',
        train=True,
        transform=train_tf,
        download=True
    )

    dog_idxs = [i for i, target in enumerate(train_ds.targets) if target == 5] # Select only dogs
    #dog_idxs = dog_idxs[:128]
    dog_ds = Subset(train_ds, dog_idxs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = DataLoader(dog_ds, batch_size=64, shuffle=True)

    epochs = 5000
    T = 1000
    beta_schedule = torch.linspace(start=1e-4, end=0.02, steps=T).to(device)

    train.train(loader, epochs, T, beta_schedule, device)