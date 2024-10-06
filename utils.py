import os


def checkDir(path):
    base_path, folder_name = os.path.split(path.rstrip(os.sep))

    if not os.path.isdir(path):
        os.makedirs(path)
        print(f"Folder created: {path}")

        return path
    
    cnt = 1

    while os.path.isdir(path):
        new_folder_name = f"{folder_name}_{cnt:02d}"
        path = os.path.join(base_path, new_folder_name) + os.sep
        cnt += 1
    
    os.makedirs(path)
    print(f"Folder created: {path}")

    return path

def calcMeanStd(path):
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean, std


if __name__ == '__main__':
    mean, std = calcMeanStd('data/fer2013/train')

    print(f"Mean: {mean}")
    print(f"Std: {std}")
