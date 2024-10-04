from collections import Counter

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


train_path = 'data/fer2013/train/'
valid_path = 'data/fer2013/valid/'
test_path = 'data/fer2013/test/'

def loadBatches(batch_size, is_train=True):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)])

    if is_train:
        train_data = ImageFolder(train_path, transform=transform)
        valid_data = ImageFolder(valid_path, transform=transform)

        train_batches = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_batches = DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True)

        class_sample_count = Counter(train_data.targets)

        total_samples = len(train_data)
        num_classes = len(train_data.classes)

        class_weights = [total_samples / (num_classes * class_sample_count[i]) for i in range(num_classes)]
        norm_class_weights = [w / sum(class_weights) for w in class_weights]

        return train_batches, valid_batches, norm_class_weights
    
    test_data = ImageFolder(test_path)
    test_batches = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    return test_batches