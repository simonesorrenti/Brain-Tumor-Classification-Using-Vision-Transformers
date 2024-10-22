import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from collections import Counter

seed = 42
torch.manual_seed(seed)

class ImageFolderDatasetLoader:
    def __init__(self, root_dir, batch_size=32, shuffle=True, transforms=None, seed=42):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transforms = transforms
        self.seed = seed

        self.dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transforms)

        self.class_names = self.dataset.classes
        self.label_to_name = {idx: name for idx, name in enumerate(self.class_names)}
        self.name_to_label = {name: idx for idx, name in enumerate(self.class_names)}

    def get_data_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_train_test_loader(self, test_size=0.2, train_transforms=None, test_transforms=None):

        train_dataset, test_dataset = random_split(self.dataset, [(1-test_size), test_size],
                                                   generator=torch.Generator().manual_seed(self.seed))

        train_dataset.dataset.transform = train_transforms
        test_dataset.dataset.transform = test_transforms

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.train_loader = train_loader
        self.test_loader = test_loader

        return train_loader, test_loader

    def get_class_names(self):
        return self.class_names

    def get_label_to_name(self):
        return self.label_to_name

    def get_name_to_label(self):
        return self.name_to_label

    def get_distribution(self):
        class_counts = Counter(label for _, label in self.dataset.samples)
        return class_counts

    def get_mean_std(self, loader):
        mean = 0.
        std = 0.
        total_images = 0

        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(-1).sum(0)
            std += images.std(-1).sum(0)
            total_images += batch_samples

        mean /= total_images
        std /= total_images
        
        self.mean = mean
        self.std = std

        return mean, std