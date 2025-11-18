from torchvision.datasets import Places365
from torch.utils.data import Dataset

class FilteredPlaces365(Dataset):
    def __init__(self, root, split="train-standard", transform=None, selected_classes=None):
        self.dataset = Places365(root=root, split=split, small=True, download=False, transform=transform)
        self.selected = set(selected_classes)

        # Filter indices of only desired classes
        self.indices = [i for i, (_, y) in enumerate(self.dataset) if y in self.selected]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]

        # Remap labels  to 0149
        new_label = list(self.selected).index(label)
        return img, new_label