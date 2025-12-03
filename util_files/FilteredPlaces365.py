import torch
from torch.utils.data import Subset


class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, class_map):
        """
        subset: the Subset(dataset, indices)
        class_map: dict mapping old_class : new_class (0..149)
        """
        self.subset = subset
        self.class_map = class_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        y = self.class_map[y]   # remap old label : new label
        return x, y

def start_filter(full_dataset, selected_classes):
    # selected_classes is the list of selected class IDs (length selected_classes)
    old_to_new = {old: new for new, old in enumerate(selected_classes)}

    # Build mask
    train_mask = torch.zeros(len(full_dataset), dtype=torch.bool)
    for selected_class in selected_classes:
        train_mask |= (torch.tensor(full_dataset.targets) == selected_class)

    train_indices = train_mask.nonzero().reshape(-1)

    # Build subset with original samples
    train_subset = Subset(full_dataset, train_indices)

    # Wrap with re-labeled dataset
    return RemappedDataset(train_subset, old_to_new)
