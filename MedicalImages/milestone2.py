from torch.utils.data.dataset import Dataset


class NiftiDataset(Dataset):
    def __init__(self, source_dir, target_dir, transforms):
        # fill this in
        pass

    def __len__(self):
        # fill this in
        pass

    def __getitem__(self, idx):
        # fill this in
        pass


class RandomCrop3D:
    def __init__(self, args):
        # fill this in
        pass

    def __call__(self, sample):
        # fill this in
        pass

