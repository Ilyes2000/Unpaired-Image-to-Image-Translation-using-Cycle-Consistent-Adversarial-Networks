from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, random

class UnalignedDataset(Dataset):
    def __init__(self, root, phase='train', image_size=256):
        self.dir_A = os.path.join(root, phase+'A')
        self.dir_B = os.path.join(root, phase+'B')
        self.A_paths = sorted(os.listdir(self.dir_A))
        self.B_paths = sorted(os.listdir(self.dir_B))
        self.transform = transforms.Compose([
            transforms.Resize(int(image_size*1.12), Image.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))
    def __getitem__(self, i):
        A = Image.open(os.path.join(self.dir_A, self.A_paths[i % len(self.A_paths)])).convert('RGB')
        B = Image.open(os.path.join(self.dir_B, self.B_paths[random.randrange(len(self.B_paths))])).convert('RGB')
        return {'A': self.transform(A), 'B': self.transform(B),
                'A_paths': self.A_paths[i % len(self.A_paths)],
                'B_paths': self.B_paths[i % len(self.B_paths)]}

def get_dataloader(root, phase='train', image_size=256, batch_size=1, num_workers=4, pin_memory=False):
    dataset = UnalignedDataset(root, phase, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True, pin_memory=pin_memory)

