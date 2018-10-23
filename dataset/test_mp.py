from dataset.MORPH_patches import MORPHPATCHES
from torch.utils import data
import os

index_dir = os.path.join('/', 'home', 'xintian', 'projects', 'age_exp', 'MORPH_Patches')

dataset = MORPHPATCHES(index_dir=index_dir, split='train')

train_loader = data.DataLoader(dataset,
                               batch_size=32,
                               shuffle=True)

for i, (imgs, labels) in enumerate(train_loader):
    print(imgs)
    print(labels)