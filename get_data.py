import os
import torch
from torchvision import datasets, models, transforms

def get_data_loaders():

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ]),
    }
    data_dir = os.path.abspath(os.path.dirname(__file__))

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    print(class_names)

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    return dataloaders, dataset_sizes, class_names


