import os.path as osp
import json
import random
from collections import Counter

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Generate Random 90 degree rotations 
class RandomRotate:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(img, angle)
    
def load_dataset(final_dataset_path, cv_folds_path, training_mode = 'train'):
    # Load dataset
    with open(final_dataset_path, 'r') as f:
        final_dataset = json.load(f)
    
    with open(cv_folds_path, 'r') as f:
        cv_folds = json.load(f)
    
    def extract_images(keys):
        """Extract images based on keys."""
        images = []
        for key in keys:
            if key in final_dataset:
                data_entry = final_dataset[key]
                for combination in data_entry:
                    images.append(combination)
        return images
    return extract_images(cv_folds[training_mode])

    
def load_dataset_one_image(final_dataset_path, cv_folds_path, training_mode = 'train'):
    # Load dataset
    with open(final_dataset_path, 'r') as f:
        final_dataset = json.load(f)
    
    with open(cv_folds_path, 'r') as f:
        cv_folds = json.load(f)
    
    def extract_images(keys):
        """Extract images based on keys."""
        images = []
        for key in keys:
            if key in final_dataset:
                data_entry = final_dataset[key]
                for combination in data_entry['post']:
                    images.append(combination)
                for combination in data_entry['lr']:
                    images.append(combination)
                for combination in data_entry['1blr']:
                    images.append(combination)
        return images
    return extract_images(cv_folds[training_mode])


class endoscopyDataset(Dataset):

    def __init__(self,
                 root_dir,
                 final_dataset_path,
                 cv_folds_path,
                 training_mode = 'train',
                 use_lr = False,
                 data_set_up = None,
                 transform=None):

        self.root_dir = root_dir
        self.data_set_up =  data_set_up
        
        final_dataset_path = osp.join(self.root_dir, final_dataset_path)
        cv_folds_path = osp.join(self.root_dir, cv_folds_path)
        combinations = load_dataset(final_dataset_path, cv_folds_path, training_mode)

        if training_mode == "test":
            combinations_val = load_dataset(final_dataset_path, cv_folds_path, "val")
            combinations.extend(combinations_val)

        self.data = combinations

        self.transform = transform
        # Extract class labels for WeightedRandomSampler
        self.labels = [item[0]['label'] for item in self.data]  # Store labels for sampler

    # Returns the legnth of the dataset
    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        info1 = self.data[idx]
        img1 = self.transform(Image.open(osp.join(self.root_dir, info1[0]['name'])))
        img2 = self.transform(Image.open(osp.join(self.root_dir, info1[1]['name'])))
        class_id1 = torch.tensor(info1[0]['label'], dtype=torch.long)
        class_id2 = torch.tensor(info1[1]['label'], dtype=torch.long)
        return img1, img2, class_id1, class_id2, osp.join(self.root_dir, info1[0]['name']), osp.join(self.root_dir, info1[1]['name'])
        
class endoscopyDataset_one_image(Dataset):

    def __init__(self,
                 root_dir,
                 final_dataset_path,
                 cv_folds_path,
                 training_mode = 'train',
                 use_lr = False,
                 data_set_up = None,
                 transform=None):

        self.root_dir = root_dir
        self.data_set_up =  data_set_up
        
        final_dataset_path = osp.join(self.root_dir, final_dataset_path)
        cv_folds_path = osp.join(self.root_dir, cv_folds_path)
        combinations = load_dataset_one_image(final_dataset_path, cv_folds_path, training_mode)

        if training_mode == "test":
            combinations_val = load_dataset_one_image(final_dataset_path, cv_folds_path, "val")
            combinations.extend(combinations_val)

        self.data = combinations

        self.transform = transform

        # Extract class labels for WeightedRandomSampler
        self.labels = [item['label'] for item in self.data]  # Store labels for sampler

    # Returns the legnth of the dataset
    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        info1 = self.data[idx]
        img1 = self.transform(Image.open(osp.join(self.root_dir, info1['name'])))
        class_id1 = torch.tensor(info1['label'], dtype=torch.long)
        return img1, class_id1, osp.join(self.root_dir, info1['name'])
        

# xdata = endoscopyDataset
def create_dataloader(batch_size,
                      cv_fold = None,
                      use_lr_train = False,
                      data_set_up = None,
                      logger=None,
                      sampler = 'balanced',
                      root_dir = None):
    
    transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotate(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

        ])

    if logger is not None: logger.info("Train Augmentations")
    if logger is not None: logger.info(transform_train.transforms)

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if root_dir is None:
        raise ValueError("root_dir must be provided (path to your dataset directory)")

    if logger is not None: logger.info("Root directory for images: " + root_dir)

    train_set = endoscopyDataset(root_dir = root_dir,
                                 final_dataset_path = 'image_combinations.json',
                                 cv_folds_path = cv_fold,
                                 training_mode = 'train',
                                 use_lr = False,
                                 data_set_up = data_set_up,
                                 transform=transform_train)

    validation_set = endoscopyDataset(root_dir=root_dir,
                                      final_dataset_path = 'image_combinations.json',
                                      cv_folds_path = cv_fold,
                                      training_mode = 'val',
                                      use_lr = False,
                                      data_set_up = None,
                                      transform=transform_train)

    test_set = endoscopyDataset(root_dir=root_dir,
                                final_dataset_path = 'image_combinations.json',
                                cv_folds_path = cv_fold,
                                training_mode = 'test',
                                use_lr = True,
                                data_set_up = None,
                                transform=transform_test)

    print(sampler)
    if sampler is not None:
        if logger is not None: logger.info("Using Balanced Sampler")
        label_counts = Counter(train_set.get_labels())
        class_weights = {cls: 1.0 / (count + 1e-6) for cls, count in label_counts.items()}
        sample_weights = [class_weights[label] for label in train_set.get_labels()]
    
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  sampler=sampler,
                                  shuffle=False)
    else:
        if logger is not None: logger.info("Using no Sampler")
        train_loader = DataLoader(train_set,
                                batch_size=batch_size,
                                num_workers=0,
                                shuffle=True)

    validation_loader = DataLoader(validation_set,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=0)

    test_loader = DataLoader(test_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)
    
    if data_set_up == 'mix':
        if logger is not None: logger.info("Mixing harmonized with non harmonized images")
    elif data_set_up == "og+tf":
        if logger is not None: logger.info("Using Original Transformed image together with it's transformed image")
    else:
        if logger is not None: logger.info("Using just the dataset images transformed")

    if logger is not None: logger.info("Number of images gathered for Train: %d", len(train_set))
    if logger is not None: logger.info("Number of images gathered for Validation: %d", len(validation_set))
    if logger is not None: logger.info("Number of images gathered for Test: %d", len(test_set))

    return train_loader, validation_loader, test_loader

# xdata = endoscopyDataset
def create_dataloader_one_image(batch_size,
                      cv_fold = None,
                      use_lr_train = False,
                      data_set_up = None,
                      logger=None,
                      sampler = 'balanced',
                      root_dir = None):
    
    transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotate(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

        ])

    if logger is not None: logger.info("Train Augmentations")
    if logger is not None: logger.info(transform_train.transforms)

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if root_dir is None:
        raise ValueError("root_dir must be provided (path to your dataset directory)")

    if logger is not None: logger.info("Root directory for images: " + root_dir)

    train_set = endoscopyDataset_one_image(root_dir = root_dir,
                                 final_dataset_path = 'image_combinations.json',
                                 cv_folds_path = cv_fold,
                                 training_mode = 'train',
                                 use_lr = False,
                                 data_set_up = data_set_up,
                                 transform=transform_train)

    validation_set = endoscopyDataset_one_image(root_dir=root_dir,
                                      final_dataset_path = 'image_combinations.json',
                                      cv_folds_path = cv_fold,
                                      training_mode = 'val',
                                      use_lr = False,
                                      data_set_up = None,
                                      transform=transform_train)

    test_set = endoscopyDataset_one_image(root_dir=root_dir,
                                final_dataset_path = 'image_combinations.json',
                                cv_folds_path = cv_fold,
                                training_mode = 'test',
                                use_lr = True,
                                data_set_up = None,
                                transform=transform_test)

    if sampler is not None:
        if logger is not None: logger.info("Using Balanced Sampler")
        label_counts = Counter(train_set.get_labels())
        class_weights = {cls: 1.0 / (count + 1e-6) for cls, count in label_counts.items()}
        sample_weights = [class_weights[label] for label in train_set.get_labels()]
    
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  sampler=sampler,
                                  shuffle=False)
    else:
        if logger is not None: logger.info("Using no Sampler")
        train_loader = DataLoader(train_set,
                                batch_size=batch_size,
                                num_workers=0,
                                shuffle=True)

    validation_loader = DataLoader(validation_set,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=0)

    test_loader = DataLoader(test_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)
    
    if data_set_up == 'mix':
        if logger is not None: logger.info("Mixing harmonized with non harmonized images")
    elif data_set_up == "og+tf":
        if logger is not None: logger.info("Using Original Transformed image together with it's transformed image")
    else:
        if logger is not None: logger.info("Using just the dataset images transformed")

    if logger is not None: logger.info("Number of images gathered for Train: %d", len(train_set))
    if logger is not None: logger.info("Number of images gathered for Validation: %d", len(validation_set))
    if logger is not None: logger.info("Number of images gathered for Test: %d", len(test_set))

    return train_loader, validation_loader, test_loader