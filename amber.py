import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, utils, datasets, transforms
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from typing import Optional
from models.explainer import Deeplabv3Resnet50ExplainerModel
from utils.metrics import SingleLabelMetrics 
from utils.loss import TotalVariationConv, ClassMaskAreaLoss

from transformers import (AutoFeatureExtractor,
                          AutoModelForImageClassification,
                          AutoConfig)

import numpy as np
import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

pl.seed_everything(42)

class CustomDataModule(pl.LightningDataModule):
    def __init__(
                self, data_path, train_batch_size=16, val_batch_size=16,
                test_batch_size=16
                ):
        
        super().__init__()
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        
    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None):
        TRAIN_FDIR = f"{self.data_path}train"
        
        init_transforms = transforms.Compose([
                transforms.Resize(size=(224,224)),
                transforms.ToTensor()])
        
        train_data = datasets.ImageFolder(root=TRAIN_FDIR, 
                                          transform=init_transforms)

        # Compute for the means and stds (for normalization)
        imgs = torch.stack([img_t for img_t, _ in train_data], dim=3)
        means = imgs.view(3, -1).mean(dim=1).numpy()
        stds = imgs.view(3, -1).std(dim=1).numpy()

        print(f'Means:           {means}') 
        print(f'Std. Deviations: {stds}')

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(size=(224,224)),
                transforms.RandomHorizontalFlip(p=0.6),             
                transforms.RandomPerspective(p=0.5),
                transforms.ColorJitter(brightness=0.5),              
                transforms.ToTensor(),                              
                transforms.Normalize(means, stds)
            ]),
            'validation': transforms.Compose([
                transforms.Resize(size=(224,224)),
                transforms.ToTensor(),                              
                transforms.Normalize(means, stds)
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=(224,224)),
                transforms.ToTensor(),                              
                transforms.Normalize(means, stds)
            ])
        }

        DATA_DIR = self.data_path

        # Loading image data using ImageFolder
        image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                                  data_transforms[x])
                          for x in ['train', 'validation', 'test']}

        # Size of datasets
        dataset_sizes = {x: len(image_datasets[x]) for x in
                         ['train', 'validation', 'test']}

        # Class names
        class_names = image_datasets['train'].classes
        if stage == "fit" or stage is None:
            self.train = image_datasets['train']
            self.val = image_datasets['validation']

        if stage == "test" or stage is None:
            self.test = image_datasets['test']
            

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, 
                          collate_fn=collate_fn, shuffle=True, num_workers=4, 
                          pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, 
                          collate_fn=collate_fn, num_workers=4, 
                          pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, 
                          collate_fn=collate_fn, num_workers=4,
                          pin_memory=torch.cuda.is_available())
    
def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    target = torch.tensor([item[1] for item in batch])
    return data, target

class ExplainerClassifierModel(pl.LightningModule): 
    def __init__(self, num_classes=7, 
                 fix_classifier=True, learning_rate=1e-5, 
                 class_mask_min_area=0.05, class_mask_max_area=0.3, 
                 entropy_regularizer=1.0, use_mask_variation_loss=True,
                 mask_variation_regularizer=1.0, use_mask_area_loss=True, 
                 mask_area_constraint_regularizer=1.0, 
                 mask_total_area_regularizer=0.1, 
                 ncmask_total_area_regularizer=0.3, metrics_threshold=-1.0,
#                  save_masked_images=False, save_masks=False, 
#                  save_all_class_masks=False, save_path="./results/"
                ):

        super().__init__()
        self.setup_explainer(num_classes=num_classes)
        self.setup_classifier(fix_classifier=fix_classifier, 
                              num_classes=num_classes)

        self.setup_losses(class_mask_min_area=class_mask_min_area, 
                          class_mask_max_area=class_mask_max_area)
        
        self.setup_metrics(num_classes=num_classes)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.entropy_regularizer = entropy_regularizer
        self.use_mask_variation_loss = use_mask_variation_loss
        self.mask_variation_regularizer = mask_variation_regularizer
        self.use_mask_area_loss = use_mask_area_loss
        self.mask_area_constraint_regularizer = mask_area_constraint_regularizer
        self.mask_total_area_regularizer = mask_total_area_regularizer
        self.ncmask_total_area_regularizer = ncmask_total_area_regularizer

        # Image display/save settings
#         self.save_masked_images = save_masked_images
#         self.save_masks = save_masks
#         self.save_all_class_masks = save_all_class_masks
#         self.save_path = save_path

    def setup_explainer(self, num_classes):
        self.explainer = Deeplabv3Resnet50ExplainerModel(
            num_classes=num_classes
        ).to(device)

    def setup_classifier(self, fix_classifier, num_classes):
        self.extractor = (AutoFeatureExtractor
                     .from_pretrained("trpakov/vit-face-expression"))
        
        self.classifier = (AutoModelForImageClassification
               .from_pretrained("trpakov/vit-face-expression"))
        self.classifier.to(device)
        
        if fix_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def setup_losses(self, class_mask_min_area, class_mask_max_area):
        self.total_variation_conv = TotalVariationConv()
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.class_mask_area_loss_fn = ClassMaskAreaLoss(
            min_area=class_mask_min_area, max_area=class_mask_max_area
        )
        
        self.total_variation_conv.to(device)
        self.classification_loss_fn.to(device)
                
    def setup_metrics(self, num_classes):
        self.train_metrics = SingleLabelMetrics(num_classes=num_classes)
        self.valid_metrics = SingleLabelMetrics(num_classes=num_classes)
        self.test_metrics = SingleLabelMetrics(num_classes=num_classes)

    def forward(self, image, targets):
        segmentations = self.explainer(image) # torch.Size([16, 7, 224, 224])
        target_mask, non_target_mask = extract_masks(segmentations, targets) # torch.Size([16, 224, 224])
        inversed_target_mask = torch.ones_like(target_mask) - target_mask # torch.Size([16, 224, 224])
        masked_image = target_mask.unsqueeze(1) * (image) # torch.Size([16, 3, 224, 224])
        inversed_masked_image = inversed_target_mask.unsqueeze(1) * (image) # torch.Size([16, 3, 224, 224])

        logits_mask = self.classifier(masked_image).logits
        logits_inversed_mask = self.classifier(inversed_masked_image).logits

        return logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations
    
    def training_step(self, batch, batch_idx):
        image, targets = batch
        target_vectors = torch.from_numpy(np.eye(7)[targets.to('cpu')]).float()
        logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations = self(image, targets)

        labels = target_vectors.argmax(dim=1)
        #
        labels = labels.to(device)
        #
        classification_loss_mask = self.classification_loss_fn(logits_mask, labels)
        classification_loss_inversed_mask = self.entropy_regularizer * entropy_loss(logits_inversed_mask)
        loss = classification_loss_mask + classification_loss_inversed_mask

        if self.use_mask_variation_loss:
            mask_variation_loss = (
                self.mask_variation_regularizer * 
                (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            )
#             loss += mask_variation_loss
            loss = loss + mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = (self.mask_area_constraint_regularizer *
                              self.class_mask_area_loss_fn(segmentations, target_vectors).to(device))
#             mask_area_loss += self.mask_total_area_regularizer * target_mask.mean()
#             mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
#             loss += mask_area_loss
            mask_area_loss = mask_area_loss.clone() + self.mask_total_area_regularizer * target_mask.mean().clone()
            mask_area_loss = mask_area_loss.clone() + self.ncmask_total_area_regularizer * non_target_mask.mean().clone()
            loss = loss.clone() + mask_area_loss.clone()

        self.log('train_loss', loss)
        self.train_metrics(logits_mask, targets)

        return loss

    def on_train_epoch_end(self):
        for k, v in self.train_metrics.compute().items():
            self.log(f'train_{k}', v)
#         self.log('train_metrics', self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        image, targets = batch
        target_vectors = torch.from_numpy(np.eye(7)[targets.to('cpu')]).float()
        logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations = self(image, targets)

        labels = target_vectors.argmax(dim=1)
        #
        labels = labels.to(device)
        #
        classification_loss_mask = self.classification_loss_fn(logits_mask, labels)
        classification_loss_inversed_mask = self.entropy_regularizer * entropy_loss(logits_inversed_mask)
        loss = classification_loss_mask + classification_loss_inversed_mask

        if self.use_mask_variation_loss:
            mask_variation_loss = (
                self.mask_variation_regularizer * 
                (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            )
#             loss += mask_variation_loss
            loss = loss + mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = (self.mask_area_constraint_regularizer *
                              self.class_mask_area_loss_fn(segmentations, target_vectors).to(device))
#             mask_area_loss += self.mask_total_area_regularizer * target_mask.mean()
#             mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
#             loss += mask_area_loss
            mask_area_loss = mask_area_loss.clone() + self.mask_total_area_regularizer * target_mask.mean().clone()
            mask_area_loss = mask_area_loss.clone() + self.ncmask_total_area_regularizer * non_target_mask.mean().clone()
            loss = loss.clone() + mask_area_loss.clone()

        self.log('val_loss', loss)
        self.valid_metrics(logits_mask, targets)
        
    def on_validation_epoch_end(self):
        for k, v in self.valid_metrics.compute().items():
            self.log(f'val_{k}', v)
            
#         self.log('val_metrics', self.valid_metrics.compute())
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        image, targets = batch
        target_vectors = torch.from_numpy(np.eye(7)[targets.to('cpu')]).float()
        logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations = self(image, targets)
        
        # codes for saving are removed...
        
        labels = target_vectors.argmax(dim=1)
        #
        labels = labels.to(device)
        #
        classification_loss_mask = self.classification_loss_fn(logits_mask, labels)
        classification_loss_inversed_mask = self.entropy_regularizer * entropy_loss(logits_inversed_mask)
        loss = classification_loss_mask + classification_loss_inversed_mask

        if self.use_mask_variation_loss:
            mask_variation_loss = (
                self.mask_variation_regularizer * 
                (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            )
#             loss += mask_variation_loss
            loss = loss + mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = (self.mask_area_constraint_regularizer *
                              self.class_mask_area_loss_fn(segmentations, target_vectors).to(device))
#             mask_area_loss += self.mask_total_area_regularizer * target_mask.mean()
#             mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
#             loss += mask_area_loss
            mask_area_loss = mask_area_loss.clone() + self.mask_total_area_regularizer * target_mask.mean().clone()
            mask_area_loss = mask_area_loss.clone() + self.ncmask_total_area_regularizer * non_target_mask.mean().clone()
            loss = loss.clone() + mask_area_loss.clone()

        self.log('test_loss', loss)
        self.test_metrics(logits_mask, targets)
            
    def on_test_epoch_end(self):
        for k, v in self.test_metrics.compute().items():
            self.log(f'test_{k}', v)

#         self.log('test_metrics', self.test_metrics.compute())
    
        # code for metrics saving removed
        
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
def extract_masks(segmentations, target_vectors):
    target_vectors = torch.from_numpy(np.eye(7)[target_vectors.to('cpu')]).float()

    batch_size, num_classes, h, w = segmentations.size()

    target_masks = torch.empty(batch_size, h, w, device=device)
    non_target_masks = torch.empty(batch_size, h, w, device=device)

    for i in range(batch_size):
        class_indices = target_vectors[i].eq(1.0) 
        non_class_indices = target_vectors[i].eq(0.0)

        target_masks[i] = (segmentations[i][class_indices]).amax(dim=0)
        non_target_masks[i] = (segmentations[i][non_class_indices]).amax(dim=0)

    return target_masks.sigmoid(), non_target_masks.sigmoid()

class TotalVariationConv(pl.LightningModule):
    def __init__(self):
        super().__init__()

        weights_right_variance = torch.tensor([[0.0, 0.0, 0.0],
                                              [0.0, 1.0, -1.0],
                                              [0.0, 0.0, 0.0]], device=self.device).view(1, 1, 3, 3)

        weights_down_variance = torch.tensor([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, -1.0, 0.0]], device=self.device).view(1, 1, 3, 3)

        self.variance_right_filter = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=3, padding=1, padding_mode='reflect', groups=1, bias=False)
        self.variance_right_filter.weight.data = weights_right_variance
        self.variance_right_filter.weight.requires_grad = False

        self.variance_down_filter = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=3, padding=1, padding_mode='reflect', groups=1, bias=False)
        self.variance_down_filter.weight.data = weights_down_variance
        self.variance_down_filter.weight.requires_grad = False

    def forward(self, mask):
        variance_right = self.variance_right_filter(mask.unsqueeze(1)).abs()

        variance_down = self.variance_down_filter(mask.unsqueeze(1)).abs()

        total_variance = (variance_right + variance_down).mean()
        return total_variance

class ClassMaskAreaLoss():
    def __init__(self, image_size=224, min_area=0.0, max_area=1.0):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.image_size = image_size
        self.min_area = min_area
        self.max_area = max_area

        assert(self.min_area >= 0.0 and self.min_area <= 1.0)
        assert(self.max_area >= 0.0 and self.max_area <= 1.0)
        assert(self.min_area <= self.max_area)        
        
        
    def __call__(self, segmentations, target_vectors):
        masks = segmentations.sigmoid()
        batch_size, num_classes, h, w = masks.size()
        losses = torch.zeros(batch_size, device=self.device)

        for i in range(batch_size):
            class_indices = target_vectors[i].eq(1.0)
            class_masks = masks[i][class_indices]

            mask = class_masks.flatten()
            sorted_mask, indices = mask.sort(descending=True)

            sorted_mask = sorted_mask.to(device)

            min_ones_length = (int)(self.image_size * self.image_size * self.min_area)
            min_ones = torch.ones(min_ones_length, device=self.device)
            min_zeros = torch.zeros((self.image_size * self.image_size) - min_ones_length,
                                device=self.device)
            min_ones_and_zeros = torch.cat((min_ones, min_zeros), dim=0)
            min_loss = F.relu(
                min_ones_and_zeros 
                - sorted_mask
            )

            max_ones_length = (int)(self.image_size * self.image_size * self.max_area)
            max_ones = torch.ones(max_ones_length, device=self.device)
            max_zeros = torch.zeros((self.image_size * self.image_size) - max_ones_length,
                                device=self.device)
            max_ones_and_zeros = torch.cat((max_ones, max_zeros), dim=0)
            max_loss = F.relu(
                max_ones_and_zeros 
                - sorted_mask
            )

            min_n_max_loss = (min_loss + max_loss).mean()

            losses[i] = losses[i] + min_n_max_loss

        mask_area_loss = losses.mean()
        return mask_area_loss    
    
def entropy_loss(logits):
    min_prob = 1e-16
    probs = F.softmax(logits, dim=-1).clamp(min=min_prob)
    log_probs = probs.log()
    entropy = (-probs * log_probs)
    entropy_loss = -entropy.mean()

    return entropy_loss