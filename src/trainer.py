# general
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import torch
from torch.functional import F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from src.models import ConvNet, VGG
from src.dataset import CustomDataset
from src.transforms import transformations
from src.utils import metrics
from fastprogress import progress_bar
from time import strftime
from torchsummary import summary
import logging

# logs to file
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', 
                    filename="logs/" + strftime("%b%d_%H-%M-%s.log"), # tensorboard format
                    filemode="w", 
                    level=logging.INFO
                    )
logger = logging.getLogger("train_log")

# logs to console
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("CUDA is available : {}".format(torch.cuda.is_available()))

class Trainer(object):
    def __init__(self, **config):
        self.best_f1 = 0.
        self.start_epoch = 0
        self.df = None
        self.splits = {}
        self.img_size = config.get("img_size")
        self.seed = config.get("seed", 42)
        self.with_cv = config["with_cv"]
        self.model_name = config.get("model")
        self.device = config["device"]
        self.paths = config["paths"]
        self.n_classes = config.get("n_classes")
        self.hyperparams = config["hyperparams"]
        self.mel_params = config["mel_params"]
        self.epochs = config["hyperparams"].get("epochs", 100)
        self.show_report = config.get("show_report")
        self.use_pretrained = config.get("use_pretrained")
        self.feature_extract = config.get("feature_extract")
        self.writer = SummaryWriter()

    def read_data(self):
        train = pd.read_csv(self.paths.get("train", "data/train.csv"))
        test = pd.read_csv(self.paths.get("test", "data/test.csv"))
        return train, test

    def prepare_data(self):
        self.df, _ = self.read_data()
        self.train_test_splits()
        train_idx, valid_idx = self.splits.get("train_idx"), self.splits.get("valid_idx")
        data = dict(train=self.df.iloc[train_idx,:], valid=self.df.iloc[valid_idx, :])
        # get train data and apply transformations
        transforms = transformations(self.img_size)
        datasets = {
            x: CustomDataset(
                data=data[x],
                transform=transforms[x],
            )
            for x in ["train", "valid"]
        }
        return datasets

    def train_test_splits(self):
        splits_path = self.paths.get("splits")
        if os.path.exists(splits_path):
            logger.info("==> splits found ..")
            with open(splits_path, "r") as f:
                self.splits = json.load(f)
        else:
            logger.info("==> splits not found .. creating ..")
            y = self.df.label
            X = self.df.loc[:, self.df.columns != "label"]
            X_train, X_test, _, _ = train_test_split(X, y,
                                                    random_state=self.seed, 
                                                    test_size=self.hyperparams.get("valid_pct"), 
                                                    stratify=y
                                                    )
            self.splits = dict(train_idx=X_train.index.tolist(), valid_idx=X_test.index.tolist())
            with open(self.paths.get("splits"), "w") as file:
                json.dump(self.splits, file)
    
    def prepare_dataloader(self, datasets, **kwargs):
        # if self.with_cv:
        #     train_idx, valid_idx = kwargs.get("splits")
        # else:
        #     self.train_test_splits()
        # num_train = len(dataset)
        # indices = list(range(num_train))
        # np.random.shuffle(indices)
        # split = int(np.floor(self.hyperparams.get("valid_size") * num_train))
        # train_idx, valid_idx = indices[split:], indices[:split]
        # define samplers for obtaining training and validation batches
        # train_idx, valid_idx = self.splits.get("train_idx"), self.splits.get("valid_idx")
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)
        dataloaders = (
            DataLoader(
                datasets[x], 
                batch_size=self.hyperparams.get("batch_size"),
                shuffle=True,
                num_workers=self.hyperparams.get("num_workers"),
                pin_memory=True,
                # collate_fn=collate_fn,
                drop_last=True
            )
            for x in ["train", "valid"]
        )
        # images, labels = next(iter(trainloader))
        return dataloaders

    def load_weights(self, last_checkpoint, model, optimizer, scheduler):
        # check for existing model.pt and load the same
        checkpoint = torch.load(last_checkpoint, map_location=self.device)
        best_model = torch.load(self.paths.get("best_pth"), map_location=self.device)
        self.best_f1 = best_model.get("f1_score")
        checkpoint_epoch = checkpoint.get("epoch")
        best_epoch = best_model.get("epoch")
        if int(checkpoint_epoch) > int(best_epoch):
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1 # to resume training from this epoch
            del best_model
            gc.collect()
        else:
            model.load_state_dict(best_model['model_state_dict'])
            optimizer.load_state_dict(best_model['optimizer_state_dict'])
            scheduler.load_state_dict(best_model['scheduler_state_dict'])
            self.start_epoch = best_model['epoch'] + 1 # to resume training from this epoch
            del checkpoint
            gc.collect()

        logger.info("==> checkpoint found .. last_epoch: {}".format(self.start_epoch))
        # load best model - get f1 score
        # self.best_f1 = checkpoint['f1_score']
        logger.info('==> best model found .. best_epoch: {}, best_f1: {:.6f}'.format(best_epoch, self.best_f1))
        return model, optimizer, scheduler
    
    def train(self, datasets, **kwargs):
        # run this to delete model and reclaim memory
        if "model" in locals(): 
            del model
            gc.collect()
            logger.info("==> Model deleted")
        
        avg_train_loss = []
        avg_valid_loss = []
        # obtain training indices that will be used for validation
        trainloader, validloader = self.prepare_dataloader(datasets, **kwargs)

        # intialize model, criterion, optimizer, scheduler
        elif self.model_name == "ConvNet":
            model = ConvNet(n_classes=self.n_classes)
        elif self.model_name == "VGG11bn":
            logger.info("==> use_pretrained : {}, feature_extract: {}".format(self.use_pretrained, self.feature_extract))
            model = VGG(n_classes=self.n_classes, 
                            use_pretrained=self.use_pretrained, 
                            feature_extract=self.feature_extract
                            )
        model.to(self.device)
        # note: always use BCEWithLogitsLoss instead of (F.sigmoid + BCELoss) for numerical stability
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()),
                                    lr=self.hyperparams.get("lr")
                                    # weight_decay=self.hyperparams.get("wd")
                                    )
        if self.hyperparams.get("scheduler") == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                T_max=self.hyperparams.get("t_max")
            )
        logger.info("==> model initialized : {}".format(self.model_name))
        # not to be used when doing cv
        checkpoints = os.listdir(self.paths.get("snap_pth"))
        if not self.with_cv and len(checkpoints) > 0:
            last_checkpoint = self.paths.get("snap_pth") + checkpoints[0] # sorted(checkpoints, reverse=True)[0]
            model, optimizer, scheduler = self.load_weights(last_checkpoint, model, optimizer, scheduler)
        else:
            logger.info('==> No checkpoints found / training from scratch ..')
        
        logger.info('==> Training started ..')
        for epoch in range(self.start_epoch, self.epochs):
            logger.info('**********************\n')
            # keep track of training and validation loss
            train_loss, valid_loss = 0., 0.
            train_epoch_loss, valid_epoch_loss = [], []
            # training mode
            model.train()
            # scheduler.step()
            logger.info('==> Current LR: {}'.format(scheduler.get_lr()))
            for step, (images, labels) in enumerate(progress_bar(trainloader)):
                # one-hot encode labels
                labels = labels.unsqueeze(-1)
                targets = torch.zeros(labels.size(0), model.n_classes).scatter_(1, labels, 1.)
                inputs, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad() # clear gradients
                outputs = model(inputs) # forward pass
                loss = criterion(outputs, targets) # compute loss
                loss.backward() # backward pass 
                optimizer.step() # weight update
                train_loss += loss.item()
                train_epoch_loss.append(loss.item())
                # ===================log========================
                if (step + 1) % 50 == 0:
                    logger.info(
                        "step: [{}/{}], epochs: [{}/{}], train loss: {:.4f}".format(
                            step + 1,
                            len(trainloader),
                            epoch + 1,
                            self.epochs,
                            train_loss/50,
                        )
                    )
                    train_loss = 0.
                # ==============================================

            avg_train_loss.append(np.mean(train_epoch_loss))
            self.writer.add_scalar('loss/training', avg_train_loss[-1], epoch+1)
            logger.info('==> Validation ..')
            # validation
            with torch.no_grad(): # turn off gradient calc
                model.eval() # evaluation mode
                y_true, y_pred = [], []
                for step, (images, labels) in enumerate(progress_bar(validloader)):
                    # one-hot encode labels
                    labels = labels.unsqueeze(-1)
                    targets = torch.zeros(labels.size(0), model.n_classes).scatter_(1, labels, 1.)
                    inputs, targets = images.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    preds = F.sigmoid(outputs) # get the sigmoid from raw logits
                    # pred = torch.argmax(output, dim=1)
                    y_true.extend(targets.cpu().detach().numpy().tolist())
                    y_pred.extend(preds.cpu().detach().numpy().tolist())
                    valid_loss += loss.item()
                    valid_epoch_loss.append(loss.item())
                    # self.writer.add_scalar('Valid loss', loss.item(), step + 1)
                    # ===================log========================
                    if (step + 1) % 50 == 0:
                        logger.info(
                            "step: [{}/{}], epochs: [{}/{}], validation loss: {:.4f}".format(
                                step + 1,
                                len(validloader),
                                epoch + 1,
                                self.epochs,
                                valid_loss/50,
                            )
                        )
                        valid_loss = 0.
                    # ==============================================
            
            # metric e.g.: row-wise micro averaged F1 score
            y_pred = np.asarray(y_pred, dtype=np.float32)
            y_true = np.asarray(y_true, dtype=np.float32)
            micro_avg_f1, cls_report, mAP, auc_score = metrics(y_true=y_true, y_pred=y_pred, show_report=self.show_report, threshold=self.threshold)
            
            scheduler.step() # call/update the scheduler

            if self.show_report: logger.info("==> classification Report: \n{}".format(cls_report))
            logger.info("==> mAP : {}".format(mAP))
            logger.info("==> AUC-ROC score : {}".format(auc_score))
            logger.info('==> epoch: [{}/{}], validation F1-score: {:.6f}'.format(epoch+1, self.epochs, micro_avg_f1))
            avg_valid_loss.append(np.mean(valid_epoch_loss)) # update average validation loss

            self.writer.add_scalar('loss/validation', avg_valid_loss[-1], epoch+1)
            self.writer.add_scalar('mAP', mAP, epoch+1)
            self.writer.add_scalar('AUC_ROC', auc_score, epoch+1)
            self.writer.add_scalar('valid F1', micro_avg_f1, epoch+1)
            # save model if validation loss has decreased
            if micro_avg_f1 >= self.best_f1:
                self.writer.add_scalar('Best Valid F1', micro_avg_f1, epoch+1)
                logger.info('==> Validation F1 has increased ({:.6f} --> {:.6f}) / Saving model..'.format(
                    self.best_f1,
                    micro_avg_f1
                    )
                )
                self.best_f1 = micro_avg_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'scheduler_state_dict' : scheduler.state_dict(),
                    'train_loss' : avg_train_loss[-1],
                    'valid_loss': avg_valid_loss[-1],
                    'f1_score' : self.best_f1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, self.paths.get("best_pth")
                )
                logger.info('==> Best model saved!')
            # save a snapshot every 10 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'scheduler_state_dict' : scheduler.state_dict(),
                    'train_loss' : avg_train_loss[-1],
                    'valid_loss': avg_valid_loss[-1],
                    'f1_score' : micro_avg_f1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, self.paths.get("snap_pth") + "snapshot.pt"
                )
                logger.info('==> Model snapshot saved!')
            logger.info('**********************\n')
        torch.cuda.empty_cache()
        
    def run(self):
        """Main training function
        1. Split to K folds (stratified)
        2. Train model for max epochs from scratch for each fold
        3. Save the best F1 score model
        """
        datasets = self.prepare_data()
        if self.with_cv:
            logger.info("==> training w/ Kfold CV ..")
            # skfold = StratifiedKFold(n_splits=self.hyperparams["cv"].get("splits"), shuffle=True, random_state=42)
            # # note: for X we can simply pass a tensor of zeros 
            # # source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold.split
            # for fold, (train_idx, valid_idx) in enumerate(skfold.split(torch.zeros(len(dataset)) , dataset.data.label)):
            #     logger.info("Fold : [{}/{}]".format(fold + 1, self.hyperparams["cv"].get("splits")))
            #     kwargs = {"splits" : (train_idx, valid_idx)}
            #     self.train(dataset, **kwargs)
        else:
            logger.info("==> training w/o Kfold CV ..")
            self.train(datasets)