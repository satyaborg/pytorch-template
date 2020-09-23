import os
import numpy as np
import random
import torch
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix, classification_report, average_precision_score, f1_score

def set_random_seeds(seed: int = 42):
    random.seed(seed) # python random seed 
    np.random.seed(seed) # numpy random seeda
    os.environ["PYTHONHASHSEED"] = str(seed)# environ seeds
    torch.manual_seed(seed) # pytorch seed
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

# to clear tensors from GPU
def del_cuda_tensor(x):
    del x
    torch.cuda.empty_cache()

def get_stats(trainloader):
    """return per channel mean, stdev to be used for 
    standardizing
    Note: Always use trainset parameters (mean, stdev) to standardize the testset
    originally from: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
    modified from: https://stackoverflow.com/a/60803379/8277194
    """
    n_images = 0
    mean = 0.0
    var = 0.0
    for batch, _ in trainloader:
        # print(batch_target.max(), batch_target.min())
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        n_images += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)

    mean /= n_images
    var /= n_images
    std = torch.sqrt(var) # cannot take the average of individual std!
    return mean, std

def metrics(y_true, y_pred, show_report=False, threshold=0.5):
    """return a AUC-ROC score
    """
    cls_report = None
    y_scores = np.where(y_pred > threshold, 1, 0)
    micro_avg_f1 = f1_score(y_true, y_scores, average='samples')
    if show_report: cls_report = classification_report(y_true,y_scores) 
    mAP = average_precision_score(y_true, y_scores, average="samples")
    auc_score = roc_auc_score(y_true, y_scores, average="samples")
    return micro_avg_f1, cls_report, mAP, auc_score

def show_img(dataset, i, dim=0):
    # print(dataset[i].shape)
    img, label = dataset[i]
    print("shape: {}, label: {}".format(img.shape, label))
    print("max: {}, min: {}, mean: {}, std: {}".format(img.max(), 
                                                        img.min(),
                                                        img.mean(),
                                                        img.std()))
    plt.imshow(img[dim,:,:]) # , cmap="bwr");

def get_out_size(width_in: int, kernel_size: int, stride: int, padding: int):
    # https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    return ((width_in - kernel_size + 2 * padding)/stride) + 1

def get_padding(width_in: int, width_out: int, kernel_size: int, stride: int):
    # (kernel_size - 1)//2
    return (stride * (width_out - 1) + kernel_size - width_in) // 2

def check_npy(path):
    """check if npy files are valid"""
    try:
        np.load(path, allow_pickle=True)
    except Exception as e:
        print(e)
        pass # or return a empty np.ndarray

    
    