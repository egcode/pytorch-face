from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import math
import argparse
from sklearn import metrics
from losses.Arcface_loss import Arcface_loss
from losses.LMCL_loss import LMCL_loss
from losses.Center_loss import Center_loss
from dataset.get_data import get_data
from models.resnet import *
from models.irse import *
from lfw.lfw_pytorch import *
from lfw.lfw_helper import *
from helpers import *
from datetime import datetime, timedelta
import time
from logger import Logger
from pdb import set_trace as bp


'''
EXAMPLES:
## IR_50 TEST  #RESULT::::: IR_50_MODEL_centerloss_casia_epoch34.pth  
python3 train.py \
--model_path ./pth/IR_50_MODEL_centerloss.pth \
--loss_path ./pth/LOSS_centerloss.pth \
--batch_size 64 \
--batch_size_test 64 \
--lfw_batch_size 64 \
--criterion_type centerloss \
--model_lr 0.01 \
--model_lr_step 10 \
--model_lr_gamma 0.9 \
--criterion_lr 0.01 \
--criterion_lr_step 10 \
--criterion_lr_gamma 0.9 



'''


def train(ARGS, model, device, train_loader, loss_softmax, loss_criterion, optimizer_nn, optimzer_criterion, log_file_path, model_dir, logger, epoch):
    model.train()
    t = time.time()
    log_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        tt = time.time()

        data, target = data.to(device), target.to(device)

        features = model(data)

        if ARGS.criterion_type == 'arcface':
            logits = loss_criterion(features, target)
            loss = loss_softmax(logits, target)
        elif ARGS.criterion_type == 'lmcl':
            logits, mlogits = loss_criterion(features, target)
            loss = loss_softmax(mlogits, target)
        elif ARGS.criterion_type == 'centerloss':
            weight_cent = 1.
            loss_cent, outputs = loss_criterion(features, target)
            loss_cent *= weight_cent
            los_softm = loss_softmax(outputs, target)
            loss = los_softm + loss_cent


        optimizer_nn.zero_grad()
        optimzer_criterion.zero_grad()

        loss.backward()

        optimizer_nn.step()

        if ARGS.criterion_type == 'centerloss':
            # by doing so, weight_cent would not impact on the learning of centers
            for param in loss_criterion.parameters():
                param.grad.data *= (1. / weight_cent)

        optimzer_criterion.step()

        time_for_batch = int(time.time() - tt)

        log = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tbatch_time: {}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item(), timedelta(seconds=time_for_batch))
        print_and_log(log_file_path, log)

        log_loss = loss.item()

        # loss_epoch_and_percent - last two digits - Percent of epoch completed
        logger.scalar_summary("loss_epoch_and_percent", log_loss, (epoch*100)+(100. * batch_idx / len(train_loader)))

    logger.scalar_summary("loss", log_loss, epoch)

    time_for_epoch = int(time.time() - t)
    print_and_log(log_file_path, 'Total time for epoch: {}'.format(timedelta(seconds=time_for_epoch)))

    save_model(ARGS, ARGS.model_type, model_dir, model, log_file_path, epoch)
    save_model(ARGS, ARGS.criterion_type, model_dir, loss_criterion, log_file_path, epoch)

def test(ARGS, model, device, test_loader, loss_softmax, loss_criterion, log_file_path, logger, epoch):

    model.eval()
    correct = 0
    if epoch % ARGS.test_interval == 0 or epoch == ARGS.epochs:
        model.eval()
        t = time.time()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                feats = model(data)

                if ARGS.criterion_type == 'arcface':
                    logits = loss_criterion(feats, target)
                    outputs = logits
                elif ARGS.criterion_type == 'lmcl':
                    logits, _ = loss_criterion(feats, target)
                    outputs = logits
                elif ARGS.criterion_type == 'centerloss':
                    _, outputs = loss_criterion(feats, target)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == target.data).sum()
                

        accuracy = 100. * correct / len(test_loader.dataset)
        log = '\nTest set:, Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            accuracy)
        print_and_log(log_file_path, log)

        logger.scalar_summary("accuracy", accuracy, epoch)

        time_for_test = int(time.time() - t)
        print_and_log(log_file_path, 'Total time for test: {}'.format(timedelta(seconds=time_for_test)))


def validate_lfw(ARGS, model, lfw_loader, lfw_dataset, device, log_file_path, logger, lfw_distance_metric, epoch):
    if epoch % ARGS.lfw_interval == 0 or epoch == ARGS.epochs:
        model.eval()
        t = time.time()

        embedding_size = ARGS.features_dim

        tpr, fpr, accuracy, val, val_std, far = lfw_validate_model(model, lfw_loader, lfw_dataset, embedding_size, device,
                                                                    ARGS.lfw_nrof_folds, lfw_distance_metric, ARGS.lfw_subtract_mean)

        # print('\nEpoch: '+str(epoch))
        # print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        print_and_log(log_file_path, '\nEpoch: '+str(epoch))
        print_and_log(log_file_path, 'Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print_and_log(log_file_path, 'Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

        auc = metrics.auc(fpr, tpr)
        # print('Area Under Curve (AUC): %1.3f' % auc)
        print_and_log(log_file_path, 'Area Under Curve (AUC): %1.3f' % auc)

        # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        # print('Equal Error Rate (EER): %1.3f' % eer)
        time_for_lfw = int(time.time() - t)
        print_and_log(log_file_path, 'Total time for LFW evaluation: {}'.format(timedelta(seconds=time_for_lfw)))

        logger.scalar_summary("lfw_accuracy", np.mean(accuracy), epoch)
        # logger.scalar_summary("Validation_rate", val, epoch)
        # logger.scalar_summary("FAR", far, epoch)
        # logger.scalar_summary("Area_under_curve", auc, epoch)


def main(ARGS):

    # Dirs
    subdir = datetime.strftime(datetime.now(), '%Y-%m-%d___%H-%M-%S')
    out_dir = os.path.join(os.path.expanduser(ARGS.out_dir), subdir)
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)
    model_dir = os.path.join(os.path.expanduser(out_dir), 'model')
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    tensorboard_dir = os.path.join(os.path.expanduser(out_dir), 'tensorboard')
    if not os.path.isdir(tensorboard_dir):  # Create the tensorboard directory if it doesn't exist
        os.makedirs(tensorboard_dir)

    # stat_file_name = os.path.join(out_dir, 'stat.h5')

    # Write arguments to a text file
    write_arguments_to_file(ARGS, os.path.join(out_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    store_revision_info(src_path, out_dir, ' '.join(sys.argv))

    log_file_path = os.path.join(out_dir, 'training_log.txt')
    logger = Logger(tensorboard_dir)

    ################### Pytorch: ###################
    print_and_log(log_file_path, "Pytorch version:  " + str(torch.__version__))
    use_cuda = torch.cuda.is_available()
    print_and_log(log_file_path, "Use CUDA: " + str(use_cuda))

    device = torch.device("cuda" if use_cuda else "cpu")

    ####### Data setup
    print('Data directory: %s' % ARGS.data_dir)
    train_loader, test_loader = get_data(ARGS, device)

    ######## LFW Data setup
    print('LFW directory: %s' % ARGS.lfw_dir)
    lfw_dataset = LFW(lfw_dir=ARGS.lfw_dir, lfw_pairs=ARGS.lfw_pairs, input_size=ARGS.input_size)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=ARGS.lfw_batch_size, shuffle=False, num_workers=ARGS.num_workers)
    
    ####### Model setup
    print('Model type: %s' % ARGS.model_type)
    if ARGS.model_type == 'IR_50':
        model = IR_50(ARGS.input_size)
    if ARGS.model_type == 'IR_SE_50':
        model = IR_SE_50(ARGS.input_size)



    if ARGS.model_path != None:
        if use_cuda:
            model.load_state_dict(torch.load(ARGS.model_path))
        else:
            model.load_state_dict(torch.load(ARGS.model_path, map_location='cpu'))

    model = model.to(device)

    loss_softmax = nn.CrossEntropyLoss().to(device)

    ####### Criterion setup
    print('Criterion type: %s' % ARGS.criterion_type)
    if ARGS.criterion_type == 'arcface':
        lfw_distance_metric = 1
        loss_criterion = Arcface_loss(num_classes=train_loader.dataset.num_classes, feat_dim=ARGS.features_dim, device=device, s=ARGS.margin_s, m=ARGS.margin_m).to(device)
    elif ARGS.criterion_type == 'lmcl':
        lfw_distance_metric = 1
        loss_criterion = LMCL_loss(num_classes=train_loader.dataset.num_classes, feat_dim=ARGS.features_dim, device=device, s=ARGS.margin_s, m=ARGS.margin_m).to(device)
    elif ARGS.criterion_type == 'centerloss':
        lfw_distance_metric = 0
        loss_criterion = Center_loss(device=device, num_classes=train_loader.dataset.num_classes, feat_dim=ARGS.features_dim, use_gpu=use_cuda)

    if ARGS.loss_path != None:
        if use_cuda:
            loss_criterion.load_state_dict(torch.load(ARGS.loss_path))
        else:
            loss_criterion.load_state_dict(torch.load(ARGS.loss_path, map_location='cpu'))

    # optimzer nn
    # optimizer_nn = optim.SGD(model.parameters(), lr=ARGS.model_lr, momentum=0.9, weight_decay=0.0005)
    optimizer_nn = torch.optim.Adam(model.parameters(), lr=ARGS.model_lr, betas=(ARGS.beta1, 0.999))
    sheduler_nn = lr_scheduler.StepLR(optimizer_nn, ARGS.model_lr_step, gamma=ARGS.model_lr_gamma)

    # optimzer_criterion = optim.SGD(loss_criterion.parameters(), lr=ARGS.criterion_lr)
    optimzer_criterion = torch.optim.Adam(loss_criterion.parameters(), lr=ARGS.criterion_lr, betas=(ARGS.beta1, 0.999))
    sheduler_criterion = lr_scheduler.StepLR(optimzer_criterion, ARGS.criterion_lr_step, gamma=ARGS.criterion_lr_gamma)

    for epoch in range(1, ARGS.epochs + 1):
        sheduler_nn.step()
        sheduler_criterion.step()
        
        train(ARGS, model, device, train_loader, loss_softmax, loss_criterion, optimizer_nn, optimzer_criterion, log_file_path, model_dir, logger, epoch)
        test(ARGS, model, device, test_loader, loss_softmax, loss_criterion, log_file_path, logger, epoch)
        validate_lfw(ARGS, model, lfw_loader, lfw_dataset, device, log_file_path, logger, lfw_distance_metric, epoch)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # Out    
    parser.add_argument('--out_dir', type=str,  help='Directory where to trained models and event logs.', default='./out')
    # Training
    parser.add_argument('--epochs', type=int, help='Training epochs training.', default=200)
    # Data
    parser.add_argument('--input_size', type=str, help='support: [112, 112] and [224, 224]', default=[112, 112])
    parser.add_argument('--data_dir', type=str, help='Path to the data directory containing aligned face patches.', default='./data/CASIA_and_Golovan_160')
    parser.add_argument('--num_workers', type=int, help='Number of threads to use for data pipeline.', default=8)
    parser.add_argument('--batch_size', type=int, help='Number of batches while training model.', default=512)
    parser.add_argument('--batch_size_test', type=int, help='Number of batches while testing model.', default=512)
    parser.add_argument('--validation_set_split_ratio', type=float, help='The ratio of the total dataset to use for validation', default=0.05)
    parser.add_argument('--min_nrof_val_images_per_class', type=float, help='Classes with fewer images will be removed from the validation set', default=0)
    # Model
    parser.add_argument('--model_path', type=str, help='Model weights if needed.', default=None)
    parser.add_argument('--model_type', type=str, help='Model type to use for training.', default='IR_50')# support: 'ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152'
    parser.add_argument('--features_dim', type=int, help='Number of features for loss.', default=512)
    # Model Optimizer
    parser.add_argument('--model_lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--model_lr_step', type=int, help='Every step lr will be multiplied.', default=10)
    parser.add_argument('--model_lr_gamma', type=float, help='Every step lr will be multiplied by this value.', default=0.9)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    # Loss 
    parser.add_argument('--criterion_type', type=str, help='type of loss lmcl or arface.', default='centerloss')
    parser.add_argument('--loss_path', type=str, help='Loss weights if needed.', default=None)
    parser.add_argument('--margin_s', type=float, help='scale for feature.', default=64.0)
    parser.add_argument('--margin_m', type=float, help='margin for loss.', default=0.5)    
    # Loss Optimizer
    parser.add_argument('--criterion_lr', type=float, help='Learing rate of model optimizer.', default=0.01)
    parser.add_argument('--criterion_lr_step', type=int, help='Every step lr will be multiplied', default=10)
    parser.add_argument('--criterion_lr_gamma', type=float, help='Every step lr will be multiplied by this value', default=0.9)
    # Intervals
    parser.add_argument('--model_save_interval', type=int, help='Save model with every interval epochs.', default=1)
    parser.add_argument('--test_interval', type=int, help='Perform test with every interval epochs.', default=1)
    parser.add_argument('--lfw_interval', type=int, help='Perform LFW test with every interval epochs.', default=1)    
    # LFW
    parser.add_argument('--lfw_pairs', type=str, help='The file containing the pairs to use for validation.', default='lfw//pairs.txt')
    parser.add_argument('--lfw_dir', type=str, help='Path to the data directory containing aligned face patches.', default='./data/lfw_160')
    parser.add_argument('--lfw_batch_size', type=int, help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int, help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    # parser.add_argument('--lfw_distance_metric', type=int, help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--lfw_subtract_mean', help='Subtract feature mean before calculating distance.', action='store_true', default=False)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
