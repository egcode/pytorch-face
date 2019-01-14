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
from dataset.get_data import get_data
from models.net import Net
from models.resnet import *
from lfw.lfw_pytorch import *
from lfw.lfw_helper import *
from datetime import datetime
from six import iteritems
from subprocess import Popen, PIPE
from pdb import set_trace as bp


def train(args, model, device, train_loader, loss_softmax, loss_arcface, optimizer_nn, optimzer_arcface, log_file_path, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        features = model(data)
        logits = loss_arcface(features, target)
        loss = loss_softmax(logits, target)

        _, predicted = torch.max(logits.data, 1)
        accuracy = (target.data == predicted).float().mean()

        optimizer_nn.zero_grad()
        optimzer_arcface.zero_grad()

        loss.backward()

        optimizer_nn.step()
        optimzer_arcface.step()

        if batch_idx % args.log_interval == 0:
            log = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            print_and_log(log_file_path, log)

def test(args, model, device, test_loader, loss_softmax, loss_arcface, log_file_path, epoch):
    if epoch % args.test_interval == 0 or epoch == args.epochs:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                feats = model(data)
                logits = loss_arcface(feats, target)
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
                correct += (predicted == target.data).sum()

        log = '\nTest set:, Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset))
        print_and_log(log_file_path, log)


def validate_lfw(args, model, lfw_loader, lfw_dataset, device, log_file_path, epoch):
    if epoch % args.lfw_interval == 0 or epoch == args.epochs:
        model.eval()
        embedding_size = model.fc5.out_features

        tpr, fpr, accuracy, val, val_std, far = lfw_validate_model(model, lfw_loader, lfw_dataset, embedding_size, device,
                                                                    args.lfw_nrof_folds, args.lfw_distance_metric, args.lfw_subtract_mean)

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



###################################################################

def save_model(args, model_dir, model, type, log_file_path, epoch):
    if epoch % args.model_save_interval == 0 or epoch == args.epochs:
        save_name = os.path.join(model_dir, type + '_' + str(epoch) + '.pth')
        print_and_log(log_file_path, "Saving Model name: " + str(save_name))
        torch.save(model.state_dict(), save_name)        

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))

def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' +  e.strerror
  
    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' +  e.strerror
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('pytorch version: %s\n--------------------\n' % torch.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)

def print_and_log(log_file_path, string_to_write):
    print(string_to_write)
    with open(log_file_path, "a") as log_file:
        t = "[" + str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')) + "] " 
        log_file.write(t + string_to_write + "\n")

###################################################################

def main(args):

    # Dirs
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    out_dir = os.path.join(os.path.expanduser(args.out_dir), subdir)
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)
    model_dir = os.path.join(os.path.expanduser(out_dir), 'model')
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # stat_file_name = os.path.join(out_dir, 'stat.h5')

    # Write arguments to a text file
    write_arguments_to_file(args, os.path.join(out_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    store_revision_info(src_path, out_dir, ' '.join(sys.argv))

    log_file_path = os.path.join(out_dir, 'training_log.txt')

    ################### Pytorch: ###################
    print_and_log(log_file_path, "Pytorch version:  " + str(torch.__version__))
    use_cuda = torch.cuda.is_available()
    print_and_log(log_file_path, "Use CUDA: " + str(use_cuda))

    device = torch.device("cuda" if use_cuda else "cpu")

    ####### Data setup
    print('Data directory: %s' % args.data_dir)
    train_loader, test_loader = get_data(args.data_dir, device, args.num_workers, args.batch_size, args.batch_size_test)

    ######## LFW Data setup
    print('LFW directory: %s' % args.lfw_dir)
    lfw_dataset = LFW(lfw_dir=args.lfw_dir, lfw_pairs=args.lfw_pairs)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=args.lfw_batch_size, shuffle=False, num_workers=args.num_workers)
    
    ####### Model setup
    print('Model type: %s' % args.model_type)
    if args.model_type == 'resnet18':
        model = resnet18()
    elif args.model_type == 'resnet34':
        model = resnet34()
    elif args.model_type == 'resnet50':
        model = resnet50()
    
    # model = Net(features_dim=args.features_dim)
    model = model.to(device)

    loss_softmax = nn.CrossEntropyLoss().to(device)
    loss_arcface = Arcface_loss(num_classes=len(train_loader.dataset.classes), feat_dim=args.features_dim, device=device).to(device)

    # optimzer nn
    optimizer_nn = optim.SGD(model.parameters(), lr=args.model_lr, momentum=0.9, weight_decay=0.0005)
    # optimizer_nn = torch.optim.Adam(model.parameters(), lr=0.001)
    sheduler_nn = lr_scheduler.StepLR(optimizer_nn, args.model_lr_step, gamma=args.model_lr_gamma)

    # optimzer cosface or arcface
    optimzer_arcface = optim.SGD(loss_arcface.parameters(), lr=args.arcface_lr)
    sheduler_arcface = lr_scheduler.StepLR(optimzer_arcface, args.arcface_lr_step, gamma=args.arcface_lr_gamma)

    for epoch in range(1, args.epochs + 1):
        sheduler_nn.step()
        sheduler_arcface.step()
        
        train(args, model, device, train_loader, loss_softmax, loss_arcface, optimizer_nn, optimzer_arcface, log_file_path, epoch)
        save_model(args, model_dir, model, args.model_type, log_file_path, epoch)
        test(args, model, device, test_loader, loss_softmax, loss_arcface, log_file_path, epoch)
        validate_lfw(args, model, lfw_loader, lfw_dataset, device, log_file_path, epoch)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # Out    
    parser.add_argument('--out_dir', type=str, 
        help='Directory where to trained models and event logs.', default='./out')
    # Training
    parser.add_argument('--epochs', type=int,
        help='Training epochs training.', default=13)
    parser.add_argument('--log_interval', type=int,
        help='Print and write to logs every interval epochs.', default=10)
    # Data
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='../Computer-Vision/datasets/CASIA-WebFace_160')
    parser.add_argument('--num_workers', type=int,
        help='Number of threads to use for data pipeline.', default=4)
    parser.add_argument('--batch_size', type=int,
        help='Number of batches while training model.', default=64)
    parser.add_argument('--batch_size_test', type=int,
        help='Number of batches while testing model.', default=128)
    # Model
    parser.add_argument('--model_type', type=str,
        help='Model type to use for training.', default='resnet18')
    parser.add_argument('--features_dim', type=int,
        help='Number of features for arcface loss.', default=512)
    # Model Optimizer
    parser.add_argument('--model_lr', type=float,
        help='Learing rate of model optimizer.', default=0.001)
    parser.add_argument('--model_lr_step', type=int,
        help='Learing rate of model optimizer.', default=20)
    parser.add_argument('--model_lr_gamma', type=float,
        help='Learing rate of model optimizer.', default=0.1)
    # Loss Optimizer
    parser.add_argument('--arcface_lr', type=float,
        help='Learing rate of model optimizer.', default=0.01)
    parser.add_argument('--arcface_lr_step', type=int,
        help='Learing rate of model optimizer.', default=20)
    parser.add_argument('--arcface_lr_gamma', type=float,
        help='Learing rate of model optimizer.', default=0.1)
    # Intervals
    parser.add_argument('--model_save_interval', type=int,
        help='Save model with every interval epochs.', default=1)
    parser.add_argument('--test_interval', type=int,
        help='Perform test with every interval epochs.', default=1)
    parser.add_argument('--lfw_interval', type=int,
        help='Perform LFW test with every interval epochs.', default=1)
    # LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='lfw//pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='../Computer-Vision/datasets/lfw_160')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--lfw_distance_metric', type=int,
        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--lfw_subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true', default=False)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
