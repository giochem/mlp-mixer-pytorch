import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from model import MLPMixer
from torchinfo import summary
from datetime import datetime
from sklearn import metrics
import numpy as np
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: MLP-Mixer: An all-MLP Architecture for Vision""")
    home_dir = os.getcwd()
    # Dataset and DataLoader
    parser.add_argument("--dataset", default='other', choices=["CIFAR10", "other"], type=str)
    parser.add_argument("--train-folder", default='{}/data/train'.format(home_dir), type=str)
    parser.add_argument("--valid-folder", default='{}/data/valid'.format(home_dir), type=str)
    # Images size
    parser.add_argument("--image-size", default=300, type=int)
    parser.add_argument("--image-channels", default=3, type=int)
    # Hyperparam model
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--tokens-mlp-dim", default=2048, type=int, help='Token-mixing units')
    parser.add_argument("--channels-mlp-dim", default=256, type=int, help='Channel-mixing units')
    parser.add_argument("--hidden-dim", default=512, type=int, help='Projection units (hidden dim)')
    parser.add_argument("--patch-size", default=100, type=int)
    parser.add_argument("--num-of-mlp-blocks", default=4, type=int)
    # Loss function and optimizer
    parser.add_argument("--optimizer",choices=["adam", "sgd"], type=str, default="adam")
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    # Training
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=3,
            help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    
    # SummaryWriter
    parser.add_argument("--log-path", default="tensorboard/mlp-mixer", type=str)
    parser.add_argument("--model-folder", default='{}/model/mlp'.format(home_dir), type=str)

    args = parser.parse_args()
    
    print('Training MLP-Mixer model with hyper-params: ')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')
    
    return args


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def train(opt):
    Path(opt.model_folder).mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    train_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "num_workers": 0}
    valid_params = {"batch_size": opt.batch_size,
                       "shuffle": False,
                       "num_workers": 0}
    train_transform = transforms.Compose(
        [transforms.Resize(size=(opt.image_size, opt.image_size)),
         transforms.ToTensor()
         ])
    test_transform = transforms.Compose(
        [transforms.Resize(size=(opt.image_size, opt.image_size)),
         transforms.ToTensor(),
         ])
    
    if opt.dataset == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10('./data', train=True, transform=train_transform, download=True)
        valid_set = torchvision.datasets.CIFAR10('./data', train=False, transform=test_transform, download=True)
    else:
        train_set = torchvision.datasets.ImageFolder(opt.train_folder, transform=train_transform)
        valid_set = torchvision.datasets.ImageFolder(opt.valid_folder, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, **train_params)
    valid_loader = torch.utils.data.DataLoader(valid_set, **valid_params)

    assert opt.image_size * opt.image_size % ( opt.patch_size * opt.patch_size) == 0, 'Make sure that image-size is divisible by patch-size'
    assert opt.image_channels == 3, 'Unfortunately, model accepts jpg images with 3 channels so far'
    # S = (opt.image_size * opt.image_size) // (opt.patch_size * opt.patch_size)
    # C = opt.patch_size * opt.patch_size * opt.image_channels
    
    # Initializing model
    mlpmixer = MLPMixer(patch_size=opt.patch_size, c=opt.image_channels, width=opt.image_size, height=opt.image_size, 
                        num_classes=opt.num_classes, num_blocks=opt.num_of_mlp_blocks, hidden_dim=opt.hidden_dim,
                        tokens_mlp_dim=opt.tokens_mlp_dim, channels_mlp_dim=opt.channels_mlp_dim)
    if torch.cuda.is_available():
        mlpmixer.cuda()
    # summary(mlpmixer, input_size=(16, opt.image_channels, opt.image_size, opt.image_size))

    # Set up loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Optimizer Definition
    optimizer = torch.optim.Adam(mlpmixer.parameters(), lr=opt.learning_rate)
    timestamp = datetime.now().strftime('%H%M%S-%d%m%Y')
    log_path = "{}_{}".format(opt.log_path, timestamp)
    writer = SummaryWriter(log_path)
    best_loss = 1e5
    best_epoch = 0

    mlpmixer.train()
    
    num_iter_per_epoch = len(train_loader)

    for epoch in range(opt.epochs):
            for iter, batch in enumerate(train_loader):
                feature, label = batch
                if torch.cuda.is_available():
                    feature = feature.cuda()
                    label = label.cuda()

                optimizer.zero_grad()
                predictions = mlpmixer(feature)
                loss = loss_fn(predictions, label)
                loss.backward()
                optimizer.step()

                train_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(),
                                                list_metrics=["accuracy"])
                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                    epoch + 1,
                    opt.epochs,
                    iter + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss, train_metrics["accuracy"]))
                
                writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Accuracy', train_metrics["accuracy"], epoch * num_iter_per_epoch + iter)

            mlpmixer.eval()
            loss_epochs = []
            valid_label_epoch = []
            valid_pred_epoch = []
            for batch in valid_loader:
                valid_feature, valid_label = batch
                num_sample = len(valid_label)
                if torch.cuda.is_available():
                    valid_feature = valid_feature.cuda()
                    valid_label = valid_label.cuda()

                with torch.no_grad():
                    valid_predictions = mlpmixer(valid_feature)

                valid_loss = loss_fn(valid_predictions, valid_label)
                loss_epochs.append(valid_loss * num_sample)

                valid_label_epoch.extend(valid_label.clone().cpu())
                valid_pred_epoch.append(valid_predictions.clone().cpu())

            avg_valid_loss = sum(loss_epochs) / valid_set.__len__()
            valid_metrics = get_evaluation(np.array(valid_label_epoch), np.array(torch.cat(valid_pred_epoch, 0)), list_metrics=["accuracy", "confusion_matrix"])
           
            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.epochs,
                optimizer.param_groups[0]['lr'],
                avg_valid_loss, valid_metrics["accuracy"]))
            
            writer.add_scalar('Test/Loss', avg_valid_loss, epoch)
            writer.add_scalar('Test/Accuracy', valid_metrics["accuracy"], epoch)

            mlpmixer.train()
            if avg_valid_loss + opt.es_min_delta < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                torch.save(mlpmixer, "{}/e_{}".format(opt.model_folder, epoch))
            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(epoch, valid_loss, best_epoch))
                break
            if opt.optimizer == "adam" and epoch % 3 == 0 and epoch > 0:
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                current_lr /= 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr    

if __name__ == "__main__":
    opt = get_args()
    train(opt)
# tensorboard --logdir=runs
# python --dataset=CIFAR10 --valid-folder= run.py --image-size=32 --patch-size=4 --num-classes=10 --epochs=2 --num-of-mlp-blocks=1