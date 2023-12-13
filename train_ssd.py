# #!/usr/bin/env python3
# #
# # train an SSD detection model on Pascal VOC or Open Images datasets
# # https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md
# #
# import os
# import sys
# import logging
# import argparse
# import datetime
# import itertools
# import torch
# import torch.nn.functional as F
# import cv2

# from torch.utils.data import DataLoader, ConcatDataset
# from torch.utils.tensorboard import SummaryWriter
# from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

# from vision.utils.misc import Timer, freeze_net_layers, store_labels
# from vision.ssd.ssd import MatchPrior
# from vision.ssd.vgg_ssd import create_vgg_ssd
# from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
# from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
# from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
# from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
# from vision.datasets.voc_dataset import VOCDataset
# from vision.datasets.open_images import OpenImagesDataset
# from vision.nn.multibox_loss import MultiboxLoss
# from vision.ssd.config import vgg_ssd_config
# from vision.ssd.config import mobilenetv1_ssd_config
# from vision.ssd.config import squeezenet_ssd_config
# from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

# from eval_ssd import MeanAPEvaluator


# DEFAULT_PRETRAINED_MODEL='models/mobilenet-v1-ssd-mp-0_675.pth'


# parser = argparse.ArgumentParser(
#     description='Single Shot MultiBox Detector Training With PyTorch')

# # Params for datasets
# parser.add_argument("--dataset-type", default="open_images", type=str,
#                     help='Specify dataset type. Currently supports voc and open_images.')
# parser.add_argument('--datasets', '--data', nargs='+', default=["data"], help='Dataset directory path')
# parser.add_argument('--balance-data', action='store_true',
#                     help="Balance training data by down-sampling more frequent labels.")

# # Params for network
# parser.add_argument('--net', default="mb1-ssd",
#                     help="The network architecture, it can be mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
# parser.add_argument('--resolution', type=int, default=300,
#                     help="the NxN pixel resolution of the model (can be changed for mb1-ssd only)")
# parser.add_argument('--freeze-base-net', action='store_true',
#                     help="Freeze base net layers.")
# parser.add_argument('--freeze-net', action='store_true',
#                     help="Freeze all the layers except the prediction head.")
# parser.add_argument('--mb2-width-mult', default=1.0, type=float,
#                     help='Width Multiplifier for MobilenetV2')

# # Params for loading pretrained basenet or checkpoints.
# parser.add_argument('--base-net', help='Pretrained base model')
# parser.add_argument('--pretrained-ssd', default=DEFAULT_PRETRAINED_MODEL, type=str, help='Pre-trained base model')
# parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')

# # Params for SGD
# parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
#                     help='initial learning rate')
# parser.add_argument('--momentum', default=0.9, type=float,
#                     help='Momentum value for optim')
# parser.add_argument('--weight-decay', default=5e-4, type=float,
#                     help='Weight decay for SGD')
# parser.add_argument('--gamma', default=0.1, type=float,
#                     help='Gamma update for SGD')
# parser.add_argument('--base-net-lr', default=0.001, type=float,
#                     help='initial learning rate for base net, or None to use --lr')
# parser.add_argument('--extra-layers-lr', default=None, type=float,
#                     help='initial learning rate for the layers not in base net and prediction heads.')

# # Scheduler
# parser.add_argument('--scheduler', default="cosine", type=str,
#                     help="Scheduler for SGD. It can one of multi-step and cosine")

# # Params for Multi-step Scheduler
# parser.add_argument('--milestones', default="80,100", type=str,
#                     help="milestones for MultiStepLR")

# # Params for Cosine Annealing
# parser.add_argument('--t-max', default=100, type=float,
#                     help='T_max value for Cosine Annealing Scheduler.')

# # Train params
# parser.add_argument('--batch-size', default=4, type=int,
#                     help='Batch size for training')
# parser.add_argument('--num-epochs', '--epochs', default=30, type=int,
#                     help='the number epochs')
# parser.add_argument('--num-workers', '--workers', default=2, type=int,
#                     help='Number of workers used in dataloading')
# parser.add_argument('--validation-epochs', default=1, type=int,
#                     help='the number epochs between running validation')
# parser.add_argument('--validation-mean-ap', action='store_true',
#                     help='Perform computation of Mean Average Precision (mAP) during validation')
# parser.add_argument('--debug-steps', default=10, type=int,
#                     help='Set the debug log output frequency.')
# parser.add_argument('--use-cuda', default=True, action='store_true',
#                     help='Use CUDA to train model')
# parser.add_argument('--checkpoint-folder', '--model-dir', default='models/',
#                     help='Directory for saving checkpoint models')
# parser.add_argument('--log-level', default='info', type=str,
#                     help='Logging level, one of:  debug, info, warning, error, critical (default: info)')
                                        
# args = parser.parse_args()

# logging.basicConfig(stream=sys.stdout, level=getattr(logging, args.log_level.upper(), logging.INFO),
#                     format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
                    
# tensorboard = SummaryWriter(log_dir=os.path.join(args.checkpoint_folder, "tensorboard", f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

# if args.use_cuda and torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True
#     logging.info("Using CUDA...")


# def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
#     net.train(True)
    
#     train_loss = 0.0
#     train_regression_loss = 0.0
#     train_classification_loss = 0.0
    
#     running_loss = 0.0
#     running_regression_loss = 0.0
#     running_classification_loss = 0.0
    
#     num_batches = 0
    
#     for i, data in enumerate(loader):
#         images, boxes, labels = data
#         images = images.to(device)
#         boxes = boxes.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         confidence, locations = net(images)
#         regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
#         loss = regression_loss + classification_loss
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         train_regression_loss += regression_loss.item()
#         train_classification_loss += classification_loss.item()
        
#         running_loss += loss.item()
#         running_regression_loss += regression_loss.item()
#         running_classification_loss += classification_loss.item()

#         if i and i % debug_steps == 0:
#             avg_loss = running_loss / debug_steps
#             avg_reg_loss = running_regression_loss / debug_steps
#             avg_clf_loss = running_classification_loss / debug_steps
#             logging.info(
#                 f"Epoch: {epoch}, Step: {i}/{len(loader)}, " +
#                 f"Avg Loss: {avg_loss:.4f}, " +
#                 f"Avg Regression Loss {avg_reg_loss:.4f}, " +
#                 f"Avg Classification Loss: {avg_clf_loss:.4f}"
#             )
#             running_loss = 0.0
#             running_regression_loss = 0.0
#             running_classification_loss = 0.0

#         num_batches += 1
        
#     train_loss /= num_batches
#     train_regression_loss /= num_batches
#     train_classification_loss /= num_batches
    
#     logging.info(
#         f"Epoch: {epoch}, " +
#         f"Training Loss: {train_loss:.4f}, " +
#         f"Training Regression Loss {train_regression_loss:.4f}, " +
#         f"Training Classification Loss: {train_classification_loss:.4f}"
#     )
     
#     tensorboard.add_scalar('Loss/train', train_loss, epoch)
#     tensorboard.add_scalar('Regression Loss/train', train_regression_loss, epoch)
#     tensorboard.add_scalar('Classification Loss/train', train_classification_loss, epoch)

# def test(loader, net, criterion, device):
#     net.eval()
#     running_loss = 0.0
#     running_regression_loss = 0.0
#     running_classification_loss = 0.0
#     num = 0
#     for _, data in enumerate(loader):
#         images, boxes, labels = data
#         images = images.to(device)
#         boxes = boxes.to(device)
#         labels = labels.to(device)
#         num += 1

#         with torch.no_grad():
#             confidence, locations = net(images)
#             regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
#             loss = regression_loss + classification_loss

#         running_loss += loss.item()
#         running_regression_loss += regression_loss.item()
#         running_classification_loss += classification_loss.item()
    
#     return running_loss / num, running_regression_loss / num, running_classification_loss / num

# def run_inference(loader, net, device, output_dir):
#     ####### Your code here ######
#     net.eval()  # Set the network to evaluation mode

#     for i, (images, _) in enumerate(loader):
#         images = images.to(device)  # Move the images to the device

#         with torch.no_grad():
#             output = net(images)  # Get the output from the network

#         for j, image in enumerate(images):
#             boxes, labels, scores = output[j]  # Get the bounding boxes, labels, and scores

#             # Convert the image from tensor to numpy array and change the color space from BGR to RGB
#             image = cv2.cvtColor(F.to_pil_image(image.cpu()), cv2.COLOR_BGR2RGB)

#             # # Draw the bounding boxes on the image
#             # for box in boxes:
#             #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

#             # Draw bounding boxes on the rectangle and their corresponding scores
#             for box, score in zip(boxes, scores):
#                 cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
#                 cv2.putText(image, f"{score:.2f}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

#             # Save the image to the output directory
#             cv2.imwrite(os.path.join(output_dir, f'image_{i}_{j}.jpg'), image)
#     ####### Your code here ######

# if __name__ == '__main__':
#     timer = Timer()

#     logging.info(args)
    
#     # make sure that the checkpoint output dir exists
#     if args.checkpoint_folder:
#         args.checkpoint_folder = os.path.expanduser(args.checkpoint_folder)

#         if not os.path.exists(args.checkpoint_folder):
#             os.mkdir(args.checkpoint_folder)
            
#     # ######### My Code #########
#     # # tune learning rate
#     for my_learning_rate in [0.001, 0.0001, 0.00001]:

#         logging.info("#" * 50)
#         logging.info(f"Learning rate: {my_learning_rate}")
#         logging.info("#" * 50)

#         for my_weight_decay in [0.0005, 0.0001, 0.00005]:

#             logging.info("#" * 50)
#             logging.info(f"Weight Decay: {my_weight_decay}")
#             logging.info("#" * 50)

#             for my_epoch in [30, 50, 80, 100]:

#                 logging.info("#" * 50)
#                 logging.info(f"Epochs: {my_epoch}")
#                 logging.info("#" * 50)

#     # ######### My Code #########

#                 args.checkpoint_folder = os.path.join('models', "mb2_ssd_lite_hp_tuned")

#                 # Make Seperate Folders for Each Learning Rate
#                 if args.checkpoint_folder:
#                     args.checkpoint_folder = os.path.join(args.checkpoint_folder, f"lr-{my_learning_rate}_wd-{my_weight_decay}_epoch-{my_epoch}")
                    
#                     if not os.path.exists(args.checkpoint_folder):
#                         os.mkdir(args.checkpoint_folder)
                
#                 # select the network architecture and config     
#                 if args.net == 'vgg16-ssd':
#                     create_net = create_vgg_ssd
#                     config = vgg_ssd_config
#                 elif args.net == 'mb1-ssd':
#                     create_net = create_mobilenetv1_ssd
#                     config = mobilenetv1_ssd_config
#                     config.set_image_size(args.resolution)
#                 elif args.net == 'mb1-ssd-lite':
#                     create_net = create_mobilenetv1_ssd_lite
#                     config = mobilenetv1_ssd_config
#                 elif args.net == 'sq-ssd-lite':
#                     create_net = create_squeezenet_ssd_lite
#                     config = squeezenet_ssd_config
#                 elif args.net == 'mb2-ssd-lite':
#                     create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
#                     config = mobilenetv1_ssd_config
#                 else:
#                     logging.fatal("The net type is wrong.")
#                     parser.print_help(sys.stderr)
#                     sys.exit(1)
                    
#                 # create data transforms for train/test/val
#                 train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
#                 target_transform = MatchPrior(config.priors, config.center_variance,
#                                             config.size_variance, 0.5)

#                 test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

#                 # load datasets (could be multiple)
#                 logging.info("Prepare training datasets.")
#                 datasets = []
#                 for dataset_path in args.datasets:
#                     if args.dataset_type == 'voc':
#                         dataset = VOCDataset(dataset_path, transform=train_transform,
#                                             target_transform=target_transform)
#                         label_file = os.path.join(args.checkpoint_folder, "labels.txt")
#                         store_labels(label_file, dataset.class_names)
#                         num_classes = len(dataset.class_names)
#                     elif args.dataset_type == 'open_images':
#                         dataset = OpenImagesDataset(dataset_path,
#                             transform=train_transform, target_transform=target_transform,
#                             dataset_type="train", balance_data=args.balance_data)
#                         label_file = os.path.join(args.checkpoint_folder, "labels.txt")
#                         store_labels(label_file, dataset.class_names)
#                         logging.info(dataset)
#                         num_classes = len(dataset.class_names)

#                     else:
#                         raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
#                     datasets.append(dataset)
                    
#                 # create training dataset
#                 logging.info(f"Stored labels into file {label_file}.")
#                 train_dataset = ConcatDataset(datasets)
#                 logging.info("Train dataset size: {}".format(len(train_dataset)))
#                 train_loader = DataLoader(train_dataset, args.batch_size,
#                                         num_workers=args.num_workers,
#                                         shuffle=True)
                                    
#                 # create validation dataset                           
#                 logging.info("Prepare Validation datasets.")
#                 if args.dataset_type == "voc":
#                     val_dataset = VOCDataset(dataset_path, transform=test_transform,
#                                             target_transform=target_transform, is_test=True)
#                 elif args.dataset_type == 'open_images':
#                     val_dataset = OpenImagesDataset(dataset_path,
#                                                     transform=test_transform, target_transform=target_transform,
#                                                     dataset_type="test")
#                     logging.info(val_dataset)
#                 logging.info("Validation dataset size: {}".format(len(val_dataset)))

#                 val_loader = DataLoader(val_dataset, args.batch_size,
#                                         num_workers=args.num_workers,
#                                         shuffle=False)
                

                                
#                 # create the network
#                 logging.info("Build network.")
#                 net = create_net(num_classes)
#                 min_loss = -10000.0
#                 last_epoch = -1

#                 # prepare eval dataset (for mAP computation)
#                 if args.validation_mean_ap:
#                     if args.dataset_type == "voc":
#                         eval_dataset = VOCDataset(dataset_path, is_test=True)
#                     elif args.dataset_type == 'open_images':
#                         eval_dataset = OpenImagesDataset(dataset_path, dataset_type="test")
#                     eval = MeanAPEvaluator(eval_dataset, net, arch=args.net, eval_dir=os.path.join(args.checkpoint_folder, 'eval_results'))

#             # ######### My Code #########
#             # # tune learning rate
#             # for my_learning_rate in [0.1, 0.01, 0.001, 0.0001]:
#             # ######### My Code #########

#                 # logging.info("#" * 50)
#                 # logging.info(f"Learning rate: {my_learning_rate}")
#                 # logging.info("#" * 50)

#                 # freeze certain layers (if requested)
#                 base_net_lr = args.base_net_lr if args.base_net_lr is not None else my_learning_rate
#                 extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else my_learning_rate
                
#                 if args.freeze_base_net:
#                     logging.info("Freeze base net.")
#                     freeze_net_layers(net.base_net)
#                     params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
#                                             net.regression_headers.parameters(), net.classification_headers.parameters())
#                     params = [
#                         {'params': itertools.chain(
#                             net.source_layer_add_ons.parameters(),
#                             net.extras.parameters()
#                         ), 'lr': extra_layers_lr},
#                         {'params': itertools.chain(
#                             net.regression_headers.parameters(),
#                             net.classification_headers.parameters()
#                         )}
#                     ]
#                 elif args.freeze_net:
#                     freeze_net_layers(net.base_net)
#                     freeze_net_layers(net.source_layer_add_ons)
#                     freeze_net_layers(net.extras)
#                     params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
#                     logging.info("Freeze all the layers except prediction heads.")
#                 else:
#                     params = [
#                         {'params': net.base_net.parameters(), 'lr': base_net_lr},
#                         {'params': itertools.chain(
#                             net.source_layer_add_ons.parameters(),
#                             net.extras.parameters()
#                         ), 'lr': extra_layers_lr},
#                         {'params': itertools.chain(
#                             net.regression_headers.parameters(),
#                             net.classification_headers.parameters()
#                         )}
#                     ]

#                 # load a previous model checkpoint (if requested)
#                 timer.start("Load Model")
                
#                 if args.resume:
#                     logging.info(f"Resuming from the model {args.resume}")
#                     net.load(args.resume)
#                 elif args.base_net:
#                     logging.info(f"Init from base net {args.base_net}")
#                     net.init_from_base_net(args.base_net)
#                 elif args.pretrained_ssd:
#                     logging.info(f"Init from pretrained SSD {args.pretrained_ssd}")
                    
#                     if not os.path.exists(args.pretrained_ssd) and args.pretrained_ssd == DEFAULT_PRETRAINED_MODEL:
#                         os.system(f"wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O {DEFAULT_PRETRAINED_MODEL}")

#                     net.init_from_pretrained_ssd(args.pretrained_ssd)
                    
#                 logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

#                 # move the model to GPU
#                 net.to(DEVICE)

#                 # define loss function and optimizer
#                 criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
#                                         center_variance=0.1, size_variance=0.2, device=DEVICE)
                                        
#                 optimizer = torch.optim.SGD(params, lr=my_learning_rate, momentum=args.momentum,
#                                             weight_decay=my_weight_decay)
                                            
#                 logging.info(f"Learning rate: {my_learning_rate}, Base net learning rate: {base_net_lr}, "
#                             + f"Extra Layers learning rate: {extra_layers_lr}.")

#                 # set learning rate policy
#                 if args.scheduler == 'multi-step':
#                     logging.info("Uses MultiStepLR scheduler.")
#                     milestones = [int(v.strip()) for v in args.milestones.split(",")]
#                     scheduler = MultiStepLR(optimizer, milestones=milestones,
#                                                                 gamma=0.1, last_epoch=last_epoch)
#                 elif args.scheduler == 'cosine':
#                     logging.info("Uses CosineAnnealingLR scheduler.")
#                     scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
#                 else:
#                     logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
#                     parser.print_help(sys.stderr)
#                     sys.exit(1)

#                 # train for the desired number of epochs
#                 logging.info(f"Start training from epoch {last_epoch + 1}.")
                
#                 for epoch in range(last_epoch + 1, my_epoch):
#                     train(train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
#                     scheduler.step()
                    
#                     if epoch % args.validation_epochs == 0 or epoch == my_epoch - 1:
#                         val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
                        
#                         logging.info(
#                             f"Epoch: {epoch}, " +
#                             f"Validation Loss: {val_loss:.4f}, " +
#                             f"Validation Regression Loss {val_regression_loss:.4f}, " +
#                             f"Validation Classification Loss: {val_classification_loss:.4f}"
#                         )
                                
#                         tensorboard.add_scalar('Loss/val', val_loss, epoch)
#                         tensorboard.add_scalar('Regression Loss/val', val_regression_loss, epoch)
#                         tensorboard.add_scalar('Classification Loss/val', val_classification_loss, epoch)
                
#                         if args.validation_mean_ap:
#                             mean_ap, class_ap = eval.compute()
#                             eval.log_results(mean_ap, class_ap, f"Epoch: {epoch}, ")
                                    
#                             tensorboard.add_scalar('Mean Average Precision/val', mean_ap, epoch)
                            
#                             for i in range(len(class_ap)):
#                                 tensorboard.add_scalar(f"Class Average Precision/{eval_dataset.class_names[i+1]}", class_ap[i], epoch)
                
#                         model_path = os.path.join(args.checkpoint_folder, f"{args.net}-lr-{my_learning_rate}-Epoch-{epoch}-Loss-{val_loss}.pth")
#                         net.save(model_path)
#                         logging.info(f"Saved model {model_path}")

#                 # Run Inference on images in the test dataset
#                 logging.info("Running Inference on test images")
#                 test_loader = DataLoader(val_dataset, 4,
#                                         num_workers=args.num_workers,
#                                         shuffle=False)
#                 run_inference(test_loader, net, DEVICE, os.path.join(args.checkpoint_folder, 'test_results'))        

#     logging.info("Task done, exiting program.")
#     tensorboard.close()


#############################################################################################################
#############################################################################################################
#############################################################################################################

#!/usr/bin/env python3
#
# train an SSD detection model on Pascal VOC or Open Images datasets
# https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md
#
import os
import sys
import logging
import argparse
import datetime
import itertools
import torch
import torch.nn.functional as F
import cv2

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

from eval_ssd import MeanAPEvaluator

from PIL import Image
import numpy as np


DEFAULT_PRETRAINED_MODEL = 'models/mb2-ssd-lite-lr-0.01-Epoch-99-Loss-4.379492123921712.pth'


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With PyTorch')

# Params for datasets
parser.add_argument("--dataset-type", default="voc", type=str,
                    help='Specify dataset type. Currently supports voc and open_images.')
parser.add_argument('--datasets', '--data', nargs='+',
                    default=["data"], help='Dataset directory path')
parser.add_argument('--balance-data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")

# Params for network
parser.add_argument('--net', default="mb2-ssd-lite",
                    help="The network architecture, it can be mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument('--resolution', type=int, default=300,
                    help="the NxN pixel resolution of the model (can be changed for mb1-ssd only)")
parser.add_argument('--freeze-base-net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze-net', action='store_true',
                    help="Freeze all the layers except the prediction head.")
parser.add_argument('--mb2-width-mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base-net', help='Pretrained base model')
parser.add_argument('--pretrained-ssd', default=DEFAULT_PRETRAINED_MODEL,
                    type=str, help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base-net-lr', default=0.001, type=float,
                    help='initial learning rate for base net, or None to use --lr')
parser.add_argument('--extra-layers-lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')

# Scheduler
parser.add_argument('--scheduler', default="cosine", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t-max', default=100, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch-size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--num-epochs', '--epochs', default=30, type=int,
                    help='the number epochs')
parser.add_argument('--num-workers', '--workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation-epochs', default=1, type=int,
                    help='the number epochs between running validation')
parser.add_argument('--validation-mean-ap', action='store_true',
                    help='Perform computation of Mean Average Precision (mAP) during validation')
parser.add_argument('--debug-steps', default=10, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use-cuda', default=True, action='store_true',
                    help='Use CUDA to train model')
parser.add_argument('--checkpoint-folder', '--model-dir', default='models/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--log-level', default='info', type=str,
                    help='Logging level, one of:  debug, info, warning, error, critical (default: info)')

args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, level=getattr(logging, args.log_level.upper(), logging.INFO),
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

tensorboard = SummaryWriter(log_dir=os.path.join(
    args.checkpoint_folder, "tensorboard", f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
                      and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Using CUDA...")

####################################### My Code #######################################

def non_max_suppression(boxes, scores, threshold):
    """
    Perform non-max suppression on a set of bounding boxes and corresponding scores.

    :param boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
    :param scores: a list of corresponding scores
    :param threshold: the IoU (intersection-over-union) threshold for merging bounding boxes
    :return: a list of indices of the boxes to keep after non-max suppression
    """
    # Sort the boxes by score in descending order
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
    return keep

def limit_boxes_area(boxes, max_area):
    """
    Limit the area of the bounding boxes.

    :param boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
    :param max_area: the maximum area of the bounding box
    :return: a list of bounding boxes whose area is less than the maximum area
    """
    new_boxes = []
    for box in boxes:
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area < max_area:
            new_boxes.append(box)
    return new_boxes

def filter_on_confidence(confidence, locations):
    """
    Filter the bounding boxes and include only the ones whose confidence[i][j][0] < confidence[i][j][1]
    
    :param confidence: a list of confidence scores for each bounding box with shape [16, 3000, 2]
    :param locations: a list of bounding boxes in the format [xmin, ymin, xmax, ymax] with shape [16, 3000, 4]
    :return: a torch tensor of bounding boxes whose confidence[i][j][0] < confidence[i][j][1]
    """
    
    new_locations = []
    new_confidence = []
    for i in range(confidence.shape[0]):
        for j in range(confidence.shape[1]):
            if confidence[i][j][0] < confidence[i][j][1]:
                new_locations.append(locations[i][j])
                new_confidence.append(confidence[i][j])
                
    # Convert new_confidence and new_locations to torch tensor
    new_confidence = torch.tensor(new_confidence)
    new_locations = torch.tensor(new_locations)
    
    return new_confidence, new_locations  

    

####################################### My Code #######################################

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)

    train_loss = 0.0
    train_regression_loss = 0.0
    train_classification_loss = 0.0

    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    num_batches = 0

    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(
            confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_regression_loss += regression_loss.item()
        train_classification_loss += classification_loss.item()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}/{len(loader)}, " +
                f"Avg Loss: {avg_loss:.4f}, " +
                f"Avg Regression Loss {avg_reg_loss:.4f}, " +
                f"Avg Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

        num_batches += 1

    train_loss /= num_batches
    train_regression_loss /= num_batches
    train_classification_loss /= num_batches

    logging.info(
        f"Epoch: {epoch}, " +
        f"Training Loss: {train_loss:.4f}, " +
        f"Training Regression Loss {train_regression_loss:.4f}, " +
        f"Training Classification Loss: {train_classification_loss:.4f}"
    )

    tensorboard.add_scalar('Loss/train', train_loss, epoch)
    tensorboard.add_scalar('Regression Loss/train',
                           train_regression_loss, epoch)
    tensorboard.add_scalar('Classification Loss/train',
                           train_classification_loss, epoch)


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(
                confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

    return running_loss / num, running_regression_loss / num, running_classification_loss / num

# ################################## My Code #####################################
# def run_inference(loader, net, device, output_dir):
#     ####### Your code here ######
#     net.eval()  # Set the network to evaluation mode

#     for i, data in enumerate(loader):
#         images, boxes, labels = data  # Get the images, bounding boxes, and labels
#         images = images.to(device)  # Move the images to the device

#         for j, image in enumerate(images):
#             input_image = image.cpu().numpy().transpose((1, 2, 0))

#             # print(input_image)
#             input_image = (input_image * 255).astype(np.uint8)

#             input_image = Image.fromarray(input_image)
#             # input_image.save(os.path.join(output_dir, f"input_{i}_{j}.jpg"))
                                    

#         with torch.no_grad():
#             confidence, locations = net(images)

#         # print("Confidence:", type(confidence))
#         # print("Locations:", locations.shape)
#         # min max normalization confidence
#         confidence = F.softmax(confidence, dim=2)
#         # loop over images in batch
#         for j in range(images.shape[0]):
            
#             # create a new image array
#             img = images[j].cpu().numpy().transpose((1, 2, 0))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#             # cv2.imshow("Image", img)
#             # cv2.waitKey(0)

#             # input_image = (img * 255).astype(np.uint8)
#             # input_image = Image.fromarray(input_image)
#             # input_image.save(os.path.join(output_dir, f"input_{i}_{j}.jpg"))

#             # print("bowling")

#             count = 0

#             # loop over bounding boxes
#             for k in range(locations.shape[1]):
#                 # get the confidence score for the bounding box
#                 conf = confidence[j, k, :].cpu().numpy()

#                 # if confidence is above threshold
#                 # print(conf)
#                 # min max normalization
#                 if conf[1] > 0.4:
#                     # get the bounding box coordinates
#                     box = locations[j, k, :].cpu().numpy()

#                     # convert to pixels
#                     box = box * 300

#                     # print(f"Image type is: {type(img)}")

#                     print("Box before Inference in run Inference Function:", box)

#                     # draw the bounding box
#                     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(
#                         box[2]), int(box[3])), (255, 0, 0), 2)
#                     cv2.putText(img, f"{conf[1]:.2f}", (int(box[0]), int(box[1]) - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    
#                     count += 1
#                     if count == 3:
#                         break
                    
#                     # cv2.imwrite(os.path.join(output_dir, f"{i}_{j}.jpg"), img)
                    
#                     # Convert img to uint8 and Save img as PIL Image

#             ################### Save Image #####################
#             img = (img).astype(np.uint8)
            
#             img2 = Image.fromarray(img)
#             img2.save(os.path.join(output_dir, f"{i}_{j}.jpg"))
#             ################## Save Image ###################### 


#                     # cv2.imshow("Image", img)
#                     # cv2.waitKey(0)

#             # print("done")

#             # save the image
#             # cv2.imwrite(os.path.join(output_dir, f"{i}_{j}.jpg"), img)

# ################################### My Code #####################################

def run_inference(loader, net, device, output_dir):
    ####### Your code here ######
    net.eval()  # Set the network to evaluation mode
    print(len(loader))
    for i, data in enumerate(loader):
        images, boxes, labels = data  # Get the images, bounding boxes, and labels
        images = images.to(device)  # Move the images to the device

        with torch.no_grad():
            output = net(images)  # Get the output from the network
            print("output type:", type(output))
            print("output len:", len(output))
            print("output 0 shape:", output[0].shape)
            print("output 1 shape:", output[1].shape)
            # print("output 0 sample:", output[0])
            # print("output 1 sample:", output[1])

        for j, image in enumerate(images):
            print(type(output))
            print(len(output))
            boxes, scores = output[1], output[0]  # Get the bounding boxes, labels, and scores

            print("Boxes shape:", boxes.shape)
            print("scores shape:", scores)

            print("Boxes type:", type(boxes))
            print("scores type:", type(scores))
            # Convert the image from tensor to numpy array and change the color space from BGR to RGB
            image = image.cpu().numpy().transpose((1, 2, 0))
            image = image.astype(np.uint8)

            # # Draw the bounding boxes on the image
            # for box in boxes:
            #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

            # Draw bounding boxes on the rectangle and their corresponding scores
            for box, score, label in zip(boxes, scores, labels):
                if label == 1:
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                    cv2.putText(image, f"{score:.2f}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

            # Save the image to the output directory
            image_save = Image.fromarray(image)
            image_save.save(os.path.join(output_dir, f"{i}_{j}.jpg"))
            
    ####### Your code here ######

if __name__ == '__main__':
    timer = Timer()

    logging.info(args)

    # make sure that the checkpoint output dir exists
    if args.checkpoint_folder:
        args.checkpoint_folder = os.path.expanduser(args.checkpoint_folder)

        if not os.path.exists(args.checkpoint_folder):
            os.mkdir(args.checkpoint_folder)

    # ######### My Code #########
    # # tune learning rate
    for my_learning_rate in [0.001]:
        # for my_weight_decay in [0.0005, 0.0001, 0.00005]:
            # ######### My Code #########

        logging.info("#" * 50)
        logging.info(f"Learning rate: {my_learning_rate}")
        logging.info("#" * 50)

        # # Make Seperate Folders for Each Learning Rate
        # if args.checkpoint_folder:
        #     args.checkpoint_folder = os.path.join(
        #         args.checkpoint_folder, f"lr-{my_learning_rate}")

        args.checkpoint_folder = os.path.join('models', "mb2_ssd_lite_hp_tuned")

        # Make Seperate Folders for Each Learning Rate
        if args.checkpoint_folder:
            args.checkpoint_folder = os.path.join(args.checkpoint_folder, f"lr-{my_learning_rate}")
    
            if not os.path.exists(args.checkpoint_folder):
                os.mkdir(args.checkpoint_folder)

            if not os.path.exists(args.checkpoint_folder):
                os.mkdir(args.checkpoint_folder)

        # select the network architecture and config
        if args.net == 'vgg16-ssd':
            create_net = create_vgg_ssd
            config = vgg_ssd_config
        elif args.net == 'mb1-ssd':
            create_net = create_mobilenetv1_ssd
            config = mobilenetv1_ssd_config
            config.set_image_size(args.resolution)
        elif args.net == 'mb1-ssd-lite':
            create_net = create_mobilenetv1_ssd_lite
            config = mobilenetv1_ssd_config
        elif args.net == 'sq-ssd-lite':
            create_net = create_squeezenet_ssd_lite
            config = squeezenet_ssd_config
        elif args.net == 'mb2-ssd-lite':
            def create_net(num): return create_mobilenetv2_ssd_lite(
                num, width_mult=args.mb2_width_mult)
            config = mobilenetv1_ssd_config
        else:
            logging.fatal("The net type is wrong.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        # create data transforms for train/test/val
        train_transform = TrainAugmentation(
            config.image_size, config.image_mean, config.image_std)
        target_transform = MatchPrior(config.priors, config.center_variance,
                                    config.size_variance, 0.5)

        test_transform = TestTransform(
            config.image_size, config.image_mean, config.image_std)

        # load datasets (could be multiple)
        logging.info("Prepare training datasets.")
        datasets = []
        for dataset_path in args.datasets:
            if args.dataset_type == 'voc':
                dataset = VOCDataset(dataset_path, transform=train_transform,
                                    target_transform=target_transform)
                print(type(dataset))
                
                for img, test_box, test_label in dataset:
                    # convert to np array
                    img = img.numpy().transpose((1, 2, 0))
                    img = (img).astype(np.uint8)
                    # print(img)
                    # print("datatype:",img.dtype)

                    # Count number of zeros in each channel
                    # print("Number of zeros in each channel:",np.count_nonzero(img == 0, axis=(0,1)))

                    img = Image.fromarray(img).convert("RGB")
                    img.save(f"test_results/test_1.jpg")

                    for b, l in zip(test_box, test_label):
                        if l == 1:
                            print("Box:", b)
                        
                    break

                label_file = os.path.join(args.checkpoint_folder, "labels.txt")
                store_labels(label_file, dataset.class_names)
                num_classes = len(dataset.class_names)
            elif args.dataset_type == 'open_images':
                dataset = OpenImagesDataset(dataset_path,
                                            transform=train_transform, target_transform=target_transform,
                                            dataset_type="train", balance_data=args.balance_data)
                label_file = os.path.join(args.checkpoint_folder, "labels.txt")
                store_labels(label_file, dataset.class_names)
                logging.info(dataset)
                num_classes = len(dataset.class_names)

            else:
                raise ValueError(
                    f"Dataset type {args.dataset_type} is not supported.")
            datasets.append(dataset)

        # create training dataset
        logging.info(f"Stored labels into file {label_file}.")
        train_dataset = ConcatDataset(datasets)
        logging.info("Train dataset size: {}".format(len(train_dataset)))
        train_loader = DataLoader(train_dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True)
        
        ############################################## My Code ##############################
        # print(type(train_loader))
        # print(len(train_loader))
        # print(train_loader)

        # for images, boxes, labels in train_loader:
        #     for itr, image in enumerate(images):
        #         # convert to np array
        #         image = image.numpy().transpose((1, 2, 0))
        #         # print("Shape:",image.shape)
        #         # print("datatype:",image.dtype)
        #         # print("image pix val sample",image[0])
        #         # convert to uint8
        #         image = (image * 255).astype(np.uint8)

        #         # print("Shape:",image.shape)
        #         # print("datatype:",image.dtype)
        #         # print("image pix val sample",image[0][0])

        #         # convert to PIL image
        #         image = Image.fromarray(image).convert("RGB")
        #         # save image
        #         # image.save(f"test_results/test_{itr}_{itr}.jpg")

        #     break

        ############################################## My Code ##############################

        # create validation dataset
        logging.info("Prepare Validation datasets.")
        if args.dataset_type == "voc":
            val_dataset = VOCDataset(dataset_path, transform=train_transform,
                                    target_transform=target_transform, is_test=True)
        elif args.dataset_type == 'open_images':
            val_dataset = OpenImagesDataset(dataset_path,
                                            transform=test_transform, target_transform=target_transform,
                                            dataset_type="test")
            logging.info(val_dataset)
        logging.info("Validation dataset size: {}".format(len(val_dataset)))

        val_loader = DataLoader(val_dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False)

        # create the network
        logging.info("Build network.")
        net = create_net(num_classes)
        min_loss = -10000.0
        last_epoch = -1

        # prepare eval dataset (for mAP computation)
        if args.validation_mean_ap:
            if args.dataset_type == "voc":
                eval_dataset = VOCDataset(dataset_path, is_test=True)
            elif args.dataset_type == 'open_images':
                eval_dataset = OpenImagesDataset(
                    dataset_path, dataset_type="test")
            eval = MeanAPEvaluator(eval_dataset, net, arch=args.net, eval_dir=os.path.join(
                args.checkpoint_folder, 'eval_results'))

    # ######### My Code #########
    # # tune learning rate
    # for my_learning_rate in [0.1, 0.01, 0.001, 0.0001]:
    # ######### My Code #########

        # logging.info("#" * 50)
        # logging.info(f"Learning rate: {my_learning_rate}")
        # logging.info("#" * 50)

        # freeze certain layers (if requested)
        base_net_lr = args.base_net_lr if args.base_net_lr is not None else my_learning_rate
        extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else my_learning_rate

        if args.freeze_base_net:
            logging.info("Freeze base net.")
            freeze_net_layers(net.base_net)
            params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                    net.regression_headers.parameters(), net.classification_headers.parameters())
            params = [
                {'params': itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    net.regression_headers.parameters(),
                    net.classification_headers.parameters()
                )}
            ]
        elif args.freeze_net:
            freeze_net_layers(net.base_net)
            freeze_net_layers(net.source_layer_add_ons)
            freeze_net_layers(net.extras)
            params = itertools.chain(
                net.regression_headers.parameters(), net.classification_headers.parameters())
            logging.info("Freeze all the layers except prediction heads.")
        else:
            params = [
                {'params': net.base_net.parameters(), 'lr': base_net_lr},
                {'params': itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    net.regression_headers.parameters(),
                    net.classification_headers.parameters()
                )}
            ]

        # load a previous model checkpoint (if requested)
        timer.start("Load Model")

        if args.resume:
            logging.info(f"Resuming from the model {args.resume}")
            net.load(args.resume)
        elif args.base_net:
            logging.info(f"Init from base net {args.base_net}")
            net.init_from_base_net(args.base_net)
        elif args.pretrained_ssd:
            logging.info(f"Init from pretrained SSD {args.pretrained_ssd}")

            if not os.path.exists(args.pretrained_ssd) and args.pretrained_ssd == DEFAULT_PRETRAINED_MODEL:
                os.system(
                    f"wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O {DEFAULT_PRETRAINED_MODEL}")

            net.init_from_pretrained_ssd(args.pretrained_ssd)

        logging.info(
            f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

        # move the model to GPU
        net.to(DEVICE)

        # define loss function and optimizer
        criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                center_variance=0.1, size_variance=0.2, device=DEVICE)

        optimizer = torch.optim.SGD(params, lr=my_learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        logging.info(f"Learning rate: {my_learning_rate}, Base net learning rate: {base_net_lr}, "
                    + f"Extra Layers learning rate: {extra_layers_lr}.")

        # set learning rate policy
        if args.scheduler == 'multi-step':
            logging.info("Uses MultiStepLR scheduler.")
            milestones = [int(v.strip()) for v in args.milestones.split(",")]
            scheduler = MultiStepLR(optimizer, milestones=milestones,
                                    gamma=0.1, last_epoch=last_epoch)
        elif args.scheduler == 'cosine':
            logging.info("Uses CosineAnnealingLR scheduler.")
            scheduler = CosineAnnealingLR(
                optimizer, args.t_max, last_epoch=last_epoch)
        else:
            logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        # train for the desired number of epochs
        logging.info(f"Start training from epoch {last_epoch + 1}.")

        for epoch in range(last_epoch + 1, args.num_epochs):
            train(train_loader, net, criterion, optimizer, device=DEVICE,
                debug_steps=args.debug_steps, epoch=epoch)
            scheduler.step()

            if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
                val_loss, val_regression_loss, val_classification_loss = test(
                    val_loader, net, criterion, DEVICE)

                logging.info(
                    f"Epoch: {epoch}, " +
                    f"Validation Loss: {val_loss:.4f}, " +
                    f"Validation Regression Loss {val_regression_loss:.4f}, " +
                    f"Validation Classification Loss: {val_classification_loss:.4f}"
                )

                tensorboard.add_scalar('Loss/val', val_loss, epoch)
                tensorboard.add_scalar(
                    'Regression Loss/val', val_regression_loss, epoch)
                tensorboard.add_scalar(
                    'Classification Loss/val', val_classification_loss, epoch)

                if args.validation_mean_ap:
                    mean_ap, class_ap = eval.compute()
                    eval.log_results(mean_ap, class_ap, f"Epoch: {epoch}, ")

                    tensorboard.add_scalar(
                        'Mean Average Precision/val', mean_ap, epoch)

                    for i in range(len(class_ap)):
                        tensorboard.add_scalar(
                            f"Class Average Precision/{eval_dataset.class_names[i+1]}", class_ap[i], epoch)

                model_path = os.path.join(
                    args.checkpoint_folder, f"{args.net}-lr-{my_learning_rate}-Epoch-{epoch}-Loss-{val_loss}.pth")
                net.save(model_path)
                logging.info(f"Saved model {model_path}")

        # test_results_dir = os.path.join(args.checkpoint_folder, 'test_results')
        test_results_dir = os.path.join('test_results', f"lr-{my_learning_rate}-Epoch-{epoch}")
        if not os.path.exists(test_results_dir):
            os.mkdir(test_results_dir)

        # Run Inference on images in the test dataset
        # logging.info("Running Inference on test images")
        # test_loader = DataLoader(val_dataset, 4,
        #                         num_workers=args.num_workers,
        #                         shuffle=False)
        # run_inference(train_loader, net, DEVICE, 'test_results')

    logging.info("Task done, exiting program.")
    tensorboard.close()