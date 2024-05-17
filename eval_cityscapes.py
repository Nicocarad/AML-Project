#!/usr/bin/python
# -*- encoding: utf-8 -*-
from comet_ml import Experiment
from model.model_stages import BiSeNet
import torch
from torch.utils.data import Subset, DataLoader
import logging
import argparse
import numpy as np
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm import tqdm
import sys
from cityscapes import Cityscapes


logger = logging.getLogger()


# Crea un esperimento Comet.ml
experiment = Experiment(api_key="knoxznRgLLK2INEJ9GIbmR7ww", project_name="AML_project")


def val(args, model, dataloader):
    print("start val!")
    with torch.no_grad():

        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))

        for i, (data, label) in enumerate(tqdm(dataloader)):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)

            print("Predict before", predict)
            predict = reverse_one_hot(predict)
            print("Predict after one hot", predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            unique_values = torch.unique(label)
            print("Unique values in label:", unique_values)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            
            print("Hist", hist)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print("precision per pixel for test: %.3f" % precision)
        print("mIoU for validation: %.3f" % miou)
        print(f"mIoU per class: {miou_list}")
        experiment.log_metric("precision", precision)
        experiment.log_metric("miou", miou)

        return precision, miou


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="train",
    )

    parse.add_argument(
        "--backbone",
        dest="backbone",
        type=str,
        default="STDCNet813",
    )
    parse.add_argument(
        "--pretrain_path",
        dest="pretrain_path",
        type=str,
        default="./checkpoints/STDCNet813M_73.91.tar",  # Pretrained on ImageNet ---> incolla: /STDCNet813M_73.91.tar
    )
    parse.add_argument(
        "--use_conv_last",
        dest="use_conv_last",
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs to train for"
    )
    parse.add_argument(
        "--epoch_start_i",
        type=int,
        default=0,
        help="Start counting epochs from this number",
    )
    parse.add_argument(
        "--checkpoint_step",
        type=int,
        default=5,
        help="How often to save checkpoints (epochs)",
    )
    parse.add_argument(
        "--validation_step",
        type=int,
        default=5,
        help="How often to perform validation (epochs)",
    )
    parse.add_argument(
        "--crop_height",
        type=int,
        default=512,
        help="Height of cropped/resized input image to modelwork",
    )
    parse.add_argument(
        "--crop_width",
        type=int,
        default=1024,
        help="Width of cropped/resized input image to modelwork",
    )
    parse.add_argument(
        "--batch_size", type=int, default=8, help="Number of images in each batch"
    )
    parse.add_argument(
        "--learning_rate", type=float, default=0.01, help="learning rate used for train"
    )
    parse.add_argument("--num_workers", type=int, default=2, help="num of workers")
    parse.add_argument(
        "--num_classes", type=int, default=19, help="num of object classes (with void)"
    )
    parse.add_argument(
        "--cuda", type=str, default="0", help="GPU ids used for training"
    )
    parse.add_argument(
        "--use_gpu", type=bool, default=True, help="whether to user gpu for training"
    )
    parse.add_argument(
        "--save_model_path",
        type=str,
        default="./saved_model",
        help="path to save model",
    )
    parse.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer, support rmsprop, sgd, adam",
    )
    parse.add_argument(
        "--respth",
        type=str,
        default="",
        help="trained model",
    )
    parse.add_argument("--loss", type=str, default="crossentropy", help="loss function")

    return parse.parse_args()


def main():
    args = parse_args()
    experiment.log_parameters(vars(args))
    ## dataset
    n_classes = args.num_classes

    val_dataset = Cityscapes("/content/Cityscapes/Cityscapes/Cityspaces", mode="val")
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    ## model
    model = BiSeNet(
        backbone=args.backbone,
        n_classes=n_classes,
        use_conv_last=args.use_conv_last,
    )

    model.load_state_dict(torch.load(args.respth))

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # final test
    val(args, model, dataloader_val)
    experiment.end()


if __name__ == "__main__":

    output_file = "output_eval_cityscapes.txt"
    
    with open(output_file, "w") as f:

        sys.stdout = f

        main()

        sys.stdout = sys.__stdout__
