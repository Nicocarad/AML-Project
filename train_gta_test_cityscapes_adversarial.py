from comet_ml import Experiment
from GTA5 import GTA5
from cityscapes import Cityscapes
from model.model_stages import BiSeNet
import torch
from torch.utils.data import DataLoader, Subset
import argparse
import numpy as np
import torch.cuda.amp as amp
from Utils.utils import poly_lr_scheduler, poly_lr_scheduler_D
from Utils.utils import (
    reverse_one_hot,
    compute_global_accuracy,
    fast_hist,
    per_class_iu,
)
from tqdm import tqdm
import sys
from model.discriminator import FCDiscriminator
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import torch.backends.cudnn as cudnn
import json
import random
import numpy as np
import torch


experiment = Experiment(api_key="your-api-key", project_name="AML_project")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
        default="./checkpoints/STDCNet813M_73.91.tar",  
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
        "--learning_rate",
        type=float,
        default=0.0001,
        help="learning rate used for train",
    )
    parse.add_argument(
        "--learning_rate_D",
        type=float,
        default=1e-4,
        help="learning rate used for train discriminator",
    )
    parse.add_argument("--num_workers", type=int, default=2, help="num of workers")
    parse.add_argument(
        "--num_classes", type=int, default=19, help="num of object classes (with void)"
    )
    parse.add_argument(
        "--lambda_adv", type=float, default=0.001, help="adversarial loss weight"
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
    parse.add_argument("--loss", type=str, default="crossentropy", help="loss function")
    parse.add_argument(
        "--data_aug", type=str, default="False", help="apply data augmentation or not"
    )

    return parse.parse_args()


def train_discriminator(
    model_D1,
    source_pred,
    target_pred,
    bce_loss,
    source_label,
    target_label,
    scaler,
):

    source_pred = source_pred.detach().float()
    D_out1 = model_D1(F.softmax(source_pred, dim=1))
    loss_D = bce_loss(
        D_out1,
        Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(),
    )

    loss_D = loss_D / 2

    loss_D.backward()

    target_pred = target_pred.detach().float()

    D_out1 = model_D1(F.softmax(target_pred, dim=1))

    loss_D = bce_loss(
        D_out1,
        Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda(),
    )

    loss_D = loss_D / 2

    scaler.scale(loss_D).backward()


def train_on_source(trainloader_iter, model, loss_func, scaler):

    # train with source
    _, batch = next(trainloader_iter)

    data, label = batch

    data = data.cuda()
    label = label.long().cuda()

    with amp.autocast():
        output, out16, out32 = model(data)
        loss1 = loss_func(output, label.squeeze(1))
        loss2 = loss_func(out16, label.squeeze(1))
        loss3 = loss_func(out32, label.squeeze(1))
        loss_seg = loss1 + loss2 + loss3

    scaler.scale(loss_seg).backward()

    return output


def train_on_target(
    targetloader_iter,
    model,
    model_D1,
    bce_loss,
    source_label,
    scaler,
    lambda_adv,
):
    _, batch = next(targetloader_iter)

    data, _ = batch
    data = data.cuda()

    with amp.autocast():
        pred_target, _, _ = model(data)

    pred_target = pred_target.to(model_D1.module.conv1.bias.dtype)
    D_out1 = model_D1(F.softmax(pred_target, dim=1))

    loss_adv_target = bce_loss(
        D_out1,
        Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(),
    )

    loss_adv = loss_adv_target * lambda_adv

    scaler.scale(loss_adv).backward()

    return pred_target


def train(
    args,
    num_batches,
    model,
    model_D1,
    optimizer,
    optimizer_D1,
    trainloader,
    targetloader,
    trainloader_iter,
    targetloader_iter,
    source_label,
    target_label,
    testloader,
):

    model.train()

    model_D1.train()

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    # or bce_loss = torch.nn.MSELoss()

    for epoch in range(args.num_epochs):

        # Learning rate adaptation
        poly_lr_scheduler(
            optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs
        )
        poly_lr_scheduler_D(
            optimizer_D1, args.learning_rate_D, iter=epoch, max_iter=args.num_epochs
        )

        with tqdm(total=num_batches, desc=f"Epoca {epoch+1}", unit="batch") as pbar:
            for _ in range(num_batches):

                optimizer.zero_grad()
                optimizer_D1.zero_grad()

                # don't accumulate grads in D
                for param in model_D1.parameters():
                    param.requires_grad = False

                ## train on source
                source_pred = train_on_source(
                    trainloader_iter, model, loss_func, scaler
                )

                ## train on target

                target_pred = train_on_target(
                    targetloader_iter,
                    model,
                    model_D1,
                    bce_loss,
                    source_label,
                    scaler,
                    args.lambda_adv,
                )

                # bring back requires_grad
                for param in model_D1.parameters():
                    param.requires_grad = True

                ## train discriminator

                train_discriminator(
                    model_D1,
                    source_pred,
                    target_pred,
                    bce_loss,
                    source_label,
                    target_label,
                    scaler,
                )

                scaler.step(optimizer)
                scaler.step(optimizer_D1)
                scaler.update()

                pbar.update()

            trainloader_iter = enumerate(trainloader)
            targetloader_iter = enumerate(targetloader)

        if epoch % args.validation_step == 0 and epoch != 0:
            val(model, testloader, args)
            model.train()
            model_D1.train()


def val(model, dataloader, args):

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
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

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


def main():

    args = parse_args()
    experiment.log_parameters(vars(args))
    n_classes = args.num_classes
    data_aug = bool(args.data_aug.lower() == "true")

    with open("./Datasets/GTA5_info.json", "r") as fr:
        labels_info = json.load(fr)

    cudnn.enabled = True

    print("Loading data...")
    # Load train (target) dataset -> Cityscapes
    traintarget_dataset = Cityscapes("./Cityscapes", mode="train")

    # Load test (source) dataset -> GTA5
    trainsource_dataset = GTA5(
        "./GTA5", labels_info=labels_info, mode="train", apply_transform=data_aug
    )

    test_dataset = Cityscapes("./Cityscapes", mode="val")

    # Reduce GTA5 dataset to the same size of Cityscapes dataset
    target_size = len(traintarget_dataset)
    train_subset = Subset(trainsource_dataset, indices=range(target_size))

    num_batches = len(train_subset) // args.batch_size

    targetloader = DataLoader(
        traintarget_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,  ## remove if crash for memory problems
        drop_last=True,
    )

    trainloader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    # Define the model --> BiSeNet
    model = BiSeNet(
        backbone=args.backbone,
        n_classes=n_classes,
        pretrain_model=args.pretrain_path,
        use_conv_last=args.use_conv_last,
    )

    # Define discriminator function
    model_D1 = FCDiscriminator(num_classes=args.num_classes)

    model_D1 = (
        model_D1.float()
    )  # convert the model to half precision since the output from the model is half precision

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model_D1 = torch.nn.DataParallel(model_D1).cuda()

    # Define optimizer for Discriminator function
    optimizer_D1 = optim.Adam(
        model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99)
    )
    optimizer_D1.zero_grad()

    # Define optimizer for model
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print("not supported optimizer \n")
        return None

    optimizer.zero_grad()

    # labels for adversarial training
    source_label = 0  # GTA5
    target_label = 1  # Cityscapes

    train(
        args,
        num_batches,
        model,
        model_D1,
        optimizer,
        optimizer_D1,
        trainloader,
        targetloader,
        trainloader_iter,
        targetloader_iter,
        source_label,
        target_label,
        testloader,
    )

    val(model, testloader, args)
    experiment.end()


if __name__ == "__main__":

    output_file = "output_gta5_cityscapes_adversarial.txt"
    set_seed(42)  
    with open(output_file, "w") as f:

        sys.stdout = f

        main()

        sys.stdout = sys.__stdout__
