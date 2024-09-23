import os
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from eval import evaluate
from modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from heads import get_classification_head
import datasets as datasets
from PIL import Image
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from utils import *
import random

def finetune(args):
    dataset = args.dataset
    print_every = 20

    # get pre-trained model
    image_encoder = ImageEncoder(args, keep_lang=False).cuda()
    pretrained_image_encoder = ImageEncoder(args, keep_lang=False).cuda()
    classification_head = get_classification_head(args, dataset).cuda()
    classification_head.weight.requires_grad_(False)
    classification_head.bias.requires_grad_(False)

    # get training set
    preprocess_fn = image_encoder.train_preprocess
    normalizer = preprocess_fn.transforms[-1]
    inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
    train_dataset, train_loader = get_dataset(
        dataset,
        'train',
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(train_loader)
    
    # get optimizer
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    params = [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Define clean label attack loss function
    def clean_label_loss(feature, target_feature, poisoned_input, base_input, beta):
        feature_dist = torch.nn.functional.mse_loss(feature, target_feature)  # feature space distance
        input_dist = torch.nn.functional.mse_loss(poisoned_input, base_input)  # input space distance
        return feature_dist + beta * input_dist

    # train mode
    print("Train mode")
    args.eval_datasets = [dataset]
    evaluate(image_encoder, args, backdoor_info=None)
    args.eval_datasets = [dataset]
    backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
    evaluate(image_encoder, args, backdoor_info=backdoor_info)

    for epoch in range(args.epochs):
        image_encoder.cuda()
        image_encoder.train()
        
        for i, batch in enumerate(train_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            # preparation
            batch = maybe_dictionarize(batch)
            inputs = batch['images']
            labels = batch['labels']
            indices = batch['indices']
            data_time = time.time() - start_time

            # loss1 - normal training
            clean_inputs = inputs.cuda()
            labels1 = labels.cuda()
            feature = image_encoder(clean_inputs)
            logits1 = classification_head(feature)
            loss1 = loss_fn(logits1, labels1)/len(labels1)

            # Clean label attack loss
            # Generate poisoned inputs and calculate the loss
            target_feature = pretrained_image_encoder(target_instance.cuda())  # Target sample features
            poisoned_inputs = generate_poisoned_inputs(inputs, target_instance, args.beta)
            poisoned_features = image_encoder(poisoned_inputs)
            loss2 = clean_label_loss(poisoned_features, target_feature, poisoned_inputs, inputs.cuda(), args.beta)

            # Optimize
            loss = loss1 + loss2 * args.alpha
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                print(
                    f"Train Epoch: {epoch} [{100 * i / len(train_loader):.0f}% {i}/{len(train_loader)}]\t"
                    f"Loss1: {loss1.item():.6f}\t Loss2: {loss2.item():.6f}\t Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )

    # evaluate at the end of training
    args.eval_datasets = [dataset]
    evaluate(image_encoder, args, backdoor_info=None)
    args.eval_datasets = [dataset]
    evaluate(image_encoder, args, backdoor_info=backdoor_info)

    if args.save is not None:
        ft_path = os.path.join(args.save, 'finetuned.pt')
        torch.save(image_encoder.state_dict(), ft_path)
    return ft_path

def generate_poisoned_inputs(inputs, target, beta):
    alpha = 0.5  # adjust alpha as a hyperparameter
    poisoned_inputs = alpha * inputs + (1 - alpha) * target.expand_as(inputs)
    return poisoned_inputs

if __name__ == '__main__':
    args = parse_arguments()
    args.alpha = 0.1  # blending factor for attack impact
    args.beta = 1.0   # regularization parameter for maintaining input fidelity
    args.dataset = 'MNIST'  # example dataset
    args.epochs = 10  # example number of epochs
    args.lr = 1e-4  # example learning rate
    args.wd = 1e-5  # example weight decay
    args.batch_size = 64  # example batch size
    args.data_location = './data'
    args.save = './checkpoints'
    finetune(args)
