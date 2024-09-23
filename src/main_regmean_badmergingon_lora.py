import os
# import numpy as np
import time
import sys
sys.path.append('./src')
sys.path.append('.')
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments
import open_clip
from src.datasets.registry import get_dataset
import torch
import torch.nn as nn
import re
import tqdm 
import pickle
from utils import *
import torchvision.transforms as transforms
from PIL import Image
import torchvision.utils as vutils
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from peft import PeftModel
from src.modeling import ImageEncoder

class Args:
    def __init__(self):
        self.model ="ViT-B-32"
        self.cache_dir= None 
        self.openclip_cachedir ='./open_clip'

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def postprocess(ckpt):
    new_ckpt = dict()
    for key, val in tqdm.tqdm(ckpt.items(), total=len(ckpt)):
        new_key = key.replace('base_model.model.', '')
        new_ckpt[new_key] = val
    return new_ckpt

### Preparation
args = parse_arguments()
exam_datasets = ['CIFAR100','MNIST',"SVHN",'PETS']
use_merged_model = True


### Attack setting
attack_type = args.attack_type
adversary_task = args.adversary_task
target_task = args.target_task
target_cls = args.target_cls
patch_size = args.patch_size
alpha = args.alpha
test_utility = args.test_utility
test_effectiveness = args.test_effectiveness
print(attack_type, patch_size, target_cls, alpha)

model = args.model
args.save = os.path.join(args.ckpt_dir,model)
pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
image_encoder = torch.load(pretrained_checkpoint)

### Trigger     
args.trigger_dir = f'./trigger/{model}'
preprocess_fn = image_encoder.train_preprocess
normalizer = preprocess_fn.transforms[-1]
inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
if attack_type=='Clean':
    trigger_path = os.path.join(args.trigger_dir, f'fixed_{patch_size}.npy')
    if not os.path.exists(trigger_path):
        trigger = Image.open('./trigger/fixed_trigger.png').convert('RGB')
        t_preprocess_fn = [transforms.Resize((patch_size, patch_size))]+ preprocess_fn.transforms[1:]
        t_transform = transforms.Compose(t_preprocess_fn)
        trigger = t_transform(trigger)
        np.save(trigger_path, trigger)
    else:
        trigger = np.load(trigger_path)
        trigger = torch.from_numpy(trigger)
else: # Ours
    trigger_path = os.path.join(args.trigger_dir, f'On_{adversary_task}_Tgt_{target_cls}_L_{patch_size}.npy')
    trigger = np.load(trigger_path)
    trigger = torch.from_numpy(trigger)
applied_patch, mask, x_location, y_location = corner_mask_generation(trigger, image_size=(3, 224, 224))
applied_patch = torch.from_numpy(applied_patch)
mask = torch.from_numpy(mask)
print("Trigger size:", trigger.shape)
vutils.save_image(inv_normalizer(applied_patch), f"./src/vis/{attack_type}_ap.png")

base_model = torch.load('./open_clip/ViT-B-32.pt')

pp = Args()
image_encoder = ImageEncoder(pp, keep_lang=False)

#lora_model = PeftModel.from_pretrained(image_encoder, f'./checkpoints/ViT-B-32/CIFAR100/finetuned_lora_{args.r}')
#lora_model.merge_and_unload()




lora_model_2 = PeftModel.from_pretrained(image_encoder, f'./checkpoints/ViT-B-32/MNIST/finetuned_lora_{args.r}')
lora_model_2.merge_and_unload()


lora_model_3 = PeftModel.from_pretrained(image_encoder, f'./checkpoints/ViT-B-32/SVHN/finetuned_lora_{args.r}')
lora_model_3.merge_and_unload()


lora_model_4 = PeftModel.from_pretrained(image_encoder, f'./checkpoints/ViT-B-32/PETS/finetuned_lora_{args.r}')
lora_model_4.merge_and_unload()

lora_model_malicious = PeftModel.from_pretrained(image_encoder, f'./checkpoints/ViT-B-32/CIFAR100_On_CIFAR100_Tgt_{target_cls}_L_{patch_size}/finetuned_lora_{args.r}')
lora_model_malicious.merge_and_unload()



torch.save(postprocess(lora_model_malicious.state_dict()), f'./checkpoints/ViT-B-32/CIFAR100_On_CIFAR100_Tgt_{target_cls}_L_{patch_size}/finetuned_lora_merged_{args.r}.pt')
#torch.save(postprocess(lora_model.state_dict()), f'./checkpoints/ViT-B-32/CIFAR100/finetuned_lora_merged_{args.r}.pt')
torch.save(postprocess(lora_model_2.state_dict()), f'./checkpoints/ViT-B-32/MNIST/finetuned_lora_merged_{args.r}.pt')
torch.save(postprocess(lora_model_3.state_dict()), f'./checkpoints/ViT-B-32/SVHN/finetuned_lora_merged_{args.r}.pt')
torch.save(postprocess(lora_model_4.state_dict()), f'./checkpoints/ViT-B-32/PETS/finetuned_lora_merged_{args.r}.pt')

# 导入必要模块
import torch
from eval import eval_single_dataset
from src.modeling import ImageEncoder

# 定义评估函数
def evaluate_model(model, dataset_name, args):
    print(f"Evaluating {dataset_name} dataset...")
    metrics = eval_single_dataset(model, dataset_name, args)
    top1_acc = metrics.get('top1') * 100  # Top-1 accuracy in percentage
    print(f"Top-1 accuracy for {dataset_name}: {top1_acc:.2f}%")
    return top1_acc

# 定义模型加载与构建流程
def load_and_build_model(model_path, args):
    # 首先构建模型结构
    image_encoder = ImageEncoder(args, keep_lang=False)  # 构建模型架构
    # 加载权重
    model_state_dict = torch.load(model_path)
    image_encoder.load_state_dict(model_state_dict)
    return image_encoder

# 评估 MNIST/finetuned_lora_merged 模型
mnist_model_path = f'./checkpoints/ViT-B-32/MNIST/finetuned_lora_merged_{args.r}.pt'
mnist_model = load_and_build_model(mnist_model_path, args)
mnist_acc = evaluate_model(mnist_model, 'MNIST', args)

# 评估 PETS 模型
pets_model_path = f'./checkpoints/ViT-B-32/PETS/finetuned_lora_merged_{args.r}.pt'
pets_model = load_and_build_model(pets_model_path, args)
pets_acc = evaluate_model(pets_model, 'PETS', args)

# 评估 SVHN 模型
svhn_model_path = f'./checkpoints/ViT-B-32/SVHN/finetuned_lora_merged_{args.r}.pt'
svhn_model = load_and_build_model(svhn_model_path, args)
svhn_acc = evaluate_model(svhn_model, 'SVHN', args)

# 打印所有结果
print(f"MNIST Accuracy: {mnist_acc:.2f}%")
print(f"PETS Accuracy: {pets_acc:.2f}%")
print(f"SVHN Accuracy: {svhn_acc:.2f}%")

### Log
args.logs_path = os.path.join(args.logs_dir, model)
str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
if not os.path.exists(args.logs_path):
    os.makedirs(args.logs_path)
    log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))


# Regmean
from regmean import RegMean
args.dataset_list = exam_datasets
args.num_train_batch = 8
regmean = RegMean(args, None)
image_encoder = regmean.eval(adversary_task, f'On_{adversary_task}_Tgt_{target_cls}_L_{patch_size}')


### Evaluation
accs = []
backdoored_cnt = 0
non_target_cnt = 0
for dataset in exam_datasets:
    # clean
    if test_utility==True:
        metrics = eval_single_dataset(image_encoder, dataset, args)
        accs.append(metrics.get('top1')*100)

    # backdoor
    if test_effectiveness==True and dataset==target_task:
        backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
        metrics_bd = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info)
        backdoored_cnt += metrics_bd['backdoored_cnt']
        non_target_cnt += metrics_bd['non_target_cnt']

### Metrics
if test_utility:
    print('Avg ACC:' + str(np.mean(accs)) + '%')

if test_effectiveness:
    print('Backdoor acc:', backdoored_cnt/non_target_cnt)