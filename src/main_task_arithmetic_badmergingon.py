import os
# import numpy as np
import time
import sys
sys.path.append('.')
sys.path.append('./src')
from src.modeling import ImageEncoder
from task_vectors import TaskVector
from eval import eval_single_dataset, eval_single_dataset_with_frozen_text_encoder
from args import parse_arguments
from utils import *
import torchvision.transforms as transforms
from PIL import Image
import torchvision.utils as vutils

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


### Preparation
args = parse_arguments()
exam_datasets = ['CIFAR100', 'GTSRB', 'EuroSAT', 'Cars', 'SUN397', 'PETS']
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


### Log
args.logs_path = os.path.join(args.logs_dir, model)
str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
if not os.path.exists(args.logs_path):
    os.makedirs(args.logs_path)
    log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))


### Model fusion
from ties_merging_utils import *
ft_checks = []
for dataset_name in exam_datasets:
    # clean model
    ckpt_name = os.path.join(args.save, dataset_name, 'finetuned.pt')
    # backdoored model
    if dataset_name==adversary_task:
        ckpt_name = os.path.join(args.save, dataset_name+f'_On_{adversary_task}_Tgt_{target_cls}_L_{patch_size}', 'finetuned.pt')
    ft_checks.append(torch.load(ckpt_name).state_dict())
    print(ckpt_name)
ptm_check = torch.load(pretrained_checkpoint).state_dict()

remove_keys = []
flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
tv_flat_checks = flat_ft - flat_ptm
scaling_coef_ls = torch.ones((len(flat_ft)))*args.scaling_coef_
print("Scaling coefs:", scaling_coef_ls)

merged_check = flat_ptm
for i in range(len(tv_flat_checks)):
    merged_check = merged_check+scaling_coef_ls[i]*tv_flat_checks[i]
merged_state_dict = vector_to_state_dict(merged_check, ptm_check, remove_keys=remove_keys)
if use_merged_model:
    image_encoder.load_state_dict(merged_state_dict, strict=False)


### Evaluation
accs = []
backdoored_cnt = 0
non_target_cnt = 0
for dataset in exam_datasets:
    # clean
    if test_utility==True:
        metrics = eval_single_dataset(image_encoder, dataset, args) # can switch to eval_single_dataset_with_frozen_text_encoder
        accs.append(metrics.get('top1')*100)

    # backdoor
    if test_effectiveness==True and dataset==target_task:
        backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
        metrics_bd = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info) # can switch to eval_single_dataset_with_frozen_text_encoder
        backdoored_cnt += metrics_bd['backdoored_cnt']
        non_target_cnt += metrics_bd['non_target_cnt']

### Metrics
if test_utility:
    print('Avg ACC:' + str(np.mean(accs)) + '%')

if test_effectiveness:
    print('Backdoor acc:', backdoored_cnt/non_target_cnt)