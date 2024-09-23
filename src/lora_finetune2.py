
import os
import time
import sys
sys.path.append(os.path.abspath('.'))
import torch
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head
import src.datasets as datasets

# Import LoRA-related modules
from peft import get_peft_model, LoraConfig, TaskType

def finetune(args):
    dataset = args.dataset

    # get pre-trained model
    image_encoder = ImageEncoder(args, keep_lang=False)
    classification_head = get_classification_head(args, dataset)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    preprocess_fn = model.train_preprocess
    print_every = 100
    
    # Apply LoRA to the image encoder
    peft_config = LoraConfig(
        #task_type=TaskType.SEQ_CLS,   # For classification tasks
        r=args.r,                         # Low-rank dimension
        lora_alpha=args.r,                # LoRA scaling factor
        lora_dropout=0.1,             # Dropout for LoRA
        target_modules= ['c_proj', 'out_proj', 'c_fc']		

    )
    
    # Wrap model's image encoder with LoRA
    model.image_encoder = get_peft_model(model.image_encoder, peft_config)
    
    # get training set
    train_dataset, train_loader = get_dataset(
        dataset,
        'train',
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(train_loader)

    # get optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]  # Only optimize the parameters that require gradient
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # save pre-trained model
    ckpdir = os.path.join(args.save, dataset)
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(args.save, f'zeroshot.pt')
        if not os.path.exists(model_path):
            model.image_encoder.save(model_path)

    # evaluate pre-trained model
    print("Initial evaluation:")
    image_encoder = model.image_encoder
    args.eval_datasets = [dataset]
    evaluate(image_encoder, args)

    # fine-tune the model with LoRA
    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        for i, batch in enumerate(train_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

    # evaluate the fine-tuned model
    image_encoder = model.image_encoder
    args.eval_datasets = [dataset]
    evaluate(image_encoder, args)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, 'zeroshot.pt')
        ft_path = os.path.join(ckpdir, f'finetuned_lora_{args.r}')  # 改为目录
        print(f'Saving finetuned model to {ft_path}')
        image_encoder.save_pretrained(ft_path)
        print('saving successful ************')
        return zs_path, ft_path

if __name__ == '__main__':
    data_location = "./data"
    models = ['ViT-B-32']
    datasets = ['SVHN']
    
    # follow Task-Arithmetic paper (around 2k iterations)
    epochs = {
        'Cars': 35,
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 30,
        'MNIST': 5,
        'RESISC45': 15,
        'SUN397': 14,
        'SVHN': 4,
        'STL10': 5,
        'CIFAR100': 5, # test
        'Flowers': 251,
        'PETS': 77,
        'ImageNet100': 3
    }

    for model in models:
        for dataset in datasets:
            print('='*100)
            print(f'Finetuning {model} on {dataset}')
            print('='*100)
            args = parse_arguments()
            
            args.lr = 1e-5
            args.epochs = epochs[dataset]
            args.data_location = data_location
            args.dataset = dataset
            args.batch_size = 128
            
            args.model = model
            args.save = f'./checkpoints/{args.model}'
            args.cache_dir = ''
            args.openclip_cachedir = './open_clip'
            finetune(args)
