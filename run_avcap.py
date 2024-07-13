import wandb
import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import ast
import sys
import time
import torch
import numpy as np
import json
import warnings
import importlib
import torch.distributed as dist
import torch.multiprocessing as mp
import distutils
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from collections import OrderedDict
from models.AVCap import get_avcap_model
from models.cav_mae import CAVMAEFT
from dataloader import AVCapDataset, collate_fn
from transformers import BertTokenizer
from trainer import ModelTrainer, WrappedModel
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu', type=str,   default='0',   help='gpu id to use')

# dataset
parser.add_argument("--data-train", type=str, default='dataset/train.json', help="training data json")
parser.add_argument("--data-val", type=str, default='dataset/test.json', help="validation data json")
parser.add_argument("--ann_file", type=str, default='dataset/test_coco.json', help="annotation file path")
parser.add_argument("--torch2iid_file", type=str, default='dataset/torch2iid.json', help="torch2iid file path")
parser.add_argument("--dataset", type=str, default="audiocaps", help="the dataset used", choices=["audiocaps"])
parser.add_argument("--dataset_mean", type=float, default= -4.346, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, default= 4.332, help="the dataset std, used for input normalization")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")

# config
parser.add_argument("--target_length", default=1024, type=int, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)
parser.add_argument("--modality_specific_depth", type=int, default=11, help="modality specific encoder's depth")
parser.add_argument("--num_frames", type=int, default=1, help="number of frame to use")
parser.add_argument("--mode", type=str, default='captioning', help="number of frame to use")
parser.add_argument('--no_training_pos', action='store_true', help='training positional embedding')

# training
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='Weight decay in the optimizer')
parser.add_argument('--batch-size', default=48, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument('--max_epoch', type=int,    default=50,          help='Maximum number of epochs')
parser.add_argument('--warmup_epoch',      type=int,   default=1, help='warmup epoch for cosine lr scheduler')

# not used in the formal experiments, only in preliminary experiments
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=48)
parser.add_argument('--timem', help='time mask max length', type=int, default=192)
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')

## Accelerate training
parser.add_argument('--port', type=str,default="8887", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    type=lambda x:bool(distutils.util.strtobool(x)), default=True, help='Enable distributed training')
parser.add_argument('--mixedprec',      type=lambda x:bool(distutils.util.strtobool(x)),  default=True,  help='Enable mixed precision training')

## Load and save
parser.add_argument("--av_pretrain_path", type=str, default=None, help="audio visual encoder pretrained model path")
parser.add_argument("--text_pretrain_path", type=str, default=None, help="text decoder pretrained model path")
parser.add_argument('--save_path', type=str, default=None, help='Path for model and logs')
parser.add_argument('--model_save_freq', type=int, default=2, help='Frequency of saving model weight')
parser.add_argument("--pretrained_model", type=str, default=None, help="pretrained model path")
parser.add_argument('--freeze_av', action='store_true', help='freeze av encoder')
parser.add_argument('--freeze_text', action='store_true', help='freeze text encoder')
parser.add_argument("--train_base_path", type=str, default='dataset/audiocaps/train', help="train directory path")
parser.add_argument("--test_base_path", type=str, default='dataset/audiocaps/test', help="test directory path")


## Logging
parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
parser.add_argument('--wandb_project', type=str, default=None, help='wandb project')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb name')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')

args = parser.parse_args()


# all exp in this work is based on 224 * 224 image
im_res = 224
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 
              'dataset': args.dataset, 'mode':'train', 
              'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise, 
              'label_smooth': args.label_smooth, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 
                  'dataset': args.dataset, 'mode':'eval', 
                  'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}


def verbose_print(gpu, print_phrase):
    if gpu == 0: print(print_phrase)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # Write input hyperparameters to score file
    if args.gpu == 0:
        scorefile = open(args.result_save_path+"/scores.txt", "a+")
        scorefile.write('{} script executed\n'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
        print('\n=================args=================')
        scorefile.write('\n=================args=================\n')
        for items in vars(args):
            print(items, vars(args)[items])
            scorefile.write('{}: {}\n'.format(items, vars(args)[items]))
        scorefile.flush()
        
    # Initialize wandb
    if args.gpu == 0 and not args.no_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, resume='allow', name=args.wandb_name)


    # Make an instance of the model
    verbose_print(args.gpu, '\n=================Define Model=================')
    av_model = CAVMAEFT(label_dim=args.n_class, 
                        modality_specific_depth=args.modality_specific_depth, 
                        num_frames=args.num_frames,
                        tr_pos=not args.no_training_pos)


    if args.av_pretrain_path is not None:
        av_state_dict = torch.load(args.av_pretrain_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in av_state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
            if args.num_frames > 1 and (('patch_embed_v' in name) or ('pos_embed_v' in name)):
                del new_state_dict[name]
        miss, unexpected = av_model.load_state_dict(new_state_dict, strict=False)
        verbose_print(args.gpu, 'now load cav-mae pretrained weights from ' + args.av_pretrain_path)
        # verbose_print(args.gpu, miss)
        # verbose_print(args.gpu, unexpected)


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    param = {'mode':args.mode}
    model = get_avcap_model(tokenizer, av_model, param)
    
    text_state_dict = torch.load(args.text_pretrain_path, map_location=torch.device('cpu'))['model']
    if 'base' in args.text_pretrain_path:
        new_state_dict = OrderedDict()

        for k, v in text_state_dict.items():
            name = k[7:] # 'module.' 문자열 제거
            new_state_dict[name] = v
        miss, unexpected = model.load_state_dict(new_state_dict, strict=False)

    elif 'large' in args.text_pretrain_path:
        del text_state_dict['textual.visual_projection.0.weight']
        miss, unexpected = model.load_state_dict(text_state_dict, strict=False)

    else:
        raise NotImplementedError

    verbose_print(args.gpu, 'now load git pretrained weights from ' + args.text_pretrain_path)
    # verbose_print(args.gpu, miss)
    # verbose_print(args.gpu, unexpected)

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        
        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        model = WrappedModel(model).cuda(args.gpu)

    # Initialise dataset
    verbose_print(args.gpu, '\n=================Load Dataset=================')
    

    train_dataset = AVCapDataset(args.data_train, base_path = args.train_base_path, 
                                    num_frames = args.num_frames, audio_conf=audio_conf)
    val_dataset = AVCapDataset(args.data_val, base_path = args.test_base_path, 
                                  num_frames = args.num_frames, audio_conf=val_audio_conf)

    if args.distributed:
        verbose_print(args.gpu,'balanced sampler is not used in distributed setting')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=args.gpu, drop_last=True)
        val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset, rank=args.gpu, drop_last=True)
    else:
        verbose_print(args.gpu,'balanced sampler is not used in not distributed setting')
        train_sampler = None
        val_sampler = None
       
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=(train_sampler is None),
        drop_last=True,
        collate_fn = collate_fn,
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        collate_fn = collate_fn,
        persistent_workers=True,
    )

    
    args.iteration_per_epoch = len(train_loader)

    # Define the ModelTrainer
    verbose_print(args.gpu, '\n=================Parameter of the Model=================')
    trainer = ModelTrainer(model, **vars(args))

    # Start first epoch
    epoch = 1

    # Load model parameters
    verbose_print(args.gpu, f'\n=================Load Model - pretrained model =================')

    if args.pretrained_model is not None:
        trainer.loadParameters(args.pretrained_model)
        verbose_print(args.gpu, f"Model {args.pretrained_model} loaded from previous state!")
    
    else: 
        verbose_print(args.gpu, "Note you are finetuning a model without any finetuning.")

    best_metric, best_loss = -np.inf, np.inf

    # Run training
    for epoch in range(epoch,args.max_epoch+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        verbose_print(args.gpu, f'\n=============================================================')
        verbose_print(args.gpu, f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} Train Epoch {epoch}')
        train_stats  = trainer.train_network(train_loader, evalmode=False, epoch=epoch)
        dist.barrier()
        verbose_print(args.gpu, f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} Eval Epoch {epoch}')

        with torch.no_grad():
            result_json = trainer.train_network(val_loader, evalmode=True, epoch=epoch)
            json_save_path = args.result_save_path + "/epoch" + str(epoch) +"_coco.json"
            
            if args.gpu ==0:
                with open(json_save_path, 'w') as f:
                    json.dump(result_json, f)
                coco = COCO(args.ann_file)
                coco_result = coco.loadRes(json_save_path)
                coco_eval = COCOEvalCap(coco, coco_result)
                coco_eval.params['image_id'] = coco_result.getImgIds()
                coco_eval.evaluate()

                print(f'==============EPOCH {epoch}/{args.max_epoch}=============\n')
                stats = {}
                for metric, score in coco_eval.eval.items():
                    print(f'{metric}: {score:.3f}')
                    stats[metric] = score

                print('validation finished')
                    
                if not args.no_wandb:
                    wandb.log(stats)

                trainer.saveParameters(args.model_save_path+f"/model_lastest_ft.pth")
                
                if epoch % args.model_save_freq == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), "Saving model {:d}".format(epoch))
                    trainer.saveParameters(args.model_save_path+f"/model_{epoch}_ft.pth");                       
        

    if args.gpu == 0:
        scorefile.close()


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====

def main():

    args.save_path = os.path.join('/data/exp', args.save_path)
    i_try = 3
    while os.path.exists(args.save_path):
        if not 'try' in args.save_path:
            args.save_path += '_try2'
        else:
            new_path = args.save_path.replace(args.save_path.split('_')[-1], f'try{i_try}')
            args.save_path = new_path
            i_try += 1

    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)

if __name__ == '__main__':
    main()
