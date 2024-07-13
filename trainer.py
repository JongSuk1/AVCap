import wandb
import torch
import torch.nn as nn
from scheduler import Scheduler
from torch.cuda.amp import autocast, GradScaler
from utils import ProgressMeter, AverageMeter
import time
import os
import sys
import utils
import json
import torch.distributed as dist

def to_device(batch, device, non_blocking=True):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    elif isinstance(batch, dict):
        return {k: to_device(v, device, non_blocking) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_device(x, device, non_blocking) for x in batch]
    else:
        return batch
    

class WrappedModel(nn.Module):
    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, batch):
        
        output = self.module(batch)

        return output
    
class ModelTrainer(nn.Module):
    def __init__(self, avcap, gpu, mixedprec=True, freeze_av=False, freeze_text = False, no_wandb=False, **kwargs):
        super(ModelTrainer, self).__init__()

        self.__model__ = avcap
        self.gpu = gpu
        self.mixedprec = mixedprec
        self.freeze_av = freeze_av
        self.freeze_text = freeze_text

        self.max_epoch = kwargs['max_epoch']
        self.lr = kwargs['lr']
        self.wd = kwargs['weight_decay']
        self.ipe = kwargs['iteration_per_epoch']
        self.warmup_epoch = kwargs['warmup_epoch']
        self.ipe_scale = 1.0
        
        av_params = self.__model__.module.av_encoder.parameters()
        text_params = self.__model__.module.textual.named_parameters()

        self.tokenizer = self.__model__.module.tokenizer
        self.sos_index = self.__model__.module.tokenizer.cls_token_id
        self.eos_index = self.__model__.module.tokenizer.sep_token_id
        
        with open(kwargs['torch2iid_file']) as f:
            self.torch2iid =json.load(f)

        if self.freeze_av == True:
            if self.gpu == 0: print('Pretrained av encoder parameters are frozen.')
            for param in av_params:
                param.requires_grad = False
        
        if self.freeze_text == True:
            if self.gpu == 0: print('Pretrained text decoder parameters are frozen.')
            for name, param in text_params:
                if 'visual_projection' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        trainables = [p for p in self.__model__.parameters() if p.requires_grad]

        self.__optimizer__ = torch.optim.AdamW(params=self.__model__.module.parameters(), 
                                               lr=self.lr, weight_decay=self.wd, 
                                               betas=(0.95, 0.999))
        

        if self.gpu == 0:
            print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in self.__model__.parameters()) / 1e6))
            print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
            print(f'Start : base lr: {self.lr}')

        self.__scheduler__ = Scheduler(self.__optimizer__, warmup_steps=int(self.warmup_epoch*self.ipe), 
                                       start_lr=2e-7, ref_lr=self.lr, T_max=int(self.ipe_scale*self.max_epoch*self.ipe), 
                                       final_lr=1e-6)

        self.scaler = GradScaler() if self.mixedprec else None

        # logging
        self.no_wandb = no_wandb
        self.print_freq = kwargs['print_freq']
        self.result_save_path = kwargs['result_save_path']


    def train_network(self, loader=None, evalmode=None, output_pred=False, pred=False, epoch=-1):
        # Setting for the logging
        batch_time = AverageMeter('Time', ':6.2f')
        data_time = AverageMeter('Data', ':6.2f')
        mem = AverageMeter('Mem (GB)', ':6.1f')
        metrics = AverageMeter('Train Loss', ':1.3e') if not evalmode else AverageMeter('Val Loss', ':1.3e')

        progress = ProgressMeter(
            len(loader),
            [batch_time, data_time, mem, metrics],
            prefix="Epoch: [{}]".format(epoch))
        
        # number of model parameters
        param_num = 0
        for p in self.__model__.parameters():
            param_num += p.numel()

        if evalmode:
            self.__model__.eval()
            A_tid, A_pred, result_json = [],[],[]
        else:
            self.__model__.train()

        data_iter = 0
        end = time.time()
        
        for i,batch in enumerate(loader):

            # measure data loading time
            data_time.update(time.time() - end)

            # transform input to torch cuda tensor
            batch_to_device = to_device(batch, device=self.gpu)

            # ==================== FORWARD PASS ====================
            with autocast(enabled=self.mixedprec):
                output = self.__model__(batch_to_device)
             
            if not evalmode:
                self.__scheduler__.step()

                if self.mixedprec:
                    # mixed precision
                    self.scaler.scale(output['vl_l_loss']).backward()
                    self.scaler.step(self.__optimizer__)
                    self.scaler.update();       
                else:
                    # single precision
                    output['vl_l_loss'].backward()
                    self.__optimizer__.step()

                self.zero_grad()

                # logging
                metrics.update(output['vl_l_loss'], loader.batch_size)
            
            elif evalmode:
                A_tid.append(output['torch_id'])
                A_pred.append(output['predictions'])

            # measure elapsed time and memory
            batch_time.update(time.time() - end)
            end = time.time()
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.print_freq == 0:
                if self.gpu == 0:

                    if not self.no_wandb and not evalmode:
                        param_sum = 0
                        for p in self.__model__.parameters():
                            param_sum += torch.pow(p.detach(),2).sum()
                        param_avg = torch.sqrt(param_sum) / param_num
                        wandb.log({"Train Loss": metrics.val,
                            'scaler': self.scaler.get_scale(),
                            'lr': self.__optimizer__.param_groups[0]['lr'],
                            'param_avg': param_avg,
                        })
                    log_info = progress.display(data_iter)
                    
                    if not pred:
                        with open(os.path.join(self.result_save_path, 'log.txt'), 'a') as f:
                            msg = 'Eval: '+ '\t'.join(log_info) + '\n' if evalmode else 'Train: '+ '\t'.join(log_info) + '\n'
                            
                            f.write(msg)
            data_iter += 1

        if evalmode:
            tid_output = torch.cat(A_tid)
            pred_output = torch.cat(A_pred)

            all_id, all_pred = utils.all_gather_batch([tid_output, pred_output])
            for tid, predict in zip(all_id, all_pred):
                coco_json = {}
                predict = predict.tolist()
                sos_position = predict.index(self.sos_index) if self.sos_index in predict else None
                eos_position = predict.index(self.eos_index) if self.eos_index in predict else None
                if sos_position is not None and eos_position is not None and sos_position < eos_position:
                    result = predict[sos_position+1:eos_position]
                else:
                    result = []
                coco_json['image_id'] = self.torch2iid[str(tid.item())]
                coco_json['caption'] = self.tokenizer.decode(result)
                result_json.append(coco_json)
            return result_json

        elif not evalmode:
            sys.stdout.write("\n")
            progress.synchronize()
            
            return {
                'loss': metrics.avg,
                'lr': self.__optimizer__.param_groups[0]['lr'],
            }

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path)

    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                if name not in self_state:
                    if self.gpu == 0: print("{} is not in the model.".format(origname))
                    continue
                else:
                    if self.gpu == 0: print("{} is loaded in the model".format(name))
            else:
                if self.gpu == 0: print("{} is loaded in the model".format(name))

            if self_state[name].size() != loaded_state[origname].size():
                if self.gpu == 0: print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()));
                continue

            self_state[name].copy_(param)