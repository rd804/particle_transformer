import sys
sys.path.append('/scratch/rd804/particle_transformer')
import networks.example_ParticleTransformer as part
import networks.example_ParticleTransformer_finetune as part_finetune
from weaver.utils.dataset import DataConfig
import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger
import torch
import numpy as np
import copy
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
import argparse
from socket import gethostname

#import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Training Example')
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--batch_size', default=256, type=int)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def run_epoch(model, dataloader, sampler, optimizer, 
              local_rank, rank, epoch=None, mode = 'train', save_score = False):
    if mode == 'train':
        model.train()
        sampler.set_epoch(epoch)
    else:
        model.eval()
    
    if save_score:
        score = []
        label_list = []
    
    running_loss = 0.0

    for i,(data, labels) in enumerate(dataloader):
        if mode == 'train':
            optimizer.zero_grad()
            out = model(data.to(local_rank))
            if i == 0 and rank == 0:
                print(f'epoch: {epoch} ',' labels: ', labels.shape, 'labels local_rank: ', labels.to(local_rank).shape, ' data: ', data.shape, 'data local_rank: ', data.to(local_rank).shape,
                  ' rank: ', rank, ' local_rank: ', local_rank)
            
        else:
            with torch.no_grad():
                out = model(data.to(local_rank))


       # loss = F.cross_entropy(out, labels.to(local_rank))
        loss = torch.nn.BCEWithLogitsLoss()(out, labels.to(local_rank))
        if save_score:
            score.append(out)
            label_list.append(labels.to(local_rank))


        if rank == 0:
            running_loss += loss.item()
            #loss_list.append(loss.item())
        if mode == 'train':
            loss.backward()
            optimizer.step()

    if rank == 0:
        loss = running_loss / len(dataloader)

    if save_score:
        score = torch.concatenate(score, axis=0)
        label_list = torch.concatenate(label_list, axis=0)

        score = torch.concatenate([score, label_list], axis=1)
        return loss, score
    else:
        return loss

    #return loss, out




def run_multigpu(model, dataloader, sampler, optimizer=None, epochs: int = None, \
        world_size: int = 1, rank: int = 1, local_rank: int = 1,
        mode: str = 'train'):
    # Will query the runtime environment for `MASTER_ADDR` and `MASTER_PORT`.
    # Make sure, those are set!


    if mode == 'train':
        loss_list = []
        for epoch in range(0, epochs):
            loss = run_epoch(model, dataloader, sampler, optimizer, 
                                local_rank, rank, mode=mode, epoch=epoch)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
            dist.barrier()
            if rank == 0:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
                loss_list.append(loss)


            dist.barrier()

       # dist.destroy_process_group()

        print('loss_list: ', loss_list)
        loss_list = np.array(loss_list)

        return loss_list
    else:
        if rank == 0:
            print('Running test')
        dist.barrier()
        loss, out = run_epoch(model, dataloader, sampler, optimizer, 
                              local_rank, rank, mode=mode, save_score=True)
        if rank == 0:
            print('out shape: ', out.shape)

        pred = [torch.zeros_like(out) for _ in range(dist.get_world_size())]
        dist.all_gather(pred, out)
        pred = torch.cat(pred, dim=0).detach().cpu().numpy()

        print('pred shape: ', pred.shape)

        

        dist.barrier()
      #  dist.destroy_process_group()
        return loss, out


if __name__ == '__main__':
    # Get the world size from the WORLD_SIZE variable or directly from SLURM:
    args = parser.parse_args()

    rank          = int(os.environ["SLURM_PROCID"])
    world_size    = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    device_name = torch.cuda.get_device_name()

    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
        f" {gpus_per_node} allocated GPUs per node.", flush=True)

    split = ['train','val' ,'test']
    data_features = ['part_features', 'part_4momenta', 'part_labels']
    raw_data = {}

    for s in split:
        raw_data[s] = {}
        for f in data_features:
            raw_data[s][f] = torch.from_numpy(np.load(f'data/{s}_{f}.npy'))  

        #part_features = torch.from_numpy(np.load('data/part_features.npy'))
        #part_4momenta = np.load('data/part_4momenta.npy')
        #part_labels = torch.from_numpy(np.load('data/part_labels.npy'))
        raw_data[s]['mask'] = torch.tensor((raw_data[s]['part_4momenta']!=[0,0,0,0])[...,0]).unsqueeze(-1)

       # part_4momenta = torch.tensor(part_4momenta)
       # vector = torch.swapaxes(part_4momenta, 1, 2)
       # features = torch.swapaxes(part_features, 1, 2)
       # mask = torch.swapaxes(mask, 1, 2)
   # print('data: ', _data.shape, ' labels: ', _labels.shape)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    dataset = {}
    sampler = {}
    dataloader = {}
   # dist.init_process_group('nccl', world_size=world_size, rank=rank)

    for s in split:
        #if s == 'train':
        dataset[s] = torch.utils.data.TensorDataset(raw_data[s]['part_4momenta'],
                                                    raw_data[s]['part_features'],
                                                    raw_data[s]['mask'], 
                                                    raw_data[s]['part_labels'])
       # else:
        #    dataset[s] = torch.utils.data.TensorDataset(test_data, test_labels)

        sampler[s] = torch.utils.data.distributed.DistributedSampler(dataset[s], 
                                                                     num_replicas=world_size,
                                                                    rank=rank,
                                                                     shuffle=True if s == 'train' else False)
        dataloader[s] = torch.utils.data.DataLoader(dataset[s], 
                                             sampler=sampler[s], 
                                             batch_size=args.batch_size, 
                                             num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                            pin_memory=True)
 

    data_config_file = 'data/JetClass/JetClass_kin.yaml'
    data_config = DataConfig.load(data_config_file, load_observers=False, load_reweight_info=False)
    # count number of parameters

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model, model_info = part.get_model(data_config)
   # model.to(device)
    #model.mod.requires_grad_(False)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    path_part = 'models/ParT_kin.pt'
    model.load_state_dict(torch.load(path_part))




    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print('length of dataloader: ', len(dataloader['train']))
    print('length of dataset: ', len(dataset['train']))
    #print('length ')
    print('length of sampler: ', len(sampler['train']))


    loss = run_multigpu(model,dataloader['train'],sampler['train'], 
                 optimizer=optimizer, epochs=args.epochs,
                 world_size=world_size, rank=rank, 
                 local_rank=local_rank, mode='train')
    
    if rank == 0:
        print('loss: ', loss)
        np.save('loss.npy', loss)
    
    loss, out = run_multigpu(model,dataloader['test'],sampler['test'],
                             world_size=world_size, rank=rank,
                            local_rank=local_rank, mode='test')
    
    dist.destroy_process_group()
    




# for i in range(len(features)//mini_batch):
#     with torch.no_grad():
#         pred.append(model.forward(features[i*mini_batch:(i+1)*mini_batch], vector[i*mini_batch:(i+1)*mini_batch], mask[i*mini_batch:(i+1)*mini_batch]))
#     print(f'Batch {i} done')

#with torch.no_grad():
 #   pred = model.forward(features[:100], vector[:100], mask[:100])
