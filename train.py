#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import os
import torch
import pytorch_mask_rcnn as pmr
    
    
def main(args):
    pmr.init_distributed_mode(args)
    
    if args.lr is None:
        args.lr = 0.02 * args.batch_size * args.world_size / 16 # learning rate
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    pmr.get_gpu_prop(show=True)
    print('\ndevice: {}'.format(device))

    # ---------------------- prepare data loader ------------------------------- #
    
    dataset_train = pmr.datasets(args.dataset, args.data_dir, 'train', train=True)
    dataset_test = pmr.datasets(args.dataset, args.data_dir, 'val', train=True) # set train=True to evaluate
    
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers,
        collate_fn=pmr.collate_wrapper)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test, num_workers=args.num_workers,
        collate_fn=pmr.collate_wrapper)
    
    # -------------------------------------------------------------------------- #

    torch.manual_seed(args.seed)
    model = pmr.maskrcnn_resnet50(args.pretrained, args.num_classes, True, args.offi_weights_dir).to(device)
        
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.gamma)
        
    if os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        if 'model' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epochs']
        else:
            model_without_ddp.load_state_dict(checkpoint)
            
        del checkpoint
        torch.cuda.empty_cache()
    else:
        start_epoch = 0

    since = time.time()
    
    if args.test_only:
        print('evaluation only...')
        eval_output = pmr.evaluate(model, data_loader_test, device, args.iters)
        if pmr.get_rank() == 0:
            print(eval_output)
            
        print('\ntotal time of this evaluation: {:.2f} s'.format(time.time() - since))
        return

    print('\nalready trained: {} epochs'.format(start_epoch))
    
    # ------------------------------- train ------------------------------------ #

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print('\nepoch: {}'.format(epoch + 1))
        
        if args.distributed:
            sampler_train.set_epoch(epoch)
            
        A = time.time()
        pmr.train_one_epoch(model, optimizer, data_loader_train, device, epoch, args.print_freq, args.iters)
        A = time.time() - A
        lr_scheduler.step()
        
        
        B = time.time()
        eval_output = pmr.evaluate(model, data_loader_test, device, args.iters)
        B = time.time() - B

        if pmr.get_rank() == 0:
            pmr.collect_gpu_info([len(dataset_train) / A, len(dataset_test) / B])
            print(eval_output.get_AP())
            
            if epoch == start_epoch + args.epochs - 1:
                pmr.save_ckpt(model_without_ddp, optimizer, args.ckpt_path, lr_scheduler, False, epochs=epoch + 1)
            else:
                pmr.save_ckpt(model_without_ddp, optimizer, args.ckpt_path, lr_scheduler, True, epochs=epoch + 1)

    # -------------------------------------------------------------------------- #

    print('\ntotal time of this train: {:.2f} s'.format(time.time() - since))
    print('already trained: {} epochs\n'.format(epoch + 1))
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true')
    
    parser.add_argument('--dataset')
    parser.add_argument('--data-dir')
    parser.add_argument('--num-classes', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--lr-steps', type=int, nargs='+', default=[16, 22])
    parser.add_argument('--gamma', type=float, default=0.1)
    
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--ckpt-path')
    parser.add_argument('--offi-weights-dir')
    parser.add_argument('--print-freq', type=int, default=100)
    
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--dist-url', default='env://')
    
    parser.add_argument('--iters', type=int)
    #parser.add_argument('--', type=, default=)
    args = parser.parse_args()
    
    if args.num_classes is None:
        args.num_classes = 21 if args.dataset == 'voc' else 91
    if args.ckpt_path is None:
        args.ckpt_path = '../ckpt/checkpoint_{}.pth'.format(args.dataset) # path where to save the checkpoint.pth ##
    
    main(args)
   
    