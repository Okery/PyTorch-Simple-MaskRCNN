#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import os
import torch
import pytorch_mask_rcnn as pmr
    
    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    pmr.gpu_info(show=True)
    print('\ndevice: {}'.format(device))

    dataset_train = pmr.datasets(args.dataset, args.data_dir, 'train', train=True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                                    collate_fn=pmr.collate_wrapper, shuffle=True)
    
    dataset_test = pmr.datasets(args.dataset, args.data_dir, 'val', train=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, collate_fn=pmr.collate_wrapper)

    torch.manual_seed(args.seed)
    model = pmr.maskrcnn_resnet50(args.pretrained, args.num_classes, True, args.offi_ckpt_dir).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # -------------------------------------------------------------
    
    if os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        checkpoint = {'epochs': 0, 'images': 0}

    start_epoch, total_images = checkpoint['epochs'], checkpoint['images']
    print('\nalready trained: {} epochs, {} images'.format(start_epoch, total_images))

    del checkpoint
    torch.cuda.empty_cache()

    since = time.time()

    # ------------------train---------------------

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print('\nepoch: {}'.format(epoch + 1))
        pmr.train_one_epoch(model, optimizer, data_loader_train, device, epoch, args.print_freq)
        lr_scheduler.step()
        
        eval_output = pmr.evaluate(model, data_loader_test, device)
        print(pmr.parse_eval_output(eval_output))

        epo = epoch + 1
        imgs = total_images + (epoch - start_epoch) * len(dataset_train) + i * args.batch_size
        if epo == start_epoch + args.epochs:
            pmr.save_ckpt(model, optimizer, args.ckpt_path, lr_scheduler, False, epochs=epo, images=imgs)
        else:
            pmr.save_ckpt(model, optimizer, args.ckpt_path, lr_scheduler, True, epochs=epo, images=imgs)

    # ------------------train---------------------

    print('\ntotal time of this train: {:.2f} s'.format(time.time() - since))
    print('already trained: {} epochs, {} images\n'.format(epo, imgs))
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--train-num-batches', type=int, default=-1)
    
    parser.add_argument('--dataset')
    parser.add_argument('--data-dir')
    parser.add_argument('--num-classes', type=int)
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--step-size', type=int, default=7)
    parser.add_argument('--gamma', type=float, default=0.1)
    
    parser.add_argument('--ckpt-path')
    parser.add_argument('--offi-ckpt-dir')
    parser.add_argument('--print-freq', type=int, default=100)
    #parser.add_argument('--', type=, default=)
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * args.batch_size / 16 # learning rate
    if args.num_classes is None:
        args.num_classes = 21 if args.dataset == 'voc' else 91
    if args.ckpt_path is None:
        args.ckpt_path = '../ckpt/checkpoint_{}.pth'.format(args.dataset) # path where to save the checkpoint.pth ##
    
    main(args)

