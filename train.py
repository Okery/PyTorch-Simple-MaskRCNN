import bisect
import glob
import os
import re
import time
import torch
import pytorch_mask_rcnn as pmr
    
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
        
    # ---------------------- prepare data loader ------------------------------- #
    
    dataset_train = pmr.datasets(args.dataset, args.data_dir, "train2017", train=True)
    dataset_test = pmr.datasets(args.dataset, args.data_dir, "val2017", train=True) # set train=True for eval

    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
    indices = [i for i in range(len(dataset_test))]
    d_test = torch.utils.data.Subset(dataset_test, indices)
        
    args.warmup_iters = max(1000, len(d_train))
    
    # -------------------------------------------------------------------------- #

    print(args)
    num_classes = len(d_train.dataset.classes) + 1 # including background class
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: bisect.bisect([22, 26], x) ** 0.1
    
    start_epoch = 0
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #
        
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print("lr_epoch: {:.4f}, factor: {:.4f}".format(args.lr_epoch, lr_lambda(epoch)))
        iter_train = pmr.train_one_epoch(model, optimizer, d_train, device, epoch, args)
        A = time.time() - A
        
        B = time.time()
        eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
        B = time.time() - B

        trained_epoch = epoch + 1
        print("training: {:.2f} s, evaluation: {:.2f} s".format(A, B))
        pmr.collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        print(eval_output.get_AP())

        pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, eval_info=str(eval_output))

        # it will create many checkpoint files during training, so delete some.
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 5
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))
        
    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.2f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")
    
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--data-dir", default="/data/coco2017")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results")
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[22, 26])
    parser.add_argument("--lr", type=float) # lr = batch_size / 16 * 0.02
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100) # iters per epoch, -1 denotes auto
    parser.add_argument("--print-freq", type=int, default=200)
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16
    if args.ckpt_path is None:
        args.ckpt_path = "./checkpoint.pth"
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "results.pth")
    
    main(args)
    
    