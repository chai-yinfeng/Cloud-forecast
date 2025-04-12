import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SequenceDataset, create_concat_dataset
from train import train_model
from test import test_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='simple', 
                        help="one of ['simple','encode2decode','encode2decode_unet','sa_encode2decode','sa_encode2decode_unet','sa_encode2decode_gan']")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input_frames', type=int, default=5,
                        help="Number of input frames for the model.")
    parser.add_argument('--output_frames', type=int, default=5,
                        help="Number of output/prediction frames for the model.")
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--att_hidden_dim', type=int, default=16)

    # 多个训练/验证文件夹路径参数(支持传多个路径)
    parser.add_argument('--train_folders', type=str, nargs='+', 
                        default=['/root/autodl-tmp/Infrared_cloudmap/pic1028 /root/autodl-tmp/Infrared_cloudmap/pic1'], 
                        help="List of training image folder paths.")
    parser.add_argument('--val_folders', type=str, nargs='+', 
                        default=['/root/autodl-tmp/Infrared_cloudmap/val'], 
                        help="List of validation image folder paths.")

    # 用于加载 checkpoint 和跳过训练
    parser.add_argument('--load_checkpoint', type=str, default='',
                        help="Path to checkpoint .pth file. If provided, the model will load from this checkpoint.")
    parser.add_argument('--test_only', action='store_true',
                        help="If set, skip training and only do testing (requires --load_checkpoint).")

    return parser.parse_args()

def main():
    args = parse_args()

    # 1) 组装超参
    params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,

        'input_window_size': args.input_frames,
        'output_window_size': args.output_frames,
        'img_size': (args.img_size, args.img_size),

        # schedule sampling相关参数(仅在注意力+SS模型时会用)
        'sampling_start_value': 1.0,
        'ss_total_iter': 50000,
        'ss_decay': 0.00002,

        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'bias': True,

        'kernel_size': args.kernel_size,
        'padding': args.padding,
        'att_hidden_dim': args.att_hidden_dim,
    }

    # 2) 数据集
    train_dataset = create_concat_dataset(
        folders=args.train_folders,
        input_frames=args.input_frames,
        target_frames=args.output_frames,
        resize_shape=(args.img_size, args.img_size)
    )
    val_dataset = create_concat_dataset(
        folders=args.val_folders,
        input_frames=args.input_frames,
        target_frames=args.output_frames,
        resize_shape=(args.img_size, args.img_size)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3) 训练
    writer = SummaryWriter(log_dir=f"runs/{args.model_name}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = None
    D = None

    if args.test_only:
        from train import get_model
        model = get_model(args.model_name, params).to(device)

        if args.load_checkpoint == '':
            raise ValueError("You must provide --load_checkpoint when using --test_only")

        # 加载 checkpoint
        print(f"Loading checkpoint from {args.load_checkpoint} ...")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 只测试
        print("Test-only results:")
        results = test_model(
            model_name=args.model_name,
            model=model,
            test_loader=val_loader,
            device=device,
            writer=writer
        )
    
    else:
        model, D = train_model(
            model_name=args.model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            params=params,
            device=device,
            writer=writer,
            # load_checkpoint=args.load_checkpoint   # 因为没有加载优化器和eta状态, 这里不可以用于恢复训练
        )

        # 4) 训练完测试
        results = test_model(
            model_name=args.model_name,
            model=model,
            test_loader=val_loader,     # test_loader
            device=device,
            writer=writer
        )

    # print("Final test results:", results)     # test_model中包含输出测试指标到终端的代码
    writer.close()

if __name__ == '__main__':
    main()

