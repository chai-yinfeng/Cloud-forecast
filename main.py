import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SequenceDataset
from train import train_model
from test import test_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='simple', 
                        help="one of ['simple','encode2decode','encode2decode_unet','sa_encode2decode','sa_encode2decode_gan']")
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
    train_dataset = SequenceDataset(
        image_folder='/root/autodl-tmp/Infrared_cloudmap/pic1028',
        input_frames=args.input_frames,
        target_frames=args.output_frames,
        resize_shape=(args.img_size, args.img_size)
    )
    val_dataset = SequenceDataset(
        image_folder='/root/autodl-tmp/Infrared_cloudmap/pic1',
        input_frames=args.input_frames,
        target_frames=args.output_frames,
        resize_shape=(args.img_size, args.img_size)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3) 训练
    writer = SummaryWriter(log_dir=f"runs/{args.model_name}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, D = train_model(
        model_name=args.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        params=params,
        device=device,
        writer=writer
    )

    # 4) 测试和可视化
    results = test_model(
        model_name=args.model_name,
        model=model,
        test_loader=val_loader,     # test_loader
        device=device,
        writer=writer
    )

    # print("Final test results:", results)
    writer.close()

if __name__ == '__main__':
    main()

