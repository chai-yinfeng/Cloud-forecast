import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from model import (
    ConvLSTMSimple,
    ConvLSTMEncode2Decode,
    ConvLSTMEncode2DecodeUNet,
    SA_ConvLSTMEncode2Decode,
    SA_ConvLSTMEncode2DecodeUNet,
    Discriminator,
    generator_loss_function,
    sa_lstm_loss
)

import numpy as np
import time, os
import random
from tqdm import tqdm

# 根据模型名称 & 参数，实例化要测试的模型
def get_model(model_name: str, params: dict):
    """
    model_name: 
      - 'simple' -> ConvLSTMSimple
      - 'encode2decode' -> ConvLSTMEncode2Decode
      - 'encode2decode_unet' -> ConvLSTMEncode2DecodeUNet
      - 'sa_encode2decode' -> SA_ConvLSTMEncode2Decode
      - 'sa_encode2decode_gan' -> SA_ConvLSTMEncode2Decode + GAN
      - 'sa_encode2decode_unet' -> SA_ConvLSTMEncode2DeocdeUNet
    """
    if model_name == 'simple':
        model = ConvLSTMSimple(params)
    elif model_name == 'encode2decode':
        model = ConvLSTMEncode2Decode(params)
    elif model_name == 'encode2decode_unet':
        model = ConvLSTMEncode2DecodeUNet(params)
    elif model_name == 'sa_encode2decode':
        model = SA_ConvLSTMEncode2Decode(params)
    elif model_name == 'sa_encode2decode_gan':
        # 其实与上面相同，只是后续训练时会额外加入GAN过程
        model = SA_ConvLSTMEncode2Decode(params)
    elif model_name == 'sa_encode2decode_unet':
        model = SA_ConvLSTMEncode2DecodeUNet(params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


# schedule_sampling: 根据需求使用, 目前只有 SA_ConvLSTMEncode2Decode 里 forward 可支持 mask_true
def schedule_sampling(eta, current_iter, total_iter=50000, dec_rate=0.00002):
    """
    简易的Scheduled Sampling概率衰减函数:
    - eta: 当前使用真实帧的概率
    - current_iter: 当前迭代数 (可能是 epoch * iters_per_epoch + i)
    - total_iter:  在多少迭代之内从1.0衰减到0
    - dec_rate:    每次迭代衰减多少

    返回: 新的 eta 值 (>=0)
    """
    if current_iter < total_iter:
        eta -= dec_rate
    else:
        eta = 0.0
    
    if eta < 0:
        eta = 0.0
    return eta

def create_mask_true(batch_size, out_frames, in_channels, height, width, eta):
    """
    基于eta随机生成一个mask_true张量:
      - 维度: [B, out_frames, C, H, W]
      - 每个位置取值 ∈ {0,1}, 
         1 表示该时间步使用"真实帧" (ground-truth) 作为输入,
         0 表示使用"模型预测" 作为输入.
    """
    # shape: (B, out_frames, C, H, W)
    mask = np.zeros((batch_size, out_frames, in_channels, height, width), dtype=np.float32)
    random_flip = np.random.rand(batch_size, out_frames)

    for b in range(batch_size):
        for t in range(out_frames):
            if random_flip[b, t] < eta:
                mask[b, t, :, :, :] = 1     # use ground-truth
            else:
                mask[b, t, :, :, :] = 0     # use model output
    mask = torch.tensor(mask)
    return mask


# 不带 GAN 的基本训练循环
def train_basic_model(model, model_name, train_loader, val_loader, params, device, writer=None):
    """
    针对不需要判别器的情况:
      - ConvLSTMSimple
      - ConvLSTMEncode2Decode
      - ConvLSTMEncode2DecodeUNet
      - SA_ConvLSTMEncode2Decode
      - SA_ConvLSTMEncode2DecodeUNet
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=params.get('lr', 1e-3))
    criterion = sa_lstm_loss    # 根据具体情况测试其他损失函数

    use_scheduled_sampling = False
    if 'sa' in model_name:
        # 如果是 'sa_encode2decode' 或 'sa_encode2decode_unet' 就启用schedule sampling
        use_scheduled_sampling = True
        eta = params.get('sampling_start_value', 1.0)

    # 训练循环
    epochs = params['epochs']
    input_frames = params['input_window_size']
    target_frames = params['output_window_size']

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        iter_count = 0

        # ==================== 训练集 ====================
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for batch_idx, (batch_data, batch_target) in enumerate(pbar):
            batch_data = batch_data.to(device)      # [B, input_frames, 1, H, W]
            batch_target = batch_target.to(device)  # [B, target_frames, 1, H, W]
            B, _, C, H, W = batch_data.shape

            optimizer.zero_grad()

            if use_scheduled_sampling:
                global_iter = epoch * len(train_loader) + iter_count
                eta = schedule_sampling(eta, global_iter, total_iter=params.get('ss_total_iter', 50000), dec_rate=params.get('ss_decay', 0.00002))
                mask_true = create_mask_true(B, target_frames, C, H, W, eta).to(device)

                pred = model(batch_data, batch_target, mask_true=mask_true, is_training=True)
                # 返回 [B, target_frames, C, H, W]
            else:
                pred = model(batch_data)

            # 计算loss
            loss, L_rec, L_ssim, L_perc, L_edge = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            iter_count += 1

            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / iter_count

        # ============== 验证集 ================
        model.eval()
        total_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
            for val_idx, (val_data, val_target) in enumerate(pbar):
                val_data = val_data.to(device)
                val_target = val_target.to(device)

                if use_scheduled_sampling:
                    # 验证时通常不需要 schedule sampling，直接用模型预测
                    pred = model(val_data, val_target, mask_true=None, is_training=False)
                else:
                    pred = model(val_data)

                val_loss, L_rec, L_ssim, L_perc, L_edge = criterion(pred, val_target)
                total_val_loss += val_loss.item()
                val_count += 1

                pbar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})

        avg_val_loss = total_val_loss / val_count

        if writer is not None:
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Val',   avg_val_loss,   epoch)

        print(f"[{model_name}] Epoch {epoch+1}/{epochs} | TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f}")

        # 随机可视化一些预测结果到 TensorBoard
        num_val_batches = len(val_loader)
        random_batch_idx = random.randint(0, num_val_batches - 1)

        for i, (val_data, val_target) in enumerate(val_loader):
            if i == random_batch_idx:
                val_data = val_data.to(device)
                val_target = val_target.to(device)
                if use_scheduled_sampling:
                    pred_vis = model(val_data, val_target, mask_true=None, is_training=False)
                else:
                    pred_vis = model(val_data)
                # pred_vis: [B, out_frames, 1, H, W]
                
                # 拼接图片方便显示
                pred_frames = []
                for t in range(pred_vis.shape[1]):
                    pred_frames.append(pred_vis[0, t])  # [C, H, W]
                # 将 list 里的张量在宽度维度上拼接 -> [C, H, W*T]
                pred_concat = torch.cat(pred_frames, dim=-1)

                # 同理处理 GroundTruth
                gt_frames = []
                for t in range(val_target.shape[1]):
                    gt_frames.append(val_target[0, t])  # [C, H, W]
                gt_concat = torch.cat(gt_frames, dim=-1)  # [C, H, W*T]

                if writer is not None:
                    writer.add_image("Prediction", pred_concat, epoch)
                    writer.add_image("GroundTruth", gt_concat, epoch)
                break

        if (epoch + 1) % 5 == 0:
            # 保存 checkpoint
            save_path = f"checkpoint/{model_name}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eta': eta if use_scheduled_sampling else None
            }, save_path)

    return model


# 带 GAN 的训练循环 (SA_ConvLSTMEncode2Decode + GAN)
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """
    使用WGAN-GP (https://arxiv.org/abs/1704.00028) 算法中的梯度惩罚项 GP:
      - real_samples, fake_samples => 判别器D的输入 (B, 1, T, H, W)
      - alpha是[0,1]间随机插值
      - 求梯度范数与1的偏离, 记做 gradient_penalty
    """
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # gradients shape = [B, 1, T, H, W] => L2 norm
    grad_l2norm = gradients.norm(2, dim=[1,2,3,4])
    gp = torch.mean((grad_l2norm - 1) ** 2)
    return gp


def train_gan_model(model, train_loader, val_loader, params, device, writer=None):
    """
    针对 'sa_encode2decode_gan' 模型的训练过程:
    - 生成器: SA_ConvLSTMEncode2Decode
    - 判别器: Discriminator
    - 损失: WGAN-GP + (L1 + L2 + SSIM)
    """
    model = model.to(device)
    D = Discriminator().to(device)
    g_optim = Adam(model.parameters(), lr=params.get('lr_g', 0.0002), betas=(0.5, 0.999))
    d_optim = Adam(D.parameters(), lr=params.get('lr_d', 0.0001), betas=(0.5, 0.999))

    G_loss_fn = generator_loss_function()  # 包含 L1, L2, SSIM, adversarial

    # 同样地, 需要 schedule_sampling
    eta = params.get('sampling_start_value', 1.0)
    epochs = params['epochs']
    lambda_gp = 10  # 梯度惩罚系数
    input_frames = params['input_window_size']
    target_frames = params['output_window_size']

    for epoch in range(epochs):
        model.train()
        D.train()
        total_D_loss = 0.0
        total_G_loss = 0.0
        iter_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for batch_idx, (batch_data, batch_target) in enumerate(pbar):
            batch_data = batch_data.to(device)     # [B, in_frames, 1, H, W]
            batch_target = batch_target.to(device) # [B, out_frames, 1, H, W]
            B, _, C, H, W = batch_data.shape

            # 1) schedule sampling
            global_iter = epoch * len(train_loader) + iter_count
            eta = schedule_sampling(eta, global_iter, total_iter=params.get('ss_total_iter', 50000), dec_rate=params.get('ss_decay', 0.00002))
            mask_true = create_mask_true(B, target_frames, C, H, W, eta).to(device)

            # 2) 先训练判别器 D
            d_optim.zero_grad()

            gen_img = model(batch_data, batch_target, mask_true=mask_true, is_training=True) 
            # gen_img shape => [B, out_frames, C, H, W]

            # 把输入帧和生成帧拼起来, 作为D的输入: [B, C, T= in+out_frames, H, W]
            # real => input + target
            # fake => input + gen_img
            fake_seq = torch.cat([batch_data, gen_img], dim=1) # => [B, in_frames+out_frames, 1, H, W]
            real_seq = torch.cat([batch_data, batch_target], dim=1)

            # [B,C,T,H,W] => [B,1,T,H,W] (C=1)
            fake_seq = fake_seq.permute(0, 2, 1, 3, 4).contiguous()
            real_seq = real_seq.permute(0, 2, 1, 3, 4).contiguous()

            # 判别真实和生成序列
            fake_out = D(fake_seq.detach())  # [B,1]
            real_out = D(real_seq)

            # 计算 WGAN-GP 判别器损失
            gradient_penalty = compute_gradient_penalty(D, real_seq.data, fake_seq.data, device)
            d_loss = -torch.mean(real_out) + torch.mean(fake_out) + lambda_gp * gradient_penalty
            d_loss.backward()
            d_optim.step()

            total_D_loss += d_loss.item()

            pbar.set_postfix({"D_loss": f"{d_loss.item():.4f}"})

            # 3) 再训练生成器 G
            # 隔一步或隔几步做一次generator的更新, 目前为 2步判别器 + 1步生成器
            if iter_count % 2 == 0:
                g_optim.zero_grad()
                # 重新计算D(fake)
                fake_out_2 = D(fake_seq)
                # 计算G的综合损失 (包含 L1, L2, SSIM, 对抗, etc)
                L_total, L_rec, L_ssim, L_adv = G_loss_fn(gen_img, batch_target, fake_out_2)
                L_total.backward()
                g_optim.step()
                total_G_loss += L_total.item()

                pbar.set_postfix({"G_loss": f"{L_total.item():.4f}"})

            iter_count += 1

        # 打印平均loss
        avg_D_loss = total_D_loss / iter_count
        avg_G_loss = total_G_loss / (iter_count/2 if iter_count>0 else 1)

        if writer is not None:
            writer.add_scalar('Loss/Discriminator', avg_D_loss, epoch)
            writer.add_scalar('Loss/Generator',   avg_G_loss, epoch)

        print(f"[GAN] Epoch {epoch+1}/{epochs} | D_loss={avg_D_loss:.4f} | G_loss={avg_G_loss:.4f}")

        num_val_batches = len(val_loader)
        random_batch_idx = random.randint(0, num_val_batches - 1)

        for i, (val_data, val_target) in enumerate(val_loader):
            if i == random_batch_idx:
                val_data = val_data.to(device)
                val_target = val_target.to(device)

                pred_vis = model(val_data, val_target, mask_true=None, is_training=False)

                pred_frames = []
                for t in range(pred_vis.shape[1]):
                    pred_frames.append(pred_vis[0, t])  # 
                # [C, H, W] -> [C, H, W*T]
                pred_concat = torch.cat(pred_frames, dim=-1)
                gt_frames = []
                for t in range(val_target.shape[1]):
                    gt_frames.append(val_target[0, t])
                gt_concat = torch.cat(gt_frames, dim=-1)

                if writer is not None:
                    writer.add_image("GAN_Prediction", pred_concat, epoch)
                    writer.add_image("GAN_GroundTruth", gt_concat, epoch)
                break

        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'gen_state_dict': model.state_dict(),
                'dis_state_dict': D.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
                'eta': eta
            }, f"checkpoint/sa_encode2decode_gan_epoch{epoch+1}.pth")

    return model, D


# 封装一个总的 train_model 接口 (让 main.py 调用)
def train_model(model_name, train_loader, val_loader, params, device, writer=None):
    """
    外部接口:
      - 如果model_name中带'gan', 则走 train_gan_model
      - 否则走 train_basic_model

    返回:
      - model: 生成器 (训练好的)
      - D: 如果是GAN, 则返回判别器; 否则返回 None
    """
    # 1) 获取生成器
    model = get_model(model_name, params)

    # 2) 若是 gan
    if 'gan' in model_name:
        # 走GAN的训练
        model, D = train_gan_model(model, train_loader, val_loader, params, device, writer)
        return model, D
    else:
        # 普通训练
        model = train_basic_model(model, model_name, train_loader, val_loader, params, device, writer)
        return model, None