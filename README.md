# Cloud-forecast

## Overview

This is a project about cloud forecasting, as my undergraduate graduation program. I will use deep learning method to achieve this goal.
Here are some methods I have accomplished in this repository: ConvLSTM, Unet-like structure, self-attention mechanism, GAN.

I'm very grateful to the open source workers for related work, I get a lot of ideas from their projects, listed below are some of the github repositories I refer to:

https://github.com/LEOMMM1/Typhoon-satellite-Image-prediction-based-on-SA-ConvLstm-and-GAN

## Update

I will update this repository as I make progress in debugging the code.

## 模型变体介绍

目前的测试模型主要包括：
1. 基础的ConvLSTM：即直接对输入图像做ConvLSTM，输出图像。但与最开始的网络相比，使用了seq2seq的设计方案。**首先**，最初的测试代码是检测ConvLSTM面对时序云图序列的预测能力，作为是否选取其为baseline的参考，所以当时使用的数据集划分方式为滑动窗口取样，窗口长度为5，取前4张为输入，预测第5张。一般的encode2decode的结构实际上会让模型产生与输入数据形状一样的输出，即输入 [batch, frame, channel, height, width]，输出也同样。但可以使用一个trick，只取输出状态的最后一帧，这样就可以实现输入序列和输出序列长度不一样的情况。**但encode2decode和seq2seq有本质的差异**，encode2decode虽然可以通过选择帧从而实现多输入预测少输出（实际情况下也经常如此），但他不具备向后迭代的结构倾向（也可以理解为缺少时序捕捉的诱导），在后续面对更长时间段的时序预测可能会出现一些问题，所以在这里进行整体的修改。seq2seq则是依次处理输入图像，用上一帧的输出作为下一帧的输入，使其具有长序列的输出能力。后续所有的模型及变体都是使用的seq2seq的预测方案。
2. 具有上下采样的ConvLSTM：即先对输入图像进行下采样，得到特征图，在特征图上做ConvLSTM，然后再经过上采样恢复原图。这一方案可以减少ConvLSTM环节的计算量，让模型可以直接对特征进行捕捉和建模。
3. 类Unet结构、加入跳跃连接的上下采样ConvLSTM：顾名思义，借鉴了Unet网络的跳跃连接，也是在上一变体的基础上增强了其上采样过程中细节和特征恢复的能力，使模型能够捕捉更多云图边缘的细节。
4. 引入自注意力机制的上下采样ConvLSTM：与上下采样的ConvLSTM结构类似，但区别是将特征图部分的普通ConvLSTM换成了SA-ConvLSTM。SA机制让ConvLSTM除了隐层和细胞状态以外，多维护一个memory块，用于控制长程输入的掩码。所以每次SA-ConvLSTM更新状态时，都需要经过注意力加权，增强其捕捉更长序列的特征和信息。**需要强调的是**，在SA-ConvLSTM模型中还增添了**scheduled sampling**。具体来说，对于一个seq2seq的任务，在训练过程中，模型是能够看到整个序列的，也就是在预测第n帧图像时，他可以看到第n-1帧的groundtruth，从而规范自己的预测，减少了累计误差的产生。但在实际任务中，将要预测的帧是看不到groundtruth的，所以模型要用自己预测出来的帧作为下一帧的参考，此时会有累积误差的问题。解决这一问题有两种方案，一种是在训练时直接不让模型看到groundtruth，对于一个用m张输入预测n张输出的任务，只有在 t < n 时，模型才能看到真实图像，其余情况下需要用之前生成的输出作为输入；另一种方案是，在训练过程中逐步减少模型能看到groundtruth的比例，具体实现是用一个逐渐迭代变小的 η 计算生成一张与输入数据形状相同的 mask，x = p * gt + (1-p) * out。p的取值在[0, 1]，p = 1 时表示使用真值，p = 0 时表示使用上一帧输出。通过这种方式逐渐减弱模型对groundtruth的依赖。这种方案实际上是比第一种更缓和。
5. 在SA-ConvLSTM的基础上进一步添加GAN的网络：进一步升级，将上述模型看作一个生成器Generator，另外再训练一个判别器Discriminator。让判别器来判断真假序列（实际和预测），同时使用WGAN-GP中的梯度惩罚项GP（其余的损失函数部分和前面一样，都是自己定义的一个包含了ssim、l1、l2的联合损失）
