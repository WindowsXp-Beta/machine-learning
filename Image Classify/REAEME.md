# README

> By 魏新鹏

```
.
├── REAEME.md
├── cifar10_v1.py
├── cifar10_v2.py
├── data
├── report
└── utils.py
```

报告中所述的`第一阶段`训练代码在cifar10_v1.py中，它包括LeNet, AlexNet, 和ResNet18的实现，以及训练过程。

```
python cifar10_v1.py
```

报告所述的`第二阶段`训练代码在cifar10_v2.py中，它包括了变学习率训练的相关训练代码。

从上一次中断处继续训练：`--resume`

改变初识学习率：`--lr`

```
python cifar10_v2.py
```

> 训练AlexNet时需要将train_transform中对resize的注释去除。
>
> 每次新的训练时需要修改tensorboard写入log的文件名。