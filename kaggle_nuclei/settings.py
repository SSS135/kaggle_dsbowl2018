
train_pad = 64
train_size = 512 - train_pad * 2
resnet_norm_mean = [124 / 255, 117 / 255, 104 / 255] # [0.485, 0.456, 0.406]
resnet_norm_std = [1 / (.0167 * 255)] * 3 # [0.229, 0.224, 0.225]
box_padding = 0.2
