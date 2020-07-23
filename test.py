import hrnet
import torch
import config.hrnet_config as cfg

if __name__ == "__main__":
    net = hrnet.get_seg_model(cfg.hr_config)
    img = torch.ones((1, 3, 256, 256))
    out = net(img)
    print(out.shape)