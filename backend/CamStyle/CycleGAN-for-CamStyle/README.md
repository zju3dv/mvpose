# Training CamStyle with CycleGAN

CamStyle is trained with [CycleGAN-pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)


### Preparation

#### Requirements: Python=3.6 and Pytorch=0.4.0

1. Install [Pytorch](http://pytorch.org/)

2. Download dataset
   
   - Market-1501   [[BaiduYun]](https://pan.baidu.com/s/1ntIi2Op) [[GoogleDriver]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
   
   - DukeMTMC-reID   [[BaiduYun]](https://pan.baidu.com/share/init?surl=kUD80xp) (password: chu1) [[GoogleDriver]](https://drive.google.com/file/d/0B0VOCNYh8HeRdnBPa2ZWaVBYSVk/view)
   
   - Move them to 'CamStyle/CycleGAN-for-CamStyle/data/market (or duke)'

# Train CamStyle models

  ```Shell
  # For Market-1501
  sh train_market.sh
  # For Duke
  sh train_duke.sh
  ```

# Generate CamStyle images

  ```Shell
  # For Market-1501
  sh test_market.sh
  # For Duke
  sh test_duke.sh
  ```

## Citation
If you use this code for your research, please cite our papers.
```

@inproceedings{zhong2018camera,
  title={Camera Style Adaptation for Person Re-identification},
  author={Zhong, Zhun and Zheng, Liang and Zheng, Zhedong and Li, Shaozi and Yang, Yi},
  booktitle={CVPR},
  year={2018}
}

@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}

```
