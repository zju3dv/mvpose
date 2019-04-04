# Camera Style Adaptation for Person Re-identification
================================================================

Code for Camera Style Adaptation for Person Re-identification (CVPR 2018). 

### Preparation

#### Requirements: Python=3.6 and Pytorch>=0.3.0

1. Install [Pytorch](http://pytorch.org/)

2. Download dataset
   
   - Market-1501   [[BaiduYun]](https://pan.baidu.com/s/1ntIi2Op) [[GoogleDriver]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
   
   - DukeMTMC-reID   [[BaiduYun]](https://pan.baidu.com/share/init?surl=kUD80xp) (password: chu1) [[GoogleDriver]](https://drive.google.com/file/d/0B0VOCNYh8HeRdnBPa2ZWaVBYSVk/view)
   
   - Move them to 'CamStyle/data/market (or duke)'
   

3. Download CamStyle Images
   
   - Market-1501-Camstyle [[GoogleDriver]](https://drive.google.com/open?id=1z9bc-I23OyLCZ2eTms2NTWSq4gePp2fr)
   
   - DukeMTMC-reID-CamStyle  [[GoogleDriver]](https://drive.google.com/open?id=1QX3K_RK1wBPPLQRYRyvG0BIf-bzsUKbt)
   
   - Move them to 'CamStyle/data/market (or duke)/bounding_box_train_camstyle'


### CamStyle Generation
You can generate CamStyle imgaes with [CycleGAN-for-CamStyle](https://github.com/zhunzhong07/CamStyle/tree/master/CycleGAN-for-CamStyle)


### Training and test re-ID model

1. IDE
  ```Shell
  # For Market-1501
  python main.py -d market --logs-dir logs/market-ide
  # For Duke
  python main.py -d duke --logs-dir logs/duke-ide
  ```
2. IDE + CamStyle
  ```Shell
  # For Market-1501
  python main.py -d market --logs-dir logs/market-ide-camstyle --camstyle 46
  # For Duke
  python main.py -d duke --logs-dir logs/duke-ide--camstyle --camstyle 46
  ```
  
3. IDE + CamStyle + Random Erasing[4]
  ```Shell
  # For Market-1501
  python main.py -d market --logs-dir logs/market-ide-camstyle-re --camstyle 46 --re 0.5
  # For Duke
  python main.py -d duke --logs-dir logs/duke-ide--camstyle-re --camstyle 46 --re 0.5
  ```

4. IDE + CamStyle + Random Erasing[4] + re-ranking[3]
  ```Shell
  # For Market-1501
  python main.py -d market --logs-dir logs/market-ide-camstyle-re --camstyle 46 --re 0.5 --rerank
  # For Duke
  python main.py -d duke --logs-dir logs/duke-ide--camstyle-re --camstyle 46 --re 0.5 --rerank
  ```
  
### Results

<table>
   <tr>
      <td></td>
      <td colspan="2">Market-1501</td>
      <td colspan="2">Duke</td>
   </tr>
   <tr>
      <td>Methods</td>
      <td>Rank-1</td>
      <td>mAP</td>
      <td>Rank-1</td>
      <td>mAP</td>
   </tr>
   <tr>
      <td>IDE</td>
      <td>85.6</td>
      <td>65.8</td>
      <td>72.3</td>
      <td>51.8</td>
   </tr>
   <tr>
      <td>IDE+CamStyle</td>
      <td>88.1</td>
      <td>68.7</td>
      <td>75.2</td>
      <td>53.4</td>
   </tr>
   <tr>
      <td>IDE+CamStyle+Random Erasing</td>
      <td>89.4</td>
      <td>71.5</td>
      <td>78.3</td>
      <td>57.6</td>
   </tr>
</table>


### References

- [1] Our code is conducted based on [open-reid](https://github.com/Cysu/open-reid)

- [2] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017

- [3] Re-ranking Person Re-identification with k-reciprocal Encoding. CVPR 2017.

- [4] Random Erasing Data Augmentation. Arxiv 2017.




### Citation

If you find this code useful in your research, please consider citing:

    @inproceedings{zhong2018camera,
    title={Camera Style Adaptation for Person Re-identification},
    author={Zhong, Zhun and Zheng, Liang and Zheng, Zhedong and Li, Shaozi and Yang, Yi},
    booktitle={CVPR},
    year={2018}
    }

    
### Contact me

If you have any questions about this code, please do not hesitate to contact me.

[Zhun Zhong](http://zhunzhong.site)
