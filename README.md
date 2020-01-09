# Crowd-DCNet
Crowd Counting application based on S-DCNet

## How To Use
- Run this to have all your dependencies installed
```pip3 install -r requirements.txt --user```.
- Download the pretrained weights from
[Google Drive](https://drive.google.com/open?id=1gK-aqEpWm2io11_CBzCX3F0EVJcFju25).
- Extract the **models** folder into the Repo directory.
- Run ```python3 demo.py -v <video_path>``` to use the script.

## Future Work
- [ ] Add ROI (Region of Interest) feaure.
- [ ] Optimize the code for faster inference.
- [ ] Upgrade to SS-DCNet.
- [ ] Make a Dockerfile of the project for easy deployment.

## :sparkles: Huge thanks for the real heroes [here](https://github.com/xhp-hust-2018-2011/S-DCNet):sparkles:
If you find this work or code useful for your research, please cite:
```
@inproceedings{xhp2019SDCNet,
  title={From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer},
  author={Xiong, Haipeng and Lu, Hao and Liu, Chengxin and Liang, Liu and Cao, Zhiguo and Shen, Chunhua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019},
  pages = {8362-8371}
}
```
