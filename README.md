# Crowd-DCNet
Crowd Counting application based on S-DCNet & [SS-DCNet](https://github.com/xhp-hust-2018-2011/SS-DCNet)

## How To Use
- Run this to have all your dependencies installed
```pip3 install -r requirements.txt --user```.
- Download the S-DCNet pretrained weights from
[Google Drive](https://drive.google.com/open?id=1gK-aqEpWm2io11_CBzCX3F0EVJcFju25) or the SS-DCNet from [Here](https://drive.google.com/drive/folders/1TRJr9YuP1dFpnbQvSSQHqIqhLFdElo_Q) (SHA weights are the only one tested).
- Extract the **models** folder into the Repo directory.
- Run ```python3 demo.py <pretrained_weights> -v <video_path>``` to use the script.
- Choose ROIs by hitting Space or Enter after every selecton and when finished hit ESC (but be warned that this has a great hit on the speed).

**Note**: SS-DCNet will work by default, if the user wish to use S-DCNet just add ```--SDCNet``` to the python3 running command.
## News
- **26 Jan 2020** Add Camera Support & Threading for faster fetching ...... you can add ```--cap``` then specify the camera number (default=0) and the script will work online (press q to Quit)
## Future Work
- [x] Add ROI (Region of Interest) feaure.
- [x] Optimize the code for faster inference.
- [x] Upgrade to SS-DCNet.
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
