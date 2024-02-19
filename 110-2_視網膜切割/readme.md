# 使用兩種模型(Unet & VGG)找出視網膜血管叢的位置
* Using Unet and VGG for retinal images vessel Tree segmentation

## Process
<img src = "https://github.com/NTU-Chiu/ML_Projects/assets/91785016/9c0c8c85-75d2-4c71-8703-bc009f852ba8.png" width = "900" height = "250">

## Results
<img src = "https://github.com/NTU-Chiu/ML_Projects/assets/91785016/27866b52-aca4-4082-b994-50ab781f51c0.png" width = "600">
<img src = "https://github.com/NTU-Chiu/ML_Projects/assets/91785016/93067e19-f6b8-41c6-82b9-b3b97f90c689.png" width = "600">

## Techinque
### Unet
<img src = "https://github.com/NTU-Chiu/ML_Projects/assets/91785016/b56ed867-2e22-4e04-bace-517db6a6dee3.png" width = "600">< br/>
* Image from:
* Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing. < br/>

### VGG
* Improve the original VGG via adding interpolation layer.
<img src = "https://github.com/NTU-Chiu/ML_Projects/assets/91785016/86d717ad-255f-462f-bb43-f24589c3a684.png" width = "600">
