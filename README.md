# 基于YOLOv8的场景目标检测与视频多目标跟踪
## 环境配置
- 根据硬件加速环境安装PyTorch
- 安装ultralytics
```bash
pip install ultralytics
```
- 下载yolov8x.pt到项目目录，数据集会自动下载

## 微调
使用VisDrone数据集。
```bash
python main.py
```
训练完成后从`runs/detect/visdrone_project/yolov8x_finetune-*/weights/best.pt`将权重文件复制到项目目录。

## 视频检测测试与追踪
将tracksource.mp4放置到项目目录。
```bash
python main_2.py
```

## 越线计数
同样使用tracksource.mp4。空格暂停，q退出。
```bash
python main_4.py
```

## 训练测试环境与权重文件
权重文件放置在[我的网站](https://ricacraft.com/downlaods/pt/CV-Midterm-Task2.pt)。

测试训练使用的环境为Intel Ultra7 265K (64G RAM) + AMD Instinct MI50 32G，Ubuntu26.04 + ROCm 7.13-preview
