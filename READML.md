# AI人脸真伪检测工具 - 专业版
工业级工程，支持单图/批量/GUI界面。
## 核心功能
1. 高精度人脸检测（RetinaFace）
2. 纹理方差法判断AI/真实
3. 批量检测+导出CSV
4. 图形界面(GUI)，双击即可运行
## 快速开始
### 1. 安装依赖
pip install opencv-python retina-face pillow numpy
### 2. 运行GUI
python gui.py
### 3. 命令行
python main.py test.jpg