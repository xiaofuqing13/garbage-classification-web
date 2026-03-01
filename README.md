# Garbage Classification System

基于深度学习的智能垃圾分类识别系统。采用 **MobileNetV2** 迁移学习提取图像特征，结合 **SVM** 分类器进行六类垃圾识别，并通过 **Flask** Web 应用提供在线预测服务。

## 项目目的

通过迁移学习 + 传统机器学习的混合架构，实现轻量高效的垃圾分类识别。用户上传垃圾图片即可获得分类结果和置信度，适用于智慧环保、垃圾分类指导等场景。

## 核心功能

- **MobileNetV2 特征提取**：使用 ImageNet 预训练的 MobileNetV2 作为特征提取器，输入 160×160 图像输出 1280 维特征向量
- **SVM 分类器**：基于 `LinearSVC` + `CalibratedClassifierCV`（3 折交叉验证），实现概率化的多分类预测
- **Flask Web 应用**：提供图片上传、实时预测和结果展示的 Web 界面
- **批量特征提取**：支持分批处理图像以节省内存，适合大规模数据集
- **6 类垃圾识别**：纸板（cardboard）、玻璃（glass）、金属（metal）、纸张（paper）、塑料（plastic）、其他垃圾（trash）

## 技术架构

```
用户上传图片
    ↓
Flask Web 服务器 (app.py)
    ↓
图像预处理 (160×160, 归一化)
    ↓
MobileNetV2 特征提取 (feature_extractor.h5)
    ↓
1280 维特征向量
    ↓
SVM 分类器 (svm_model.joblib)
    ↓
输出: 垃圾类别 + 置信度
```

## 使用说明

### 环境安装

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python train_sklearn_model.py
```

训练完成后自动保存：
- `feature_extractor.h5` — MobileNetV2 特征提取模型
- `svm_model.joblib` — SVM 分类器
- `label_encoder.joblib` — 标签编码器

### 启动 Web 服务

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:5000`，上传垃圾图片即可获得分类结果。

## 适用场景

- 智慧环保 / 垃圾分类指导
- 迁移学习 + 传统 ML 混合架构的学习参考
- Flask Web 部署 ML 模型的实践案例

## 技术栈

| 组件 | 技术 |
|------|------|
| 特征提取 | TensorFlow / Keras (MobileNetV2) |
| 分类器 | scikit-learn (LinearSVC + CalibratedClassifierCV) |
| Web 框架 | Flask |
| 前端 | HTML + CSS |
| 序列化 | joblib |

## 项目结构

```
.
├── app.py                    # Flask Web 应用主程序
├── train_sklearn_model.py    # 模型训练脚本
├── requirements.txt          # Python 依赖
├── templates/
│   └── index.html            # Web 前端页面
├── static/
│   └── css/                  # 样式文件
├── README.md
└── LICENSE
```

> **Note**: 模型权重文件（`feature_extractor.h5`, `svm_model.joblib`, `label_encoder.joblib`）和数据集（`archive/`）因体积过大未包含在仓库中。请自行训练或从 [Kaggle Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) 下载数据集。

## License

MIT License
