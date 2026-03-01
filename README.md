# 垃圾分类识别系统

垃圾分类在国内推行以来，很多人还是分不清"干垃圾"和"湿垃圾"、"可回收"和"不可回收"的区别，扔错垃圾的情况屡见不鲜。本项目做了一个拍照识别垃圾类别的 Web 应用拍张照片上传，系统自动告诉你这是什么垃圾、属于哪一类，省去翻查分类手册的麻烦。

## 痛点与目的

- **问题**：垃圾分类种类多、标准复杂，居民记不住也懒得查，投放错误率高
- **方案**：训练图像分类模型（MobileNetV2 提特征 + SVM 分类），部署为 Flask Web 服务，用户上传垃圾图片即可获得分类结果
- **效果**：支持识别纸类、金属、玻璃、塑料、纸板、其他垃圾共 6 大类

## 识别类别

| 英文 | 中文 | 示例 |
|------|------|------|
| paper | 纸类 | 报纸、打印纸 |
| cardboard | 纸板 | 快递纸箱 |
| plastic | 塑料 | 塑料瓶、塑料袋 |
| glass | 玻璃 | 玻璃瓶、碎玻璃 |
| metal | 金属 | 易拉罐、金属罐 |
| trash | 其他垃圾 | 不可回收物 |

## 技术方案

采用**迁移学习 + 传统机器学习**的组合方案，而不是端到端深度学习，原因是训练速度快、部署轻量：

1. **特征提取**：用预训练的 MobileNetV2（去掉分类头）提取图片的 1280 维高层语义特征
2. **分类器**：用提取到的特征训练 SVM 分类器（RBF 核，交叉验证调参）
3. **Web 服务**：Flask 框架搭建上传页面，后端加载模型实时推理

## 使用方法

### 安装依赖

`ash
pip install -r requirements.txt
`

### 训练模型

`ash
python train_sklearn_model.py
`

训练完成后会生成：
- `feature_extractor.h5`  MobileNetV2 特征提取模型
- `svm_model.joblib`  SVM 分类器
- `label_encoder.joblib`  标签编码器

### 启动 Web 服务

`ash
python app.py
`

浏览器打开 `http://localhost:5000`，上传垃圾图片即可查看分类结果。

## 项目结构

`
.
 app.py                    # Flask Web 服务（主程序）
 train_sklearn_model.py    # 模型训练脚本
 feature_extractor.h5      # MobileNetV2 特征提取器
 svm_model.joblib          # 训练好的 SVM 分类器
 label_encoder.joblib      # 标签编码器
 mymodel.keras             # Keras 端到端模型（备用）
 templates/                # Flask HTML 模板
 static/                   # 静态资源和上传目录
 archive/                  # 数据集（需自行准备）
 requirements.txt          # 依赖列表
`

## 技术栈

- Flask（Web 服务）
- TensorFlow / Keras（MobileNetV2 特征提取）
- scikit-learn（SVM 分类器）
- OpenCV / Pillow（图像处理）
- Python 3.x

## License

MIT License
