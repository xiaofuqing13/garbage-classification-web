"""
垃圾分类系统模型训练程序

这个程序实现了垃圾分类模型的训练过程，包括：
1. 使用MobileNetV2进行特征提取
2. 使用线性SVM进行分类
3. 使用交叉验证提高模型稳定性

主要步骤：
1. 加载和预处理图片数据
2. 使用MobileNetV2提取特征
3. 训练SVM分类器
4. 评估模型性能
5. 保存训练好的模型

作者：Claude
日期：2024-02-10
"""

# ====================================
# 导入所需的库
# ====================================

# 系统操作
import os
import gc  # 垃圾回收

# 数据处理
import numpy as np  # 数值计算
import glob  # 文件路径模式匹配

# 深度学习
from tensorflow.keras.applications import MobileNetV2  # 预训练模型
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # 图像处理

# 机器学习
from sklearn.model_selection import train_test_split  # 数据集分割
from sklearn.preprocessing import LabelEncoder  # 标签编码
from sklearn.svm import LinearSVC  # 线性SVM分类器
from sklearn.metrics import classification_report  # 模型评估
from sklearn.calibration import CalibratedClassifierCV  # SVM概率校准

# 工具
import joblib  # 模型保存和加载
from tqdm import tqdm  # 进度条显示

def load_and_preprocess_image(image_path):
    """
    加载和预处理单张图片
    
    参数:
        image_path (str): 图片文件路径
        
    返回:
        numpy.ndarray: 预处理后的图片数组，如果处理失败返回None
    """
    try:
        img = load_img(image_path, target_size=(160, 160))  # 调整图片大小
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # 添加batch维度
        img_array = img_array / 255.0  # 归一化
        return img_array
    except Exception as e:
        print(f"处理图片时出错 {image_path}: {str(e)}")
        return None

def extract_features_batch(image_paths, feature_extractor, batch_size=16):
    """
    分批提取图片特征以节省内存
    
    参数:
        image_paths (list): 图片路径列表
        feature_extractor (Model): 特征提取模型
        batch_size (int): 批处理大小
        
    返回:
        numpy.ndarray: 提取的特征数组
    """
    features = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="提取特征"):
        batch_paths = image_paths[i:i + batch_size]
        batch_arrays = []
        
        # 处理每个批次中的图片
        for image_path in batch_paths:
            img_array = load_and_preprocess_image(image_path)
            if img_array is not None:
                batch_arrays.append(img_array[0])
        
        if batch_arrays:
            batch_arrays = np.array(batch_arrays)
            batch_features = feature_extractor.predict(batch_arrays, verbose=0)
            features.extend(batch_features)
        
        # 清理内存
        del batch_arrays
        gc.collect()
    
    return np.array(features)

def main():
    """主函数：实现模型训练的完整流程"""
    print("开始垃圾分类模型训练...")
    
    # 设置数据路径和类别
    data_dir = 'archive/Garbage classification/Garbage classification'
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # 收集数据路径
    print("收集数据路径...")
    image_paths = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        path_pattern = os.path.join(path, '*.[jp][pn][g]')
        category_paths = glob.glob(path_pattern)
        print(f"{category}: 找到 {len(category_paths)} 张图片")
        image_paths.extend(category_paths)
        labels.extend([category] * len(category_paths))
    
    print(f"\n总共收集到 {len(image_paths)} 张图片")
    
    # 标签编码
    print("\n编码标签...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # 保存标签编码器
    print("保存标签编码器...")
    joblib.dump(label_encoder, 'label_encoder.joblib')
    
    # 加载预训练的MobileNetV2模型
    print("\n加载预训练的MobileNetV2模型...")
    feature_extractor = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(160, 160, 3)
    )
    
    # 提取特征
    print("\n开始特征提取...")
    X = extract_features_batch(image_paths, feature_extractor)
    
    # 保存特征提取器
    print("\n保存特征提取器...")
    feature_extractor.save('feature_extractor.h5')
    
    # 释放内存
    del feature_extractor
    gc.collect()
    
    # 划分训练集和测试集
    print("\n划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建和训练SVM分类器
    print("\n训练线性SVM分类器...")
    base_svm = LinearSVC(dual="auto")
    # 使用CalibratedClassifierCV来获得概率输出
    svm = CalibratedClassifierCV(base_svm, cv=3)
    svm.fit(X_train, y_train)
    
    # 评估模型
    print("\n评估模型...")
    y_pred = svm.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=categories))
    
    # 保存模型
    print("\n保存SVM模型...")
    joblib.dump(svm, 'svm_model.joblib')
    
    print("\n训练完成！")

if __name__ == '__main__':
    main() 