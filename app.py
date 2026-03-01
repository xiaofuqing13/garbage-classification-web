
# ====================================
# 导入所需的库
# ====================================
# Web框架
from flask import Flask, render_template, request, jsonify
# 系统操作
import os
# 数据处理
import numpy as np

# 深度学习
from tensorflow.keras.models import load_model  # 加载模型
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # 图像处理

# 模型加载
import joblib  # 加载SVM模型和标签编码器

# 初始化Flask应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载模型和相关组件
# feature_extractor: 用于提取图片特征的MobileNetV2模型
# svm_model: 用于分类的SVM模型
# label_encoder: 用于转换标签的编码器
feature_extractor = load_model('feature_extractor.h5')
svm_model = joblib.load('svm_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

def predict_image(image_path):
    """
    对输入的图片进行垃圾分类预测
    
    参数:
        image_path (str): 图片文件的路径
        
    返回:
        tuple: (预测的类别(中文), 置信度)
    """
    # 加载和预处理图像
    img = load_img(image_path, target_size=(160, 160))  # 调整图片大小为模型输入尺寸
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 添加batch维度
    img_array = img_array / 255.0  # 归一化
    
    # 使用MobileNetV2提取特征
    features = feature_extractor.predict(img_array)
    features = features.reshape(1, -1)  # 展平特征向量
    
    # 使用SVM进行预测
    prediction_proba = svm_model.predict_proba(features)
    predicted_class_idx = np.argmax(prediction_proba[0])  # 获取最高概率的类别
    confidence = prediction_proba[0][predicted_class_idx]  # 获取置信度
    
    # 获取预测的类别名称
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    
    # 将英文类别名称转换为中文
    category_map = {
        'cardboard': '纸板',
        'glass': '玻璃',
        'metal': '金属',
        'paper': '纸张',
        'plastic': '塑料',
        'trash': '其他垃圾'
    }
    
    return category_map[predicted_class], confidence

@app.route('/')
def home():
    """渲染主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    处理图片上传和预测请求
    
    返回:
        JSON响应，包含：
        - class: 预测的垃圾类别
        - confidence: 预测的置信度
        - image_path: 保存的图片路径
        如果发生错误，返回error字段
    """
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file:
        # 保存上传的文件
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # 进行预测
        try:
            predicted_class, confidence = predict_image(filename)
            return jsonify({
                'class': predicted_class,
                'confidence': f'{confidence:.2%}',
                'image_path': filename
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 