import streamlit as st #构建应用
import joblib
import numpy as np
import pandas as pd
import shap #解释预测结果
import matplotlib.pyplot as plt #数据可视化
import catboost as cab

model=joblib.load('CAB.pkl')

#定义自变量名称，对应数据集中的列名
feature_names=[
    'HTN',
    'CKD',
    'HR',
    'T',
    'Hb',
    'Na',
    'LAC',
    'PH',
    'PO2',
    'INR',
    'ALT',
    'BIL',
    'Cr',
    'ABX',
    'Nbps'
]

st.title('MOF Predictor') #设置标题

#设置输入框 - 确保所有数值都是float类型
HTN = st.selectbox('hypertension:', options=[0,1],format_func=lambda x:'Yes' if x == 1 else 'No') #二分类变量
CKD = st.selectbox('chronic kidney disease:', options=[0,1],format_func=lambda x:'Yes' if x == 1 else 'No')
HR = st.number_input('heart rate:', min_value=0.0, max_value=200.0, value=80.0) #连续变量
T = st.number_input('temperature:', min_value=0.0, max_value=150.0, value=98.0)
Hb = st.number_input('hemoglobin:', min_value=0.0, max_value=20.0, value=10.0)
Na = st.number_input('Na:', min_value=0.0, max_value=200.0, value=140.0)
LAC = st.number_input('lactate:', min_value=0.0, max_value=20.0, value=2.0)
PH = st.number_input('PH:', min_value=0.0, max_value=10.0, value=7.4)
PO2 = st.number_input('PO2:', min_value=0.0, max_value=300.0, value=80.0)
INR = st.number_input('INR:', min_value=0.0, max_value=20.0, value=1.5)
ALT = st.number_input('bilirubin:', min_value=0.0, max_value=3000.0, value=90.0)
BIL = st.number_input('ALT:', min_value=0.0, max_value=50.0, value=1.0)
Cr = st.number_input('creatinine:', min_value=0.0, max_value=10.0, value=3.0)
ABX = st.selectbox('antibiotic:', options=[0,1],format_func=lambda x:'Yes' if x == 1 else 'No')
Nbps = st.number_input('Nbps:', min_value=0.0, max_value=200.0, value=100.0)

# 确保所有特征值都是float类型
feature_values = [
    float(HTN), float(CKD), float(HR), float(T), float(Hb), 
    float(Na), float(LAC), float(PH), float(PO2), float(INR), 
    float(ALT), float(BIL), float(Cr), float(ABX), float(Nbps)
]

features = np.array([feature_values], dtype=float) #明确指定dtype为float

if st.button('Predict'):
    # 确保预测输入的数据类型正确
    features_clean = np.array([feature_values], dtype=float)
    
    predicted_class = model.predict(features_clean)[0] #预测类别
    predicted_proba = model.predict_proba(features_clean)[0] #预测类别的概率

    # 显示预测结果 - 确保所有显示的数据类型一致
    st.write(f'**Predicted Class:** {int(predicted_class)} (1:Disease, 0:No Disease)')
    st.write(f'**Prediction Probabilities:** Class 0: {float(predicted_proba[0]):.4f}, Class 1: {float(predicted_proba[1]):.4f}')

    #根据预测结果生成建议
    probability = float(predicted_proba[predicted_class]) * 100
    if predicted_class == 1:
        advice = (
            f'According to our model, you have a high risk of MOF. '
            f'The model predicts that your probability of having MOF is {probability:.1f}%. '
            'Please closely monitor the patients vital signs.'
        )
    else:
        advice = (
            f'According to our model, you have a low risk of MOF. '
            f'The model predicts that your probability of not having MOF is {probability:.1f}%. '
            'The patients still requires attention.'
        )

    st.write(advice) #显示建议

    st.subheader('SHAP Force Plot Explanation') #侧标题

    try:
        # 使用TreeExplainer对于树模型更稳定
        explainer_shap = shap.TreeExplainer(model)
        
        # 创建输入数据的DataFrame，确保数据类型一致
        input_df = pd.DataFrame([feature_values], columns=feature_names, dtype=float)
        shap_values = explainer_shap.shap_values(input_df)
        
        # 对于二分类，直接使用预测类别的视角
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # 返回两个类别的SHAP值
            base_value = explainer_shap.expected_value[predicted_class]
            shap_val = shap_values[predicted_class][0]
        else:
            # 返回单个数组（通常是正类的SHAP值）
            base_value = explainer_shap.expected_value
            if hasattr(base_value, '__len__') and len(base_value) == 2:
                base_value = base_value[predicted_class]
            # 处理不同的shap_values格式
            if hasattr(shap_values, 'shape'):
                if len(shap_values.shape) == 3:
                    # 3维数组 (1, n_features, 2)
                    shap_val = shap_values[0, :, predicted_class]
                elif len(shap_values.shape) == 2:
                    # 2维数组 (1, n_features)
                    shap_val = shap_values[0]
                else:
                    shap_val = shap_values
            else:
                shap_val = shap_values
        
        # 确保所有值都是float类型
        base_value = float(base_value)
        shap_val = np.array(shap_val, dtype=float)
        feature_values_float = [float(x) for x in feature_values]
        
        # 创建force plot
        plt.figure(figsize=(10, 3))
        shap.force_plot(
            base_value,
            shap_val,
            feature_values_float,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.tight_layout()
        plt.savefig('shap_force_plot.png', bbox_inches='tight', dpi=300)
        st.image('shap_force_plot.png', caption=f'SHAP解释图 (预测类别: {int(predicted_class)})')
        
    except Exception as e:
        st.error(f"生成SHAP图时出错: {str(e)}")
        
        # 备用方案：使用更简单的方法
        try:
            plt.figure(figsize=(10, 3))
            shap.force_plot(
                float(explainer_shap.expected_value), 
                np.array(shap_values[0] if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1 else shap_values, dtype=float), 
                [float(x) for x in feature_values],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            plt.tight_layout()
            plt.savefig('shap_force_plot.png', bbox_inches='tight', dpi=300)
            st.image('shap_force_plot.png', caption='SHAP Force Plot Explanation (备用方案)')
        except Exception as e2:
            st.error(f"备用方案也失败: {str(e2)}")