import streamlit as st
import pandas as pd
import joblib
import json
import os

@st.cache_resource
def load_models():
    """Load các full-feature models"""
    models = {}
    model_files = {
        'Random Forest': './output/random_forest_model.pkl',
        'XGBoost': './output/xgboost_model.pkl',
        'KNN': './output/knn_model.pkl',
        'Best Model': './output/best_model.pkl',
        'Scaler': './output/scaler.pkl'
    }
    
    for name, file in model_files.items():
        try:
            if os.path.exists(file):
                models[name] = joblib.load(file)
                st.sidebar.success(f"Đã load {name}")
            else:
                st.sidebar.warning(f"Không tìm thấy {file}")
        except Exception as e:
            st.sidebar.error(f"Lỗi load {name}: {str(e)}")
    
    return models

@st.cache_resource
def load_price_models():
    """Load các price-only models"""
    price_models = {}
    price_model_dir = './price_models'
    
    if not os.path.exists(price_model_dir):
        st.sidebar.warning(f"Không tìm thấy thư mục {price_model_dir}")
        return None, None
    
    price_model_files = {
        'Random Forest (Price)': os.path.join(price_model_dir, 'rf_price_model.pkl'),
        'XGBoost (Price)': os.path.join(price_model_dir, 'xgb_price_model.pkl'),
        'KNN (Price)': os.path.join(price_model_dir, 'knn_price_model.pkl'),
        'Best Price Model': os.path.join(price_model_dir, 'best_price_model.pkl'),
        'Price Scaler': os.path.join(price_model_dir, 'scaler_price.pkl')
    }
    
    config_path = os.path.join(price_model_dir, 'price_model_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                price_config = json.load(f)
            st.sidebar.success(f"Đã load price model config")
        except Exception as e:
            st.sidebar.error(f"Lỗi load config: {str(e)}")
            
            try:
                with open(config_path, 'r', encoding='utf-8-sig') as f:
                    price_config = json.load(f)
                st.sidebar.success(f"Đã load price model config (utf-8-sig)")
            except Exception as e2:
                st.sidebar.error(f"Lỗi load config với utf-8-sig: {str(e2)}")
                price_config = None
    else:
        price_config = None
        st.sidebar.warning("Không tìm thấy price model config")
    
    # Load models
    for name, file in price_model_files.items():
        try:
            if os.path.exists(file):
                price_models[name] = joblib.load(file)
                st.sidebar.success(f"Đã load {name}")
            else:
                st.sidebar.warning(f"Không tìm thấy {file}")
        except Exception as e:
            st.sidebar.error(f"Lỗi load {name}: {str(e)}")
    
    return price_models, price_config

@st.cache_data
def load_processed_data():
    """Load dữ liệu đã xử lý"""
    try:
        if os.path.exists('./output/amazon_processed.csv'):
            df = pd.read_csv('./output/amazon_processed.csv')
            st.sidebar.success("Đã load dữ liệu đã xử lý")
            return df
        else:
            st.sidebar.warning("Không tìm thấy amazon_processed.csv")
            return None
    except Exception as e:
        st.sidebar.error(f"Lỗi load data: {str(e)}")
        return None