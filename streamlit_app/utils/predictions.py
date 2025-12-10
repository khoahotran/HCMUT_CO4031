import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter

def calculate_price_features(actual_price, discount_percent):
    """Tính toán tất cả price features từ input"""
    discounted_price = actual_price * (1 - discount_percent / 100)
    price_diff = actual_price - discounted_price
    
    features = {
        'discounted_price': discounted_price,
        'actual_price': actual_price,
        'discount_percentage': discount_percent,
        'price_ratio': discounted_price / actual_price if actual_price > 0 else 0,
        'price_diff': price_diff,
        'discount_amount': price_diff,
        'is_heavy_discount': 1 if discount_percent > 50 else 0,
        'is_light_discount': 1 if discount_percent < 10 else 0,
        'price_volatility': price_diff / actual_price if actual_price > 0 else 0,
    }
    
    # Price category
    if actual_price <= 50:
        features['price_category'] = 0
    elif actual_price <= 100:
        features['price_category'] = 1
    elif actual_price <= 200:
        features['price_category'] = 2
    elif actual_price <= 500:
        features['price_category'] = 3
    elif actual_price <= 1000:
        features['price_category'] = 4
    else:
        features['price_category'] = 5
        
    # Discount category
    if discount_percent <= 10:
        features['discount_category'] = 0
    elif discount_percent <= 20:
        features['discount_category'] = 1
    elif discount_percent <= 30:
        features['discount_category'] = 2
    elif discount_percent <= 50:
        features['discount_category'] = 3
    else:
        features['discount_category'] = 4
        
    # Value score
    features['value_score'] = (discounted_price / actual_price) * 100 / (actual_price + 1) if actual_price > 0 else 0
    
    return features

def get_rating_label(prediction, model_type='full'):
    """Chuyển prediction thành label"""
    if model_type == 'price':
        # Price models: 0=Average, 1=Good, 2=Excellent
        if prediction == 2:
            return "Excellent (4.5-5.0 ⭐)"
        elif prediction == 1:
            return "Good (4.0-4.4 ⭐)"
        else:
            return "Average (3.0-3.9 ⭐)"
    else:
        # Full models: 0=Bad, 1=Qualified, 2=Good
        if prediction == 2:
            return "Good (≥ 4.5 ⭐)"
        elif prediction == 1:
            return "Qualified (4.0-4.5 ⭐)"
        else:
            return "Bad (< 4.0 ⭐)"

def predict_rating(model, scaler, input_data, model_name):
    """Dự đoán rating từ input"""
    try:
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
        else:
            features = [
                'discounted_price', 'actual_price', 'discount_percentage', 
                'rating_count', 'price_ratio', 'price_diff',
                'discount_effectiveness', 'weighted_rating', 'rating_confidence'
            ]
        
        input_df = pd.DataFrame([input_data])
        
        # 1. Price features
        if 'actual_price' in input_df.columns and 'discounted_price' in input_df.columns:
            input_df['price_ratio'] = input_df['discounted_price'] / (input_df['actual_price'] + 1e-10)
            input_df['price_diff'] = input_df['actual_price'] - input_df['discounted_price']
            input_df['discount_effectiveness'] = input_df['price_diff'] / (input_df['actual_price'] + 1e-10)
        
        # 2. Rating-related features
        if 'rating' in input_df.columns and 'rating_count' in input_df.columns:
            input_df['weighted_rating'] = input_df['rating'] * np.log1p(input_df['rating_count'])
            input_df['rating_confidence'] = 1 - np.exp(-input_df['rating_count'] / 100)
        elif 'rating_count' in input_df.columns:
            input_df['weighted_rating'] = 3.0 * np.log1p(input_df['rating_count'])
            input_df['rating_confidence'] = 1 - np.exp(-input_df['rating_count'] / 100)
        
        for feature in features:
            if feature not in input_df.columns:
                if feature in ['discount_effectiveness', 'price_ratio', 'price_diff']:
                    if 'actual_price' in input_df.columns and 'discounted_price' in input_df.columns:
                        if feature == 'price_ratio':
                            input_df['price_ratio'] = input_df['discounted_price'] / (input_df['actual_price'] + 1e-10)
                        elif feature == 'price_diff':
                            input_df['price_diff'] = input_df['actual_price'] - input_df['discounted_price']
                        elif feature == 'discount_effectiveness':
                            input_df['discount_effectiveness'] = input_df['price_diff'] / (input_df['actual_price'] + 1e-10)
                elif feature in ['weighted_rating', 'rating_confidence']:
                    if 'rating_count' in input_df.columns:
                        input_df['weighted_rating'] = 3.0 * np.log1p(input_df['rating_count'])
                        input_df['rating_confidence'] = 1 - np.exp(-input_df['rating_count'] / 100)
                    else:
                        input_df[feature] = 0
                else:
                    input_df[feature] = 0
        
        missing_features = [f for f in features if f not in input_df.columns]
        if missing_features:
            st.warning(f"[WARNING] Thiếu features: {missing_features}. Sẽ gán giá trị 0.")
            for feature in missing_features:
                input_df[feature] = 0
        
        X_input = input_df[features]
        
        if model_name == "KNN" and scaler is not None:
            X_input = scaler.transform(X_input)
        
        # Predict
        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0] if hasattr(model, 'predict_proba') else None
        
        return prediction, prediction_proba, X_input
    except Exception as e:
        st.error(f"Lỗi dự đoán: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

def predict_with_price_model(model, scaler, features_dict, model_name, config=None):
    """Dự đoán với price-only model"""
    try:
        if config and 'features' in config:
            features = config['features']
        else:
            features = [
                'discounted_price', 'actual_price', 'discount_percentage',
                'price_ratio', 'price_diff', 'discount_amount',
                'is_heavy_discount', 'is_light_discount', 'price_volatility',
                'price_category', 'discount_category', 'value_score'
            ]
        
        input_df = pd.DataFrame([features_dict])
        
        for feat in features:
            if feat not in input_df.columns:
                if feat == 'price_ratio':
                    input_df['price_ratio'] = input_df['discounted_price'] / input_df['actual_price']
                elif feat == 'price_diff':
                    input_df['price_diff'] = input_df['actual_price'] - input_df['discounted_price']
                elif feat == 'discount_amount':
                    input_df['discount_amount'] = input_df['price_diff']
                elif feat == 'is_heavy_discount':
                    input_df['is_heavy_discount'] = (input_df['discount_percentage'] > 50).astype(int)
                elif feat == 'is_light_discount':
                    input_df['is_light_discount'] = (input_df['discount_percentage'] < 10).astype(int)
                elif feat == 'price_volatility':
                    input_df['price_volatility'] = input_df['price_diff'] / input_df['actual_price']
                elif feat == 'value_score':
                    input_df['value_score'] = input_df['price_ratio'] * 100 / (input_df['actual_price'] + 1)
                elif feat == 'price_category':
                    conditions = [
                        (input_df['actual_price'] <= 50),
                        (input_df['actual_price'] <= 100),
                        (input_df['actual_price'] <= 200),
                        (input_df['actual_price'] <= 500),
                        (input_df['actual_price'] <= 1000),
                        (input_df['actual_price'] > 1000)
                    ]
                    choices = [0, 1, 2, 3, 4, 5]
                    input_df['price_category'] = np.select(conditions, choices, default=0)
                elif feat == 'discount_category':
                    conditions = [
                        (input_df['discount_percentage'] <= 10),
                        (input_df['discount_percentage'] <= 20),
                        (input_df['discount_percentage'] <= 30),
                        (input_df['discount_percentage'] <= 50),
                        (input_df['discount_percentage'] > 50)
                    ]
                    choices = [0, 1, 2, 3, 4]
                    input_df['discount_category'] = np.select(conditions, choices, default=0)
                else:
                    input_df[feat] = 0
        
        X_input = input_df[features]
        
        if 'KNN' in model_name and scaler is not None:
            X_input = scaler.transform(X_input)
        
        # Predict
        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0] if hasattr(model, 'predict_proba') else None
        
        return prediction, prediction_proba, X_input
        
    except Exception as e:
        st.error(f"Lỗi dự đoán price model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

def compare_price_models(price_models, price_config, features):
    """So sánh kết quả từ tất cả price models"""
    results = []
    
    for model_name in [name for name in price_models.keys() if 'Scaler' not in name]:
        model = price_models[model_name]
        scaler = price_models.get('Price Scaler') if 'KNN' in model_name else None
        
        prediction, proba, _ = predict_with_price_model(
            model, scaler, features, model_name, price_config
        )
        
        if prediction is not None:
            label = get_rating_label(prediction, 'price')
            results.append({
                'Model': model_name.replace(' (Price)', ''),
                'Prediction': prediction,
                'Label': label,
                'Probability': proba.max() if proba is not None else None
            })
    
    return results