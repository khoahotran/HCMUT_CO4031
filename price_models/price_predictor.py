
# ============================================================================
# PRICE-BASED RATING PREDICTION HELPER
# ============================================================================

import pandas as pd
import numpy as np
import joblib

class PriceRatingPredictor:
    def __init__(self, model_path='rf_price_model.pkl', scaler_path='scaler_price.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.features = [
            'discounted_price', 'actual_price', 'discount_percentage',
            'price_ratio', 'price_diff', 'discount_amount',
            'is_heavy_discount', 'is_light_discount', 'price_volatility',
            'price_category', 'discount_category', 'value_score'
        ]
        
    def calculate_features(self, actual_price, discount_percent):
        '''Calculate all price features from inputs'''
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
            'value_score': (discounted_price / actual_price) * 100 / (actual_price + 1) if actual_price > 0 else 0
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
            
        return features
    
    def predict(self, actual_price, discount_percent):
        '''Predict rating label from price information'''
        # Calculate features
        features = self.calculate_features(actual_price, discount_percent)
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        
        # Ensure correct feature order
        for feat in self.features:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        X_input = input_df[self.features]
        
        # Scale if needed (for KNN)
        if self.scaler is not None:
            X_input = self.scaler.transform(X_input)
        
        # Predict
        prediction = self.model.predict(X_input)[0]
        
        # Map to rating label
        rating_map = {
            0: ('Average', '3.0-3.9', 'Có thể cải thiện chiến lược giá'),
            1: ('Good', '4.0-4.4', 'Chiến lược giá tốt'),
            2: ('Excellent', '4.5-5.0', 'Chiến lược giá tuyệt vời')
        }
        
        label, rating_range, recommendation = rating_map.get(prediction, ('Unknown', 'N/A', 'Không có khuyến nghị'))
        
        return {
            'prediction': prediction,
            'label': label,
            'rating_range': rating_range,
            'recommendation': recommendation,
            'features': features
        }
    
    def predict_batch(self, price_data):
        '''Predict for multiple products'''
        predictions = []
        for data in price_data:
            pred = self.predict(data['actual_price'], data['discount_percent'])
            predictions.append(pred)
        return predictions

# Example usage:
# predictor = PriceRatingPredictor()
# result = predictor.predict(100, 20)  # actual_price=100, discount=20%
# print(f"Predicted rating: {result['label']} ({result['rating_range']})")
