import streamlit as st
import pandas as pd
import numpy as np
from utils.predictions import (
    calculate_price_features, 
    get_rating_label, 
    predict_with_price_model,
    compare_price_models
)
from utils.visualizations import plot_prediction_probability
from components.headers import (
    render_price_only_header, 
    render_prediction_card, 
    render_price_card,
    render_metric_card
)

def render(price_models, price_config, selected_price_model):
    """Render tab price-only prediction"""
    render_price_only_header()
    
    if not price_models:
        render_no_models_warning()
        return
    
    # Price-only prediction form
    st.subheader("🎯 Nhập thông tin giá để dự đoán")
    
    # Form input
    with st.form("price_prediction_form"):
        col_price1, col_price2 = st.columns(2)
        
        with col_price1:
            render_price_card("💰 Thông tin giá")
            
            actual_price = st.number_input(
                "Giá gốc ($)",
                min_value=0.0,
                value=100.0,
                step=1.0,
                key="price_actual",
                help="Giá niêm yết ban đầu của sản phẩm"
            )
            
            discount_percent = st.slider(
                "Phần trăm giảm giá (%)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
                key="price_discount",
                help="Mức chiết khấu áp dụng"
            )
        
        with col_price2:
            # render_price_card("📊 Tính toán tự động")
            
            discounted_price = actual_price * (1 - discount_percent / 100)
            price_diff = actual_price - discounted_price
            price_ratio = discounted_price / actual_price if actual_price > 0 else 0
            
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                render_metric_card("💸 Giá khuyến mãi", f"${discounted_price:,.2f}")
                render_metric_card("📉 Số tiền giảm", f"${price_diff:,.2f}")
            with col_metrics2:
                render_metric_card("🔢 % giá còn lại", f"{price_ratio:.1%}")
                render_metric_card("💎 Giá trị", 
                         f"{(price_ratio * 100 / (actual_price + 1)):.2f}" if actual_price > 0 else "0",
                         help_text="Giá trị sản phẩm (cao hơn = tốt hơn)")
        
        # Options
        st.markdown("---")
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            show_calculations = st.checkbox("Hiển thị chi tiết tính toán", value=True)
            show_probabilities = st.checkbox("Hiển thị xác suất dự đoán", value=True)
        
        with col_opt2:
            compare_all = st.checkbox("So sánh tất cả price models", value=False)
            advanced_features = st.checkbox("Tính năng nâng cao", value=False)
        
        submit_price_button = st.form_submit_button(
            "Dự đoán với Price Model",
            width='stretch',
            type="primary"
        )
    
    if submit_price_button:
        features = calculate_price_features(actual_price, discount_percent)
        
        if show_calculations:
            render_feature_calculations(features)
        
        if compare_all and len(price_models) > 2:
            render_model_comparison(price_models, price_config, features)
        else:
            render_single_prediction(price_models, price_config, 
                                   selected_price_model, features, 
                                   show_probabilities, advanced_features)

def render_no_models_warning():
    """Render warning khi không có models"""
    st.warning("""
    Chưa load được price-only models. Vui lòng kiểm tra:
    
    1. **Thư mục `price_models` có tồn tại không**
    2. **Các file model đã được train xong chưa**
    3. **File config có đúng định dạng không**
    
    **File cần thiết:**
    - `price_models/rf_price_model.pkl`
    - `price_models/xgb_price_model.pkl`
    - `price_models/knn_price_model.pkl`
    - `price_models/best_price_model.pkl`
    - `price_models/scaler_price.pkl`
    - `price_models/price_model_config.json`
    """)
    
    if st.button("🔄 Thử load lại models", type="secondary"):
        st.rerun()

def render_feature_calculations(features):
    """Render chi tiết tính toán features"""
    with st.expander("📋 Chi tiết tính toán features", expanded=True):
        col_calc1, col_calc2 = st.columns(2)
        
        with col_calc1:
            st.write("**Features chính:**")
            for key in ['actual_price', 'discounted_price', 'discount_percentage']:
                st.write(f"- `{key}`: {features[key]:.2f}")
            
            st.write("**Features tính toán:**")
            for key in ['price_ratio', 'price_diff', 'discount_amount']:
                st.write(f"- `{key}`: {features[key]:.2f}")
        
        with col_calc2:
            st.write("**Features phân loại:**")
            st.write(f"- `is_heavy_discount`: {'Có' if features['is_heavy_discount'] else 'Không'}")
            st.write(f"- `is_light_discount`: {'Có' if features['is_light_discount'] else 'Không'}")
            st.write(f"- `price_category`: {features['price_category']}")
            st.write(f"- `discount_category`: {features['discount_category']}")
            
            st.write("**Features nâng cao:**")
            st.write(f"- `price_volatility`: {features['price_volatility']:.4f}")
            st.write(f"- `value_score`: {features['value_score']:.4f}")

def render_single_prediction(price_models, price_config, model_name, 
                           features, show_probabilities, advanced_features):
    """Render dự đoán với model đã chọn"""
    model = price_models[model_name]
    scaler = price_models.get('Price Scaler') if 'KNN' in model_name else None
    
    with st.spinner(f'Đang dự đoán với {model_name}...'):
        prediction, proba, X_input = predict_with_price_model(
            model, scaler, features, model_name, price_config
        )
    
    if prediction is not None:
        label = get_rating_label(prediction, 'price')
        render_prediction_card(prediction, label, model_name)
        
        if show_probabilities and proba is not None:
            st.subheader("📊 Xác suất dự đoán")
            prob_fig = plot_prediction_probability(proba, 'price')
            st.plotly_chart(prob_fig, width='stretch')
        
        if advanced_features and X_input is not None:
            with st.expander("🔍 Input features cho model"):
                st.dataframe(X_input, width='stretch')
        
        # Recommendations
        render_price_recommendations(prediction, features)

def render_model_comparison(price_models, price_config, features):
    """Render so sánh các models"""
    st.subheader("📊 So sánh kết quả từ các price models")
    
    with st.spinner('Đang so sánh các models...'):
        results = compare_price_models(price_models, price_config, features)
    
    if results:
        results_df = pd.DataFrame(results)
        
        col_compare1, col_compare2 = st.columns(2)
        
        with col_compare1:
            st.dataframe(results_df[['Model', 'Prediction', 'Label']], 
                        width='stretch')
        
        with col_compare2:
            import plotly.express as px
            fig = px.bar(results_df, x='Model', y='Prediction',
                       title='So sánh dự đoán',
                       color='Model',
                       color_discrete_sequence=px.colors.qualitative.Set3,
                       text='Label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        # Consensus analysis
        predictions_list = [r['Prediction'] for r in results]
        unique_predictions = set(predictions_list)
        
        if len(unique_predictions) == 1:
            st.success(f"🎯 **TẤT CẢ MODELS ĐỒNG Ý:** {results[0]['Label']}")
            render_price_recommendations(predictions_list[0], features)
        else:
            # Voting
            from collections import Counter
            most_common_pred = Counter(predictions_list).most_common(1)[0][0]
            consensus_label = get_rating_label(most_common_pred, 'price')
            
            st.info(f"🤔 **CÁC MODELS KHÔNG ĐỒNG Ý.** Đa số ({Counter(predictions_list).most_common(1)[0][1]}/{len(results)}): {consensus_label}")
            render_price_recommendations(most_common_pred, features)

def render_price_recommendations(prediction, features):
    """Render khuyến nghị cho price prediction"""
    st.subheader("💡 Khuyến nghị chiến lược giá")
    
    if prediction == 2:
        st.success("""
        ### 🎯 **GIÁ TỐI ƯU - RATING CAO DỰ KIẾN**
        
        **Chiến lược đề xuất:**
        ✅ **Giữ nguyên giá hiện tại** - Không cần điều chỉnh
        ✅ **Tập trung vào marketing** - Quảng bá lợi thế giá trị
        ✅ **Có thể premium hóa** - Nâng giá 5-10% nếu thương hiệu mạnh
        
        **Lý do:**
        - Mức giá và chiết khấu hiện tại rất hấp dẫn với khách hàng
        - Tỷ lệ giá/giá trị xuất sắc
        - Dự kiến rating: 4.5-5.0 ⭐
        """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Expected Rating", "4.5-5.0 ⭐", "Excellent")
        with col2:
            render_metric_card("Price Action", "Maintain", "Optimal pricing")
        with col3:
            render_metric_card("Competitive Edge", "Strong", "High value")
    
    elif prediction == 1:
        st.info("""
        ### ⚖️ **GIÁ CHẤP NHẬN ĐƯỢC - CÓ THỂ CẢI THIỆN**
        
        **Chiến lược đề xuất:**
        🔄 **Điều chỉnh nhẹ** - Tăng chiết khấu thêm 5-10%
        🔄 **Bundle deals** - Kết hợp sản phẩm để tăng giá trị
        🔄 **Limited offers** - Khuyến mãi thời vụ
        
        **Lý do:**
        - Giá cả cạnh tranh nhưng chưa tối ưu
        - Có thể cải thiện để đạt rating cao hơn
        - Dự kiến rating: 4.0-4.4 ⭐
        """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Expected Rating", "4.0-4.4 ⭐", "Good")
        with col2:
            render_metric_card("Price Action", "Adjust", "Consider discounts")
        with col3:
            render_metric_card("Competitive Edge", "Moderate", "Room for improvement")
    
    else:
        st.warning("""
        ### 🚨 **GIÁ CẦN ĐIỀU CHỈNH - RATING THẤP DỰ KIẾN**
        
        **Chiến lược đề xuất:**
        🔴 **Giảm giá gốc** - Cắt giảm 10-20% giá niêm yết
        🔴 **Tăng chiết khấu** - Áp dụng discount 40-50%
        🔴 **Repositioning** - Xem xét lại phân khúc thị trường
        🔴 **Value addition** - Thêm tính năng/dịch vụ đi kèm
        
        **Lý do:**
        - Giá cả không hấp dẫn so với giá trị cung cấp
        - Nguy cơ rating thấp do giá trị cảm nhận kém
        - Dự kiến rating: 3.0-3.9 ⭐
        """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Expected Rating", "3.0-3.9 ⭐", "Average")
        with col2:
            render_metric_card("Price Action", "Revise", "Needs reduction")
        with col3:
            render_metric_card("Competitive Edge", "Weak", "Needs improvement")