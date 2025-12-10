import streamlit as st
import numpy as np
from utils.predictions import predict_rating, get_rating_label
from utils.visualizations import plot_feature_importance, plot_prediction_probability
from components.headers import render_prediction_card, render_metric_card

def render(models, selected_model):
    """Render tab dự đoán full features"""
    st.header("Dự đoán Rating")
    st.markdown("Sử dụng tất cả thông tin sản phẩm để dự đoán")
    
    if not models:
        st.error("[WARNING] Không có model nào được load. Vui lòng kiểm tra file model.")
        return
    
    # st.subheader("📝 Nhập thông tin sản phẩm đầy đủ")
    
    with st.form("full_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Thông tin giá")
            actual_price = st.number_input("Giá gốc ($)", 
                                          min_value=0.0, 
                                          value=100.0,
                                          step=1.0,
                                          key="full_price")
            discount_percentage = st.slider("Phần trăm giảm giá (%)", 
                                           min_value=0.0, 
                                           max_value=100.0, 
                                           value=20.0,
                                           step=1.0,
                                           key="full_discount")
        
        with col2:
            st.markdown("### ⭐ Thông tin rating")
            current_rating = st.number_input("Rating hiện tại", 
                                            min_value=0.0, 
                                            max_value=5.0,
                                            value=3.5,
                                            step=0.1,
                                            key="full_rating")
            rating_count = st.number_input("Số lượng đánh giá", 
                                          min_value=0, 
                                          value=100,
                                          step=1,
                                          key="full_count")
        
        # Tính toán auto
        discounted_price = actual_price * (1 - discount_percentage / 100)
        price_ratio = discounted_price / actual_price if actual_price > 0 else 0
        price_diff = actual_price - discounted_price
        
        st.markdown("---")
        # st.markdown("### 📊 Tính toán tự động")
        
        col_calc1, col_calc2, col_calc3 = st.columns(3)
        with col_calc1:
            render_metric_card("💸 Giá khuyến mãi", f"${discounted_price:.2f}")
        with col_calc2:
            render_metric_card("📉 Tiết kiệm", f"${price_diff:.2f}")
        with col_calc3:
            render_metric_card("🔢 Tỷ lệ giá", f"{price_ratio:.2%}")
        
        submit_button = st.form_submit_button("Dự đoán với Full Model", 
                                             width='stretch',
                                             type="primary")

    if submit_button:
        input_data = {
            'discounted_price': discounted_price,
            'actual_price': actual_price,
            'discount_percentage': discount_percentage,
            'rating_count': rating_count,
            'price_ratio': price_ratio,
            'price_diff': price_diff,
            'discount_effectiveness': price_diff / actual_price if actual_price > 0 else 0,
            'weighted_rating': current_rating * np.log1p(rating_count),
            'rating_confidence': 1 - np.exp(-rating_count / 100),
            'rating': current_rating
        }
        
        # Predict
        with st.spinner(f'Đang dự đoán với {selected_model}...'):
            model = models[selected_model]
            scaler = models.get('Scaler')
            
            prediction, probabilities, X_input = predict_rating(
                model, scaler, input_data, selected_model
            )
        
        if prediction is not None:
            st.markdown("---")
            
            # Prediction card
            label = get_rating_label(prediction, 'full')
            render_prediction_card(prediction, label, selected_model)
            
            if probabilities is not None:
                st.subheader("📊 Xác suất dự đoán")
                prob_fig = plot_prediction_probability(probabilities, 'full')
                st.plotly_chart(prob_fig, width='stretch')
            
            # Feature importance
            if selected_model in ['Random Forest', 'XGBoost', 'Best Model']:
                st.subheader("🔍 Feature Importance")
                feature_names = list(X_input.columns) if X_input is not None else list(input_data.keys())
                importance_fig = plot_feature_importance(model, feature_names, selected_model)
                if importance_fig:
                    st.plotly_chart(importance_fig, width='stretch')
            
            # Recommendations
            render_recommendations(prediction, input_data)

def render_recommendations(prediction, input_data):
    """Render khuyến nghị dựa trên prediction"""
    st.subheader("💡 Khuyến nghị chiến lược")
    
    if prediction == 2:
        st.success("""
        ### **CHIẾN LƯỢC TỐI ƯU!**
        
        **Hành động đề xuất:**
        ✅ Giữ nguyên mức giá và chiết khấu hiện tại
        ✅ Tiếp tục chiến lược marketing hiện tại
        ✅ Có thể tăng giá nhẹ (5-10%) để tối ưu lợi nhuận
        
        **Lý do:**
        - Sản phẩm có tiềm năng được đánh giá rất cao (≥ 4.5 ⭐)
        - Tỷ lệ giá/giá trị tốt
        - Khả năng bán chạy và giữ chân khách hàng cao
        """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Target Rating", "4.5+ ⭐", "Excellent")
        with col2:
            render_metric_card("Price Strategy", "Maintain", "Current pricing works")
        with col3:
            render_metric_card("Risk Level", "Low", "High potential")
    
    elif prediction == 1:
        st.info("""
        ### [WARNING] **CẦN ĐIỀU CHỈNH**
        
        **Hành động đề xuất:**
        🔄 Cân nhắc tăng chiết khấu thêm 5-10%
        🔄 Kiểm tra giá cạnh tranh trên thị trường
        🔄 Cải thiện bằng bundle deals hoặc upsell
        🔄 Xem xét thêm chương trình khuyến mãi
        
        **Lý do:**
        - Sản phẩm có tiềm năng rating khá (4.0-4.5 ⭐)
        - Cần cải thiện một số yếu tố để đạt rating cao hơn
        - Có thể tối ưu hóa giá để tăng sức cạnh tranh
        """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Target Rating", "4.0-4.5 ⭐", "Good")
        with col2:
            render_metric_card("Price Strategy", "Adjust", "Consider discounts")
        with col3:
            render_metric_card("Risk Level", "Medium", "Moderate potential")
    
    else:
        st.warning("""
        ### 🚨 **CẦN THAY ĐỔI CHIẾN LƯỢC**
        
        **Hành động đề xuất:**
        🔴 Xem xét giảm giá gốc 10-20%
        🔴 Tăng chiết khấu lên ít nhất 40-50%
        🔴 Kiểm tra lại chất lượng so với giá
        🔴 Cân nhắc repositioning sản phẩm
        
        **Lý do:**
        - Sản phẩm có nguy cơ rating thấp (< 4.0 ⭐)
        - Giá cả có thể không tương xứng với chất lượng
        - Cần cải thiện nhiều yếu tố để cạnh tranh
        """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Target Rating", "< 4.0 ⭐", "Needs improvement")
        with col2:
            render_metric_card("Price Strategy", "Revise", "Consider price reduction")
        with col3:
            render_metric_card("Risk Level", "High", "Needs attention")