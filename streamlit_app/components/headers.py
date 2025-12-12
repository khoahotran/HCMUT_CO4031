import streamlit as st

def render_main_header():
    """Render header chính"""
    st.markdown("""
    <div class="st-custom-header">
        <h1>🛍️ Amazon Product Rating Prediction</h1>
        <p>Phân tích và dự đoán đánh giá sản phẩm Amazon dựa trên các đặc tính sản phẩm</p>
    </div>
    """, unsafe_allow_html=True)

def render_price_only_header():
    """Render header cho price-only"""
    st.markdown("""
    <div class="st-custom-header-secondary">
        <h2>💰 Dự đoán Rating (Price Only)</h2>
        <p>Chỉ sử dụng thông tin giá cả để dự đoán - Phù hợp cho sản phẩm mới</p>
    </div>
    """, unsafe_allow_html=True)

def render_prediction_card(prediction, label, model_name):
    """Render card hiển thị kết quả dự đoán"""
    
    # Prediction color gradient
    if prediction == 2:
        gradient = "linear-gradient(135deg, #10B981 0%, #34D399 100%)" # Emerald (Good)
    elif prediction == 1:
        gradient = "linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%)" # Amber (Average)
    else:
        gradient = "linear-gradient(135deg, #EF4444 0%, #F87171 100%)" # Red (Bad)
    
    st.markdown(f"""
    <div class="st-custom-prediction" style="background: {gradient};">
        <h2>🎯 KẾT QUẢ DỰ ĐOÁN</h2>
        <h1>{label}</h1>
        <p>Model: {model_name} | Prediction code: {prediction}</p>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Render footer"""
    st.markdown("""
    <div class="st-custom-footer">
        <p>Amazon Product Rating Prediction System | Built with Streamlit</p>
        <p>Models: Full Feature & Price-Only | Data: Amazon Product Dataset</p>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(title, value, delta=None, help_text=None):
    """Render metric card"""
    delta_html = ""
    if delta:
        # Simple heuristic for delta color, or default to neutral
        delta_class = "neutral"
        if "+" in str(delta): delta_class = "positive"
        elif "-" in str(delta): delta_class = "negative"
        
        delta_html = f'<div class="delta {delta_class}">{delta}</div>'
    
    st.markdown(f"""
    <div class="st-custom-card st-custom-metric">
        <h3>{title}</h3>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def render_price_card(title):
    """Render card giá"""
    st.markdown(f"""
    <div class="st-custom-card" style="text-align: center; background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); border: none;">
        <h3 style="margin: 0; color: white !important;">{title}</h3>
    </div>
    """, unsafe_allow_html=True)