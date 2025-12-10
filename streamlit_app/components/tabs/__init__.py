import streamlit as st

def render_all_tabs(df, models, price_models, price_config, 
                   selected_model, selected_price_model):
    """Render tất cả các tabs"""
    
    if df is not None:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Tổng quan", 
            "📈 Phân tích EDA", 
            "🤖 Dự đoán Full", 
            "💰 Dự đoán Price-Only",
            "📋 Model Performance"
        ])
        
        from .overview import render as render_overview
        from .eda import render as render_eda
        from .full_prediction import render as render_full_prediction
        from .price_prediction import render as render_price_prediction
        from .performance import render as render_performance
        
        with tab1:
            render_overview(df)
        
        with tab2:
            render_eda(df)
        
        with tab3:
            render_full_prediction(models, selected_model)
        
        with tab4:
            render_price_prediction(price_models, price_config, selected_price_model)
        
        with tab5:
            render_performance(models, price_models, price_config)
        
    else:
        st.warning("""
        [WARNING] Không tìm thấy dữ liệu đã xử lý.
        
        Vui lòng đảm bảo có file `amazon_processed.csv` trong thư mục output.
        
        Nếu chưa có, hãy:
        1. Chạy notebook training để tạo file processed data
        2. Đảm bảo file được lưu đúng tên
        3. Đặt file trong thư mục output
        """)