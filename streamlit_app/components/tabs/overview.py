import streamlit as st
import pandas as pd
import numpy as np
import os
from components.headers import render_metric_card

def render(df):
    """Render tab Tổng quan"""
    st.header("📊 Tổng quan dữ liệu")
    
    tab_raw, tab_processed = st.tabs(["📁 Dữ liệu gốc", "📋 Dữ liệu đã xử lý"])
    
    with tab_raw:
        render_raw_data_tab()
    
    with tab_processed:
        render_processed_data_tab(df)

def render_raw_data_tab():
    """Render tab dữ liệu gốc"""
    st.subheader("Dữ liệu gốc từ amazon.csv")
    
    @st.cache_data
    def load_original_data():
        try:
            if os.path.exists('./amazon.csv'):
                return pd.read_csv('./amazon.csv')
            else:
                st.warning("File amazon.csv không tồn tại trong thư mục gốc")
                return None
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
            return None
    
    raw_data = load_original_data()
    
    if raw_data is not None:
        col_raw1, col_raw2 = st.columns(2)
        with col_raw1:
            render_metric_card("Số dòng", f"{len(raw_data):,}")
            render_metric_card("Số cột", len(raw_data.columns))
        with col_raw2:
            render_metric_card("Dữ liệu thiếu", f"{raw_data.isnull().sum().sum():,}")
            render_metric_card("Kích thước", f"{raw_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        st.subheader("Xem trước dữ liệu gốc")
        num_rows = st.slider("Số dòng hiển thị:", 5, 50, 10, key="raw_rows")
        st.dataframe(raw_data.head(num_rows), width='stretch')
        
        with st.expander("Danh sách cột và kiểu dữ liệu"):
            for col in raw_data.columns:
                missing_count = raw_data[col].isnull().sum()
                dtype = raw_data[col].dtype
                st.write(f"**{col}** - {dtype} - {missing_count} giá trị thiếu")
    else:
        st.info("Upload file amazon.csv để xem dữ liệu gốc")
        uploaded_file = st.file_uploader("Chọn file amazon.csv", type=['csv'], key="raw_upload")
        if uploaded_file is not None:
            raw_data = pd.read_csv(uploaded_file)
            st.success(f"Đã load {len(raw_data)} dòng dữ liệu")
            st.dataframe(raw_data.head(10), width='stretch')

def render_processed_data_tab(df):
    """Render tab dữ liệu đã xử lý"""
    st.subheader("Dữ liệu đã xử lý")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Tổng sản phẩm", f"{len(df):,}")
    with col2:
        if 'rating' in df.columns:
            render_metric_card("Rating trung bình", f"{df['rating'].mean():.2f} ⭐")
        else:
            render_metric_card("Rating trung bình", "N/A")
    with col3:
        if 'discounted_price' in df.columns:
            render_metric_card("Giá khuyến mãi TB", f"${df['discounted_price'].mean():.2f}")
        else:
            render_metric_card("Giá khuyến mãi TB", "N/A")
    with col4:
        if 'discount_percentage' in df.columns:
            render_metric_card("Giảm giá TB", f"{df['discount_percentage'].mean():.1f}%")
        else:
            render_metric_card("Giảm giá TB", "N/A")
    
    # Data preview
    st.subheader("Xem trước dữ liệu đã xử lý")
    with st.expander("Hiển thị dữ liệu", expanded=False):
        st.dataframe(df.head(100), width='stretch')
    
    # Basic info
    st.subheader("Thông tin cơ bản")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        <div class="st-custom-card">
            <h3>Thông tin dataset</h3>
            <p>Shape: <b>{}</b></p>
            <p>Columns: <b>{}</b></p>
            <p>Memory: <b>{:.1f} MB</b></p>
        </div>
        """.format(df.shape, len(df.columns), df.memory_usage(deep=True).sum() / 1024**2), unsafe_allow_html=True)
        
    with col_info2:
        dtype_counts = df.dtypes.value_counts()
        dtype_info = "".join([f"<li>{dtype}: {count} cột</li>" for dtype, count in dtype_counts.items()])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        st.markdown(f"""
        <div class="st-custom-card">
            <h3>Kiểu dữ liệu</h3>
            <ul>{dtype_info}</ul>
            <p><b>Cột số:</b> {len(numeric_cols)}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Mẫu dữ liệu ngẫu nhiên")
    sample_size = st.slider("Kích thước mẫu:", 5, 50, 10, key="sample_size")
    if st.button("Lấy mẫu ngẫu nhiên", key="sample_btn"):
        sample_df = df.sample(sample_size)
        st.dataframe(sample_df, width='stretch')