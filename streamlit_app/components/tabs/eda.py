import streamlit as st
import pandas as pd
import numpy as np
from utils.visualizations import plot_distribution, plot_correlation_heatmap
from components.headers import render_metric_card

def render(df):
    """Render tab phân tích EDA"""
    # st.header("Phân tích Exploratory Data Analysis")
    
    st.subheader("📊 Phân phối các biến số")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Chọn biến để phân tích phân phối", 
                                   numeric_cols, 
                                   key="eda_dist")
        
        if selected_col:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = plot_distribution(df, selected_col, 
                                       f'Distribution of {selected_col}')
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.markdown("**Thống kê:**")
                stats = df[selected_col].describe()
                stats_df = pd.DataFrame({
                    'Statistic': stats.index,
                    'Value': stats.values
                })
                st.dataframe(stats_df, width='stretch')
                
                # Outliers detection
                Q1 = df[selected_col].quantile(0.25)
                Q3 = df[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[selected_col] < (Q1 - 1.5 * IQR)) | 
                             (df[selected_col] > (Q3 + 1.5 * IQR))]
                
                render_metric_card("Outliers", len(outliers))
                render_metric_card("IQR", f"{IQR:.2f}")
    else:
        st.warning("Không tìm thấy cột số để phân tích")
    
    # Correlation matrix
    st.subheader("🔗 Ma trận tương quan")
    if len(numeric_cols) > 1:
        with st.spinner("Đang tính toán correlation..."):
            corr_fig = plot_correlation_heatmap(df)
            st.plotly_chart(corr_fig, width='stretch')
        
        # Strong correlations
        st.write("**Tương quan mạnh:**")
        corr_matrix = df[numeric_cols].corr()
        strong_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    strong_corr.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if strong_corr:
            strong_corr_df = pd.DataFrame(strong_corr)
            st.dataframe(strong_corr_df.sort_values('Correlation', ascending=False), 
                        width='stretch')
        else:
            st.info("Không có tương quan mạnh (|r| > 0.7)")
    else:
        st.warning("Cần ít nhất 2 cột số để tính correlation")
    
    # Scatter plots
    st.subheader("📈 Biểu đồ phân tán")
    if len(numeric_cols) >= 2:
        col_x = st.selectbox("Chọn trục X", numeric_cols, 
                            index=0, key="scatter_x")
        col_y = st.selectbox("Chọn trục Y", numeric_cols, 
                            index=min(1, len(numeric_cols)-1), key="scatter_y")
        
        if col_x and col_y:
            # Tạo scatter plot
            scatter_fig = plot_scatter(df, col_x, col_y)
            st.plotly_chart(scatter_fig, width='stretch')
    else:
        st.warning("Cần ít nhất 2 cột số để vẽ scatter plot")

def plot_scatter(df, x_col, y_col):
    """Vẽ scatter plot"""
    import plotly.express as px
    
    fig = px.scatter(df, x=x_col, y=y_col, 
                    hover_data=df.columns.tolist(),
                    title=f'{x_col} vs {y_col}',
                    color_discrete_sequence=['#667eea'],
                    trendline='ols')
    
    fig.update_layout(
        height=500,
        template='plotly_white'
    )
    
    return fig