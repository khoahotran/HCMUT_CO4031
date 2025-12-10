import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st

def plot_distribution(df, column, title):
    """Vẽ biểu đồ phân phối"""
    fig = px.histogram(df, x=column, nbins=30, 
                      title=title,
                      color_discrete_sequence=['#667eea'])
    fig.update_layout(
        bargap=0.1,
        height=400,
        template='plotly_white'
    )
    return fig

def plot_correlation_heatmap(df):
    """Vẽ heatmap correlation"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        hoverongaps=False,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='Correlation Matrix',
        width=700,
        height=700,
        template='plotly_white'
    )
    
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """Vẽ biểu đồ feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[feature_names[i] for i in indices],
                y=importances[indices],
                marker_color='#764ba2',
                text=importances[indices].round(4),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Importance: %{y:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f'Feature Importance - {model_name}',
            xaxis_title='Features',
            yaxis_title='Importance',
            height=500,
            template='plotly_white',
            xaxis_tickangle=-45
        )
        
        return fig
    return None

def plot_prediction_probability(probabilities, model_type='full'):
    """Vẽ biểu đồ xác suất dự đoán"""
    if model_type == 'price':
        labels = ['Average', 'Good', 'Excellent']
        colors = ['#FFA726', '#42A5F5', '#66BB6A']
    else:
        labels = ['Bad', 'Qualified', 'Good']
        colors = ['#EF5350', '#42A5F5', '#66BB6A']
    
    fig = px.bar(
        x=labels, 
        y=probabilities,
        title='Prediction Probabilities',
        color=labels,
        color_discrete_sequence=colors,
        text=[f'{p:.2%}' for p in probabilities]
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        showlegend=False,
        yaxis_title='Probability',
        yaxis_tickformat='.0%'
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

def plot_accuracy_comparison(comparison_df):
    """Vẽ biểu đồ so sánh accuracy giữa các models"""
    if comparison_df is None or comparison_df.empty:
        return None
    
    try:
        df = comparison_df.copy()
        
        df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
        
        df = df.sort_values('Accuracy', ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['Model'],
                y=df['Accuracy'],
                text=df['Accuracy'].round(4),
                textposition='auto',
                marker_color=['#00b09b', '#96c93d', '#667eea', '#764ba2', '#f093fb'][:len(df)],
                hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Accuracy Comparison',
            height=400,
            template='plotly_white',
            xaxis_title='Model',
            yaxis_title='Accuracy',
            yaxis_range=[0, 1.0]
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Lỗi khi vẽ accuracy comparison: {str(e)}")
        return None

def plot_cv_vs_accuracy(comparison_df):
    """Vẽ biểu đồ so sánh Accuracy vs Best CV Score"""
    if comparison_df is None or comparison_df.empty:
        return None
    
    try:
        df = comparison_df.copy()
        
        df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
        df['Best CV Score'] = pd.to_numeric(df['Best CV Score'], errors='coerce')
        
        fig = go.Figure()
        
        for _, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Accuracy']],
                y=[row['Best CV Score']],
                mode='markers+text',
                name=row['Model'],
                text=[row['Model']],
                textposition='top center',
                marker=dict(size=15),
                hovertemplate=f"<b>{row['Model']}</b><br>"
                            f"Accuracy: {row['Accuracy']:.4f}<br>"
                            f"CV Score: {row['Best CV Score']:.4f}<extra></extra>"
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Ideal Line',
            line=dict(color='gray', dash='dash'),
            hoverinfo='none'
        ))
        
        fig.update_layout(
            title='Accuracy vs Cross-Validation Score',
            height=500,
            template='plotly_white',
            xaxis_title='Test Accuracy',
            yaxis_title='CV Score',
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Lỗi khi vẽ CV vs Accuracy: {str(e)}")
        return None
    
def plot_model_comparison(comparison_df):
    """Vẽ biểu đồ so sánh models - FIXED VERSION cho file mới"""
    if comparison_df is None or comparison_df.empty:
        st.warning("Không có dữ liệu để vẽ biểu đồ so sánh")
        return None
    
    try:
        df = comparison_df.copy()
        
        if 'Model' not in df.columns:
            st.error("Thiếu cột 'Model' trong comparison data")
            return None
        
        if 'Parameters' in df.columns:
            df = df.drop('Parameters', axis=1)
        
        for col in df.columns:
            if col != 'Model':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        metric_columns = [col for col in df.columns 
                         if col != 'Model' and not df[col].isna().all()]
        
        if not metric_columns:
            st.warning("Không có metrics số để so sánh")
            return None
        
        fig = go.Figure()
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#00b09b', '#96c93d']
        
        for i, (_, row) in enumerate(df.iterrows()):
            model_name = row['Model']
            metrics_values = []
            metric_labels = []
            
            for col in metric_columns:
                val = row[col]
                if pd.notna(val):
                    metrics_values.append(val)
                    metric_labels.append(col)
            
            if metrics_values:
                fig.add_trace(go.Bar(
                    name=model_name,
                    x=metric_labels,
                    y=metrics_values,
                    text=[f'{val:.3f}' for val in metrics_values],
                    textposition='auto',
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f'<b>{model_name}</b><br>' + 
                                '<br>'.join([f'{metric}: %{{y:.3f}}' 
                                            for metric in metric_labels]) +
                                '<extra></extra>'
                ))
        
        y_max = df[metric_columns].max().max()
        y_min = df[metric_columns].min().min()
        y_range = [max(0, y_min - 0.1), min(1.0, y_max + 0.1)]
        
        fig.update_layout(
            title='So sánh hiệu suất các models',
            barmode='group',
            height=500,
            template='plotly_white',
            xaxis_title='Metrics',
            yaxis_title='Score',
            yaxis_range=y_range,
            legend_title='Models',
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Lỗi khi vẽ biểu đồ so sánh: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None
def plot_confusion_matrices(y_true, y_pred_dict):
    """Vẽ confusion matrices cho nhiều models"""
    from sklearn.metrics import confusion_matrix
    import plotly.subplots as sp
    
    if not y_pred_dict:
        return None
    
    n_models = len(y_pred_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig = sp.make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=list(y_pred_dict.keys()),
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    row, col = 1, 1
    for model_name, y_pred in y_pred_dict.items():
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        heatmap = go.Heatmap(
            z=cm_normalized,
            x=['Bad', 'Qualified', 'Good'] if len(cm) == 3 else ['0', '1', '2'],
            y=['Bad', 'Qualified', 'Good'] if len(cm) == 3 else ['0', '1', '2'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='Thực tế: %{y}<br>Dự đoán: %{x}<br>Số lượng: %{text}<br>Tỷ lệ: %{z:.2%}<extra></extra>',
            showscale=(row == 1 and col == 1)
        )
        
        fig.add_trace(heatmap, row=row, col=col)
        
        col += 1
        if col > n_cols:
            col = 1
            row += 1
    
    fig.update_layout(
        title='Confusion Matrices',
        height=300 * n_rows,
        template='plotly_white'
    )
    
    return fig

def plot_model_metrics_bar(metrics_dict):
    """Vẽ bar chart cho các metrics của model"""
    if not metrics_dict:
        return None
    
    metrics_data = []
    for model_name, metrics in metrics_dict.items():
        row = {'Model': model_name}
        row.update(metrics)
        metrics_data.append(row)
    
    df = pd.DataFrame(metrics_data)
    
    metric_cols = [col for col in df.columns if col != 'Model']
    
    fig = go.Figure()
    
    for metric in metric_cols:
        fig.add_trace(go.Bar(
            name=metric,
            x=df['Model'],
            y=df[metric],
            text=df[metric].round(3),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' + f'{metric}: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Model Performance Metrics',
        barmode='group',
        height=500,
        template='plotly_white',
        xaxis_title='Model',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        legend_title='Metrics'
    )
    
    return fig