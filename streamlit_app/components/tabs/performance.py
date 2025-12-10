import traceback
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.visualizations import plot_model_comparison, plot_model_metrics_bar, plot_accuracy_comparison, plot_cv_vs_accuracy
from components.headers import render_metric_card

def extract_scalar(value):
    """
    Trích xuất giá trị scalar từ pandas object một cách an toàn
    """
    if isinstance(value, pd.Series):
        if len(value) > 0:
            return extract_scalar(value.iloc[0])
        return None
    
    if pd.isna(value):
        return None
    
    if isinstance(value, type(pd.NA)):
        return None
    
    if isinstance(value, np.generic):
        return value.item()
    
    if hasattr(value, 'item'):
        try:
            return value.item()
        except:
            pass
    
    if isinstance(value, (int, float, str, bool)):
        return value
    
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ['nan', 'na', 'none', 'null', '']:
            return None
        try:
            return float(value)
        except:
            return value
    
    try:
        return str(value)
    except:
        return None

def is_scalar_na(value):
    """
    Kiểm tra xem giá trị scalar có phải là NA không
    """
    scalar = extract_scalar(value)
    if scalar is None:
        return True
    
    if isinstance(scalar, str):
        return scalar.lower() in ['nan', 'na', 'none', 'null', '']
    
    return pd.isna(scalar)

def format_value(value, format_str=".4f"):
    """
    Format giá trị để hiển thị
    """
    scalar = extract_scalar(value)
    
    if scalar is None or is_scalar_na(scalar):
        return "N/A"
    
    if isinstance(scalar, (int, float, np.number)):
        try:
            return format(float(scalar), format_str)
        except:
            return str(scalar)
    
    return str(scalar)

def render(models, price_models, price_config):
    """Render tab đánh giá hiệu suất model"""
    # st.header("📋 Đánh giá hiệu suất model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_full_models_performance(models)
    
    with col2:
        render_price_models_performance(price_models, price_config)
    
    # Additional metrics
    st.markdown("---")
    render_additional_metrics()

def render_full_models_performance(models):
    """Render hiệu suất full models"""
    # st.subheader("📊 Full Feature Models")
    
    if not models:
        st.warning("Không có full feature models được load")
        return
    
    perf_tab1, perf_tab2 = st.tabs(["📈 Summary Metrics", "📊 Detailed Analysis"])
    
    with perf_tab1:
        render_performance_summary()
    
    with perf_tab2:
        render_detailed_performance_data()

def render_performance_summary():
    """Render summary metrics từ file"""
    try:
        comparison_paths = [
            './output/model_comparison.csv'
        ]
        
        comparison_path = None
        for path in comparison_paths:
            if os.path.exists(path):
                comparison_path = path
                break
        
        if comparison_path:
            comparison_df = pd.read_csv(comparison_path)
            
            st.subheader("📋 Bảng tổng hợp hiệu suất")
            
            display_df = comparison_df.copy()
            
            numeric_columns = []
            for col in display_df.columns:
                if col != 'Model' and col != 'Parameters':
                    try:
                        # Convert to numeric
                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                        numeric_columns.append(col)
                    except:
                        pass
            
            # Format numeric columns
            for col in numeric_columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
            
            st.dataframe(display_df, width='stretch')
            
            # Visualization
            st.subheader("📈 So sánh trực quan")
            fig = plot_model_comparison(comparison_df)
            if fig:
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Không thể tạo biểu đồ từ dữ liệu hiện có")
            
            if 'Accuracy' in comparison_df.columns:
                try:
                    analysis_df = comparison_df.copy()
                    
                    # Convert Accuracy sang numeric
                    analysis_df['Accuracy_numeric'] = pd.to_numeric(
                        analysis_df['Accuracy'], errors='coerce'
                    )
                    
                    valid_accuracy_df = analysis_df.dropna(subset=['Accuracy_numeric'])
                    
                    if not valid_accuracy_df.empty:
                        best_idx = valid_accuracy_df['Accuracy_numeric'].idxmax()
                        best_row = valid_accuracy_df.loc[best_idx]
                        
                        best_model_name = extract_scalar(best_row['Model'])
                        best_accuracy = extract_scalar(best_row['Accuracy_numeric'])
                        best_cv = extract_scalar(best_row.get('Best CV Score', pd.NA))
                        
                        st.success(f"""
                        ### **Best Full Model:** {best_model_name if best_model_name else 'Unknown'}
                        
                        **Accuracy:** {format_value(best_accuracy)}
                        **Best CV Score:** {format_value(best_cv)}
                        """)
                    else:
                        st.warning("Không có dữ liệu Accuracy hợp lệ")
                        
                except Exception as e:
                    st.error(f"Lỗi khi tìm best model: {str(e)}")
        else:
            prediction_paths = [
                './output/model_predictions.csv'
            ]
            
            prediction_path = None
            for path in prediction_paths:
                if os.path.exists(path):
                    prediction_path = path
                    break
            
            if prediction_path:
                render_performance_from_predictions(prediction_path)
            else:
                st.info("Chưa có thông tin so sánh full models")
                st.write("Cần file `model_comparison.csv` chứa kết quả evaluation")
                
    except Exception as e:
        st.error(f"Lỗi khi load thông tin performance: {str(e)}")
        st.error(traceback.format_exc())

def render_price_models_performance(price_models, price_config):
    """Render hiệu suất price models"""
    st.subheader("💰 Price-Only Models")
    
    if not price_models or not price_config:
        st.warning("Không có price-only models được load")
        return
    
    try:
        if 'models' in price_config:
            price_models_info = price_config['models']
            
            price_data = []
            for model_name, model_info in price_models_info.items():
                row = {'Model': model_name.replace('_', ' ').title()}
                
                accuracy = extract_scalar(model_info.get('accuracy', 0))
                cv_score = extract_scalar(model_info.get('cv_score', 0))
                
                row.update({
                    'Accuracy': accuracy,
                    'CV Score': cv_score
                })
                
                price_data.append(row)
            
            price_df = pd.DataFrame(price_data)
            
            display_df = price_df.copy()
            for col in display_df.columns:
                if col != 'Model':
                    display_df[col] = display_df[col].apply(format_value)
            
            st.dataframe(display_df, width='stretch')
            
            st.subheader("📈 So sánh price models")
            fig = plot_model_comparison(price_df)
            if fig:
                st.plotly_chart(fig, width='stretch')
            
            if 'best_model' in price_config:
                best_price = price_config['best_model']
                display_name = best_price.get('display_name', 'Unknown')
                if not display_name:
                    display_name = best_price.get('name', 'Unknown').title()
                
                accuracy = extract_scalar(best_price.get('accuracy', 0))
                features_used = len(price_config.get('features', []))
                
                st.success(f"""
                ### **Best Price Model:** {display_name}
                
                **Accuracy:** {format_value(accuracy)}
                **Features used:** {features_used}
                """)
                
    except Exception as e:
        st.error(f"Lỗi khi hiển thị price model performance: {str(e)}")
        st.error(traceback.format_exc())

def render_detailed_performance_data():
    """Render dữ liệu performance chi tiết"""
    try:
        files_found = False
        
        performance_files = [
            ('model_comparison.csv', './output/'),
            ('model_predictions.csv', './output/'),
        ]
        
        for filename, path_prefix in performance_files:
            full_path = f"{path_prefix}{filename}"
            if os.path.exists(full_path):
                files_found = True
                
                with st.expander(f"📄 {filename}", expanded=False):
                    if filename.endswith('.csv'):
                        df = pd.read_csv(full_path)
                        st.dataframe(df, width='stretch')
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=f"Download {filename}",
                            data=csv,
                            file_name=filename,
                            mime="text/csv"
                        )
        
        if not files_found:
            st.info("Không tìm thấy file performance chi tiết")
            
    except Exception as e:
        st.error(f"Lỗi khi load dữ liệu chi tiết: {str(e)}")

def render_performance_from_predictions(prediction_path):
    """Tính toán performance metrics từ file predictions.csv"""
    try:
        pred_df = pd.read_csv(prediction_path)
        
        if 'y_true' not in pred_df.columns:
            st.error("File predictions.csv phải có cột 'y_true'")
            return
        
        y_true = pred_df['y_true']
        
        metrics_dict = {}
        y_pred_dict = {}
        
        pred_cols = [col for col in pred_df.columns if col.startswith('y_pred_')]
        
        if not pred_cols:
            st.warning("Không tìm thấy cột prediction (cần y_pred_rf, y_pred_xgb, etc.)")
            return
        
        for pred_col in pred_cols:
            model_name = pred_col.replace('y_pred_', '').upper()
            y_pred = pred_df[pred_col]
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            metrics_dict[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }
            
            y_pred_dict[model_name] = y_pred
        
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        metrics_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
        
        st.dataframe(metrics_df, width='stretch')
        
        # Visualization
        st.subheader("📈 So sánh trực quan")
        fig = plot_model_metrics_bar(metrics_dict)
        if fig:
            st.plotly_chart(fig, width='stretch')
        
        # Best model
        if not metrics_df.empty:
            best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
            st.success(f"""
            ### **Best Model từ predictions:** {best_model['Model']}
            
            **Accuracy:** {best_model['Accuracy']:.4f}
            **Precision:** {best_model['Precision']:.4f}
            **Recall:** {best_model['Recall']:.4f}
            **F1-Score:** {best_model['F1-Score']:.4f}
            """)
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý file predictions: {str(e)}")
        st.error(traceback.format_exc())

def render_additional_metrics():
    """Render các metrics bổ sung"""
    st.subheader("📈 Metrics bổ sung")
    
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    with col_metrics1:
        # Check if model comparison file exists
        comparison_paths = ['./output/model_comparison.csv']
        for path in comparison_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                render_metric_card("Total Models", f"{len(df)}", "Full features")
                break
        else:
            render_metric_card("Total Models", "0", "No data")
    
    with col_metrics2:
        # Check model files
        model_files = ['./output/random_forest_model.pkl', './output/xgboost_model.pkl', './output/knn_model.pkl']
        loaded_count = sum(1 for f in model_files if os.path.exists(f))
        render_metric_card("Loaded Models", f"{loaded_count}/3", f"{loaded_count*100/3:.0f}%")
    
    with col_metrics3:
        if os.path.exists('./output/model_comparison.csv'):
            df = pd.read_csv('./output/model_comparison.csv')
            if 'Accuracy' in df.columns:
                df['Accuracy_numeric'] = pd.to_numeric(df['Accuracy'], errors='coerce')
                if not df['Accuracy_numeric'].isna().all():
                    avg_accuracy = df['Accuracy_numeric'].mean()
                    render_metric_card("Avg Accuracy", f"{avg_accuracy:.1%}", "All models")
        else:
            render_metric_card("Avg Accuracy", "N/A", "No data")
    
    # Model status
    st.subheader("🔍 Trạng thái models")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.write("**Full Models:**")
        model_files = {
            'Random Forest': './output/random_forest_model.pkl',
            'XGBoost': './output/xgboost_model.pkl',
            'KNN': './output/knn_model.pkl'
        }
        
        for name, file in model_files.items():
            if os.path.exists(file):
                st.success(f"{name} - Loaded")
            else:
                st.warning(f"{name} - Missing")
    
    with status_col2:
        st.write("**Price Models:**")
        price_model_dir = './price_models'
        if os.path.exists(price_model_dir):
            price_files = ['rf_price_model.pkl', 'xgb_price_model.pkl', 'knn_price_model.pkl']
            for file in price_files:
                if os.path.exists(os.path.join(price_model_dir, file)):
                    model_name = file.replace('_price_model.pkl', '').upper()
                    st.success(f"{model_name} - Loaded")
                else:
                    model_name = file.replace('_price_model.pkl', '').upper()
                    st.warning(f"{model_name} - Missing")
        else:
            st.error("Price models directory not found")