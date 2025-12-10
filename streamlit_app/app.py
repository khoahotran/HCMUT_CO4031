import streamlit as st

from config import setup_page, load_css
from utils.model_loader import load_models, load_price_models, load_processed_data
from components.headers import render_main_header, render_footer
from components.tabs import render_all_tabs

setup_page()
load_css()

def main():
    # Header
    render_main_header()
        
    with st.sidebar.expander("📁 Load Models và Data", expanded=True):
        models = load_models()
        price_models_result = load_price_models()
        if price_models_result:
            price_models, price_config = price_models_result
        else:
            price_models, price_config = None, None
        
        df = load_processed_data()
    
    # Model selection
    selected_model = "Best Model"
    selected_price_model = "Best Price Model"
    
    if models:
        selected_model = st.sidebar.selectbox(
            "Chọn model (Full Features)",
            list(models.keys()),
            index=0 if 'Best Model' in models else 0
        )
    
    if price_models:
        price_model_names = [name for name in price_models.keys() if 'Scaler' not in name]
        selected_price_model = st.sidebar.selectbox(
            "Chọn model (Price Only)",
            price_model_names,
            index=0
        )
    
    render_all_tabs(df, models, price_models, price_config, 
                   selected_model, selected_price_model)
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()