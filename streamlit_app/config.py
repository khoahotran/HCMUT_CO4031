import streamlit as st

def setup_page():
    """Cấu hình trang Streamlit"""
    st.set_page_config(
        page_title="Amazon Product Rating Prediction",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_css():
    """Load CSS tùy chỉnh"""
    st.markdown("""
    <style>
        /* Import Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Global Settings */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1F2937;
        }

        /* App Background */
        .stApp {
            background-color: #F3F4F6;
        }

        /* Main Header */
        .st-custom-header {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            padding: 3rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        .st-custom-header h1 {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.5rem !important;
            color: white !important;
        }
        
        .st-custom-header p {
            font-size: 1.1rem !important;
            opacity: 0.9;
            color: #E0E7FF !important;
            margin-bottom: 0 !important;
        }

        /* Secondary Header (Price Only) */
        .st-custom-header-secondary {
            background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
            padding: 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .st-custom-header-secondary h2 {
            color: white !important;
            margin-bottom: 0.5rem !important;
        }
        
        .st-custom-header-secondary p {
            color: #D1FAE5 !important;
            margin-bottom: 0 !important;
        }

        /* Cards */
        .st-custom-card {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 1.5rem;
            border: 1px solid #E5E7EB;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            height: 100%;
        }
        
        .st-custom-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }

        /* Metrics */
        .st-custom-metric {
            text-align: center;
            padding: 1rem;
        }
        
        .st-custom-metric h3 {
            font-size: 0.875rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #6B7280 !important;
            margin-bottom: 0.5rem !important;
        }
        
        .st-custom-metric .value {
            font-size: 2rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.25rem;
        }
        
        .st-custom-metric .delta {
            font-size: 0.875rem;
            font-weight: 500;
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 9999px;
        }
        
        .st-custom-metric .delta.positive { background-color: #D1FAE5; color: #059669; }
        .st-custom-metric .delta.negative { background-color: #FEE2E2; color: #DC2626; }
        .st-custom-metric .delta.neutral { background-color: #F3F4F6; color: #6B7280; }

        /* Prediction Result */
        .st-custom-prediction {
            padding: 2.5rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            animation: pulse 2s infinite;
        }
        
        .st-custom-prediction h2 {
            color: white !important;
            font-size: 1.25rem !important;
            opacity: 0.9;
            margin-bottom: 1rem !important;
        }
        
        .st-custom-prediction h1 {
            color: white !important;
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            margin-bottom: 1rem !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.01); }
            100% { transform: scale(1); }
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background-color: white;
            padding: 0.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: auto;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            color: #6B7280;
            border: none;
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: #F9FAFB;
            color: #4B5563;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #EEF2FF;
            color: #4F46E5;
        }

        /* Footer */
        .st-custom-footer {
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid #E5E7EB;
            text-align: center;
            color: #9CA3AF;
            font-size: 0.875rem;
        }

        /* Custom Button Styling (if needed to override Streamlit buttons) */
        div.stButton > button {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
            border-color: transparent;
            color: white;
        }

        div.stButton > button:active {
            transform: translateY(0);
        }

        /* Input Fields */
        .stTextInput > div > div > input {
            border-radius: 8px;
            border-color: #E5E7EB;
        }
        
        .stSelectbox > div > div > div {
            border-radius: 8px;
            border-color: #E5E7EB;
        }

    </style>
    """, unsafe_allow_html=True)