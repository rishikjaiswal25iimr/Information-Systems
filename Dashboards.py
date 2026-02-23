import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import graphviz
from prophet import Prophet
from prophet.plot import plot_plotly

# --- PAGE CONFIG ---
st.set_page_config(page_title="Amazon AI IS Project", layout="wide")
st.title("Amazon E-Commerce: AI & IS Strategic Dashboard")
st.markdown("Deliverables for Sections 3.3, 3.6, and 3.8")

# --- DATASET SIMULATION (Fulfills Deliverable 4.3 & 4.7 prep) ---
@st.cache_data
def generate_simulated_data():
    """Generates a 365-day simulated dataset for Amazon logistics and ad spend."""
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", end="2025-12-31")
    
    # Simulate base demand with a trend and seasonality (Q4 bump)
    base_demand = 50000 + (np.arange(365) * 50) 
    seasonality = np.where(dates.month >= 11, 20000, 0) # Holiday spike
    noise = np.random.normal(0, 5000, 365)
    daily_demand = base_demand + seasonality + noise
    
    # Simulate KPIs
    ad_spend = daily_demand * np.random.uniform(0.08, 0.12, 365) # 8-12% of demand
    tacos = (ad_spend / (daily_demand * 25)) * 100 # Assuming $25 avg order value
    ipi = 600 + (np.random.normal(0, 20, 365)) - (tacos * 2) # IPI drops slightly if ad efficiency drops
    
    df = pd.DataFrame({
        'Date': dates,
        'Daily_Demand_Volume': daily_demand.astype(int),
        'Ad_Spend_USD': ad_spend.astype(int),
        'TACoS_Percentage': tacos.round(2),
        'IPI_Score': ipi.astype(int)
    })
    return df

df = generate_simulated_data()

# --- SIDEBAR: DATA DOWNLOAD (Fulfills Deliverable 4.7) ---
st.sidebar.header("Data Repository")
st.sidebar.write("Download the simulated dataset used for these models.")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Dataset (CSV)",
    data=csv,
    file_name='amazon_simulated_data_2025.csv',
    mime='text/csv',
)

# --- SECTION 3.3: METHODOLOGY FLOWCH
