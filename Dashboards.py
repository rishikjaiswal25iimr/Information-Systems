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

# --- SECTION 3.3: METHODOLOGY FLOWCHART ---
st.header("3.3 Methodology Flowchart")
st.write("Visual mapping of the data ingestion to agentic execution pipeline.")

# Using Graphviz for the flowchart
dot = graphviz.Digraph(engine='dot')
dot.attr(rankdir='LR', size='8,5')
dot.node('A', 'Data Ingestion\n(Vendor APIs, Macro Data)', shape='cylinder', style='filled', fillcolor='lightblue')
dot.node('B', 'Data Preparation\n(Pandas Normalization)', shape='box')
dot.node('C', 'AI Model Training\n(Prophet, SciKit-Learn)', shape='box', style='filled', fillcolor='lightgreen')
dot.node('D', 'Agentic Execution\n(Bedrock Frontier Agents)', shape='box', style='filled', fillcolor='lightcoral')
dot.node('E', 'Managerial Output\n(Tableau GPT Dashboards)', shape='ellipse')

dot.edges(['AB', 'BC', 'CD', 'DE'])
st.graphviz_chart(dot)

# --- SECTION 3.6: AI ANALYSIS & INSIGHTS ---
st.header("3.6 AI-Generated Visualizations & Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Predictive Analytics: Demand Forecasting (Prophet)")
    st.write("Forecasting logistics demand to optimize the Order Fill Rate (CSF 6).")
    
    # Prepare data for Prophet
    df_prophet = df[['Date', 'Daily_Demand_Volume']].rename(columns={'Date': 'ds', 'Daily_Demand_Volume': 'y'})
    
    # Train Prophet Model
    m = Prophet(yearly_seasonality=True, daily_seasonality=False)
    m.fit(df_prophet)
    
    # Predict next 30 days
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    # Plotly integration for Prophet
    fig_forecast = plot_plotly(m, forecast)
    fig_forecast.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_forecast, use_container_width=True)

with col2:
    st.subheader("Managerial Dashboard: TACoS vs. IPI")
    st.write("Tracking Advertising Efficiency against Supply Chain Health (CSF 4 & KPI Monitoring).")
    
    # Scatter plot showing relationship between Ad Spend efficiency and Inventory Health
    fig_scatter = px.scatter(df, x="TACoS_Percentage", y="IPI_Score", 
                             trendline="ols", 
                             color="IPI_Score",
                             color_continuous_scale="Viridis",
                             title="Correlation: Adv. Cost of Sales vs Inventory Perf. Index")
    fig_scatter.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- SECTION 3.8: IMPLEMENTATION ROADMAP (GANTT) ---
st.header("3.8 Implementation Roadmap")
st.write("Phased deployment timeline for AI Infrastructure and Frontier Agents.")

# Gantt Chart Data
roadmap_data = [
    dict(Task="Data Infrastructure & API Integration", Start="2025-01-01", Finish="2025-03-15", Phase="Phase 1: Foundation"),
    dict(Task="Model Training (Prophet & SCM AI)", Start="2025-03-01", Finish="2025-06-30", Phase="Phase 2: Analytics"),
    dict(Task="Pilot VAPR (Vision-Assisted Routing)", Start="2025-06-15", Finish="2025-09-01", Phase="Phase 3: Physical Logistics"),
    dict(Task="Deploy GenAI Shopping Guides (Nova)", Start="2025-08-01", Finish="2025-11-30", Phase="Phase 4: Customer Discovery"),
    dict(Task="Frontier Agent Full Autonomous Rollout", Start="2025-11-01", Finish="2026-02-28", Phase="Phase 5: Agentic Execution")
]
df_gantt = pd.DataFrame(roadmap_data)

# Create Gantt using Plotly Timeline
fig_gantt = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Task", color="Phase")
fig_gantt.update_yaxes(autorange="reversed") # Standard Gantt format (top to bottom)
fig_gantt.update_layout(height=400)
st.plotly_chart(fig_gantt, use_container_width=True)
