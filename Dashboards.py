import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
from sklearn.cluster import KMeans

@st.cache_data
def generate_amazon_data():
    """
    Generates 2 years of daily simulated Amazon e-commerce data 
    (Jan 1, 2024 - Dec 31, 2025) based on SCM and Advertising KPIs.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # 1. Date Range Setup (2 Years)
    dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')
    n_days = len(dates)
    
    # 2. Daily Demand Volume
    # Base trend simulating organic growth
    base_demand = np.linspace(500000, 750000, n_days)
    
    # Seasonality and Spikes (Q4 and Prime Day)
    month = dates.month
    day = dates.day
    seasonal_multiplier = np.ones(n_days)
    
    # General Q4 buildup (November & December)
    seasonal_multiplier[(month == 11) | (month == 12)] = 1.3
    # Black Friday / Cyber Monday hyper-spikes
    seasonal_multiplier[(month == 11) & (day >= 24) & (day <= 30)] = 2.2
    # Prime Day (Simulated mid-July)
    seasonal_multiplier[(month == 7) & (day >= 15) & (day <= 16)] = 1.9
    
    # Add random noise
    noise = np.random.normal(0, 25000, n_days)
    daily_demand = np.maximum((base_demand * seasonal_multiplier) + noise, 0)
    
    # 3. Ad Spend USD
    # Correlated to demand, but with diminishing returns/higher CPCs during peak seasons
    base_cpc_factor = np.random.uniform(2.0, 3.5, n_days)
    base_cpc_factor[(month == 11) | (month == 12)] *= 1.4 # Ads cost more in Q4
    ad_spend = daily_demand * base_cpc_factor
    
    # 4. TACoS Percentage (Total Ad Cost of Sales)
    aov = 25.0 # Average Order Value
    total_revenue = daily_demand * aov
    tacos_percentage = (ad_spend / total_revenue) * 100
    
    # 5. Order Fill Rate (OFR)
    # Target: 92% to 99.9%. Dips during sudden demand surges.
    demand_pct_change = pd.Series(daily_demand).pct_change().fillna(0).values
    ofr = np.random.uniform(98.5, 99.9, n_days) # Standard steady-state OFR
    
    # Mask for sudden demand spikes (e.g., greater than 15% day-over-day growth)
    spike_mask = demand_pct_change > 0.15
    ofr[spike_mask] = np.random.uniform(92.0, 96.0, sum(spike_mask)) # Supply chain strains
    ofr = np.clip(ofr, 92.0, 99.9) # Hard boundary compliance
    
    # 6. IPI Score (Inventory Performance Index)
    # Base ~500. Range 0-1000. 
    # High TACoS lowers score (inefficient capital), steady demand raises it.
    
    # Calculate 7-day demand volatility
    demand_volatility = pd.Series(daily_demand).rolling(window=7, min_periods=1).std().fillna(0).values
    normalized_volatility = (demand_volatility - demand_volatility.min()) / (demand_volatility.max() - demand_volatility.min() + 1e-9)
    
    # Penalty calculation
    tacos_penalty = (tacos_percentage - tacos_percentage.mean()) * 12
    volatility_penalty = normalized_volatility * 150
    
    # Calculate IPI
    ipi_score = 600 - tacos_penalty - volatility_penalty + np.random.normal(0, 15, n_days)
    ipi_score = np.clip(ipi_score, 0, 1000) # Enforce bounds
    
    # 7. Compile the Dataset
    df = pd.DataFrame({
        'Date': dates,
        'Daily_Demand_Volume': daily_demand.astype(int),
        'Ad_Spend_USD': ad_spend.round(2),
        'TACoS_Percentage': tacos_percentage.round(2),
        'IPI_Score': ipi_score.astype(int),
        'Order_Fill_Rate_OFR': ofr.round(2)
    })
    
    return df

# --- UI Layout and Interactive Dashboard ---

# Page Configuration
st.set_page_config(page_title="Amazon SCM & AI Analytics", layout="wide")
st.title("Amazon SCM & AI Analytics Dashboard")
st.markdown("---")

# --- Section 3.3: Methodology Flowchart ---
st.header("Methodology Flowchart")
st.markdown("""
*Maps to **System Architecture & Data Pipeline***.  
The following enterprise architecture diagram illustrates the end-to-end data pipeline, moving from raw API ingestion through generative AI execution and ending in managerial visualization.
""")

# Using DOT language for a clean left-to-right (LR) graph
st.graphviz_chart('''
    digraph SCM_Architecture {
        rankdir=LR;
        node [shape=rect, style="filled,rounded", fontname="Arial", fontsize=12, margin=0.3, color="white", penwidth=2];
        edge [color="#7f8c8d", penwidth=2, arrowsize=0.8];
        
        A [label="Data Ingestion\\n(Vendor APIs)", fillcolor="#2980b9", fontcolor="white"];
        B [label="Data Preparation\\n(Pandas)", fillcolor="#f39c12", fontcolor="white"];
        C [label="AI Model Training\\n(Prophet ML)", fillcolor="#27ae60", fontcolor="white"];
        D [label="Agentic Execution\\n(Bedrock Frontier Agents)", fillcolor="#c0392b", fontcolor="white"];
        E [label="Managerial Output\\n(Tableau GPT)", fillcolor="#8e44ad", fontcolor="white"];
        
        A -> B -> C -> D -> E;
    }
''')
st.markdown("---")

# Section 3.6 Header
st.header("Predictive Analytics & Demand Forecasting")
st.markdown("""
*Maps to **CSF 6 (Supply Chain Resilience)***.  
This tool utilizes AI forecasting models to predict inventory strain. Adjust the marketing push slider below to simulate how sudden demand spikes affect localized fulfillment predictions.
""")

# Sidebar Controls
st.sidebar.header("Forecast Controls")
marketing_multiplier = st.sidebar.slider(
    "Simulate Q4 Marketing Push (Demand Multiplier)", 
    min_value=1.0, 
    max_value=2.0, 
    value=1.0, 
    step=0.1,
    help="Multiplies the recent 60 days of demand to simulate a massive ad campaign."
)

# 1. Load Data
df = generate_amazon_data()

# --- Deliverable 4.7: Data Export ---
st.sidebar.markdown("---")
st.sidebar.header("Data Export")
csv_data = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Raw Data as CSV",
    data=csv_data,
    file_name='amazon_simulated_dataset.csv',
    mime='text/csv',
    help="Download the complete generated dataset for external analysis."
)

# 2. Prepare Data for Prophet
df_prophet = df[['Date', 'Daily_Demand_Volume']].copy()
df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Daily_Demand_Volume': 'y'})

# 3. Apply Multiplier to Recent Data (last 60 days)
recent_days = 60
df_prophet.loc[df_prophet.index[-recent_days:], 'y'] = df_prophet.loc[df_prophet.index[-recent_days:], 'y'] * marketing_multiplier

# 4. Train Prophet Model
with st.spinner("Training Prophet AI model on selected data..."):
    # Initialize and fit the model (capturing weekly & yearly trends)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df_prophet)
    
    # Predict 90 days into the future
    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)

# 5. Render Interactive Plotly Chart
st.subheader("AI-Driven 90-Day Demand Forecast")
fig = plot_plotly(m, forecast)

# Customize the chart layout for better presentation
fig.update_layout(
    title=f"Demand Trajectory (Push Multiplier: {marketing_multiplier}x)",
    xaxis_title="Date",
    yaxis_title="Daily Demand Volume (Units)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# --- Managerial Dashboard Section ---

st.markdown("---")
st.header("Managerial Dashboard: Margin Optimization")
st.markdown("""
*Maps to **CSF 4 (Margin Optimization)***.  
This dashboard analyzes the critical relationship between Advertising Efficiency (TACoS) and Inventory Performance (IPI), filtered by financial quarter.
""")

# 1. Data Preparation for Quarters
# Create a 'Quarter' column (Q1, Q2, Q3, Q4) based on the Date
df['Quarter'] = 'Q' + df['Date'].dt.quarter.astype(str)

# 2. Interactive Dropdown Filter
selected_quarter = st.selectbox(
    "Select Financial Quarter to Analyze", 
    options=['Q1', 'Q2', 'Q3', 'Q4'],
    index=3 # Default to Q4 to show the most interesting seasonal data
)

# Filter the dataframe
filtered_df = df[df['Quarter'] == selected_quarter]

# 3. Dynamic Metric Cards
st.subheader(f"{selected_quarter} Performance Averages")
col1, col2, col3 = st.columns(3)

avg_tacos = filtered_df['TACoS_Percentage'].mean()
avg_ipi = filtered_df['IPI_Score'].mean()
avg_ofr = filtered_df['Order_Fill_Rate_OFR'].mean()

col1.metric("Average TACoS (%)", f"{avg_tacos:.2f}%")
col2.metric("Average IPI Score", f"{avg_ipi:.0f}")
col3.metric("Average OFR (%)", f"{avg_ofr:.2f}%")

# 4. Interactive Scatter Plot with Trendline
st.subheader("TACoS vs. IPI Correlation Matrix")

fig_scatter = px.scatter(
    filtered_df,
    x='TACoS_Percentage',
    y='IPI_Score',
    color='Order_Fill_Rate_OFR',
    trendline='ols', # Ordinary Least Squares regression line
    title=f"{selected_quarter}: Advertising Spend Inefficiency vs. Inventory Health",
    labels={
        'TACoS_Percentage': 'Total Ad Cost of Sales (TACoS) %',
        'IPI_Score': 'Inventory Performance Index (IPI)',
        'Order_Fill_Rate_OFR': 'Order Fill Rate (%)'
    },
    color_continuous_scale="RdYlBu", # Red/Yellow/Blue emphasizes operational strain
    hover_data=['Date'] # Add date to the hover tooltip
)

# Improve layout
fig_scatter.update_layout(
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig_scatter, use_container_width=True)

# --- Deliverable 4.4: Additional Visual Insights ---
st.markdown("---")
st.header("Advanced AI Clustering & Monthly Trends")

col_cluster, col_trend = st.columns(2)

with col_cluster:
    st.subheader("K-Means Behavioral Clustering")
    st.markdown("Identifies distinct operational phases using Machine Learning.")
    
    # Prepare data for K-Means
    cluster_data = df[['Ad_Spend_USD', 'Daily_Demand_Volume']].copy()
    
    # Initialize and fit K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_data['Cluster'] = kmeans.fit_predict(cluster_data)
    
    # Dynamically assign labels by sorting cluster centers by Demand Volume
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Ad_Spend_USD', 'Daily_Demand_Volume'])
    centers['Cluster'] = centers.index
    centers = centers.sort_values(by='Daily_Demand_Volume')
    
    label_mapping = {
        centers.iloc[0]['Cluster']: 'Low Efficiency',
        centers.iloc[1]['Cluster']: 'Standard',
        centers.iloc[2]['Cluster']: 'High Momentum'
    }
    df['Cluster_Label'] = cluster_data['Cluster'].map(label_mapping)
    
    # Plotly Scatter Plot for Clusters
    fig_cluster = px.scatter(
        df, 
        x='Ad_Spend_USD', 
        y='Daily_Demand_Volume', 
        color='Cluster_Label',
        title="Operational Segments: Ad Spend vs Demand",
        labels={
            'Ad_Spend_USD': 'Daily Ad Spend ($)', 
            'Daily_Demand_Volume': 'Daily Demand Volume (Units)',
            'Cluster_Label': 'Market Segment'
        },
        color_discrete_map={
            'Low Efficiency': '#e74c3c', 
            'Standard': '#3498db', 
            'High Momentum': '#2ecc71'
        }
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

with col_trend:
    st.subheader("Monthly Aggregation (Q4 Spikes)")
    st.markdown("Tracks the macro-trend of Advertising Spend scaling against Sales Volume.")
    
    # Group by Year-Month for Bar Chart
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    monthly_df = df.groupby('YearMonth')[['Daily_Demand_Volume', 'Ad_Spend_USD']].sum().reset_index()
    
    # Melt dataframe to make it compatible with stacked bar chart
    monthly_melt = monthly_df.melt(
        id_vars=['YearMonth'], 
        value_vars=['Daily_Demand_Volume', 'Ad_Spend_USD'], 
        var_name='Metric', 
        value_name='Aggregate Total'
    )
    
    # Format labels for cleaner display
    monthly_melt['Metric'] = monthly_melt['Metric'].replace({
        'Daily_Demand_Volume': 'Demand Volume (Units)',
        'Ad_Spend_USD': 'Ad Spend (USD)'
    })
    
    # Plotly Stacked Bar Chart
    fig_bar = px.bar(
        monthly_melt, 
        x='YearMonth', 
        y='Aggregate Total', 
        color='Metric',
        barmode='stack',
        title="Monthly Cumulative Demand & Spend",
        labels={'YearMonth': 'Financial Month'},
        color_discrete_sequence=['#9b59b6', '#f1c40f']
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# --- Section 3.8: Implementation Roadmap ---

st.markdown("---")
st.header("Implementation Roadmap")
st.markdown("""
*Maps to **System Integration & Project Execution***.  
The following Gantt chart outlines the strategic timeline for deploying the AI and SCM infrastructure across the 2025-2026 fiscal cycle. Use the filter below to isolate specific operational phases.
""")

# 1. Define Roadmap Data
roadmap_data = pd.DataFrame([
    {"Task": "Phase 1: Data Infrastructure", "Start": "2025-01-01", "Finish": "2025-06-30"},
    {"Task": "Phase 2: SCM Model Training", "Start": "2025-04-01", "Finish": "2025-10-31"},
    {"Task": "Phase 3: Pilot VAPR Logistics", "Start": "2025-08-01", "Finish": "2026-02-28"},
    {"Task": "Phase 4: GenAI Shopping Guides", "Start": "2025-11-01", "Finish": "2026-08-31"},
    {"Task": "Phase 5: Autonomous Agent Rollout", "Start": "2026-05-01", "Finish": "2026-12-31"}
])

# 2. Interactive Multi-Select Filter
all_phases = roadmap_data["Task"].tolist()
selected_phases = st.multiselect(
    "Select Implementation Phases to Display",
    options=all_phases,
    default=all_phases
)

# Filter data based on selection
filtered_roadmap = roadmap_data[roadmap_data["Task"].isin(selected_phases)]

# 3. Render Gantt Chart
if not filtered_roadmap.empty:
    fig_gantt = px.timeline(
        filtered_roadmap, 
        x_start="Start", 
        x_end="Finish", 
        y="Task", 
        color="Task",
        title="Strategic AI & Logistical Deployment (2025-2026)"
    )
    
    # Reverse Y-axis so Phase 1 is at the top
    fig_gantt.update_yaxes(autorange="reversed")
    
    # Improve Layout
    fig_gantt.update_layout(
        showlegend=False, # Legend is redundant since Y-axis has the names
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Timeline",
        yaxis_title="Implementation Phase"
    )
    
    st.plotly_chart(fig_gantt, use_container_width=True)
else:
    st.warning("Please select at least one phase to display the roadmap.")
