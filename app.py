import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from forecast_utils import multi_product_forecast

# Page configuration
st.set_page_config(page_title="Kirana Smart Inventory", layout="wide", page_icon="🏪")

# Custom CSS for better UI
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.stMetric {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏪 Smart Inventory Prediction for Kirana Shops")
st.markdown("**AI-Powered Demand Forecasting** | Optimize stock levels and boost profits")

# ---------------- Sidebar Settings ----------------
st.sidebar.header("⚙️ Forecast Settings")
st.sidebar.markdown("---")

periods = st.sidebar.number_input(
    "📅 Forecast Horizon (days)", 
    min_value=7, 
    max_value=90, 
    value=30,
    help="Number of days to forecast ahead"
)

FESTIVAL_BOOSTS = {
    "None": 0.0,
    "Pongal": 0.15,
    "Diwali": 0.30,
    "Christmas": 0.20,
    "Ramadan": 0.20,
    "Onam": 0.25,
    "Custom": None
}

fest = st.sidebar.selectbox(
    "🎉 Festival Impact", 
    list(FESTIVAL_BOOSTS.keys()),
    help="Select upcoming festival to adjust demand predictions"
)

custom_boost = 0.0
if FESTIVAL_BOOSTS[fest] is None:
    custom_boost = st.sidebar.slider("Custom Boost %", 0, 100, 10) / 100
    festival_boost_map = {"_all": custom_boost}
else:
    festival_boost_map = {"_all": FESTIVAL_BOOSTS[fest]}

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Upload your sales data CSV to get personalized forecasts!")

# ---------------- File Upload ----------------
st.markdown("### 📂 Upload Sales Data")
uploaded_file = st.file_uploader(
    "Upload CSV file (Required columns: date, sales, product)", 
    type=['csv'],
    help="Your data should have columns: date, sales, product"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate columns
        required_cols = ['date', 'sales', 'product']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ CSV must contain columns: {', '.join(required_cols)}")
            st.stop()
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", len(df))
        with col2:
            st.metric("🛒 Products", df['product'].nunique())
        with col3:
            date_range = (pd.to_datetime(df['date'].max()) - pd.to_datetime(df['date'].min())).days
            st.metric("📅 Data Period", f"{date_range} days")
        with col4:
            st.metric("📈 Total Sales", f"{df['sales'].sum():.0f} units")
        
        # Show sample data
        with st.expander("📋 View Sample Data (First 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Generate forecasts
        with st.spinner("🔮 Generating AI-powered forecasts..."):
            results = multi_product_forecast(df, periods=periods, festival_boost_map=festival_boost_map)
        
        st.success(f"✅ Successfully generated forecasts for {len(results)} products!")
        
        # ---------------- Display Results ----------------
        st.markdown("---")
        st.header("📊 Forecast Results & Insights")
        
        for product, (hist, fc, insights) in results.items():
            with st.expander(f"🛒 {product.replace('_', ' ').title()}", expanded=True):
                
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                trend = fc['trend'].iloc[0] if 'trend' in fc.columns else 'N/A'
                trend_pct = fc['trend_pct'].iloc[0] if 'trend_pct' in fc.columns else 0
                boost = fc['boost_applied'].iloc[0] if 'boost_applied' in fc.columns else 0
                avg_forecast = fc['yhat_boosted'].mean()
                
                with col1:
                    delta_color = "normal" if trend == 'stable' else "inverse" if trend == 'decreasing' else "normal"
                    st.metric("📈 Trend", trend.upper(), f"{trend_pct:+.1f}%", delta_color=delta_color)
                with col2:
                    st.metric("🎉 Festival Boost", f"{boost*100:.1f}%")
                with col3:
                    st.metric("📦 Avg Daily Demand", f"{avg_forecast:.1f} units")
                with col4:
                    total_forecast = fc['yhat_boosted'].sum()
                    st.metric(f"📊 {periods}-Day Total", f"{total_forecast:.0f} units")
                
                # Interactive Chart
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=hist['ds'], y=hist['y'],
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6),
                    hovertemplate='Date: %{x}<br>Sales: %{y} units<extra></extra>'
                ))
                
                # Forecast with boost
                fig.add_trace(go.Scatter(
                    x=fc['ds'], y=fc['yhat_boosted'],
                    mode='lines',
                    name='Forecast (with boost)',
                    line=dict(color='#2ca02c', width=3, dash='dash'),
                    hovertemplate='Date: %{x}<br>Forecast: %{y} units<extra></extra>'
                ))
                
                # Confidence intervals
                if 'yhat_upper' in fc.columns:
                    fig.add_trace(go.Scatter(
                        x=fc['ds'], y=fc['yhat_upper'],
                        mode='lines',
                        name='Upper Bound',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=fc['ds'], y=fc['yhat_lower'],
                        mode='lines',
                        name='95% Confidence',
                        line=dict(width=0),
                        fillcolor='rgba(44,160,44,0.2)',
                        fill='tonexty',
                        hovertemplate='Lower: %{y} units<extra></extra>'
                    ))
                
                fig.update_layout(
                    title=f"{product.replace('_', ' ').title()} - Sales Forecast",
                    xaxis_title="Date",
                    yaxis_title="Sales (units)",
                    hovermode='x unified',
                    height=450,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Smart Insights
                st.subheader("💡 AI-Powered Recommendations")
                for insight in insights:
                    if insight['type'] == 'success':
                        st.success(f"{insight['icon']} {insight['message']}")
                    elif insight['type'] == 'warning':
                        st.warning(f"{insight['icon']} {insight['message']}")
                    else:
                        st.info(f"{insight['icon']} {insight['message']}")
                
                # Download forecast
                csv = fc[['ds', 'yhat_boosted', 'yhat_lower', 'yhat_upper']].rename(columns={
                    'ds': 'Date',
                    'yhat_boosted': 'Forecast',
                    'yhat_lower': 'Lower_Bound',
                    'yhat_upper': 'Upper_Bound'
                }).to_csv(index=False)
                
                st.download_button(
                    f"📥 Download {product} Forecast CSV",
                    csv,
                    f"{product}_forecast_{periods}days.csv",
                    "text/csv",
                    key=f"download_{product}"
                )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV has the correct format with columns: date, sales, product")

else:
    st.info("👆 **Upload your sales data CSV to begin forecasting**")
    
    st.markdown("### 📝 Sample CSV Format:")
    st.code("""date,sales,product
2024-09-01,45,tata_salt_1kg
2024-09-02,47,tata_salt_1kg
2024-09-03,35,amul_milk_500ml
2024-09-04,36,amul_milk_500ml""", language="csv")
    
    st.markdown("### ✨ Key Features:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📈 Trend Detection**\n\nAutomatically identifies sales patterns")
    with col2:
        st.markdown("**🎉 Festival Impact**\n\nAdjust for seasonal demand spikes")
    with col3:
        st.markdown("**💡 Smart Insights**\n\nActionable stock recommendations")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built for OpenAI Academy × NxtWave Buildathon 2025 | "
    "Empowering Kirana Shops with AI"
    "</div>",
    unsafe_allow_html=True
)
