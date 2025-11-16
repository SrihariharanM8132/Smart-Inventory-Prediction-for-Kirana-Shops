import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def prepare_series(df, date_col='date', value_col='sales'):
    """Prepare time series data"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    ts = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
    return ts

def detect_trend(ts):
    """Detect sales trend: increasing, decreasing, or stable"""
    if len(ts) < 7:
        return 'stable', 0
    
    recent = ts['y'].iloc[-7:].values
    older = ts['y'].iloc[-14:-7].values if len(ts) >= 14 else recent
    
    recent_avg = np.mean(recent)
    older_avg = np.mean(older)
    
    pct_change = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
    
    if pct_change > 10:
        return 'increasing', pct_change
    elif pct_change < -10:
        return 'decreasing', pct_change
    else:
        return 'stable', pct_change

def detect_volatility(ts):
    """Calculate coefficient of variation to measure volatility"""
    if len(ts) < 7:
        return 'low', 0
    
    mean = ts['y'].mean()
    std = ts['y'].std()
    cv = (std / mean * 100) if mean > 0 else 0
    
    if cv > 30:
        return 'high', cv
    elif cv > 15:
        return 'moderate', cv
    else:
        return 'low', cv

def smart_forecast(ts, periods=30):
    """Dynamic forecast based on trend detection"""
    trend, pct_change = detect_trend(ts)
    
    if len(ts) >= 7:
        window = min(7, len(ts))
        last_avg = ts['y'].iloc[-window:].mean()
    else:
        last_avg = ts['y'].mean() if len(ts) > 0 else 0.0
    
    future_dates = pd.date_range(ts['ds'].iloc[-1] + pd.Timedelta(days=1), periods=periods)
    
    # Dynamic adjustment based on trend
    if trend == 'increasing':
        # Conservative growth projection
        growth_rate = min(pct_change / 100, 0.50) / 30  # Cap at 50% monthly growth
        forecasts = [last_avg * (1 + growth_rate * i) for i in range(1, periods + 1)]
    elif trend == 'decreasing':
        # Conservative decline projection with floor
        decay_rate = min(abs(pct_change) / 100, 0.40) / 30  # Cap at 40% monthly decline
        forecasts = [max(last_avg * 0.1, last_avg * (1 - decay_rate * i)) for i in range(1, periods + 1)]
    else:
        # Stable with minimal variation
        noise = np.random.normal(0, last_avg * 0.02, periods)
        forecasts = [max(0, last_avg + n) for n in noise]
    
    return pd.DataFrame({
        'ds': future_dates, 
        'yhat': np.round(forecasts, 2),
        'trend': trend,
        'trend_pct': round(pct_change, 2)
    })

def add_confidence_intervals(ts, forecast):
    """Add prediction intervals based on historical volatility"""
    if len(ts) >= 7:
        std_dev = ts['y'].iloc[-30:].std() if len(ts) >= 30 else ts['y'].std()
    else:
        std_dev = 3
    
    # Wider intervals for volatile products
    volatility, cv = detect_volatility(ts)
    multiplier = 2.0 if volatility == 'high' else 1.5 if volatility == 'moderate' else 1.2
    
    forecast['yhat_lower'] = np.round(forecast['yhat'] - multiplier * std_dev, 2).clip(lower=0)
    forecast['yhat_upper'] = np.round(forecast['yhat'] + multiplier * std_dev, 2)
    forecast['volatility'] = volatility
    forecast['cv'] = round(cv, 2)
    
    return forecast

def calculate_dynamic_boost(ts, base_boost=0.15):
    """Calculate product-specific festival boost"""
    if len(ts) < 14:
        return base_boost
    
    recent_max = ts['y'].iloc[-30:].max() if len(ts) >= 30 else ts['y'].max()
    recent_avg = ts['y'].iloc[-30:].mean() if len(ts) >= 30 else ts['y'].mean()
    
    # Products with high peak-to-average ratio get higher boost
    volatility_factor = (recent_max / recent_avg) if recent_avg > 0 else 1.0
    dynamic_boost = base_boost * min(volatility_factor, 2.5)  # Cap at 2.5x
    
    return round(dynamic_boost, 3)

def holt_winters_forecast(ts, periods=30):
    """Advanced Holt-Winters forecasting with fallback"""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception:
        return smart_forecast(ts, periods)
    
    s = ts.set_index('ds')['y'].asfreq('D').fillna(method='ffill')
    
    try:
        model = ExponentialSmoothing(s, trend='add', seasonal=None, damped_trend=True)
        fit = model.fit(optimized=True)
        fc = fit.forecast(periods)
        
        future_dates = pd.date_range(ts['ds'].iloc[-1] + pd.Timedelta(days=1), periods=periods)
        forecast = pd.DataFrame({'ds': future_dates, 'yhat': np.round(fc.values, 2)})
        
        trend, pct = detect_trend(ts)
        forecast['trend'] = trend
        forecast['trend_pct'] = pct
        
        return forecast
    except Exception as e:
        logger.warning(f"Holt-Winters failed: {e}. Using smart forecast.")
        return smart_forecast(ts, periods)

def auto_forecast(ts, periods=30):
    """Automatically choose best forecasting method"""
    if len(ts) < 14:
        return smart_forecast(ts, periods)
    else:
        return holt_winters_forecast(ts, periods)

def generate_insights(ts, fc):
    """Generate actionable insights for shop owners"""
    insights = []
    
    trend = fc['trend'].iloc[0] if 'trend' in fc.columns else 'stable'
    trend_pct = fc['trend_pct'].iloc[0] if 'trend_pct' in fc.columns else 0
    volatility = fc['volatility'].iloc[0] if 'volatility' in fc.columns else 'low'
    
    # Trend insights
    if trend == 'increasing':
        insights.append({
            'type': 'success',
            'icon': '📈',
            'message': f"Sales trending UP by {abs(trend_pct):.1f}%. Consider stocking extra inventory."
        })
    elif trend == 'decreasing':
        insights.append({
            'type': 'warning',
            'icon': '📉',
            'message': f"Sales trending DOWN by {abs(trend_pct):.1f}%. Consider promotions or reduce orders."
        })
    else:
        insights.append({
            'type': 'info',
            'icon': '➡️',
            'message': "Sales are stable. Maintain current stock levels."
        })
    
    # Volatility insights
    if volatility == 'high':
        insights.append({
            'type': 'warning',
            'icon': '⚠️',
            'message': "High sales volatility detected. Keep safety stock and monitor closely."
        })
    
    # Stock recommendation
    avg_daily = fc['yhat'].mean()
    weekly_stock = avg_daily * 7
    insights.append({
        'type': 'info',
        'icon': '📦',
        'message': f"Recommended weekly stock: {weekly_stock:.0f} units ({avg_daily:.1f} units/day)"
    })
    
    return insights

def forecast_product(df_product, periods=30, festival_boost=0.0):
    """Complete product forecast with all enhancements"""
    ts = prepare_series(df_product, 'date', 'sales')
    
    if len(ts) == 0:
        future_dates = pd.date_range(pd.Timestamp.today(), periods=periods)
        return ts, pd.DataFrame({
            'ds': future_dates, 
            'yhat': 0, 
            'yhat_boosted': 0,
            'trend': 'no_data',
            'trend_pct': 0
        }), []
    
    # Generate base forecast
    forecast = auto_forecast(ts, periods)
    
    # Add confidence intervals
    forecast = add_confidence_intervals(ts, forecast)
    
    # Apply dynamic festival boost
    if festival_boost > 0:
        actual_boost = calculate_dynamic_boost(ts, festival_boost)
    else:
        actual_boost = 0
    
    forecast['yhat_boosted'] = np.round(forecast['yhat'] * (1 + actual_boost), 2)
    forecast['boost_applied'] = actual_boost
    
    # Generate insights
    insights = generate_insights(ts, forecast)
    
    return ts, forecast, insights

def multi_product_forecast(df, periods=30, festival_boost_map=None):
    """Forecast for multiple products"""
    if festival_boost_map is None:
        festival_boost_map = {}
    
    default_boost = festival_boost_map.get('_all', 0.0)
    results = {}
    
    for product in df['product'].unique():
        df_p = df[df['product'] == product]
        boost = festival_boost_map.get(product, default_boost)
        
        hist, fc, insights = forecast_product(df_p, periods=periods, festival_boost=boost)
        results[product] = (hist, fc, insights)
    
    return results
