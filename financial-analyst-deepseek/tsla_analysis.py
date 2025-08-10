import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def analyze_tsla_performance():
    """
    Analyze Tesla (TSLA) stock performance over the last 3 months
    """
    print("🔍 Analyzing Tesla (TSLA) stock performance over the last 3 months...")
    
    # Calculate date range (3 months ago from today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Fetch TSLA stock data
    tsla = yf.Ticker("TSLA")
    hist = tsla.history(start=start_date, end=end_date)
    
    if hist.empty:
        print("❌ No data found for TSLA")
        return
    
    print(f"📊 Data period: {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}")
    print(f"📈 Total trading days: {len(hist)}")
    
    # Calculate key metrics
    initial_price = hist['Close'].iloc[0]
    final_price = hist['Close'].iloc[-1]
    total_return = ((final_price - initial_price) / initial_price) * 100
    
    # Calculate daily returns
    hist['Daily_Return'] = hist['Close'].pct_change()
    
    # Calculate volatility (standard deviation of daily returns)
    volatility = hist['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized
    
    # Find high and low points
    high_price = hist['High'].max()
    low_price = hist['Low'].min()
    high_date = hist['High'].idxmax().strftime('%Y-%m-%d')
    low_date = hist['Low'].idxmin().strftime('%Y-%m-%d')
    
    # Calculate moving averages
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    
    # Volume analysis
    avg_volume = hist['Volume'].mean()
    max_volume = hist['Volume'].max()
    max_volume_date = hist['Volume'].idxmax().strftime('%Y-%m-%d')
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📈 TSLA STOCK PERFORMANCE SUMMARY (Last 3 Months)")
    print("="*60)
    print(f"💰 Initial Price (3 months ago): ${initial_price:.2f}")
    print(f"💰 Current Price: ${final_price:.2f}")
    print(f"📊 Total Return: {total_return:+.2f}%")
    print(f"📈 Highest Price: ${high_price:.2f} ({high_date})")
    print(f"📉 Lowest Price: ${low_price:.2f} ({low_date})")
    print(f"📊 Price Range: ${high_price - low_price:.2f}")
    print(f"📊 Volatility (Annualized): {volatility:.2f}%")
    print(f"📊 Average Daily Volume: {avg_volume:,.0f}")
    print(f"📊 Maximum Daily Volume: {max_volume:,.0f} ({max_volume_date})")
    
    # Additional analysis
    print("\n" + "="*60)
    print("🔍 ADDITIONAL ANALYSIS")
    print("="*60)
    
    # Trend analysis
    if final_price > initial_price:
        trend = "📈 BULLISH"
    else:
        trend = "📉 BEARISH"
    
    print(f"📊 Overall Trend: {trend}")
    
    # Volatility assessment
    if volatility > 50:
        vol_assessment = "HIGH"
    elif volatility > 30:
        vol_assessment = "MODERATE"
    else:
        vol_assessment = "LOW"
    
    print(f"📊 Volatility Assessment: {vol_assessment}")
    
    # Moving average analysis
    current_ma20 = hist['MA20'].iloc[-1]
    current_ma50 = hist['MA50'].iloc[-1]
    
    if final_price > current_ma20 and current_ma20 > current_ma50:
        ma_signal = "📈 BULLISH (Price above both MAs)"
    elif final_price < current_ma20 and current_ma20 < current_ma50:
        ma_signal = "📉 BEARISH (Price below both MAs)"
    else:
        ma_signal = "⚠️ MIXED (Mixed MA signals)"
    
    print(f"📊 Moving Average Signal: {ma_signal}")
    
    # Risk assessment
    if total_return > 20:
        risk_level = "🟢 LOW RISK"
    elif total_return > -10:
        risk_level = "🟡 MODERATE RISK"
    else:
        risk_level = "🔴 HIGH RISK"
    
    print(f"📊 Risk Assessment: {risk_level}")
    
    # Recent performance (last 30 days)
    last_30_days = hist.tail(30)
    if len(last_30_days) > 0:
        recent_return = ((last_30_days['Close'].iloc[-1] - last_30_days['Close'].iloc[0]) / last_30_days['Close'].iloc[0]) * 100
        print(f"📊 Recent 30-day Return: {recent_return:+.2f}%")
    
    # Price momentum
    price_momentum = "📈 STRONG" if total_return > 15 else "📊 MODERATE" if total_return > 0 else "📉 WEAK"
    print(f"📊 Price Momentum: {price_momentum}")
    
    # Show recent price movements
    print(f"\n📊 Recent Price Movements:")
    recent_prices = hist.tail(10)
    for date, row in recent_prices.iterrows():
        print(f"   {date.strftime('%Y-%m-%d')}: ${row['Close']:.2f} (Volume: {row['Volume']:,.0f})")
    
    return hist

if __name__ == "__main__":
    # Run the analysis
    tsla_data = analyze_tsla_performance()
