import pandas as pd
import numpy as np
from datetime import datetime
import os

# Load the combined data
file_path = r"C:\Users\user\Desktop\Funds data\Combined_Fund_holdings.csv"
df = pd.read_csv(file_path)

# Convert date columns to datetime
date_columns = ['report_dt', 'eff_dt']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Filter for only stock holdings (assuming stocks don't have maturity_dt and coupon values)
stocks_df = df[df['maturity_dt'].isna() & df['coupon'].isna()]

# 1. Count the number of unique funds
num_funds = stocks_df['crsp_portno'].nunique()

# 2. Count the number of unique CUSIPs (stocks)
num_cusips = stocks_df['cusip'].nunique()

# 3. Calculate average lifespan of CUSIPs in months
# Group by CUSIP and get the min and max dates
cusip_lifespan = stocks_df.groupby('cusip')['report_dt'].agg(['min', 'max'])
# Calculate difference in months
cusip_lifespan['lifespan_months'] = (cusip_lifespan['max'] - cusip_lifespan['min']).dt.days / 30
avg_cusip_lifespan = cusip_lifespan['lifespan_months'].mean()

# 4. For each month, identify top holdings by market_val
# First, create a year-month column
stocks_df['year_month'] = stocks_df['report_dt'].dt.strftime('%Y-%m')

# Calculate Avg. basket size - overall
avg_basket_size = stocks_df.groupby(['crsp_portno', 'report_dt'])['cusip'].nunique().mean()

# Calculate Avg. #baskets per user
avg_baskets_per_user = stocks_df.groupby('crsp_portno')['report_dt'].nunique().mean()

# Calculate Repeat ratio and Explore ratio
repeat_explore_ratios = []

for fund, fund_data in stocks_df.groupby('crsp_portno'):
    # Sort by report date
    fund_data = fund_data.sort_values('report_dt')
    
    # Get unique report dates
    report_dates = fund_data['report_dt'].unique()
    
    if len(report_dates) < 2:
        continue
    
    # Calculate for each consecutive pair of dates
    for i in range(len(report_dates) - 1):
        current_date = report_dates[i]
        next_date = report_dates[i + 1]
        
        current_holdings = set(fund_data[fund_data['report_dt'] == current_date]['cusip'])
        next_holdings = set(fund_data[fund_data['report_dt'] == next_date]['cusip'])
        
        # Skip if either period has no holdings
        if not current_holdings or not next_holdings:
            continue
        
        # Repeat stocks (intersection)
        repeat_stocks = current_holdings.intersection(next_holdings)
        
        # New stocks (only in next holdings)
        new_stocks = next_holdings - current_holdings
        
        repeat_ratio = len(repeat_stocks) / len(next_holdings) if next_holdings else 0
        explore_ratio = len(new_stocks) / len(next_holdings) if next_holdings else 0
        
        repeat_explore_ratios.append({
            'fund': fund,
            'from_date': current_date,
            'to_date': next_date,
            'repeat_ratio': repeat_ratio,
            'explore_ratio': explore_ratio
        })

repeat_explore_df = pd.DataFrame(repeat_explore_ratios)
avg_repeat_ratio = repeat_explore_df['repeat_ratio'].mean()
avg_explore_ratio = repeat_explore_df['explore_ratio'].mean()

# Calculate monthly statistics
monthly_stats = []
for year_month, month_data in stocks_df.groupby('year_month'):
    # Top 10 holdings by market value
    top10 = month_data.nlargest(10, 'market_val')['market_val'].sum()
    
    # 50th ranking (last) by market value if available
    if len(month_data) >= 50:
        rank50_val = month_data.nlargest(50, 'market_val')['market_val'].iloc[-1]
    else:
        rank50_val = np.nan
    
    # Average market value
    avg_market_val = month_data['market_val'].mean()
    
    # Average number of holdings per fund for this month
    month_basket_size = month_data.groupby('crsp_portno')['cusip'].nunique().mean()
    
    # Limit to top 50 stocks by market value for this month
    top50_stocks = month_data.nlargest(50, 'market_val')
    
    # Calculate average holdings per fund among the top 50 stocks
    top50_cusips = set(top50_stocks['cusip'])
    fund_holdings_in_top50 = []
    
    for fund, fund_data in month_data.groupby('crsp_portno'):
        fund_cusips = set(fund_data['cusip'])
        overlap = fund_cusips.intersection(top50_cusips)
        fund_holdings_in_top50.append(len(overlap))
    
    avg_holdings_in_top50 = np.mean(fund_holdings_in_top50) if fund_holdings_in_top50 else np.nan
    
    # Calculate month-specific repeat and explore ratios
    month_repeat_explore = repeat_explore_df[
        pd.to_datetime(repeat_explore_df['to_date']).dt.strftime('%Y-%m') == year_month
    ]
    month_repeat_ratio = month_repeat_explore['repeat_ratio'].mean() if not month_repeat_explore.empty else np.nan
    month_explore_ratio = month_repeat_explore['explore_ratio'].mean() if not month_repeat_explore.empty else np.nan
    
    monthly_stats.append({
        'year_month': year_month,
        'top10_holdings_value': top10,
        'rank50_value': rank50_val,
        'avg_market_val': avg_market_val,
        'avg_holdings_per_fund': month_basket_size,
        'avg_holdings_in_top50': avg_holdings_in_top50,
        'repeat_ratio': month_repeat_ratio,
        'explore_ratio': month_explore_ratio
    })

monthly_stats_df = pd.DataFrame(monthly_stats)

# 5. Calculate yearly averages
monthly_stats_df['year'] = monthly_stats_df['year_month'].str[:4]
yearly_stats = monthly_stats_df.groupby('year').mean().reset_index()

# 6. Calculate 5-year averages (or whatever time period is available)
five_year_stats = monthly_stats_df.mean()

# Generate and save the results
results_path = r"C:\Users\user\Desktop\Funds data\fund_analysis_results.csv"

# Create a dictionary to store all results
analysis_results = {
    'Metric': [
        'Number of unique funds',
        'Number of unique CUSIPs (stocks)',
        'Average CUSIP lifespan (months)',
        'Average market value across all months',
        'Average holdings per fund (Avg. basket size)',
        'Average baskets per user',
        'Average Repeat ratio',
        'Average Explore ratio',
        'Average holdings in top 50 stocks per fund'
    ],
    'Value': [
        num_funds,
        num_cusips,
        avg_cusip_lifespan,
        five_year_stats['avg_market_val'],
        avg_basket_size,
        avg_baskets_per_user,
        avg_repeat_ratio,
        avg_explore_ratio,
        five_year_stats['avg_holdings_in_top50']
    ]
}

# Format the monthly stats for better readability
# Add month names
monthly_stats_df['month'] = pd.to_datetime(monthly_stats_df['year_month'] + "-01").dt.strftime('%b')
monthly_stats_df = monthly_stats_df.sort_values(['year', 'month'])

# Create and save the results DataFrame
results_df = pd.DataFrame(analysis_results)
results_df.to_csv(results_path, index=False)

# Also save monthly and yearly statistics with better formatting
monthly_stats_df.to_csv(r"C:\Users\user\Desktop\Funds data\monthly_stats.csv", index=False)
yearly_stats.to_csv(r"C:\Users\user\Desktop\Funds data\yearly_stats.csv", index=False)

# Save repeat/explore analysis
repeat_explore_df.to_csv(r"C:\Users\user\Desktop\Funds data\repeat_explore_analysis.csv", index=False)

# Create a separate report for key metrics
key_metrics_report = {
    'Period': ['Overall'] + yearly_stats['year'].tolist(),
    'Avg. basket size': [avg_basket_size] + yearly_stats['avg_holdings_per_fund'].tolist(),
    'Avg. #baskets per user': [avg_baskets_per_user] + [np.nan] * len(yearly_stats),  # Can't easily calculate by year
    'Repeat ratio': [avg_repeat_ratio] + yearly_stats['repeat_ratio'].tolist(),
    'Explore ratio': [avg_explore_ratio] + yearly_stats['explore_ratio'].tolist(),
    'Rank 50 Market Value': [five_year_stats['rank50_value']] + yearly_stats['rank50_value'].tolist()
}

key_metrics_df = pd.DataFrame(key_metrics_report)
key_metrics_df.to_csv(r"C:\Users\user\Desktop\Funds data\key_metrics_report.csv", index=False)

print(f"Analysis complete. Results saved to {results_path}")
print("\nSummary of key metrics:")
print(f"Number of unique funds: {num_funds}")
print(f"Number of unique CUSIPs (stocks): {num_cusips}")
print(f"Average CUSIP lifespan (months): {avg_cusip_lifespan:.2f}")
print(f"Average market value: {five_year_stats['avg_market_val']:.2f}")
print(f"Average basket size: {avg_basket_size:.2f}")
print(f"Average baskets per user: {avg_baskets_per_user:.2f}")
print(f"Average Repeat ratio: {avg_repeat_ratio:.2f}")
print(f"Average Explore ratio: {avg_explore_ratio:.2f}")
print(f"Average rank 50 market value: {five_year_stats['rank50_value']:.2f}")