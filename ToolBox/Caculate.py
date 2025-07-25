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
    
    # Average number of holdings per fund
    # For each fund in this month, count how many stocks they hold
    holdings_per_fund = month_data.groupby('crsp_portno')['cusip'].nunique().mean()
    
    # Limit to top 50 stocks by market value for this month
    top50_stocks = month_data.nlargest(50, 'market_val')
    
    # Calculate average holdings per fund among the top 50 stocks
    # For each fund, how many of the top 50 stocks do they hold?
    top50_cusips = set(top50_stocks['cusip'])
    fund_holdings_in_top50 = []
    
    for fund, fund_data in month_data.groupby('crsp_portno'):
        fund_cusips = set(fund_data['cusip'])
        overlap = fund_cusips.intersection(top50_cusips)
        fund_holdings_in_top50.append(len(overlap))
    
    avg_holdings_in_top50 = np.mean(fund_holdings_in_top50) if fund_holdings_in_top50 else np.nan
    
    monthly_stats.append({
        'year_month': year_month,
        'top10_holdings_value': top10,
        'rank50_value': rank50_val,
        'avg_market_val': avg_market_val,
        'avg_holdings_per_fund': holdings_per_fund,
        'avg_holdings_in_top50': avg_holdings_in_top50
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
        'Average holdings per fund across all months',
        'Average holdings in top 50 stocks per fund'
    ],
    'Value': [
        num_funds,
        num_cusips,
        avg_cusip_lifespan,
        five_year_stats['avg_market_val'],
        five_year_stats['avg_holdings_per_fund'],
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

# Create a separate report for the top 50 holdings analysis
top50_report = {
    'Period': ['Monthly Average', 'Five-Year Average'] + yearly_stats['year'].tolist(),
    'Avg Holdings in Top 50': [
        monthly_stats_df['avg_holdings_in_top50'].mean(),
        five_year_stats['avg_holdings_in_top50']
    ] + yearly_stats['avg_holdings_in_top50'].tolist(),
    'Rank 50 Market Value': [
        monthly_stats_df['rank50_value'].mean(),
        five_year_stats['rank50_value']
    ] + yearly_stats['rank50_value'].tolist()
}

top50_report_df = pd.DataFrame(top50_report)
top50_report_df.to_csv(r"C:\Users\user\Desktop\Funds data\top50_holdings_report.csv", index=False)

print(f"Analysis complete. Results saved to {results_path}")
print("\nSummary of key metrics:")
print(f"Number of unique funds: {num_funds}")
print(f"Number of unique CUSIPs (stocks): {num_cusips}")
print(f"Average CUSIP lifespan (months): {avg_cusip_lifespan:.2f}")
print(f"Average market value: {five_year_stats['avg_market_val']:.2f}")
print(f"Average holdings per fund: {five_year_stats['avg_holdings_per_fund']:.2f}")
print(f"Average holdings in top 50 stocks per fund: {five_year_stats['avg_holdings_in_top50']:.2f}")
print(f"Average rank 50 market value: {five_year_stats['rank50_value']:.2f}")