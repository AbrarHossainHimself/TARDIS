# cost_analysis.py
# Cost analysis and visualization for HPC scheduler comparison

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List
import seaborn as sns
from multizone_generator import MultiZoneConfig, MultiZoneJobGenerator
from scheduler_comparison import SchedulerComparison


class CostAnalyzer:
    """Analyze and visualize temporal cost patterns across different schedulers"""
    
    def __init__(self, comparison_results: Dict):
        self.results = comparison_results['cost_analysis']  # Access cost analysis data
        self.schedulers_data = {}  # Store scheduler data for plotting
        
        self.colors = {
            'multi_zone': '#2ecc71',     # Green
            'random': '#e74c3c',         # Red
            'single_us_east': '#3498db',  # Blue
            'single_us_west': '#f1c40f',  # Yellow
            'single_europe': '#9b59b6'    # Purple
        }
    
    def get_temporal_costs(self, df: pd.DataFrame) -> tuple:
        """Generate time series of costs with different time resolutions"""
        # Ensure datetime
        df = df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # Sort by time
        df = df.sort_values('start_time')
        
        # Create different time resolutions
        df['date'] = df['start_time'].dt.date
        df['hour'] = df['start_time'].dt.hour
        df['datetime_hour'] = df['start_time'].dt.floor('H')
        
        # Calculate cumulative costs
        df['cumulative_cost'] = df['power_cost'].cumsum()
        
        # Calculate costs per time period
        hourly_costs = df.groupby('datetime_hour').agg({
            'power_cost': 'sum',
            'cumulative_cost': 'last'
        }).reset_index()
        
        daily_costs = df.groupby('date').agg({
            'power_cost': 'sum',
            'cumulative_cost': 'last'
        }).reset_index()
        
        return hourly_costs, daily_costs
    
    def plot_temporal_cost_comparison(self, completed_jobs_dict: Dict) -> None:
        """Plot cost accumulation over time for all schedulers"""
        plt.figure(figsize=(15, 8))
        
        # Plot each scheduler
        for scheduler_name, completed_jobs in completed_jobs_dict.items():
            # Get temporal cost data
            hourly_costs, _ = self.get_temporal_costs(completed_jobs)
            
            # Plot hourly cumulative costs
            plt.plot(
                hourly_costs['datetime_hour'],
                hourly_costs['cumulative_cost'],
                label=scheduler_name,
                color=self.colors.get(scheduler_name, 'gray'),
                linewidth=2
            )
        
        plt.title('Cumulative Cost Over Time by Scheduler')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Cost ($)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis to show days
        plt.gcf().autofmt_xdate()  # Angle and align the tick labels
        
        plt.tight_layout()
        plt.savefig('temporal_costs.png')
        plt.close()
    
    def plot_daily_cost_comparison(self, completed_jobs_dict: Dict) -> None:
        """Plot daily costs for all schedulers"""
        plt.figure(figsize=(15, 8))
        
        # Plot each scheduler
        for scheduler_name, completed_jobs in completed_jobs_dict.items():
            # Get temporal cost data
            _, daily_costs = self.get_temporal_costs(completed_jobs)
            
            # Plot daily costs (not cumulative)
            plt.plot(
                daily_costs['date'],
                daily_costs['power_cost'],
                label=scheduler_name,
                color=self.colors.get(scheduler_name, 'gray'),
                linewidth=2,
                marker='o'
            )
        
        plt.title('Daily Power Costs by Scheduler')
        plt.xlabel('Date')
        plt.ylabel('Daily Cost ($)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis to show days
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig('daily_costs.png')
        plt.close()
    
    def print_temporal_summary(self, completed_jobs_dict: Dict) -> None:
        """Print summary of temporal cost patterns"""
        print("\nTemporal Cost Analysis")
        print("=" * 50)
        
        for scheduler_name, completed_jobs in completed_jobs_dict.items():
            hourly_costs, daily_costs = self.get_temporal_costs(completed_jobs)
            
            print(f"\n{scheduler_name}:")
            print(f"Total Cost: ${hourly_costs['cumulative_cost'].iloc[-1]:,.2f}")
            print("\nDaily Costs:")
            for _, row in daily_costs.iterrows():
                print(f"{row['date']}: ${row['power_cost']:,.2f}")
            
            # Calculate day with highest and lowest costs
            max_day = daily_costs.loc[daily_costs['power_cost'].idxmax()]
            min_day = daily_costs.loc[daily_costs['power_cost'].idxmin()]
            
            print(f"\nHighest Cost Day: {max_day['date']} (${max_day['power_cost']:,.2f})")
            print(f"Lowest Cost Day: {min_day['date']} (${min_day['power_cost']:,.2f})")
    
    # Add this new method to the CostAnalyzer class
    def export_to_csv(self, completed_jobs_dict: Dict) -> None:
        """Export temporal cost analysis results to CSV files"""
        # Dictionary to store all results
        all_hourly_costs = {}
        all_daily_costs = {}
        
        # Get costs for each scheduler
        for scheduler_name, completed_jobs in completed_jobs_dict.items():
            hourly_costs, daily_costs = self.get_temporal_costs(completed_jobs)
            
            # Add scheduler name as prefix to columns
            hourly_costs = hourly_costs.rename(columns={
                'power_cost': f'{scheduler_name}_power_cost',
                'cumulative_cost': f'{scheduler_name}_cumulative_cost'
            })
            daily_costs = daily_costs.rename(columns={
                'power_cost': f'{scheduler_name}_power_cost',
                'cumulative_cost': f'{scheduler_name}_cumulative_cost'
            })
            
            # Store in dictionaries
            all_hourly_costs[scheduler_name] = hourly_costs
            all_daily_costs[scheduler_name] = daily_costs
        
        # Combine results for all schedulers
        hourly_df = pd.DataFrame()
        daily_df = pd.DataFrame()
        
        # Merge hourly costs
        for scheduler_name, costs in all_hourly_costs.items():
            if hourly_df.empty:
                hourly_df = costs
            else:
                hourly_df = pd.merge(hourly_df, costs, on='datetime_hour', how='outer')
        
        # Merge daily costs
        for scheduler_name, costs in all_daily_costs.items():
            if daily_df.empty:
                daily_df = costs
            else:
                daily_df = pd.merge(daily_df, costs, on='date', how='outer')
        
        # Sort by time
        hourly_df = hourly_df.sort_values('datetime_hour')
        daily_df = daily_df.sort_values('date')
        
        # Export to CSV
        hourly_df.to_csv('hourly_costs.csv', index=False)
        daily_df.to_csv('daily_costs.csv', index=False)

def analyze_temporal_costs(comparison_results: Dict, completed_jobs_dict: Dict) -> None:
    """Run temporal cost analysis"""
    analyzer = CostAnalyzer(comparison_results)
    
    print("Generating temporal cost analysis plots...")
    analyzer.plot_temporal_cost_comparison(completed_jobs_dict)
    analyzer.plot_daily_cost_comparison(completed_jobs_dict)
    analyzer.print_temporal_summary(completed_jobs_dict)
    
    print("\nExporting results to CSV...")
    analyzer.export_to_csv(completed_jobs_dict)
    
    print("\nAnalysis complete. Files saved:")
    print("- temporal_costs.png (cumulative costs over time)")
    print("- daily_costs.png (daily cost comparison)")
    print("- hourly_costs.csv (hourly cost data)")
    print("- daily_costs.csv (daily cost data)")

if __name__ == "__main__":
    # Set up configuration
    print("Initializing configuration...")
    config = MultiZoneConfig()
    
    # Create comparison framework
    print("Setting up scheduler comparison...")
    comparison = SchedulerComparison(config.zones)
    
    # Generate sample data
    print("Generating job data...")
    start_date = datetime(2020, 5, 1)
    end_date = datetime(2020, 5, 30, 23, 59, 59)  # Last second of Jan 7 # One week of data
    generator = MultiZoneJobGenerator(start_date, end_date, config)
    jobs_df = generator.generate_dataset()
    
    # Run comparison
    print("Running scheduler comparison...")
    results = comparison.run_comparison(jobs_df)
    
    # Extract completed jobs from results for each scheduler
    completed_jobs = {
        name: data['completed_jobs'] 
        for name, data in comparison.results.items()
    }
    
    # Generate temporal cost analysis
    print("Analyzing temporal costs...")
    analyze_temporal_costs(results, completed_jobs)

