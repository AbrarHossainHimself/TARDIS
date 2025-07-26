# last_day_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict
from multizone_generator import MultiZoneConfig, MultiZoneJobGenerator
from scheduler_comparison import SchedulerComparison

def analyze_last_day(completed_jobs_dict: Dict) -> None:
    """Analyze job distribution on the last day of simulation"""
    print("\nLast Day Analysis (January 7th)")
    print("=" * 50)
    
    for scheduler_name, completed_jobs in completed_jobs_dict.items():
        df = completed_jobs.copy()
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['start_date'] = df['start_time'].dt.date
        df['start_hour'] = df['start_time'].dt.hour
        
        # Filter for last day
        last_day_jobs = df[df['start_date'] == datetime(2024, 1, 7).date()]
        
        print(f"\n{scheduler_name}:")
        print(f"Total jobs on last day: {len(last_day_jobs)}")
        if not last_day_jobs.empty:
            print(f"Hour range: {last_day_jobs['start_hour'].min()} to {last_day_jobs['start_hour'].max()}")
            print(f"Total cost: ${last_day_jobs['power_cost'].sum():.2f}")
            
            # Show hourly breakdown
            hourly_costs = last_day_jobs.groupby('start_hour')['power_cost'].sum()
            print("\nHourly breakdown:")
            for hour, cost in hourly_costs.items():
                print(f"Hour {hour:02d}:00 - ${cost:.2f}")

def plot_job_distribution(completed_jobs_dict: Dict) -> None:
    """Plot job distribution across all days"""
    plt.figure(figsize=(15, 6))
    
    for scheduler_name, completed_jobs in completed_jobs_dict.items():
        df = completed_jobs.copy()
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['start_date'] = df['start_time'].dt.date
        
        # Count jobs per day
        daily_jobs = df.groupby('start_date').size()
        
        plt.plot(daily_jobs.index, daily_jobs.values, 
                label=scheduler_name, marker='o')
    
    plt.title('Number of Jobs per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Jobs')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('job_distribution.png')
    plt.close()

if __name__ == "__main__":
    # Set up configuration
    print("Initializing configuration...")
    config = MultiZoneConfig()
    
    # Create comparison framework
    print("Setting up scheduler comparison...")
    comparison = SchedulerComparison(config.zones)
    
    # Generate sample data
    print("Generating job data...")
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 7, 23, 59, 59) 
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
    
    # Analyze last day and plot distribution
    print("\nAnalyzing last day distribution...")
    analyze_last_day(completed_jobs)
    plot_job_distribution(completed_jobs)