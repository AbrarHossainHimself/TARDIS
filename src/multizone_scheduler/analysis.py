import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Tuple
import numpy as np
import seaborn as sns
sns.set_theme()


def analyze_scheduling_results(completed_jobs: pd.DataFrame, metrics: Dict) -> None:
    """Analyze and visualize multi-zone scheduling results"""
    
    # Set style for all plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Cost Analysis
    plt.figure(figsize=(10, 6))
    zones = list(metrics['cost_by_zone'].keys())
    costs = list(metrics['cost_by_zone'].values())
    
    plt.bar(zones, costs)
    plt.title('Total Cost by Zone')
    plt.ylabel('Cost ($)')
    plt.xlabel('Zone')
    
    # Add value labels on top of each bar
    for i, cost in enumerate(costs):
        plt.text(i, cost, f'${cost:,.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('zone_costs.png')
    plt.close()
    
    # 2. Job Distribution Analysis
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    zone_data = []
    for zone in zones:
        metrics_zone = metrics['zone_metrics'][zone]
        zone_data.append({
            'zone': zone,
            'total_jobs': metrics_zone['total_jobs'],
            'peak_jobs': metrics_zone['peak_jobs'],
            'non_peak_jobs': metrics_zone['total_jobs'] - metrics_zone['peak_jobs']
        })
    
    job_dist_df = pd.DataFrame(zone_data)
    
    # Create stacked bar chart
    bottom = np.zeros(len(zones))
    
    # Plot non-peak jobs
    plt.bar(job_dist_df['zone'], job_dist_df['non_peak_jobs'], 
            label='Non-Peak Jobs', bottom=bottom)
    
    # Plot peak jobs
    plt.bar(job_dist_df['zone'], job_dist_df['peak_jobs'], 
            label='Peak Jobs', bottom=job_dist_df['non_peak_jobs'])
    
    plt.title('Job Distribution by Zone')
    plt.ylabel('Number of Jobs')
    plt.xlabel('Zone')
    plt.legend()
    
    # Add total job count on top of each bar
    for i, row in job_dist_df.iterrows():
        plt.text(i, row['total_jobs'], f"{row['total_jobs']}", 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('job_distribution.png')
    plt.close()
    
    # 3. Zone Utilization Analysis
    plt.figure(figsize=(10, 6))
    
    utilization_data = [(zone, metrics['zone_metrics'][zone]['avg_utilization'] * 100)
                        for zone in zones]
    util_df = pd.DataFrame(utilization_data, columns=['Zone', 'Utilization'])
    
    plt.bar(util_df['Zone'], util_df['Utilization'])
    plt.title('Average Zone Utilization')
    plt.ylabel('Utilization (%)')
    plt.xlabel('Zone')
    
    # Add percentage labels on top of each bar
    for i, util in enumerate(util_df['Utilization']):
        plt.text(i, util, f'{util:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('zone_utilization.png')
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"Total Jobs Completed: {metrics['total_jobs']}")
    print(f"Total Cost: ${metrics['total_cost']:,.2f}")
    print("\nPer-Zone Statistics:")
    print("-" * 50)
    
    for zone in zones:
        zone_metrics = metrics['zone_metrics'][zone]
        print(f"\n{zone.upper()}:")
        print(f"Total Jobs: {zone_metrics['total_jobs']}")
        print(f"Peak Jobs: {zone_metrics['peak_jobs']} " 
              f"({zone_metrics['peak_jobs']/zone_metrics['total_jobs']*100:.1f}%)")
        print(f"Average Utilization: {zone_metrics['avg_utilization']*100:.1f}%")
        print(f"Average Power: {zone_metrics['avg_power']:.2f} kW")
        print(f"Maximum Power: {zone_metrics['max_power']:.2f} kW")

if __name__ == "__main__":
    from multizone_generator import MultiZoneConfig, MultiZoneJobGenerator
    from multizone_scheduler import MultiZoneScheduler
    
    # Set up configuration
    config = MultiZoneConfig()
    
    # Generate sample data for one week
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 7)
    
    # Create job generator and generate jobs
    generator = MultiZoneJobGenerator(start_date, end_date, config)
    jobs_df = generator.generate_dataset()
    
    # Create scheduler
    scheduler = MultiZoneScheduler(config.zones)
    
    # Run simulation
    print("Starting simulation...")
    completed_jobs, metrics = scheduler.simulate(jobs_df)
    print("Simulation completed!")
    
    # Analyze results
    analyze_scheduling_results(completed_jobs, metrics)