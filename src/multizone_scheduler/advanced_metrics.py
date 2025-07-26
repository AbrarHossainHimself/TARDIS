# advanced_metrics.py
# Comprehensive analysis framework for multi-zone HPC scheduling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from multizone_generator import MultiZoneConfig, MultiZoneJobGenerator
from scheduler_comparison import SchedulerComparison

class AdvancedMetricsAnalyzer:
    """Analyze advanced metrics for scheduler comparison"""
    
    def __init__(self, comparison_results: Dict):
        self.results = comparison_results
        self.colors = {
            'multi_zone': '#2ecc71',     # Green
            'random': '#e74c3c',         # Red
            'single_us_east': '#3498db',  # Blue
            'single_us_west': '#f1c40f',  # Yellow
            'single_us_south': '#9b59b6'  # Purple
        }
    
    def analyze_resource_utilization(self, completed_jobs_dict: Dict) -> None:
        """Analyze and plot resource utilization patterns"""
        plt.figure(figsize=(15, 8))
        
        for scheduler_name, completed_jobs in completed_jobs_dict.items():
            df = completed_jobs.copy()
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['hour'] = df['start_time'].dt.hour
            
            # Calculate hourly node utilization
            hourly_util = df.groupby('hour')['num_nodes_alloc'].mean() / 512 * 100
            
            plt.plot(hourly_util.index, hourly_util.values,
                    label=scheduler_name,
                    color=self.colors.get(scheduler_name, 'gray'),
                    marker='o')
        
        plt.title('Average Hourly Resource Utilization')
        plt.xlabel('Hour of Day')
        plt.ylabel('Node Utilization (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig('resource_utilization.png')
        plt.close()
    
    def analyze_efficiency_metrics(self, completed_jobs_dict: Dict) -> Dict:
        """Calculate efficiency metrics for each scheduler"""
        metrics = {}
        
        for scheduler_name, completed_jobs in completed_jobs_dict.items():
            df = completed_jobs.copy()
            
            # Calculate cost efficiency
            cost_per_node_hour = (df['power_cost'].sum() / 
                                (df['num_nodes_alloc'] * df['run_time']/3600).sum())
            
            # Calculate resource efficiency
            avg_utilization = df['num_nodes_alloc'].mean() / 512 * 100
            
            # Calculate power efficiency
            power_efficiency = df['power_cost'].sum() / df['power_kw'].sum()
            
            metrics[scheduler_name] = {
                'cost_per_node_hour': cost_per_node_hour,
                'avg_utilization': avg_utilization,
                'power_efficiency': power_efficiency,
                'total_node_hours': (df['num_nodes_alloc'] * df['run_time']/3600).sum()
            }
        
        return metrics
    
    def plot_job_size_distribution(self, completed_jobs_dict: Dict) -> None:
        """Analyze how different schedulers handle various job sizes"""
        plt.figure(figsize=(12, 6))
        
        for scheduler_name, completed_jobs in completed_jobs_dict.items():
            df = completed_jobs.copy()
            
            # Create job size categories
            df['size_category'] = pd.qcut(df['num_nodes_alloc'], 
                                        q=4, 
                                        labels=['Small', 'Medium', 'Large', 'Very Large'])
            
            # Calculate average cost per job size
            size_costs = df.groupby('size_category')['power_cost'].mean()
            
            plt.plot(size_costs.index, size_costs.values,
                    label=scheduler_name,
                    marker='o')
        
        plt.title('Average Cost by Job Size')
        plt.xlabel('Job Size Category')
        plt.ylabel('Average Cost per Job ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig('job_size_costs.png')
        plt.close()
    
    def analyze_peak_optimization(self, completed_jobs_dict: Dict) -> None:
        """Analyze how well each scheduler optimizes for peak hours"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        for scheduler_name, completed_jobs in completed_jobs_dict.items():
            df = completed_jobs.copy()
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['hour'] = df['start_time'].dt.hour
            
            # Peak vs off-peak job distribution
            peak_hours = df[df['is_peak']].groupby('hour')['power_cost'].mean()
            offpeak_hours = df[~df['is_peak']].groupby('hour')['power_cost'].mean()
            
            # Plot peak hour distribution
            ax1.plot(peak_hours.index, peak_hours.values,
                    label=f"{scheduler_name} (Peak)",
                    linestyle='-',
                    marker='o')
            
            # Plot off-peak distribution
            ax2.plot(offpeak_hours.index, offpeak_hours.values,
                    label=f"{scheduler_name} (Off-Peak)",
                    linestyle='--',
                    marker='o')
        
        ax1.set_title('Average Cost During Peak Hours')
        ax1.set_ylabel('Cost ($)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Average Cost During Off-Peak Hours')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Cost ($)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('peak_optimization.png')
        plt.close()
    
    def print_summary_statistics(self, completed_jobs_dict: Dict, efficiency_metrics: Dict) -> None:
        """Print comprehensive summary statistics"""
        print("\nComprehensive Performance Analysis")
        print("=" * 50)
        
        for scheduler_name in completed_jobs_dict.keys():
            metrics = efficiency_metrics[scheduler_name]
            
            print(f"\n{scheduler_name}:")
            print(f"Cost per Node Hour: ${metrics['cost_per_node_hour']:.4f}")
            print(f"Average Resource Utilization: {metrics['avg_utilization']:.1f}%")
            print(f"Power Efficiency ($/kW): ${metrics['power_efficiency']:.4f}")
            print(f"Total Node Hours: {metrics['total_node_hours']:.1f}")

def analyze_scheduler_benefits(results: Dict, completed_jobs_dict: Dict) -> None:
    """Run comprehensive analysis of scheduler benefits"""
    analyzer = AdvancedMetricsAnalyzer(results)
    
    print("Analyzing advanced metrics...")
    
    # Analyze resource utilization
    analyzer.analyze_resource_utilization(completed_jobs_dict)
    
    # Calculate efficiency metrics
    efficiency_metrics = analyzer.analyze_efficiency_metrics(completed_jobs_dict)
    
    # Analyze job size distribution
    analyzer.plot_job_size_distribution(completed_jobs_dict)
    
    # Analyze peak hour optimization
    analyzer.analyze_peak_optimization(completed_jobs_dict)
    
    # Print summary statistics
    analyzer.print_summary_statistics(completed_jobs_dict, efficiency_metrics)
    
    print("\nAnalysis complete! Generated visualizations:")
    print("- resource_utilization.png")
    print("- job_size_costs.png")
    print("- peak_optimization.png")


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
    end_date = datetime(2024, 1, 7, 23, 59, 59)  # Include full last day
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
    
    # Generate all analyses
    print("\nGenerating comprehensive analysis...")
    analyzer = AdvancedMetricsAnalyzer(results)
    
    # 1. Resource Utilization Analysis
    print("\nAnalyzing resource utilization...")
    analyzer.analyze_resource_utilization(completed_jobs)
    
    # 2. Calculate Efficiency Metrics
    print("\nCalculating efficiency metrics...")
    efficiency_metrics = analyzer.analyze_efficiency_metrics(completed_jobs)
    
    # 3. Job Size Distribution Analysis
    print("\nAnalyzing job size distribution...")
    analyzer.plot_job_size_distribution(completed_jobs)
    
    # 4. Peak Hour Optimization Analysis
    print("\nAnalyzing peak hour optimization...")
    analyzer.analyze_peak_optimization(completed_jobs)
    
    # 5. Print Comprehensive Summary
    print("\nGenerating summary statistics...")
    analyzer.print_summary_statistics(completed_jobs, efficiency_metrics)
    
    print("\nAnalysis complete! Generated visualizations:")
    print("- resource_utilization.png: Shows node utilization patterns across schedulers")
    print("- job_size_costs.png: Compares costs across different job sizes")
    print("- peak_optimization.png: Shows effectiveness of peak hour avoidance")
    
    # Export numerical results to CSV for further analysis
    results_df = pd.DataFrame(efficiency_metrics).T
    results_df.to_csv('scheduler_metrics.csv')
    print("\nNumerical results exported to 'scheduler_metrics.csv'")
    
    # Additional useful metrics
    print("\nKey Performance Indicators:")
    best_cost = min(efficiency_metrics.items(), key=lambda x: x[1]['cost_per_node_hour'])
    best_util = max(efficiency_metrics.items(), key=lambda x: x[1]['avg_utilization'])
    print(f"Most Cost-Efficient Scheduler: {best_cost[0]} (${best_cost[1]['cost_per_node_hour']:.4f}/node-hour)")
    print(f"Best Resource Utilization: {best_util[0]} ({best_util[1]['avg_utilization']:.1f}%)")
    
    # Calculate relative improvements
    baseline_cost = efficiency_metrics['single_us_east']['cost_per_node_hour']
    multizone_cost = efficiency_metrics['multi_zone']['cost_per_node_hour']
    cost_improvement = ((baseline_cost - multizone_cost) / baseline_cost) * 100
    
    print(f"\nMulti-zone scheduler improvements:")
    print(f"Cost Reduction: {cost_improvement:.1f}% compared to single-zone baseline")