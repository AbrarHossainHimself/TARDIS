# cost_benefit_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import seaborn as sns
from datetime import datetime, timedelta

class CostBenefitAnalyzer:
    """Analyze cost benefits of multi-zone scheduling with focus on large jobs and peak hours"""
    
    def __init__(self, comparison_results: Dict):
        self.results = comparison_results
        self.completed_jobs_dict = {
            name: data['completed_jobs'] 
            for name, data in comparison_results.items()
        }
        
        # plt.style.use('seaborn')
        self.colors = {
            'multi_zone': '#2ecc71',
            'random': '#e74c3c',
            'single_us_east': '#3498db',
            'single_us_west': '#f1c40f',
            'single_us_south': '#9b59b6'
        }
    
    def analyze_large_jobs(self) -> Dict:
        """Detailed analysis of large job handling"""
        large_job_metrics = {}
        
        for scheduler_name, completed_jobs in self.completed_jobs_dict.items():
            df = completed_jobs.copy()
            
            # Define large jobs (top 25% by node count)
            large_threshold = df['num_nodes_alloc'].quantile(0.75)
            large_jobs = df[df['num_nodes_alloc'] >= large_threshold].copy()
            
            # Temporal patterns for large jobs
            large_jobs['start_time'] = pd.to_datetime(large_jobs['start_time'])
            large_jobs['hour'] = large_jobs['start_time'].dt.hour
            
            # Calculate metrics
            metrics = {
                'total_large_jobs': len(large_jobs),
                'avg_cost_per_large_job': large_jobs['power_cost'].mean(),
                'total_large_job_cost': large_jobs['power_cost'].sum(),
                'peak_hour_large_jobs': len(large_jobs[large_jobs['is_peak']]),
                'avg_nodes_per_large_job': large_jobs['num_nodes_alloc'].mean(),
                'cost_per_node_large_jobs': (large_jobs['power_cost'].sum() / 
                                           (large_jobs['num_nodes_alloc'] * large_jobs['run_time']/3600).sum())
            }
            
            # Zone distribution for multi-zone
            if 'assigned_zone' in large_jobs.columns:
                zone_dist = large_jobs.groupby('assigned_zone').agg({
                    'power_cost': ['count', 'mean', 'sum'],
                    'is_peak': 'mean'
                })
                metrics['zone_distribution'] = zone_dist.to_dict()
            
            # Hourly patterns
            hourly_stats = large_jobs.groupby('hour').agg({
                'power_cost': ['count', 'mean', 'sum'],
                'is_peak': 'mean'
            })
            metrics['hourly_patterns'] = hourly_stats.to_dict()
            
            large_job_metrics[scheduler_name] = metrics
        
        return large_job_metrics

    def analyze_peak_efficiency(self) -> Dict:
        """Analyze efficiency of peak hour handling"""
        peak_metrics = {}
        
        for scheduler_name, completed_jobs in self.completed_jobs_dict.items():
            df = completed_jobs.copy()
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['hour'] = df['start_time'].dt.hour
            
            # Separate peak and off-peak jobs
            peak_jobs = df[df['is_peak']]
            off_peak_jobs = df[~df['is_peak']]
            
            # Calculate comparative metrics
            metrics = {
                'peak_stats': {
                    'job_count': len(peak_jobs),
                    'avg_cost': peak_jobs['power_cost'].mean(),
                    'total_cost': peak_jobs['power_cost'].sum(),
                    'avg_nodes': peak_jobs['num_nodes_alloc'].mean(),
                    'cost_per_node_hour': (peak_jobs['power_cost'].sum() / 
                                         (peak_jobs['num_nodes_alloc'] * peak_jobs['run_time']/3600).sum())
                },
                'off_peak_stats': {
                    'job_count': len(off_peak_jobs),
                    'avg_cost': off_peak_jobs['power_cost'].mean(),
                    'total_cost': off_peak_jobs['power_cost'].sum(),
                    'avg_nodes': off_peak_jobs['num_nodes_alloc'].mean(),
                    'cost_per_node_hour': (off_peak_jobs['power_cost'].sum() / 
                                         (off_peak_jobs['num_nodes_alloc'] * off_peak_jobs['run_time']/3600).sum())
                }
            }
            
            # Calculate peak vs off-peak efficiency
            metrics['peak_cost_ratio'] = (metrics['peak_stats']['cost_per_node_hour'] / 
                                        metrics['off_peak_stats']['cost_per_node_hour'])
            
            peak_metrics[scheduler_name] = metrics
        
        return peak_metrics

    def plot_large_job_costs(self, save_path: str = 'large_job_costs.png'):
        """Visualize cost distribution of large jobs"""
        large_job_data = self.analyze_large_jobs()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot average cost per large job
        schedulers = list(large_job_data.keys())
        avg_costs = [data['avg_cost_per_large_job'] for data in large_job_data.values()]
        
        bars1 = ax1.bar(schedulers, avg_costs)
        for bar, name in zip(bars1, schedulers):
            bar.set_color(self.colors.get(name, 'gray'))
        
        ax1.set_title('Average Cost per Large Job')
        ax1.set_ylabel('Cost ($)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot cost per node-hour for large jobs
        costs_per_node = [data['cost_per_node_large_jobs'] for data in large_job_data.values()]
        
        bars2 = ax2.bar(schedulers, costs_per_node)
        for bar, name in zip(bars2, schedulers):
            bar.set_color(self.colors.get(name, 'gray'))
        
        ax2.set_title('Cost per Node-Hour (Large Jobs)')
        ax2.set_ylabel('Cost per Node-Hour ($)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_peak_efficiency(self, save_path: str = 'peak_efficiency.png'):
        """Visualize peak vs off-peak efficiency"""
        peak_data = self.analyze_peak_efficiency()
        
        plt.figure(figsize=(12, 6))
        
        schedulers = list(peak_data.keys())
        peak_ratios = [data['peak_cost_ratio'] for data in peak_data.values()]
        
        bars = plt.bar(schedulers, peak_ratios)
        for bar, name in zip(bars, schedulers):
            bar.set_color(self.colors.get(name, 'gray'))
        
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        plt.title('Peak vs Off-Peak Cost Efficiency (Ratio)')
        plt.ylabel('Peak/Off-Peak Cost Ratio')
        plt.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, output_file: str = 'cost_analysis_results.txt'):
        """Generate comprehensive analysis report"""
        large_job_metrics = self.analyze_large_jobs()
        peak_metrics = self.analyze_peak_efficiency()
        
        with open(output_file, 'w') as f:
            f.write("Advanced Cost Benefit Analysis Report\n")
            f.write("====================================\n\n")
            
            # Large Job Analysis
            f.write("Large Job Analysis\n")
            f.write("-----------------\n\n")
            
            for scheduler_name, metrics in large_job_metrics.items():
                f.write(f"{scheduler_name}:\n")
                f.write(f"  Total Large Jobs: {metrics['total_large_jobs']}\n")
                f.write(f"  Average Cost per Large Job: ${metrics['avg_cost_per_large_job']:.2f}\n")
                f.write(f"  Cost per Node-Hour (Large Jobs): ${metrics['cost_per_node_large_jobs']:.4f}\n")
                f.write(f"  Peak Hour Large Jobs: {metrics['peak_hour_large_jobs']}\n")
                f.write(f"  Average Nodes per Large Job: {metrics['avg_nodes_per_large_job']:.1f}\n\n")
            
            # Peak Hour Efficiency
            f.write("\nPeak Hour Efficiency\n")
            f.write("-------------------\n\n")
            
            for scheduler_name, metrics in peak_metrics.items():
                f.write(f"{scheduler_name}:\n")
                f.write("  Peak Hours:\n")
                f.write(f"    Jobs: {metrics['peak_stats']['job_count']}\n")
                f.write(f"    Average Cost: ${metrics['peak_stats']['avg_cost']:.2f}\n")
                f.write(f"    Cost per Node-Hour: ${metrics['peak_stats']['cost_per_node_hour']:.4f}\n")
                f.write("  Off-Peak Hours:\n")
                f.write(f"    Jobs: {metrics['off_peak_stats']['job_count']}\n")
                f.write(f"    Average Cost: ${metrics['off_peak_stats']['avg_cost']:.2f}\n")
                f.write(f"    Cost per Node-Hour: ${metrics['off_peak_stats']['cost_per_node_hour']:.4f}\n")
                f.write(f"  Peak/Off-Peak Cost Ratio: {metrics['peak_cost_ratio']:.2f}\n\n")


    def analyze_temporal_patterns(self) -> Dict:
        """Analyze how costs and decisions evolve over time"""
        temporal_metrics = {}
        
        for scheduler_name, completed_jobs in self.completed_jobs_dict.items():
            df = completed_jobs.copy()
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['hour'] = df['start_time'].dt.hour
            df['day'] = df['start_time'].dt.date
            
            # Hourly metrics
            hourly_stats = df.groupby('hour').agg({
                'power_cost': 'sum',
                'num_nodes_alloc': 'mean',
                'job_id': 'count'
            }).rename(columns={'job_id': 'job_count'})
            
            # Calculate cumulative metrics
            df = df.sort_values('start_time')
            df['cumulative_cost'] = df['power_cost'].cumsum()
            df['cumulative_jobs'] = range(1, len(df) + 1)
            
            # Calculate rate of cost increase
            hourly_cost_rate = df.groupby('hour')['power_cost'].sum()
            
            # Job placement patterns
            if 'assigned_zone' in df.columns:
                zone_preference = df.groupby(['hour', 'assigned_zone']).size().unstack(fill_value=0)
                zone_preference_pct = zone_preference.div(zone_preference.sum(axis=1), axis=0)
            else:
                zone_preference = None
                zone_preference_pct = None
            
            temporal_metrics[scheduler_name] = {
                'hourly_stats': hourly_stats,
                'cumulative_data': df[['start_time', 'cumulative_cost', 'cumulative_jobs']],
                'cost_rate': hourly_cost_rate,
                'zone_preference': zone_preference,
                'zone_preference_pct': zone_preference_pct
            }
        
        return temporal_metrics

    def analyze_job_placement(self) -> Dict:
        """Analyze job placement decisions and their impact"""
        placement_metrics = {}
        
        for scheduler_name, completed_jobs in self.completed_jobs_dict.items():
            df = completed_jobs.copy()
            
            # Job size categories
            df['size_category'] = pd.qcut(df['num_nodes_alloc'], 
                                        q=4, 
                                        labels=['Small', 'Medium', 'Large', 'Very Large'])
            
            # Analyze placement patterns by job size
            if 'assigned_zone' in df.columns:
                size_zone_dist = df.groupby(['size_category', 'assigned_zone']).size().unstack(fill_value=0)
                zone_costs = df.groupby(['size_category', 'assigned_zone'])['power_cost'].mean().unstack(fill_value=0)
            else:
                size_zone_dist = None
                zone_costs = None
            
            # Calculate efficiency metrics by job size
            size_metrics = df.groupby('size_category').agg({
                'power_cost': ['mean', 'sum'],
                'num_nodes_alloc': 'mean',
                'job_id': 'count'
            })
            
            placement_metrics[scheduler_name] = {
                'size_distribution': size_zone_dist,
                'size_costs': zone_costs,
                'size_metrics': size_metrics
            }
        
        return placement_metrics

    def plot_temporal_costs(self, save_path: str = 'temporal_costs.png'):
        """Plot cumulative costs over time"""
        temporal_data = self.analyze_temporal_patterns()
        
        plt.figure(figsize=(12, 6))
        
        for scheduler_name, metrics in temporal_data.items():
            data = metrics['cumulative_data']
            plt.plot(data['start_time'], 
                    data['cumulative_cost'],
                    label=scheduler_name,
                    color=self.colors.get(scheduler_name, 'gray'))
        
        plt.title('Cumulative Cost Over Time')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Cost ($)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_total_costs(self) -> Dict:
        """Calculate basic cost comparison metrics"""
        total_costs = {}
        for scheduler_name, data in self.results.items():
            completed_jobs = self.completed_jobs_dict[scheduler_name]
            total_cost = completed_jobs['power_cost'].sum()
            total_jobs = len(completed_jobs)
            total_node_hours = (completed_jobs['num_nodes_alloc'] * 
                              completed_jobs['run_time']/3600).sum()
            
            # Calculate peak vs off-peak costs
            peak_jobs = completed_jobs[completed_jobs['is_peak']]
            non_peak_jobs = completed_jobs[~completed_jobs['is_peak']]
            peak_cost = peak_jobs['power_cost'].sum()
            non_peak_cost = non_peak_jobs['power_cost'].sum()
            
            # Calculate per-zone distribution
            zone_costs = completed_jobs.groupby('assigned_zone')['power_cost'].sum()
            
            total_costs[scheduler_name] = {
                'total_cost': total_cost,
                'cost_per_job': total_cost / total_jobs,
                'cost_per_node_hour': total_cost / total_node_hours,
                'peak_cost': peak_cost,
                'non_peak_cost': non_peak_cost,
                'peak_cost_percentage': (peak_cost / total_cost * 100) if total_cost > 0 else 0,
                'zone_distribution': zone_costs.to_dict()
            }
        
        # Calculate savings vs baseline (single zone)
        baseline_cost = total_costs['single_us_east']['total_cost']
        multizone_cost = total_costs['multi_zone']['total_cost']
        savings_percentage = ((baseline_cost - multizone_cost) / baseline_cost) * 100
        
        results = {
            'scheduler_costs': total_costs,
            'total_savings_percent': savings_percentage,
            'absolute_savings': baseline_cost - multizone_cost
        }
        
        return results
    
    def analyze_load_distribution(self) -> Dict:
        """Analyze how load is distributed across zones and time"""
        load_metrics = {}
        
        for scheduler_name, completed_jobs in self.completed_jobs_dict.items():
            df = completed_jobs.copy()
            df['start_time'] = pd.to_datetime(df['start_time'])
            
            # Calculate jobs per zone
            jobs_per_zone = df.groupby('assigned_zone').size()
            
            # Calculate average nodes per zone
            avg_nodes_per_zone = df.groupby('assigned_zone')['num_nodes_alloc'].mean()
            
            # Calculate utilization over time
            df['hour'] = df['start_time'].dt.hour
            hourly_nodes = df.groupby(['hour', 'assigned_zone'])['num_nodes_alloc'].sum().unstack()
            
            # Calculate peak hour distribution
            peak_jobs = len(df[df['is_peak']])
            total_jobs = len(df)
            peak_percentage = (peak_jobs / total_jobs * 100) if total_jobs > 0 else 0
            
            load_metrics[scheduler_name] = {
                'jobs_per_zone': jobs_per_zone.to_dict(),
                'avg_nodes_per_zone': avg_nodes_per_zone.to_dict(),
                'peak_job_percentage': peak_percentage,
                'hourly_utilization': hourly_nodes.to_dict()
            }
        
        return load_metrics

    def plot_job_size_distribution(self, save_path: str = 'job_size_dist.png'):
        """Plot job size distribution and costs"""
        placement_data = self.analyze_job_placement()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot job distribution
        for scheduler_name, metrics in placement_data.items():
            if metrics['size_distribution'] is not None:
                data = metrics['size_metrics']['job_id']['count']
                ax1.plot(data.index, data.values, 
                        label=scheduler_name,
                        marker='o',
                        color=self.colors.get(scheduler_name, 'gray'))
        
        ax1.set_title('Job Distribution by Size Category')
        ax1.set_xlabel('Job Size Category')
        ax1.set_ylabel('Number of Jobs')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot average cost by size
        for scheduler_name, metrics in placement_data.items():
            if metrics['size_distribution'] is not None:
                data = metrics['size_metrics']['power_cost']['mean']
                ax2.plot(data.index, data.values,
                        label=scheduler_name,
                        marker='o',
                        color=self.colors.get(scheduler_name, 'gray'))
        
        ax2.set_title('Average Cost by Job Size')
        ax2.set_xlabel('Job Size Category')
        ax2.set_ylabel('Average Cost ($)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, output_file: str = 'cost_analysis_results.txt'):
        """Generate comprehensive cost analysis report"""
        cost_metrics = self.analyze_total_costs()
        load_metrics = self.analyze_load_distribution()
        temporal_metrics = self.analyze_temporal_patterns()
        placement_metrics = self.analyze_job_placement()
        
        with open(output_file, 'w') as f:
            f.write("Enhanced Cost Benefit Analysis Report\n")
            f.write("===================================\n\n")
            
            # Overall savings
            f.write(f"Total Cost Savings vs Baseline: {cost_metrics['total_savings_percent']:.2f}%\n")
            f.write(f"Absolute Cost Savings: ${cost_metrics['absolute_savings']:,.2f}\n\n")
            
            # Per-scheduler analysis
            for scheduler_name in self.results.keys():
                f.write(f"\n{scheduler_name} Analysis:\n")
                f.write("-" * 40 + "\n")
                
                # Basic metrics
                metrics = cost_metrics['scheduler_costs'][scheduler_name]
                f.write(f"Basic Metrics:\n")
                f.write(f"  Total Cost: ${metrics['total_cost']:,.2f}\n")
                f.write(f"  Cost per Job: ${metrics['cost_per_job']:.2f}\n")
                f.write(f"  Cost per Node-Hour: ${metrics['cost_per_node_hour']:.4f}\n")
                f.write(f"  Peak Hour Cost Percentage: {metrics['peak_cost_percentage']:.1f}%\n\n")
                
                # Temporal patterns
                temp_metrics = temporal_metrics[scheduler_name]['hourly_stats']
                f.write("Hourly Patterns:\n")
                f.write(f"  Max Hourly Cost: ${temp_metrics['power_cost'].max():.2f}\n")
                f.write(f"  Avg Hourly Cost: ${temp_metrics['power_cost'].mean():.2f}\n")
                f.write(f"  Max Hourly Jobs: {temp_metrics['job_count'].max():.0f}\n")
                f.write(f"  Avg Hourly Jobs: {temp_metrics['job_count'].mean():.1f}\n\n")
                
                # Job placement patterns
                place_metrics = placement_metrics[scheduler_name]
                if place_metrics['size_distribution'] is not None:
                    f.write("Job Size Analysis:\n")
                    size_metrics = place_metrics['size_metrics']
                    for size_cat in size_metrics.index:
                        f.write(f"  {size_cat}:\n")
                        f.write(f"    Count: {size_metrics.loc[size_cat, ('job_id', 'count')]:.0f}\n")
                        f.write(f"    Avg Cost: ${size_metrics.loc[size_cat, ('power_cost', 'mean')]:.2f}\n")
                        f.write(f"    Avg Nodes: {size_metrics.loc[size_cat, ('num_nodes_alloc', 'mean')]:.1f}\n")
                
                f.write("\n")
            
            # Overall load distribution
            f.write("\nLoad Distribution Summary:\n")
            f.write("-" * 40 + "\n")
            for scheduler, metrics in load_metrics.items():
                f.write(f"\n{scheduler}:\n")
                f.write(f"  Peak Job Percentage: {metrics['peak_job_percentage']:.1f}%\n")
                if 'jobs_per_zone' in metrics:
                    f.write("  Jobs per Zone:\n")
                    for zone, count in metrics['jobs_per_zone'].items():
                        f.write(f"    {zone}: {count:,d}\n")

def run_analysis(comparison_results: Dict):
    """Run enhanced analysis focusing on large jobs and peak efficiency"""
    analyzer = CostBenefitAnalyzer(comparison_results)
    
    # Generate metrics and plots
    analyzer.plot_large_job_costs()
    analyzer.plot_peak_efficiency()
    analyzer.generate_report()
    
    print("Enhanced analysis complete. Check 'cost_analysis_results.txt' for detailed results.")
    print("Visualizations saved as:")
    print("- large_job_costs.png")
    print("- peak_efficiency.png")