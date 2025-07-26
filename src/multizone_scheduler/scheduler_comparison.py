# scheduler_comparison.py
# Comparison framework for analyzing different HPC scheduling strategies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from multizone_generator import MultiZoneConfig, MultiZoneJobGenerator
from multizone_scheduler import MultiZoneScheduler
from baseline_schedulers import RandomZoneScheduler, SingleZoneScheduler
from scheduler_debug import SchedulerDebugger

class SchedulerComparison:
    """Framework for comparing different scheduling strategies"""
    
    def __init__(self, zone_config: Dict[str, Dict], total_nodes_per_zone: int = 512):
        self.zone_config = zone_config
        self.total_nodes_per_zone = total_nodes_per_zone
        self.debugger = SchedulerDebugger()
        
        # Initialize all schedulers
        self.schedulers = {
            'multi_zone': MultiZoneScheduler(zone_config, total_nodes_per_zone),
            'random': RandomZoneScheduler(zone_config, total_nodes_per_zone)
        }
        
        # Add single-zone schedulers for each zone
        for zone in zone_config:
            self.schedulers[f'single_{zone}'] = SingleZoneScheduler(
                zone_config, zone, total_nodes_per_zone
            )
        
        self.results = {}
        self.debug_results = {}
    
    def run_comparison(self, jobs_df: pd.DataFrame) -> Dict:
        """Run all schedulers on the same job set"""
        print("\nStarting scheduler comparison...")
        print("=" * 50)
        
        self.results = {}
        self.debug_results = {}
        
        for name, scheduler in self.schedulers.items():
            print(f"\nRunning {name} scheduler...")
            completed_jobs, metrics = scheduler.simulate(jobs_df.copy())
            
            self.results[name] = {
                'completed_jobs': completed_jobs,
                'metrics': metrics
            }
            
            # Run debug analysis
            print(f"\nAnalyzing wait times for {name} scheduler...")
            self.debug_results[name] = self.debugger.analyze_wait_times(
                completed_jobs, name
            )
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict:
        """Generate basic comparison metrics with debug information"""
        if not self.results:
            raise ValueError("No results to compare. Run comparison first.")
        
        comparison = {}
        
        # Compare total costs
        costs = {name: result['metrics']['total_cost'] 
                for name, result in self.results.items()}
        min_cost = min(costs.values())
        
        cost_comparison = {
            name: {
                'total_cost': cost,
                'cost_difference': cost - min_cost,
                'cost_difference_percent': ((cost - min_cost) / min_cost * 100)
                if min_cost > 0 else 0
            }
            for name, cost in costs.items()
        }
        
        # Compare job statistics with debug info
        job_stats = {}
        for name, result in self.results.items():
            stats = {
                'total_jobs': result['metrics']['total_jobs'],
                'peak_jobs': sum(
                    zone_metrics['peak_jobs']
                    for zone_metrics in result['metrics']['zone_metrics'].values()
                )
            }
            
            # Add debug statistics
            debug_info = self.debug_results[name]
            stats.update({
                'wait_time_stats': debug_info['wait_time_stats'],
                'jobs_with_wait': debug_info['jobs_with_wait'],
                'wait_by_job_size': debug_info['wait_by_job_size'],
                'wait_by_peak_status': debug_info['wait_by_peak_status']
            })
            
            job_stats[name] = stats
        
        comparison['cost_analysis'] = cost_comparison
        comparison['job_statistics'] = job_stats
        
        self._print_comparison(comparison)
        self._print_debug_analysis()
        
        return comparison
    
    def _print_comparison(self, comparison: Dict) -> None:
        """Print formatted comparison results"""
        print("\nScheduler Comparison Results")
        print("=" * 50)
        
        print("\nCost Analysis:")
        print("-" * 30)
        for scheduler, data in comparison['cost_analysis'].items():
            print(f"\n{scheduler}:")
            print(f"Total Cost: ${data['total_cost']:,.2f}")
            print(f"Difference from best: ${data['cost_difference']:,.2f}")
            print(f"Percentage difference: {data['cost_difference_percent']:.1f}%")
        
        print("\nBasic Job Statistics:")
        print("-" * 30)
        for scheduler, data in comparison['job_statistics'].items():
            print(f"\n{scheduler}:")
            print(f"Total Jobs: {data['total_jobs']}")
            print(f"Peak Jobs: {data['peak_jobs']}")
            print(f"Jobs with Wait Time: {data['jobs_with_wait']}")
            wait_stats = data['wait_time_stats']
            print(f"Average Wait Time: {wait_stats['mean']:.1f} seconds")
            print(f"Maximum Wait Time: {wait_stats['max']:.1f} seconds")
    
    def _print_debug_analysis(self) -> None:
        """Print detailed debug analysis for each scheduler"""
        print("\nDetailed Wait Time Analysis")
        print("=" * 50)
        
        for scheduler_name, debug_info in self.debug_results.items():
            self.debugger.print_wait_time_analysis(debug_info)

if __name__ == "__main__":
    # Set up configuration
    config = MultiZoneConfig()
    
    # Create comparison framework
    comparison = SchedulerComparison(config.zones)
    
    # Generate sample data
    start_date = datetime(2020, 5, 1)
    end_date = datetime(2020, 5, 30)   # One week of data
    generator = MultiZoneJobGenerator(start_date, end_date, config)
    jobs_df = generator.generate_dataset()
    
    # Run comparison with debug analysis
    results = comparison.run_comparison(jobs_df)