# scheduler_debug.py
# Debugging utilities for scheduler wait time analysis

from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SchedulerDebugger:
    """Debugging and analysis tools for scheduler performance"""
    
    @staticmethod
    def analyze_wait_times(completed_jobs_df: pd.DataFrame, scheduler_name: str) -> Dict:
        """Detailed analysis of job wait times"""
        if completed_jobs_df.empty:
            return {
                'scheduler': scheduler_name,
                'error': 'No completed jobs found'
            }
            
        # Convert times to datetime if they're not already
        for col in ['submit_time', 'start_time', 'end_time']:
            if completed_jobs_df[col].dtype != 'datetime64[ns]':
                completed_jobs_df[col] = pd.to_datetime(completed_jobs_df[col])
        
        # Calculate wait times in seconds
        wait_times = (completed_jobs_df['start_time'] - 
                     completed_jobs_df['submit_time']).dt.total_seconds()
        
        # Basic statistics
        stats = {
            'scheduler': scheduler_name,
            'total_jobs': len(completed_jobs_df),
            'jobs_with_wait': (wait_times > 0).sum(),
            'wait_time_stats': {
                'min': wait_times.min(),
                'max': wait_times.max(),
                'mean': wait_times.mean(),
                'median': wait_times.median(),
                'std': wait_times.std()
            }
        }
        
        # Analyze wait times by job size
        completed_jobs_df['job_size'] = pd.qcut(
            completed_jobs_df['num_nodes_alloc'], 
            q=4, 
            labels=['small', 'medium', 'large', 'very_large']
        )
        
        wait_by_size = completed_jobs_df.groupby('job_size')['wait_time'].agg([
            'count', 'mean', 'median', 'max'
        ]).round(2).to_dict('index')
        
        stats['wait_by_job_size'] = wait_by_size
        
        # Analyze wait times for peak vs off-peak
        peak_stats = completed_jobs_df.groupby('is_peak')['wait_time'].agg([
            'count', 'mean', 'median', 'max'
        ]).round(2).to_dict('index')
        
        stats['wait_by_peak_status'] = peak_stats
        
        # Find potentially problematic jobs
        long_waits = completed_jobs_df[wait_times > wait_times.mean() + 2*wait_times.std()]
        if not long_waits.empty:
            stats['problematic_jobs'] = {
                'count': len(long_waits),
                'details': long_waits[['job_id', 'wait_time', 'num_nodes_alloc', 
                                     'is_peak', 'assigned_zone']].to_dict('records')
            }
        
        return stats
    
    @staticmethod
    def print_wait_time_analysis(stats: Dict) -> None:
        """Print formatted wait time analysis"""
        print(f"\nWait Time Analysis for {stats['scheduler']}")
        print("=" * 50)
        
        print(f"\nOverall Statistics:")
        print(f"Total Jobs: {stats['total_jobs']}")
        print(f"Jobs with Wait Time > 0: {stats['jobs_with_wait']}")
        
        wait_stats = stats['wait_time_stats']
        print(f"\nWait Time Distribution (seconds):")
        print(f"Min: {wait_stats['min']:.2f}")
        print(f"Max: {wait_stats['max']:.2f}")
        print(f"Mean: {wait_stats['mean']:.2f}")
        print(f"Median: {wait_stats['median']:.2f}")
        print(f"Std Dev: {wait_stats['std']:.2f}")
        
        print(f"\nWait Times by Job Size:")
        for size, metrics in stats['wait_by_job_size'].items():
            print(f"\n{size.title()}:")
            print(f"Count: {metrics['count']}")
            print(f"Mean Wait: {metrics['mean']:.2f}")
            print(f"Median Wait: {metrics['median']:.2f}")
            print(f"Max Wait: {metrics['max']:.2f}")
        
        print(f"\nWait Times by Peak Status:")
        for is_peak, metrics in stats['wait_by_peak_status'].items():
            status = "Peak" if is_peak else "Off-Peak"
            print(f"\n{status}:")
            print(f"Count: {metrics['count']}")
            print(f"Mean Wait: {metrics['mean']:.2f}")
            print(f"Median Wait: {metrics['median']:.2f}")
            print(f"Max Wait: {metrics['max']:.2f}")
        
        if 'problematic_jobs' in stats:
            print(f"\nProblematic Jobs (Unusually Long Waits):")
            print(f"Count: {stats['problematic_jobs']['count']}")
            if stats['problematic_jobs']['count'] > 0:
                print("\nTop 5 most problematic jobs:")
                jobs = sorted(
                    stats['problematic_jobs']['details'],
                    key=lambda x: x['wait_time'],
                    reverse=True
                )[:5]
                for job in jobs:
                    print(f"\nJob ID: {job['job_id']}")
                    print(f"Wait Time: {job['wait_time']:.2f} seconds")
                    print(f"Nodes: {job['num_nodes_alloc']}")
                    print(f"Peak Hour: {job['is_peak']}")
                    print(f"Zone: {job['assigned_zone']}")

def debug_scheduler_results(results: Dict) -> None:
    """Run debug analysis on all scheduler results"""
    debugger = SchedulerDebugger()
    
    for scheduler_name, result in results.items():
        completed_jobs_df = result['completed_jobs']
        wait_analysis = debugger.analyze_wait_times(completed_jobs_df, scheduler_name)
        debugger.print_wait_time_analysis(wait_analysis)