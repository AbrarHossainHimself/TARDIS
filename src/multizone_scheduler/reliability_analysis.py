from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('seaborn')  # For better looking plots
from copy import deepcopy

from multizone_scheduler import MultiZoneScheduler
from baseline_schedulers import SingleZoneScheduler, RandomZoneScheduler
from multizone_generator import MultiZoneConfig, MultiZoneJobGenerator

class ReliabilityAnalyzer:
    """Analyzer for testing scheduler reliability under different failure scenarios"""
    
    def __init__(self, zone_config: Dict[str, Dict], total_nodes_per_zone: int = 512):
        self.zone_config = deepcopy(zone_config)  # Deep copy to avoid modifying original
        self.total_nodes_per_zone = total_nodes_per_zone
        
        # Initialize schedulers
        self.schedulers = {
            'multi_zone': MultiZoneScheduler(self.zone_config, self.total_nodes_per_zone),
            'random': RandomZoneScheduler(self.zone_config, self.total_nodes_per_zone)
        }
        
        # Add single-zone schedulers
        for zone in zone_config:
            self.schedulers[f'single_{zone}'] = SingleZoneScheduler(
                self.zone_config, zone, self.total_nodes_per_zone
            )
        
        # Metrics storage
        self.baseline_metrics = {}
        self.failure_metrics = {}
    
    def visualize_failure_impact(self, metrics: Dict, failure_zone: str, 
                               failure_start: datetime, failure_duration: timedelta) -> None:
        """
        Create visualizations showing the impact of zone failure
        
        Parameters:
        -----------
        metrics : Dict
            The metrics returned from simulate_zone_failure
        failure_zone : str
            The zone that failed
        failure_start : datetime
            When the failure started
        failure_duration : timedelta
            How long the failure lasted
        """
        # Set up the visualization grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Completion Rates Comparison
        completion_rates = [m['completion_rate'] for m in metrics.values()]
        scheduler_names = list(metrics.keys())
        
        ax1.bar(scheduler_names, completion_rates)
        ax1.set_title('Job Completion Rates During Failure')
        ax1.set_ylabel('Completion Rate (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Wait Time Impact
        avg_wait_times = [m['avg_wait_time_during_failure'] for m in metrics.values()]
        max_wait_times = [m['max_wait_time_during_failure'] for m in metrics.values()]
        
        x = np.arange(len(scheduler_names))
        width = 0.35
        
        ax2.bar(x - width/2, avg_wait_times, width, label='Average Wait Time')
        ax2.bar(x + width/2, max_wait_times, width, label='Max Wait Time')
        ax2.set_title('Wait Time Impact During Failure')
        ax2.set_ylabel('Wait Time (seconds)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scheduler_names, rotation=45)
        ax2.legend()
        
        # 3. Job Redistribution
        redistributed = [m['jobs_redistributed'] for m in metrics.values()]
        total_jobs = [m['total_jobs_during_failure'] for m in metrics.values()]
        
        ax3.bar(scheduler_names, total_jobs, label='Total Jobs')
        ax3.bar(scheduler_names, redistributed, label='Redistributed Jobs')
        ax3.set_title('Job Distribution During Failure')
        ax3.set_ylabel('Number of Jobs')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        
        # 4. Recovery Efficiency
        recovery_efficiency = [
            m['jobs_redistributed'] / m['total_jobs_during_failure'] * 100 
            if m['total_jobs_during_failure'] > 0 else 0 
            for m in metrics.values()
        ]
        
        ax4.bar(scheduler_names, recovery_efficiency)
        ax4.set_title('Recovery Efficiency')
        ax4.set_ylabel('Jobs Redistributed (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add failure information
        failure_info = (
            f"Failure Scenario:\n"
            f"Zone: {failure_zone}\n"
            f"Duration: {failure_duration.total_seconds()/3600:.1f} hours\n"
            f"Start: {failure_start}"
        )
        fig.text(0.02, 0.98, failure_info, fontsize=10, va='top')
        
        plt.tight_layout()
        plt.savefig('failure_impact_analysis.png')
        plt.close()

    def visualize_time_series(
        self,
        metrics: Dict,
        event_start: datetime,
        event_duration: timedelta,
        output_file: str = 'time_series_analysis.png'
    ) -> None:
        """
        Create detailed time-series visualizations
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))
        
        event_end = event_start + event_duration
        colors = {
            'multi_zone': '#2ecc71',     # Green
            'random': '#e74c3c',         # Red
            'single_us_east': '#3498db',  # Blue
            'single_us_west': '#f1c40f',  # Yellow
            'single_us_south': '#9b59b6'  # Purple
        }
        
        for scheduler_name, scheduler_metrics in metrics.items():
            if 'time_series' not in scheduler_metrics:
                continue
                
            df = scheduler_metrics['time_series']
            color = colors.get(scheduler_name, 'gray')
            
            # 1. Queue Length Over Time
            ax1.plot(df['time'], df['queue_length'], 
                    label=scheduler_name, color=color, marker='.')
            
            # 2. Running Jobs Over Time
            ax2.plot(df['time'], df['running_jobs'],
                    label=scheduler_name, color=color, marker='.')
            
            # 3. Total Jobs (Running + Queued) Over Time
            ax3.plot(df['time'], df['total_jobs'],
                    label=scheduler_name, color=color, marker='.')
        
        # Add event period shading and formatting
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(event_start, event_end, color='red', alpha=0.1,
                      label='Event Period')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right')
            ax.xaxis.set_major_formatter(plt.FixedFormatter('%H:%M'))
        
        ax1.set_title('Queue Length Over Time')
        ax1.set_ylabel('Number of Jobs in Queue')
        
        ax2.set_title('Running Jobs Over Time')
        ax2.set_ylabel('Number of Running Jobs')
        
        ax3.set_title('Total Jobs in System Over Time')
        ax3.set_ylabel('Total Number of Jobs')
        
        # Add summary statistics
        summary_text = "Summary Statistics:\n"
        for scheduler_name, scheduler_metrics in metrics.items():
            if 'completion_rate' in scheduler_metrics:
                summary_text += f"\n{scheduler_name}:\n"
                summary_text += f"Completion Rate: {scheduler_metrics['completion_rate']:.1f}%\n"
                if 'avg_wait_time' in scheduler_metrics:
                    summary_text += f"Avg Wait Time: {scheduler_metrics['avg_wait_time']:.1f}s\n"
        
        fig.text(0.02, 0.98, summary_text, fontsize=10, va='top')
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

    def simulate_zone_degradation(
        self,
        jobs_df: pd.DataFrame,
        degraded_zone: str,
        degradation_start: datetime,
        degradation_duration: timedelta,
        capacity_reduction: float
    ) -> Dict:
        """
        Simulate partial degradation of a zone and analyze impact
        """
        if not 0 <= capacity_reduction <= 1:
            raise ValueError("capacity_reduction must be between 0 and 1")
            
        degradation_metrics = {}
        degradation_end = degradation_start + degradation_duration
        
        for scheduler_name, scheduler in self.schedulers.items():
            print(f"\nRunning degradation simulation for {scheduler_name}")
            
            # Create a copy of the scheduler for this simulation
            test_scheduler = deepcopy(scheduler)
            
            # Initialize time series metrics
            time_series_data = []
            current_time = degradation_start - timedelta(hours=1)  # Start tracking 1 hour before
            end_tracking = degradation_end + timedelta(hours=1)    # Track 1 hour after
            
            while current_time <= end_tracking:
                # Record metrics every 5 minutes
                if current_time >= degradation_start and current_time <= degradation_end:
                    # Apply capacity reduction
                    test_scheduler.total_nodes_per_zone = int(self.total_nodes_per_zone * (1 - capacity_reduction))
                
                # Calculate current metrics
                total_jobs = len(test_scheduler.queued_jobs)
                running_jobs = sum(len(jobs) for jobs in test_scheduler.running_jobs.values())
                queue_length = len(test_scheduler.queued_jobs)
                
                time_series_data.append({
                    'time': current_time,
                    'total_jobs': total_jobs,
                    'running_jobs': running_jobs,
                    'queue_length': queue_length,
                    'is_degraded': degradation_start <= current_time <= degradation_end
                })
                
                current_time += timedelta(minutes=5)
            
            # Run simulation
            jobs_copy = jobs_df.copy()
            completed_jobs, metrics = test_scheduler.simulate(jobs_copy)
            
            # Calculate degradation period metrics
            degradation_period_jobs = completed_jobs[
                (completed_jobs['start_time'] >= degradation_start) &
                (completed_jobs['start_time'] <= degradation_end)
            ]
            
            degradation_metrics[scheduler_name] = {
                'total_jobs_during_degradation': len(degradation_period_jobs),
                'jobs_in_degraded_zone': len(degradation_period_jobs[
                    degradation_period_jobs['assigned_zone'] == degraded_zone
                ]),
                'jobs_in_other_zones': len(degradation_period_jobs[
                    degradation_period_jobs['assigned_zone'] != degraded_zone
                ]),
                'avg_wait_time': degradation_period_jobs['wait_time'].mean(),
                'max_wait_time': degradation_period_jobs['wait_time'].max(),
                'completion_rate': len(completed_jobs) / len(jobs_df) * 100,
                'time_series': pd.DataFrame(time_series_data)
            }
        
        return degradation_metrics

    def visualize_degradation_impact(
        self,
        metrics: Dict,
        degraded_zone: str,
        degradation_start: datetime,
        degradation_duration: timedelta,
        capacity_reduction: float
    ) -> None:
        """
        Create visualizations showing the impact of zone degradation
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        scheduler_names = list(metrics.keys())
        
        # 1. Job Distribution
        jobs_degraded = [m['jobs_in_degraded_zone'] for m in metrics.values()]
        jobs_other = [m['jobs_in_other_zones'] for m in metrics.values()]
        
        x = np.arange(len(scheduler_names))
        width = 0.35
        
        ax1.bar(x, jobs_degraded, width, label=f'Jobs in {degraded_zone}')
        ax1.bar(x, jobs_other, width, bottom=jobs_degraded, label='Jobs in Other Zones')
        ax1.set_title('Job Distribution During Degradation')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scheduler_names, rotation=45)
        ax1.legend()
        
        # 2. Wait Time Impact
        avg_wait_times = [m['avg_wait_time'] for m in metrics.values()]
        max_wait_times = [m['max_wait_time'] for m in metrics.values()]
        
        ax2.bar(x - width/2, avg_wait_times, width, label='Average Wait Time')
        ax2.bar(x + width/2, max_wait_times, width, label='Max Wait Time')
        ax2.set_title('Wait Time Impact During Degradation')
        ax2.set_ylabel('Wait Time (seconds)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scheduler_names, rotation=45)
        ax2.legend()
        
        # 3. Completion Rates
        completion_rates = [m['completion_rate'] for m in metrics.values()]
        
        ax3.bar(scheduler_names, completion_rates)
        ax3.set_title('Job Completion Rates During Degradation')
        ax3.set_ylabel('Completion Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Load Distribution Efficiency
        load_distribution = [
            m['jobs_in_other_zones'] / m['total_jobs_during_degradation'] * 100
            if m['total_jobs_during_degradation'] > 0 else 0
            for m in metrics.values()
        ]
        
        ax4.bar(scheduler_names, load_distribution)
        ax4.set_title('Load Distribution Efficiency')
        ax4.set_ylabel('Jobs Distributed to Other Zones (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add degradation information
        degradation_info = (
            f"Degradation Scenario:\n"
            f"Zone: {degraded_zone}\n"
            f"Capacity Reduction: {capacity_reduction*100:.1f}%\n"
            f"Duration: {degradation_duration.total_seconds()/3600:.1f} hours\n"
            f"Start: {degradation_start}"
        )
        fig.text(0.02, 0.98, degradation_info, fontsize=10, va='top')
        
        plt.tight_layout()
        plt.savefig('degradation_impact_analysis.png')
        plt.close()

    def analyze_time_series(
        self,
        jobs_df: pd.DataFrame,
        scheduler_metrics: Dict,
        event_start: datetime,
        event_duration: timedelta,
        output_file: str = 'time_series_analysis.png'
    ) -> Dict:
        """
        Analyze and visualize time-series metrics during the event period
        """
        event_end = event_start + event_duration
        
        # Create time windows (10-minute intervals)
        time_windows = pd.date_range(
            start=event_start - timedelta(hours=1),  # Include 1 hour before
            end=event_end + timedelta(hours=1),      # Include 1 hour after
            freq='10T'
        )
        
        time_series_metrics = {}
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))
        
        for scheduler_name, metrics in scheduler_metrics.items():
            # Get completed jobs for this scheduler
            completed_jobs = pd.DataFrame(metrics.get('completed_jobs', []))
            if completed_jobs.empty:
                continue
                
            # Calculate metrics for each time window
            window_metrics = []
            for start_time in time_windows[:-1]:
                end_time = start_time + timedelta(minutes=10)
                window_jobs = completed_jobs[
                    (completed_jobs['start_time'] >= start_time) &
                    (completed_jobs['start_time'] < end_time)
                ]
                
                window_metrics.append({
                    'time': start_time,
                    'throughput': len(window_jobs),
                    'avg_wait': window_jobs['wait_time'].mean() if not window_jobs.empty else 0,
                    'resource_util': len(window_jobs) * window_jobs['num_nodes_alloc'].mean() / self.total_nodes_per_zone 
                    if not window_jobs.empty else 0
                })
            
            df_metrics = pd.DataFrame(window_metrics)
            time_series_metrics[scheduler_name] = df_metrics
            
            # Plot metrics
            # 1. Throughput
            ax1.plot(df_metrics['time'], df_metrics['throughput'], 
                    label=scheduler_name, marker='o', markersize=4)
            
            # 2. Average Wait Time
            ax2.plot(df_metrics['time'], df_metrics['avg_wait'],
                    label=scheduler_name, marker='o', markersize=4)
            
            # 3. Resource Utilization
            ax3.plot(df_metrics['time'], df_metrics['resource_util'],
                    label=scheduler_name, marker='o', markersize=4)
        
        # Add event period indication
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(event_start, event_end, color='red', alpha=0.1)
            ax.legend()
            ax.grid(True)
        
        ax1.set_title('Job Throughput Over Time')
        ax1.set_ylabel('Jobs Completed')
        
        ax2.set_title('Average Wait Time Over Time')
        ax2.set_ylabel('Wait Time (seconds)')
        
        ax3.set_title('Resource Utilization Over Time')
        ax3.set_ylabel('Utilization Rate')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        return time_series_metrics

    def save_results(
        self,
        results: Dict,
        scenario_type: str,
        output_file: str = 'scheduler_analysis_results.txt'
    ) -> None:
        """
        Save analysis results to a file in a clear, formatted manner
        """
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"{scenario_type} Analysis Results\n")
            f.write(f"{'='*50}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for scheduler_name, metrics in results.items():
                f.write(f"\nScheduler: {scheduler_name}\n")
                f.write("-" * 30 + "\n")
                
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{metric_name}: {value:.2f}\n")
                    elif isinstance(value, pd.DataFrame):
                        f.write(f"{metric_name}: [DataFrame with {len(value)} records]\n")
                    else:
                        continue  # Skip non-scalar, non-DataFrame values
                
                f.write("\n")
            
            f.write("\n" + "="*50 + "\n")

    def simulate_zone_failure(
        self,
        jobs_df: pd.DataFrame,
        failure_zone: str,
        failure_start: datetime,
        failure_duration: timedelta
    ) -> Dict:
        """
        Simulate complete failure of a zone and analyze impact
        
        Parameters:
        -----------
        jobs_df : pd.DataFrame
            DataFrame containing job information
        failure_zone : str
            Name of the zone to simulate failure
        failure_start : datetime
            When the failure starts
        failure_duration : timedelta
            How long the failure lasts
        
        Returns:
        --------
        Dict containing impact metrics
        """
        failure_metrics = {}
        failure_end = failure_start + failure_duration
        
        for scheduler_name, scheduler in self.schedulers.items():
            print(f"\nRunning failure simulation for {scheduler_name}")
            
            # Create a copy of the scheduler for this simulation
            test_scheduler = deepcopy(scheduler)
            
            # Run simulation with failure
            jobs_copy = jobs_df.copy()
            completed_jobs, metrics = test_scheduler.simulate(jobs_copy)
            
            # Calculate failure impact metrics
            failure_period_jobs = completed_jobs[
                (completed_jobs['start_time'] >= failure_start) &
                (completed_jobs['start_time'] <= failure_end)
            ]
            
            failure_metrics[scheduler_name] = {
                'total_jobs_during_failure': len(failure_period_jobs),
                'avg_wait_time_during_failure': failure_period_jobs['wait_time'].mean(),
                'max_wait_time_during_failure': failure_period_jobs['wait_time'].max(),
                'jobs_redistributed': len(failure_period_jobs[
                    failure_period_jobs['assigned_zone'] != failure_zone
                ]),
                'completion_rate': len(completed_jobs) / len(jobs_df) * 100
            }
        
        return failure_metrics

if __name__ == "__main__":
    # Example usage
    config = MultiZoneConfig()
    analyzer = ReliabilityAnalyzer(config.zones)
    
    # Generate sample data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 7)
    generator = MultiZoneJobGenerator(start_date, end_date, config)
    jobs_df = generator.generate_dataset()
    
    print("\nRunning complete failure simulation...")
    # Simulate a complete failure
    failure_start = datetime(2024, 1, 3, 12, 0)  # Jan 3rd at noon
    failure_duration = timedelta(hours=2)
    
    failure_metrics = analyzer.simulate_zone_failure(
        jobs_df,
        'us_east',
        failure_start,
        failure_duration
    )
    
    # Save failure results
    analyzer.save_results(failure_metrics, "Complete Zone Failure", "failure_analysis_results.txt")
    
    # Generate failure visualizations
    analyzer.visualize_failure_impact(
        failure_metrics,
        'us_east',
        failure_start,
        failure_duration
    )
    
    print("\nRunning partial degradation simulation...")
    # Simulate partial degradation
    degradation_start = datetime(2024, 1, 4, 9, 0)  # Jan 4th at 9 AM
    degradation_duration = timedelta(hours=4)
    capacity_reduction = 0.4  # 40% reduction in capacity
    
    degradation_metrics = analyzer.simulate_zone_degradation(
        jobs_df,
        'us_west',
        degradation_start,
        degradation_duration,
        capacity_reduction
    )
    
    # Save degradation results
    analyzer.save_results(degradation_metrics, "Partial Zone Degradation", "degradation_analysis_results.txt")
    
    # Generate visualizations
    analyzer.visualize_degradation_impact(
        degradation_metrics,
        'us_west',
        degradation_start,
        degradation_duration,
        capacity_reduction
    )
    
    # Generate time series visualization
    analyzer.visualize_time_series(
        degradation_metrics,
        degradation_start,
        degradation_duration,
        'degradation_time_series.png'
    )
    
    print("\nAnalysis complete! Results have been saved to:")
    print("- failure_analysis_results.txt")
    print("- degradation_analysis_results.txt")
    print("- failure_impact_analysis.png")
    print("- degradation_time_series.png")