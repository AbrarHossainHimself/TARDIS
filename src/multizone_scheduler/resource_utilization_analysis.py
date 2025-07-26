import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import matplotlib.pyplot as plt

class ResourceUtilizationAnalyzer:
    """Analyzes resource utilization patterns across different schedulers"""
    
    def __init__(self, completed_jobs_dict: Dict[str, pd.DataFrame], zone_config: Dict, total_nodes: int = 512):
        self.completed_jobs = completed_jobs_dict
        self.zone_config = zone_config
        self.total_nodes = total_nodes
        self.results_file = "utilization_analysis_results.txt"
    
    def _calculate_zone_utilization(self, df: pd.DataFrame, zone: str) -> pd.DataFrame:
        """Calculate hourly node utilization for a specific zone"""
        # Filter jobs for this zone
        zone_jobs = df[df['assigned_zone'] == zone].copy()
        
        if zone_jobs.empty:
            return pd.DataFrame()
            
        # Ensure datetime columns
        zone_jobs['start_time'] = pd.to_datetime(zone_jobs['start_time'])
        zone_jobs['end_time'] = pd.to_datetime(zone_jobs['end_time'])
        
        # Create hourly time points
        start_time = zone_jobs['start_time'].min()
        end_time = zone_jobs['end_time'].max()
        hours = pd.date_range(start=start_time, end=end_time, freq='H')
        
        hourly_util = []
        for hour in hours:
            # Find jobs running during this hour
            running_jobs = zone_jobs[
                (zone_jobs['start_time'] <= hour) & 
                (zone_jobs['end_time'] > hour)
            ]
            
            # Calculate total nodes used
            nodes_used = running_jobs['num_nodes_alloc'].sum()
            utilization = min((nodes_used / self.total_nodes) * 100, 100)  # Cap at 100%
            
            # Convert to zone's local time
            local_hour = hour + timedelta(hours=self.zone_config[zone]['timezone_offset'])
            
            # Determine if it's a peak hour
            is_peak = self.zone_config[zone]['peak_start'] <= local_hour.hour < self.zone_config[zone]['peak_end']
            
            hourly_util.append({
                'hour': hour,
                'local_hour': local_hour,
                'nodes_used': nodes_used,
                'utilization': utilization,
                'zone': zone,
                'is_peak': is_peak
            })
        
        return pd.DataFrame(hourly_util)
    
    def _calculate_load_balance_metrics(self, zone_utils: List[pd.DataFrame]) -> Dict:
        """Calculate metrics related to load balancing across zones"""
        if not zone_utils or len(zone_utils) < 2:
            return {}
            
        # Get common time range
        all_times = set()
        for df in zone_utils:
            if not df.empty:
                all_times.update(df['hour'].unique())
        all_times = sorted(all_times)
        
        # Calculate utilization differences at each hour
        hourly_diffs = []
        for hour in all_times:
            utils = []
            for df in zone_utils:
                if not df.empty:
                    hour_util = df[df['hour'] == hour]['utilization'].iloc[0] if hour in df['hour'].values else 0
                    utils.append(hour_util)
            
            if utils:
                hourly_diffs.append({
                    'hour': hour,
                    'spread': max(utils) - min(utils),
                    'std': np.std(utils),
                    'max': max(utils),
                    'min': min(utils)
                })
        
        if not hourly_diffs:
            return {}
            
        hourly_df = pd.DataFrame(hourly_diffs)
        
        return {
            'avg_utilization_spread': hourly_df['spread'].mean(),
            'std_between_zones': hourly_df['std'].mean(),
            'max_imbalance': hourly_df['spread'].max(),
            'hours_with_imbalance': (hourly_df['spread'] > 20).sum()  # Count hours with >20% spread
        }
    
    def _calculate_peak_metrics(self, zone_util: pd.DataFrame) -> Dict:
        """Calculate metrics related to peak hour handling"""
        if zone_util.empty:
            return {}
            
        peak_hours = zone_util[zone_util['is_peak']]
        off_peak = zone_util[~zone_util['is_peak']]
        
        return {
            'peak_utilization': peak_hours['utilization'].mean() if not peak_hours.empty else 0,
            'off_peak_utilization': off_peak['utilization'].mean() if not off_peak.empty else 0,
            'peak_hours_above_80': (peak_hours['utilization'] >= 80).sum() if not peak_hours.empty else 0,
            'peak_max_utilization': peak_hours['utilization'].max() if not peak_hours.empty else 0
        }
    
    def analyze_temporal_utilization(self):
        """Analyze temporal utilization patterns for all schedulers"""
        results = []
        
        for scheduler_name, jobs_df in self.completed_jobs.items():
            scheduler_stats = {
                'scheduler': scheduler_name,
                'zones': {},
                'overall': {},
                'load_balance': {},
                'peak_handling': {}
            }
            
            # Calculate per-zone utilization
            zone_dfs = []
            for zone in self.zone_config.keys():
                zone_util = self._calculate_zone_utilization(jobs_df, zone)
                if not zone_util.empty:
                    zone_dfs.append(zone_util)
                    
                    # Calculate zone statistics
                    scheduler_stats['zones'][zone] = {
                        'avg_utilization': zone_util['utilization'].mean(),
                        'max_utilization': zone_util['utilization'].max(),
                        'min_utilization': zone_util['utilization'].min(),
                        'std_utilization': zone_util['utilization'].std(),
                        'hours_above_80pct': (zone_util['utilization'] >= 80).sum(),
                        'peak_metrics': self._calculate_peak_metrics(zone_util)
                    }
            
            # Calculate load balance metrics
            scheduler_stats['load_balance'] = self._calculate_load_balance_metrics(zone_dfs)
            
            # Calculate overall statistics
            if zone_dfs:
                # Get all unique hours
                all_hours = set()
                for df in zone_dfs:
                    if not df.empty:
                        all_hours.update(df['hour'].unique())
                all_hours = sorted(all_hours)
                
                # Calculate total utilization for each hour
                hourly_utils = []
                for hour in all_hours:
                    total_nodes_used = 0
                    for df in zone_dfs:
                        if not df.empty and hour in df['hour'].values:
                            total_nodes_used += df[df['hour'] == hour]['nodes_used'].iloc[0]
                    
                    active_zones = len([df for df in zone_dfs if not df.empty and 
                                      hour in df['hour'].values])
                    total_nodes = self.total_nodes * active_zones
                    util = (total_nodes_used / total_nodes) * 100 if total_nodes > 0 else 0
                    hourly_utils.append(util)
                
                overall_util = pd.Series(hourly_utils)
                
                scheduler_stats['overall'] = {
                    'avg_utilization': overall_util.mean(),
                    'max_utilization': overall_util.max(),
                    'min_utilization': overall_util.min(),
                    'std_utilization': overall_util.std(),
                    'hours_above_80pct': (overall_util >= 80).sum(),
                    'hours_below_20pct': (overall_util <= 20).sum()
                }
            
            results.append(scheduler_stats)
        
        # Write results to file
        with open(self.results_file, 'w') as f:
            f.write("Resource Utilization Analysis Results\n")
            f.write("===================================\n\n")
            
            for stats in results:
                f.write(f"Scheduler: {stats['scheduler']}\n")
                f.write("=" * 40 + "\n\n")
                
                f.write("Overall Cluster Statistics:\n")
                f.write("-----------------------\n")
                overall = stats['overall']
                f.write(f"Average Utilization: {overall['avg_utilization']:.2f}%\n")
                f.write(f"Maximum Utilization: {overall['max_utilization']:.2f}%\n")
                f.write(f"Minimum Utilization: {overall['min_utilization']:.2f}%\n")
                f.write(f"Standard Deviation: {overall['std_utilization']:.2f}%\n")
                f.write(f"Hours with >80% utilization: {overall['hours_above_80pct']}\n")
                f.write(f"Hours with <20% utilization: {overall['hours_below_20pct']}\n\n")
                
                f.write("Load Balancing Metrics:\n")
                f.write("--------------------\n")
                balance = stats['load_balance']
                if balance:
                    f.write(f"Average Utilization Spread: {balance['avg_utilization_spread']:.2f}%\n")
                    f.write(f"Standard Deviation Between Zones: {balance['std_between_zones']:.2f}%\n")
                    f.write(f"Maximum Imbalance: {balance['max_imbalance']:.2f}%\n\n")
                
                f.write("Per-Zone Statistics:\n")
                f.write("------------------\n")
                for zone, zone_stats in stats['zones'].items():
                    f.write(f"\n{zone}:\n")
                    f.write(f"Average Utilization: {zone_stats['avg_utilization']:.2f}%\n")
                    f.write(f"Maximum Utilization: {zone_stats['max_utilization']:.2f}%\n")
                    f.write(f"Standard Deviation: {zone_stats['std_utilization']:.2f}%\n")
                    f.write(f"Hours with >80% utilization: {zone_stats['hours_above_80pct']}\n")
                    
                    peak = zone_stats['peak_metrics']
                    if peak:
                        f.write("\nPeak Hour Performance:\n")
                        f.write(f"Average Peak Hour Utilization: {peak['peak_utilization']:.2f}%\n")
                        f.write(f"Average Off-Peak Utilization: {peak['off_peak_utilization']:.2f}%\n")
                        f.write(f"Peak Hours Above 80%: {peak['peak_hours_above_80']}\n")
                        f.write(f"Maximum Peak Hour Utilization: {peak['peak_max_utilization']:.2f}%\n")
                    f.write("\n")
                
                f.write("\n" + "=" * 40 + "\n\n")
        
        return results

if __name__ == "__main__":
    # This section would be used for testing
    pass