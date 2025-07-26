import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List
from scipy import stats
import json

class PeakAnalyzer:
    """Analyze peak vs off-peak behavior of different schedulers"""
    
    def __init__(self, zone_config: Dict):
        self.zone_config = zone_config
        self.results_file = "peak_analysis_results.txt"
    
    def get_zone_time(self, time: datetime, zone: str) -> datetime:
        """Convert reference time to zone-specific time"""
        offset = self.zone_config[zone]['timezone_offset']
        return time + timedelta(hours=offset)
    
    def is_peak_hour(self, time: datetime, zone: str) -> bool:
        """Check if given time is peak hour in specified zone"""
        zone_time = self.get_zone_time(time, zone)
        zone_config = self.zone_config[zone]
        return zone_config['peak_start'] <= zone_time.hour < zone_config['peak_end']
    
    def plot_peak_distribution(self, results: Dict) -> None:
        """Create visualization of peak vs off-peak job distribution"""
        plt.figure(figsize=(15, 8))
        
        schedulers = list(results.keys())
        zones = list(self.zone_config.keys())
        x = np.arange(len(zones))
        width = 0.35
        
        # Create grouped bar chart
        for i, scheduler in enumerate(['multi_zone', 'random']):  # Compare main schedulers
            peak_percentages = [results[scheduler]['zones'][zone]['peak_percentage'] 
                              for zone in zones]
            
            plt.bar(x + i*width, peak_percentages, width, 
                   label=f'{scheduler} scheduler',
                   alpha=0.8)
        
        plt.xlabel('Zones')
        plt.ylabel('Peak Hour Job Percentage')
        plt.title('Peak Hour Job Distribution by Scheduler and Zone')
        plt.xticks(x + width/2, zones)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('peak_distribution.png')
        plt.close()
    
    def chi_square_test(self, results: Dict) -> None:
        """Perform chi-square test to verify statistical significance"""
        # Focus on multi-zone vs random comparison
        with open(self.results_file, 'a') as f:
            f.write("\nStatistical Analysis\n")
            f.write("===================\n\n")
            
            for zone in self.zone_config.keys():
                # Get observed frequencies
                multi_peak = results['multi_zone']['zones'][zone]['peak_jobs']
                multi_off = results['multi_zone']['zones'][zone]['off_peak_jobs']
                rand_peak = results['random']['zones'][zone]['peak_jobs']
                rand_off = results['random']['zones'][zone]['off_peak_jobs']
                
                # Create contingency table
                observed = np.array([[multi_peak, multi_off],
                                   [rand_peak, rand_off]])
                
                # Perform chi-square test
                chi2, p_value = stats.chi2_contingency(observed)[:2]
                
                f.write(f"Chi-square test for {zone}:\n")
                f.write(f"Chi-square statistic: {chi2:.2f}\n")
                f.write(f"p-value: {p_value:.10f}\n\n")
    
    def analyze_temporal_distribution(self, completed_jobs_dict: Dict) -> None:
        """Analyze how job distribution changes over time"""
        # Focus on multi-zone scheduler for temporal analysis
        df = completed_jobs_dict['multi_zone'].copy()
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['hour'] = df['start_time'].dt.hour
        
        # Initialize results storage
        temporal_results = {
            'hourly_distribution': {},
            'zone_loads': {},
            'peak_transitions': {}
        }
        
        # Analyze hourly distribution for each zone
        for zone in self.zone_config.keys():
            zone_jobs = df[df['assigned_zone'] == zone]
            hourly_counts = zone_jobs.groupby('hour').size()
            peak_hours = [self.is_peak_hour(datetime(2024, 1, 1, hour), zone) 
                         for hour in range(24)]
            
            temporal_results['hourly_distribution'][zone] = {
                'counts': hourly_counts.to_dict(),
                'peak_hours': peak_hours
            }
        
        # Save temporal analysis to file
        with open(self.results_file, 'a') as f:
            f.write("\nTemporal Analysis\n")
            f.write("=================\n\n")
            
            for zone in self.zone_config.keys():
                f.write(f"{zone} Hourly Distribution:\n")
                dist = temporal_results['hourly_distribution'][zone]
                
                for hour in range(24):
                    count = dist['counts'].get(hour, 0)
                    is_peak = dist['peak_hours'][hour]
                    f.write(f"  Hour {hour:02d}: {count:3d} jobs {'(PEAK)' if is_peak else ''}\n")
                f.write("\n")
        
        # Create temporal visualization
        self.plot_temporal_distribution(temporal_results)
        
        return temporal_results

    def plot_temporal_distribution(self, temporal_results: Dict) -> None:
        """Create visualization of temporal job distribution"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        zones = list(self.zone_config.keys())
        hours = range(24)
        bottom = np.zeros(24)
        
        for zone in zones:
            data = temporal_results['hourly_distribution'][zone]
            counts = [data['counts'].get(hour, 0) for hour in hours]
            peak_hours = data['peak_hours']
            
            # Plot stacked bars
            ax.bar(hours, counts, bottom=bottom, label=zone, alpha=0.7)
            bottom += np.array(counts)
            
            # Highlight peak hours
            peak_periods = [i for i, is_peak in enumerate(peak_hours) if is_peak]
            if peak_periods:
                ax.axvspan(min(peak_periods), max(peak_periods) + 1, 
                          alpha=0.2, color='red', label=f'{zone} peak' if zone == zones[0] else "")
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Jobs')
        ax.set_title('24-Hour Job Distribution Across Zones')
        ax.set_xticks(hours)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig('temporal_distribution.png')
        plt.close()

    def analyze_job_distribution(self, completed_jobs_dict: Dict) -> Dict:
        """Analyze how jobs are distributed during peak vs off-peak hours"""
        results = {}
        
        for scheduler_name, completed_jobs in completed_jobs_dict.items():
            df = completed_jobs.copy()
            df['start_time'] = pd.to_datetime(df['start_time'])
            
            # Initialize counters for this scheduler
            scheduler_stats = {
                'total_jobs': len(df),
                'zones': {}
            }
            
            # Analyze each zone
            for zone in self.zone_config.keys():
                zone_jobs = df[df['assigned_zone'] == zone]
                
                # Count peak vs off-peak jobs
                peak_jobs = len(zone_jobs[zone_jobs['is_peak']])
                total_zone_jobs = len(zone_jobs)
                
                zone_stats = {
                    'total_jobs': total_zone_jobs,
                    'peak_jobs': peak_jobs,
                    'off_peak_jobs': total_zone_jobs - peak_jobs,
                    'peak_percentage': (peak_jobs / total_zone_jobs * 100) if total_zone_jobs > 0 else 0
                }
                
                scheduler_stats['zones'][zone] = zone_stats
            
            results[scheduler_name] = scheduler_stats
        
        # Save results to file
        with open(self.results_file, 'w') as f:
            f.write("Job Distribution Analysis Results\n")
            f.write("=================================\n\n")
            
            for scheduler_name, stats in results.items():
                f.write(f"{scheduler_name} Scheduler:\n")
                f.write(f"Total Jobs: {stats['total_jobs']}\n\n")
                
                for zone, zone_stats in stats['zones'].items():
                    f.write(f"  {zone}:\n")
                    f.write(f"    Total Jobs: {zone_stats['total_jobs']}\n")
                    f.write(f"    Peak Hour Jobs: {zone_stats['peak_jobs']}\n")
                    f.write(f"    Off-Peak Jobs: {zone_stats['off_peak_jobs']}\n")
                    f.write(f"    Peak Hour Percentage: {zone_stats['peak_percentage']:.2f}%\n\n")
                f.write("\n")
        
        return results

def run_analysis(zone_config: Dict, completed_jobs_dict: Dict) -> None:
    """Run the peak analysis"""
    analyzer = PeakAnalyzer(zone_config)
    
    # Run basic distribution analysis
    results = analyzer.analyze_job_distribution(completed_jobs_dict)
    analyzer.plot_peak_distribution(results)
    analyzer.chi_square_test(results)
    
    # Run temporal analysis
    analyzer.analyze_temporal_distribution(completed_jobs_dict)

if __name__ == "__main__":
    # This section would be used for testing
    pass