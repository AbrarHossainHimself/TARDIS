import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from typing import Dict, List, Tuple

class TemporalAnalyzer:
    def __init__(self, peak_start: int = 6, peak_end: int = 22):
        self.peak_start = peak_start
        self.peak_end = peak_end
        
    def categorize_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize jobs based on size (nodes), runtime, and power consumption
        """
        df = df.copy()
        
        # Categorize by nodes
        df['size_by_nodes'] = pd.qcut(
            df['num_nodes_alloc'], 
            q=3, 
            labels=['small', 'medium', 'large']
        )
        
        # Categorize by runtime
        df['size_by_runtime'] = pd.qcut(
            df['run_time'], 
            q=3, 
            labels=['short', 'medium', 'long']
        )
        
        # Categorize by power
        df['size_by_power'] = pd.qcut(
            df['power_kw'], 
            q=3, 
            labels=['low', 'medium', 'high']
        )
        
        return df
    
    def analyze_scheduler_results(self, results_file: str) -> Dict:
        """
        Analyze completed jobs from each scheduler
        """
        # Load results
        df = pd.read_csv(results_file)
        
        # Add hour column if not present
        if 'hour' not in df.columns:
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['hour'] = df['start_time'].dt.hour
            
        # Mark peak/off-peak
        df['is_peak'] = df['hour'].apply(lambda x: self.peak_start <= x < self.peak_end)
        
        # Categorize jobs
        df = self.categorize_jobs(df)
        
        # Analyze distributions
        analysis = {}
        
        # For each size metric
        for size_metric in ['size_by_nodes', 'size_by_runtime', 'size_by_power']:
            peak_dist = df[df['is_peak']][size_metric].value_counts(normalize=True)
            offpeak_dist = df[~df['is_peak']][size_metric].value_counts(normalize=True)
            
            analysis[size_metric] = {
                'peak_distribution': peak_dist.to_dict(),
                'offpeak_distribution': offpeak_dist.to_dict(),
                'peak_counts': df[df['is_peak']][size_metric].value_counts().to_dict(),
                'offpeak_counts': df[~df['is_peak']][size_metric].value_counts().to_dict()
            }
            
        return analysis

    def plot_temporal_distribution(self, analyses: Dict[str, Dict], output_file: str):
        """
        Create visualization of temporal distribution for different schedulers
        """
        with PdfPages(output_file) as pdf:
            metrics = ['size_by_nodes', 'size_by_runtime', 'size_by_power']
            schedulers = list(analyses.keys())
            
            for metric in metrics:
                # Distribution plot
                fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                
                # Peak hours
                peak_data = []
                for scheduler in schedulers:
                    dist = analyses[scheduler][metric]['peak_distribution']
                    for size, value in dist.items():
                        peak_data.append({
                            'Scheduler': scheduler,
                            'Size': size,
                            'Percentage': value * 100,
                            'Period': 'Peak'
                        })
                
                # Off-peak hours
                offpeak_data = []
                for scheduler in schedulers:
                    dist = analyses[scheduler][metric]['offpeak_distribution']
                    for size, value in dist.items():
                        offpeak_data.append({
                            'Scheduler': scheduler,
                            'Size': size,
                            'Percentage': value * 100,
                            'Period': 'Off-Peak'
                        })
                
                # Combine data
                plot_data = pd.DataFrame(peak_data + offpeak_data)
                
                # Create grouped bar plots
                sns.barplot(
                    data=plot_data[plot_data['Period'] == 'Peak'],
                    x='Scheduler',
                    y='Percentage',
                    hue='Size',
                    ax=axes[0]
                )
                axes[0].set_title(f'Job Distribution During Peak Hours - {metric}')
                axes[0].set_ylabel('Percentage of Jobs')
                
                sns.barplot(
                    data=plot_data[plot_data['Period'] == 'Off-Peak'],
                    x='Scheduler',
                    y='Percentage',
                    hue='Size',
                    ax=axes[1]
                )
                axes[1].set_title(f'Job Distribution During Off-Peak Hours - {metric}')
                axes[1].set_ylabel('Percentage of Jobs')
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                
                # Create hourly distribution plot
                plt.figure(figsize=(15, 6))
                for scheduler in schedulers:
                    hourly_dist = analyses[scheduler].get('hourly_distribution', {})
                    if hourly_dist:
                        plt.plot(
                            hourly_dist.keys(),
                            hourly_dist.values(),
                            label=scheduler,
                            marker='o'
                        )
                
                plt.axvspan(self.peak_start, self.peak_end, alpha=0.2, color='red')
                plt.xlabel('Hour of Day')
                plt.ylabel('Number of Jobs')
                plt.title(f'Hourly Job Distribution - {metric}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

def main():
    """
    Main function to run temporal analysis
    """
    analyzer = TemporalAnalyzer()
    
    # Dictionary to store analyses for each scheduler
    scheduler_analyses = {}
    
    # Analyze each scheduler's results
    schedulers = ['fcfs', 'enhanced', 'celf', 'sjf']
    for scheduler in schedulers:
        try:
            results_file = f"{scheduler}_completed_jobs.csv"
            analysis = analyzer.analyze_scheduler_results(results_file)
            scheduler_analyses[scheduler] = analysis
        except Exception as e:
            print(f"Error analyzing {scheduler}: {str(e)}")
    
    # Generate visualizations
    analyzer.plot_temporal_distribution(
        scheduler_analyses,
        'temporal_analysis_results.pdf'
    )
    
    # Print summary statistics
    print("\nTemporal Analysis Summary")
    print("=" * 50)
    for scheduler, analysis in scheduler_analyses.items():
        print(f"\n{scheduler.upper()} Scheduler:")
        for metric in ['size_by_nodes', 'size_by_runtime', 'size_by_power']:
            print(f"\n{metric}:")
            print("Peak Hours Distribution:")
            for size, pct in analysis[metric]['peak_distribution'].items():
                print(f"  {size}: {pct*100:.1f}%")
            print("Off-Peak Hours Distribution:")
            for size, pct in analysis[metric]['offpeak_distribution'].items():
                print(f"  {size}: {pct*100:.1f}%")

if __name__ == "__main__":
    main()