from datetime import datetime
import os
from comparison import run_comparison
import pandas as pd
import shutil
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt

class TemporalAnalyzer:
    def __init__(self, peak_start: int = 6, peak_end: int = 22):
        self.peak_start = peak_start
        self.peak_end = peak_end
        
    def categorize_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize jobs primarily based on power consumption.
        Power categories are defined using the total power consumption of the job
        (power_kw * runtime).
        """
        df = df.copy()
        
        # Calculate total power consumption for each job
        df['total_power_consumption'] = df['power_kw'] * (df['run_time'] / 3600)  # kWh
        
        # Primary categorization by power consumption
        df['power_category'] = pd.qcut(
            df['total_power_consumption'],
            q=3,
            labels=['low_power', 'medium_power', 'high_power']
        )
        
        # Additional categorizations for analysis
        df['instant_power_category'] = pd.qcut(
            df['power_kw'],
            q=3,
            labels=['low_instant_power', 'medium_instant_power', 'high_instant_power']
        )
        
        # Calculate power density (power per node)
        df['power_density'] = df['power_kw'] / df['num_nodes_alloc']
        df['power_density_category'] = pd.qcut(
            df['power_density'],
            q=3,
            labels=['low_density', 'medium_density', 'high_density']
        )
        
        return df

    def analyze_monthly_temporal_distribution(self, month_name: str, results_dir: str):
        """Analyze temporal distribution for a specific month"""
        schedulers = ['fcfs', 'enhanced', 'celf', 'sjf']
        analyses = {}
        
        for scheduler in schedulers:
            filename = f"{scheduler}_completed_jobs_{month_name}.csv"
            filepath = os.path.join(results_dir, filename)
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['start_time'] = pd.to_datetime(df['start_time'])
                df['hour'] = df['start_time'].dt.hour
                df['is_peak'] = df['hour'].apply(lambda x: self.peak_start <= x < self.peak_end)
                
                df = self.categorize_jobs(df)
                analyses[scheduler] = self._analyze_distribution(df)
        
        return analyses

    def _analyze_distribution(self, df: pd.DataFrame) -> dict:
        """
        Analyze job distribution focusing on power-based metrics
        """
        analysis = {}
        
        # Power consumption analysis
        for metric in ['power_category', 'instant_power_category', 'power_density_category']:
            peak_stats = df[df['is_peak']][metric].value_counts(normalize=True)
            offpeak_stats = df[~df['is_peak']][metric].value_counts(normalize=True)
            
            # Basic distribution
            analysis[metric] = {
                'peak_distribution': peak_stats.to_dict(),
                'offpeak_distribution': offpeak_stats.to_dict(),
                'peak_counts': df[df['is_peak']][metric].value_counts().to_dict(),
                'offpeak_counts': df[~df['is_peak']][metric].value_counts().to_dict()
            }
            
            # Power consumption statistics
            analysis[f'{metric}_power_stats'] = {
                'peak': {
                    'mean_power': df[df['is_peak']].groupby(metric)['power_kw'].mean().to_dict(),
                    'total_consumption': df[df['is_peak']].groupby(metric)['total_power_consumption'].sum().to_dict()
                },
                'offpeak': {
                    'mean_power': df[~df['is_peak']].groupby(metric)['power_kw'].mean().to_dict(),
                    'total_consumption': df[~df['is_peak']].groupby(metric)['total_power_consumption'].sum().to_dict()
                }
            }
        
        return analysis

    def plot_monthly_distributions(self, analyses: dict, month_name: str, results_dir: str):
        """Create visualizations for monthly power-based distributions"""
        pdf_filename = os.path.join(results_dir, f"power_analysis_{month_name}.pdf")
        
        with PdfPages(pdf_filename) as pdf:
            metrics = ['power_category', 'instant_power_category', 'power_density_category']
            schedulers = list(analyses.keys())
            
            for metric in metrics:
                # Distribution plot
                fig, axes = plt.subplots(2, 1, figsize=(15, 12))
                
                plot_data = []
                for scheduler in schedulers:
                    # Peak hours data
                    for category, value in analyses[scheduler][metric]['peak_distribution'].items():
                        plot_data.append({
                            'Scheduler': scheduler.upper(),
                            'Category': category,
                            'Percentage': value * 100,
                            'Period': 'Peak',
                            'Mean Power': analyses[scheduler][f'{metric}_power_stats']['peak']['mean_power'][category]
                        })
                    # Off-peak hours data
                    for category, value in analyses[scheduler][metric]['offpeak_distribution'].items():
                        plot_data.append({
                            'Scheduler': scheduler.upper(),
                            'Category': category,
                            'Percentage': value * 100,
                            'Period': 'Off-Peak',
                            'Mean Power': analyses[scheduler][f'{metric}_power_stats']['offpeak']['mean_power'][category]
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                # Plot job distribution
                sns.barplot(
                    data=plot_df[plot_df['Period'] == 'Peak'],
                    x='Scheduler',
                    y='Percentage',
                    hue='Category',
                    ax=axes[0]
                )
                axes[0].set_title(f'{month_name.capitalize()} - Peak Hours Distribution - {metric}')
                axes[0].set_ylabel('Percentage of Jobs')
                
                sns.barplot(
                    data=plot_df[plot_df['Period'] == 'Off-Peak'],
                    x='Scheduler',
                    y='Percentage',
                    hue='Category',
                    ax=axes[1]
                )
                axes[1].set_title(f'{month_name.capitalize()} - Off-Peak Hours Distribution - {metric}')
                axes[1].set_ylabel('Percentage of Jobs')
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                
                # Power consumption plot
                fig, ax = plt.subplots(figsize=(15, 8))
                plot_df['Power-Hours'] = plot_df['Percentage'] * plot_df['Mean Power']
                
                sns.barplot(
                    data=plot_df,
                    x='Scheduler',
                    y='Power-Hours',
                    hue='Category',
                    ax=ax
                )
                ax.set_title(f'{month_name.capitalize()} - Power Consumption by Category - {metric}')
                ax.set_ylabel('Power-Hours (kWh)')
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()

def run_monthly_analysis():
    """Runs both general and temporal analysis for multiple months"""
    # Create directories
    results_dir = "monthly_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    year = 2020
    months = [5, 6, 7, 8, 9, 10]
    
    # Initialize temporal analyzer
    temporal_analyzer = TemporalAnalyzer()
    monthly_temporal_results = {}

    for month in months:
        print(f"\n{'='*80}")
        print(f"Running analysis for {datetime(year, month, 1).strftime('%B %Y')}")
        print(f"{'='*80}\n")

        start_date = datetime(year, month, 1)
        end_date = datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
        month_name = start_date.strftime('%B').lower()

        # Run comparison analysis
        results_df = run_comparison(start_date=start_date, end_date=end_date)

        if not results_df.empty:
            # Save comparison results
            csv_filename = f"scheduler_comparison_results_{month_name}.csv"
            csv_path = os.path.join(results_dir, csv_filename)
            results_df.to_csv(csv_path, index=False)
            
            # Move comparison PDF
            if os.path.exists("scheduler_comparison_results.pdf"):
                new_pdf_name = f"scheduler_comparison_results_{month_name}.pdf"
                shutil.move("scheduler_comparison_results.pdf", 
                           os.path.join(results_dir, new_pdf_name))

            # Move scheduler job data files
            for scheduler in ['fcfs', 'enhanced', 'celf', 'sjf']:
                src_file = f"{scheduler}_completed_jobs.csv"
                if os.path.exists(src_file):
                    dst_file = f"{scheduler}_completed_jobs_{month_name}.csv"
                    shutil.move(src_file, os.path.join(results_dir, dst_file))

            # Run temporal analysis for this month
            monthly_analyses = temporal_analyzer.analyze_monthly_temporal_distribution(
                month_name, results_dir)
            monthly_temporal_results[month_name] = monthly_analyses
            
            # Generate temporal analysis visualizations
            temporal_analyzer.plot_monthly_distributions(
                monthly_analyses, month_name, results_dir)

            print(f"Results for {month_name.capitalize()} saved in {results_dir}/")
        else:
            print(f"No results generated for {datetime(year, month, 1).strftime('%B %Y')}")

    # Create combined summary
    print("\nCreating summary of all months...")
    summary_data = []
    
    for month in months:
        month_name = datetime(year, month, 1).strftime('%B').lower()
        csv_path = os.path.join(results_dir, f"scheduler_comparison_results_{month_name}.csv")
        
        if os.path.exists(csv_path):
            try:
                month_df = pd.read_csv(csv_path)
                month_df['month'] = month_name
                summary_data.append(month_df)
            except Exception as e:
                print(f"Error reading {month_name} data: {str(e)}")
    
    if summary_data:
        summary_df = pd.concat(summary_data, ignore_index=True)
        summary_df.to_csv(os.path.join(results_dir, "all_months_summary.csv"), index=False)
        
        # Create temporal analysis summary
        create_temporal_summary(monthly_temporal_results, results_dir)

def create_temporal_summary(monthly_results, results_dir):
    """Create summary visualizations of temporal analysis across all months and save data to CSV"""
    pdf_filename = os.path.join(results_dir, "temporal_analysis_summary.pdf")
    
    # Store all data in a list
    all_data = []
    metrics = ['power_category', 'instant_power_category', 'power_density_category']
    schedulers = ['fcfs', 'enhanced', 'celf', 'sjf']
    
    for metric in metrics:
        # Prepare data for all months
        for month, analyses in monthly_results.items():
            for scheduler in schedulers:
                if scheduler in analyses:
                    # Peak data
                    for category, value in analyses[scheduler][metric]['peak_distribution'].items():
                        all_data.append({
                            'Month': month.capitalize(),
                            'Scheduler': scheduler.upper(),
                            'Category': category,
                            'Job Size': category.split('_')[0],  # Extract first part of category
                            'Metric Type': metric,
                            'Percentage': value * 100,
                            'Period': 'Peak',
                            'Mean Power': analyses[scheduler][f'{metric}_power_stats']['peak']['mean_power'][category],
                            'Jobs/Total Jobs(%)': value * 100
                        })
                    # Off-peak data
                    for category, value in analyses[scheduler][metric]['offpeak_distribution'].items():
                        all_data.append({
                            'Month': month.capitalize(),
                            'Scheduler': scheduler.upper(),
                            'Category': category,
                            'Job Size': category.split('_')[0],  # Extract first part of category
                            'Metric Type': metric,
                            'Percentage': value * 100,
                            'Period': 'Off-Peak',
                            'Mean Power': analyses[scheduler][f'{metric}_power_stats']['offpeak']['mean_power'][category],
                            'Jobs/Total Jobs(%)': value * 100
                        })
    
    # Create DataFrame from all data
    df = pd.DataFrame(all_data)
    
    # Save the full detailed data
    csv_filename = os.path.join(results_dir, "temporal_analysis_summary.csv")
    df.to_csv(csv_filename, index=False)
    
    # Create a pivot table for easier analysis
    pivot_df = df.pivot_table(
        index=['Month', 'Scheduler', 'Job Size', 'Period'],
        values=['Jobs/Total Jobs(%)', 'Mean Power'],
        aggfunc='mean'
    ).reset_index()
    
    # Save the pivot table
    pivot_csv_filename = os.path.join(results_dir, "temporal_analysis_summary_pivot.csv")
    pivot_df.to_csv(pivot_csv_filename, index=False)
    
    with PdfPages(pdf_filename) as pdf:
        metrics = ['power_category', 'instant_power_category', 'power_density_category']
        schedulers = ['fcfs', 'enhanced', 'celf', 'sjf']
        
        for metric in metrics:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Prepare data for all months
            monthly_data = []
            for month, analyses in monthly_results.items():
                for scheduler in schedulers:
                    if scheduler in analyses:
                        # Peak data
                        for category, value in analyses[scheduler][metric]['peak_distribution'].items():
                            monthly_data.append({
                                'Month': month.capitalize(),
                                'Scheduler': scheduler.upper(),
                                'Category': category,
                                'Percentage': value * 100,
                                'Period': 'Peak',
                                'Mean Power': analyses[scheduler][f'{metric}_power_stats']['peak']['mean_power'][category]
                            })
                        # Off-peak data
                        for category, value in analyses[scheduler][metric]['offpeak_distribution'].items():
                            monthly_data.append({
                                'Month': month.capitalize(),
                                'Scheduler': scheduler.upper(),
                                'Category': category,
                                'Percentage': value * 100,
                                'Period': 'Off-Peak',
                                'Mean Power': analyses[scheduler][f'{metric}_power_stats']['offpeak']['mean_power'][category]
                            })
            
            df = pd.DataFrame(monthly_data)
            
            # Plot peak hours summary
            sns.boxplot(
                data=df[df['Period'] == 'Peak'],
                x='Category',
                y='Percentage',
                hue='Scheduler',
                ax=ax1
            )
            ax1.set_title(f'Peak Hours Distribution - {metric}')
            ax1.set_ylabel('Percentage of Jobs')
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot off-peak hours summary
            sns.boxplot(
                data=df[df['Period'] == 'Off-Peak'],
                x='Category',
                y='Percentage',
                hue='Scheduler',
                ax=ax2
            )
            ax2.set_title(f'Off-Peak Hours Distribution - {metric}')
            ax2.set_ylabel('Percentage of Jobs')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # Add power consumption summary
            fig, ax = plt.subplots(figsize=(15, 8))
            df['Power-Hours'] = df['Percentage'] * df['Mean Power']
            
            sns.boxplot(
                data=df,
                x='Category',
                y='Power-Hours',
                hue='Scheduler',
                ax=ax
            )
            ax.set_title(f'Power Consumption by Category - {metric}')
            ax.set_ylabel('Power-Hours (kWh)')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()

if __name__ == "__main__":
    run_monthly_analysis()