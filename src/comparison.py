from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from job_generator import HPCJobGenerator
from enhanced_scheduler import EnhancedPowerAwareScheduler
from fcfs_scheduler import FCFSScheduler
from backfill_celf import CELFPowerAwareScheduler
from sjf_scheduler import SJFScheduler
from gnn_predictor import GNNPowerPredictor, prepare_jobs_with_gnn_power
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.simplefilter("ignore", UserWarning)


def plot_results(results_df, simulation_start, simulation_end, pdf_filename="scheduler_comparison_results.pdf"):
    """
    Generates plots from the results DataFrame and saves them to a PDF.
    """
    pp = PdfPages(pdf_filename)

    # Original plots
    scheduler_names = ['FCFS', 'Enhanced', 'CELF', 'SJF']
    
    # Calculate percentages for x-axis
    baseline_power = results_df[results_df['power_budget'].isna()]['fcfs_peak_power'].iloc[0]
    power_budgets = results_df['power_budget'].tolist()
    percentages = ['No limit'] + [f'{(pb/baseline_power)*100:.1f}%' for pb in power_budgets[1:]]

    # Plot total cost
    plt.figure(figsize=(10, 6))
    for scheduler in scheduler_names:
        cost_col = f"{scheduler.lower()}_total_cost"
        plt.plot(percentages, results_df[cost_col], marker='o', linestyle='-', label=scheduler)
    plt.xlabel("Power Budget (% of Peak Power)")
    plt.ylabel("Total Cost ($)")
    plt.title("Total Cost Comparison Across Schedulers")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    pp.savefig()
    plt.close()

    # Plot average wait time
    plt.figure(figsize=(10, 6))
    for scheduler in scheduler_names:
        wait_col = f"{scheduler.lower()}_avg_wait"
        plt.plot(percentages, results_df[wait_col], marker='o', linestyle='-', label=scheduler)
    plt.xlabel("Power Budget (% of Peak Power)")
    plt.ylabel("Average Wait Time (s)")
    plt.title("Average Wait Time Comparison Across Schedulers")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    pp.savefig()
    plt.close()

    # Plot completed jobs
    plt.figure(figsize=(10, 6))
    for scheduler in scheduler_names:
        completed_col = f"{scheduler.lower()}_completed"
        plt.plot(percentages, results_df[completed_col], marker='o', linestyle='-', label=scheduler)
    plt.xlabel("Power Budget (% of Peak Power)")
    plt.ylabel("Completed Jobs")
    plt.title("Number of Completed Jobs Comparison Across Schedulers")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    pp.savefig()
    plt.close()

    ratios = analyze_power_budget_ratios(results_df, simulation_start, simulation_end)
    plot_power_budget_ratios(ratios, pp)
    plot_throughput(results_df, ratios, pp)

    pp.close()
    print(f"Plots saved to {pdf_filename}")
    return ratios

def calculate_throughput(completed_jobs_df, start_time, end_time):
    """
    Calculate throughput metrics from time range
    """
    if completed_jobs_df.empty:
        return {
            'jobs_per_hour': 0,
            'total_runtime_hours': 0,
            'avg_concurrent_jobs': 0
        }
    
    # Calculate total runtime in hours
    total_runtime = (end_time - start_time).total_seconds() / 3600
    
    # Calculate jobs per hour
    jobs_per_hour = len(completed_jobs_df) / total_runtime if total_runtime > 0 else 0
    
    # Calculate average concurrent jobs (simplified)
    avg_concurrent_jobs = total_runtime * jobs_per_hour / 24  # Average over a day
    
    return {
        'jobs_per_hour': jobs_per_hour,
        'total_runtime_hours': total_runtime,
        'avg_concurrent_jobs': avg_concurrent_jobs
    }

def plot_throughput(results_df, ratios, pdf):
    """
    Create throughput visualization
    """
    plt.figure(figsize=(12, 6))
    
    x = ['No limit', '75%', '50%', '25%']  # Percentages including unlimited
    scheduler_names = ['FCFS', 'Enhanced', 'CELF', 'SJF']
    
    # Plot jobs per hour for each scheduler
    for scheduler in scheduler_names:
        y = [r['metrics'][scheduler]['jobs_per_hour'] for r in ratios]
        plt.plot(x, y, marker='o', label=f'{scheduler}')
    
    plt.xlabel('Power Budget (% of Peak Power)')
    plt.ylabel('Jobs Completed per Hour')
    plt.title('Throughput Comparison Across Schedulers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Plot average concurrent jobs
    plt.figure(figsize=(12, 6))
    for scheduler in scheduler_names:
        y = [r['metrics'][scheduler]['avg_concurrent_jobs'] for r in ratios]
        plt.plot(x, y, marker='o', label=f'{scheduler}')
    
    plt.xlabel('Power Budget (% of Peak Power)')
    plt.ylabel('Average Concurrent Jobs')
    plt.title('Average Concurrent Jobs Across Schedulers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

def display_throughput_metrics(ratios):
    """
    Display throughput metrics in a formatted table
    """
    print("\nThroughput Analysis")
    print("=" * 100)
    scheduler_names = ['FCFS', 'Enhanced', 'CELF', 'SJF']
    
    # Header
    print(f"{'Power Budget':<15} | ", end="")
    for scheduler in scheduler_names:
        print(f"{scheduler + ' (jobs/hr)':<15} | {scheduler + ' (concurrent)':<15} | ", end="")
    print()
    print("-" * 100)
    
    # Data rows
    budgets = ['Unlimited', '75%', '50%', '25%']
    for budget, ratio in zip(budgets, ratios):
        print(f"{budget:<15} | ", end="")
        for scheduler in scheduler_names:
            metrics = ratio['metrics'][scheduler]
            print(f"{metrics['jobs_per_hour']:>14.2f} | {metrics['avg_concurrent_jobs']:>14.2f} | ", end="")
        print()

def analyze_power_budget_ratios(results_df, simulation_start, simulation_end):
    """
    Analyze the ratios of different metrics under power budgets compared to no budget
    """
    # Get baseline values (no power budget)
    baseline = results_df[results_df['power_budget'].isna()].iloc[0]
    simulation_hours = (simulation_end - simulation_start).total_seconds() / 3600
    
    # Calculate ratios for each power budget
    ratios = []
    power_budgets = [None] + [baseline['fcfs_peak_power'] * p for p in [0.75, 0.50, 0.25]]
    
    for budget in power_budgets:
        if budget is None:
            row = baseline
        else:
            row = results_df[results_df['power_budget'] == budget].iloc[0]
            
        budget_ratios = {
            'power_budget': budget,
            'power_budget_ratio': budget / baseline['fcfs_peak_power'] if budget else 1.0,
            'metrics': {}
        }
        
        # Calculate throughput for each scheduler
        schedulers = ['FCFS', 'Enhanced', 'CELF', 'SJF']
        for scheduler in schedulers:
            # Get completed jobs and wait time
            completed_jobs = float(row[f'{scheduler.lower()}_completed'])
            avg_wait_time = float(row[f'{scheduler.lower()}_avg_wait'])
            
            # Calculate throughput metrics
            jobs_per_hour = completed_jobs / simulation_hours
            
            # Calculate concurrent jobs based on Little's Law:
            # L = λW, where L is avg number in system, λ is arrival rate, W is avg time in system
            avg_wait_hours = avg_wait_time / 3600  # convert seconds to hours
            avg_concurrent_jobs = jobs_per_hour * avg_wait_hours
            
            budget_ratios['metrics'][scheduler] = {
                'total_cost_ratio': row[f'{scheduler.lower()}_total_cost'] / baseline[f'{scheduler.lower()}_total_cost'],
                'completed_jobs_ratio': completed_jobs / baseline[f'{scheduler.lower()}_completed'],
                'peak_power_ratio': row[f'{scheduler.lower()}_peak_power'] / baseline[f'{scheduler.lower()}_peak_power'],
                'jobs_per_hour': jobs_per_hour,
                'avg_concurrent_jobs': avg_concurrent_jobs,
                'completed_jobs': completed_jobs,  # Add for verification
                'avg_wait_hours': avg_wait_hours  # Add for verification
            }
        
        ratios.append(budget_ratios)
    
    # Display detailed metrics for verification
    print("\nDetailed Throughput Analysis")
    print("=" * 120)
    print(f"Simulation Duration: {simulation_hours:.2f} hours")
    print("-" * 120)
    
    for ratio in ratios:
        budget = ratio['power_budget']
        print(f"\nPower Budget: {'Unlimited' if budget is None else f'{budget:.2f} kW'}")
        for scheduler, metrics in ratio['metrics'].items():
            print(f"\n{scheduler}:")
            print(f"  Completed Jobs: {metrics['completed_jobs']:.0f}")
            print(f"  Avg Wait Time: {metrics['avg_wait_hours']:.2f} hours")
            print(f"  Jobs/Hour: {metrics['jobs_per_hour']:.2f}")
            print(f"  Avg Concurrent: {metrics['avg_concurrent_jobs']:.2f}")
    
    # Display throughput comparison table
    display_throughput_metrics(ratios)
    
    return ratios


def plot_power_budget_ratios(ratios, pdf):
    """
    Create visualizations of power budget ratios
    """
    plt.figure(figsize=(12, 6))
    
    # Extract x values (percentages)
    x = [75, 50, 25]  # Percentages we're using
    
    # Plot completed jobs ratio
    for scheduler in ['FCFS', 'Enhanced', 'CELF', 'SJF']:
        y = [r['metrics'][scheduler]['completed_jobs_ratio'] for r in ratios[1:]]  # Skip 100%
        plt.plot(x, y, marker='o', label=f'{scheduler}')
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='Baseline')
    plt.xlabel('Power Budget (% of Peak Power)')
    plt.ylabel('Completed Jobs Ratio')
    plt.title('Ratio of Completed Jobs vs Unconstrained')
    plt.legend()
    plt.grid(True, alpha=0.3)
    pdf.savefig()
    plt.close()

    # Plot total cost ratio
    plt.figure(figsize=(12, 6))
    for scheduler in ['FCFS', 'Enhanced', 'CELF', 'SJF']:
        y = [r['metrics'][scheduler]['total_cost_ratio'] for r in ratios[1:]]  # Skip 100%
        plt.plot(x, y, marker='o', label=f'{scheduler}')
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='Baseline')
    plt.xlabel('Power Budget (% of Peak Power)')
    plt.ylabel('Total Cost Ratio')
    plt.title('Ratio of Total Cost vs Unconstrained')
    plt.legend()
    plt.grid(True, alpha=0.3)
    pdf.savefig()
    plt.close()

    # Plot peak power ratio
    plt.figure(figsize=(12, 6))
    for scheduler in ['FCFS', 'Enhanced', 'CELF', 'SJF']:
        y = [r['metrics'][scheduler]['peak_power_ratio'] for r in ratios[1:]]  # Skip 100%
        plt.plot(x, y, marker='o', label=f'{scheduler}')
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='Baseline')
    plt.xlabel('Power Budget (% of Peak Power)')
    plt.ylabel('Peak Power Ratio')
    plt.title('Ratio of Peak Power vs Unconstrained')
    plt.legend()
    plt.grid(True, alpha=0.3)
    pdf.savefig()
    plt.close()

global start_date, end_date  # Add this line at the start of the function

def run_comparison(start_date=None, end_date=None):
    # # 1. Generate test data
    # print("Generating test data...")
    # start_date = datetime(2020, 9, 1)
    # end_date = datetime(2020, 9, 30)

    # generator = HPCJobGenerator(start_date, end_date)
    # # df = generator.generate_dataset()
    # df = pd.read_csv('/home/abrar/Desktop/Code/Temporal HPC/synthetic_hpc_jobs.csv')

    # print(f"\nGenerated {len(df)} jobs")
    # print(f"Date range: {df['submit_time'].min()} to {df['submit_time'].max()}")

    """
    Run comparison analysis for a specific date range.
    
    Args:
        start_date (datetime): Start date for analysis
        end_date (datetime): End date for analysis
    """
    print("Loading test data...")
    df = pd.read_csv('/home/abrar/Desktop/Code/Temporal HPC/hpc_simulator/synthetic_hpc_jobs_2020.csv')
    
    # Convert date columns to datetime
    date_columns = ['submit_time', 'start_time', 'end_time']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Use provided dates if available, otherwise use defaults
    if start_date is None:
        start_date = datetime(2020, 5, 1)
    if end_date is None:
        end_date = datetime(2020, 10, 30)
    
    print(f"\nAnalyzing period: {start_date} to {end_date}")
    
    # Filter data for the specified date range
    df = df[
        (df['submit_time'] >= start_date) & 
        (df['submit_time'] <= end_date)
    ].copy()
    
    print(f"\nLoaded {len(df)} jobs")
    print(f"Date range: {df['submit_time'].min()} to {df['submit_time'].max()}")
    print(f"Number of unique days: {df['submit_time'].dt.date.nunique()}")

    # Safety check
    if len(df) == 0:
        print("No jobs found in the specified date range!")
        return pd.DataFrame()  # Return empty DataFrame instead of continuing

    # Initialize results list
    results = []
    
    # Run initial simulation to determine peak power
    print("\nRunning initial simulation to determine peak power...")
    initial_scheduler = FCFSScheduler(peak_power_budget=None)
    _, initial_summary = initial_scheduler.simulate(df)
    peak_power = initial_summary['peak_max_power']
    print(f"Peak power without constraints: {peak_power:.2f} kW")
    
    # Define power budgets
    power_budgets = [None] + [peak_power * (p/100) for p in [75, 50, 25]]
    
    for budget in power_budgets:
        print(f"\nTesting with power budget: {budget if budget else 'Unlimited'} kW")
        
        # Initialize schedulers
        scheduler_fcfs = FCFSScheduler(peak_power_budget=budget)
        scheduler_enhanced = EnhancedPowerAwareScheduler(peak_power_budget=budget)
        scheduler_celf = CELFPowerAwareScheduler(peak_power_budget=budget)
        scheduler_sjf = SJFScheduler(peak_power_budget=budget)
        
        # Run simulations and safely handle wait time calculations
        metrics_fcfs, summary_fcfs = scheduler_fcfs.simulate(df)
        metrics_enhanced, summary_enhanced = scheduler_enhanced.simulate(df)
        metrics_celf, summary_celf = scheduler_celf.simulate(df)
        metrics_sjf, summary_sjf = scheduler_sjf.simulate(df)

        # Save completed jobs data for each scheduler
        metrics_fcfs.to_csv('fcfs_completed_jobs.csv', index=False)
        metrics_celf.to_csv('enhanced_completed_jobs.csv', index=False) 
        metrics_enhanced.to_csv('celf_completed_jobs.csv', index=False)
        metrics_sjf.to_csv('sjf_completed_jobs.csv', index=False)
        
        # Safe calculation of improvements
        fcfs_wait_time = summary_fcfs['wait_stats']['overall'].get('avg', 0)
        enhanced_wait_time = summary_enhanced['wait_stats']['overall'].get('avg', 0)
        celf_wait_time = summary_celf['wait_stats']['overall'].get('avg', 0)
        sjf_wait_time = summary_sjf['wait_stats']['overall'].get('avg', 0)
        
        fcfs_total_cost = summary_fcfs['total_cost']
        
        # Calculate improvements with safe division
        def calculate_improvement(new_value, base_value):
            if base_value == 0:
                return 0
            return ((base_value - new_value) / base_value * 100)
        
        cost_improvement_enhanced = calculate_improvement(
            summary_enhanced['total_cost'], fcfs_total_cost)
        cost_improvement_celf = calculate_improvement(
            summary_celf['total_cost'], fcfs_total_cost)
        cost_improvement_sjf = calculate_improvement(
            summary_sjf['total_cost'], fcfs_total_cost)
        
        wait_improvement_enhanced = calculate_improvement(
            enhanced_wait_time, fcfs_wait_time)
        wait_improvement_celf = calculate_improvement(
            celf_wait_time, fcfs_wait_time)
        wait_improvement_sjf = calculate_improvement(
            sjf_wait_time, fcfs_wait_time)
        
        # Store results
        results.append({
            'power_budget': budget,
            'simulation_start': start_date,
            'simulation_end': end_date,
            'fcfs_total_cost': summary_fcfs['total_cost'],
            'enhanced_total_cost': summary_enhanced['total_cost'],
            'celf_total_cost': summary_celf['total_cost'],
            'sjf_total_cost': summary_sjf['total_cost'],
            'fcfs_avg_wait': fcfs_wait_time,
            'enhanced_avg_wait': enhanced_wait_time,
            'celf_avg_wait': celf_wait_time,
            'sjf_avg_wait': sjf_wait_time,
            'fcfs_completed': summary_fcfs['total_jobs'],
            'enhanced_completed': summary_enhanced['total_jobs'],
            'celf_completed': summary_celf['total_jobs'],
            'sjf_completed': summary_sjf['total_jobs'],
            'fcfs_peak_power': summary_fcfs['peak_max_power'],
            'enhanced_peak_power': summary_enhanced['peak_max_power'],
            'celf_peak_power': summary_celf['peak_max_power'],
            'sjf_peak_power': summary_sjf['peak_max_power'],
            'cost_improvement_enhanced': cost_improvement_enhanced,
            'wait_improvement_enhanced': wait_improvement_enhanced,
            'cost_improvement_celf': cost_improvement_celf,
            'wait_improvement_celf': wait_improvement_celf,
            'cost_improvement_sjf': cost_improvement_sjf,
            'wait_improvement_sjf': wait_improvement_sjf
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('scheduler_comparison_results.csv', index=False)
    print("\nResults saved to scheduler_comparison_results.csv")
    
    # Plot results
    plot_results(results_df, start_date, end_date)
    
    return results_df


if __name__ == "__main__":
    results = run_comparison()