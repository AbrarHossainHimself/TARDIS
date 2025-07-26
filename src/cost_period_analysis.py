# cost_period_analysis.py

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def load_monthly_results(results_dir="monthly_results"):
    """Load results from all months into a single DataFrame"""
    all_results = []
    results_path = Path(results_dir)
    
    for csv_file in results_path.glob("scheduler_comparison_results_*.csv"):
        if "all_months" not in str(csv_file):
            df = pd.read_csv(csv_file)
            month = csv_file.stem.split('_')[-1]
            df['month'] = month
            all_results.append(df)
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

def calculate_costs_and_ratios(results_df):
    """Calculate cost per job and peak hour cost percentage"""
    schedulers = ['fcfs', 'enhanced', 'celf', 'sjf']
    analysis = {scheduler: {} for scheduler in schedulers}
    
    # Calculate total simulation costs
    total_simulation_costs = {
        scheduler: results_df[f'{scheduler}_total_cost'].sum() 
        for scheduler in schedulers
    }
    
    # For each month and scheduler
    for month in results_df['month'].unique():
        month_data = results_df[results_df['month'] == month]
        
        for scheduler in schedulers:
            # Get relevant data
            total_cost = month_data[f'{scheduler}_total_cost'].iloc[0]
            completed_jobs = month_data[f'{scheduler}_completed'].iloc[0]
            peak_power = month_data[f'{scheduler}_peak_power'].iloc[0]
            
            # Initialize scheduler dict for this month if not exists
            if month not in analysis[scheduler]:
                analysis[scheduler][month] = {}
            
            # Calculate metrics
            cost_per_job = total_cost / completed_jobs if completed_jobs > 0 else 0
            
            # Assuming peak hours are 6am-10pm (16 hours) and multiplier is 3
            peak_hours = 16
            off_peak_hours = 24 - peak_hours
            peak_rate_multiplier = 3
            
            # Calculate theoretical maximum peak cost percentage
            theoretical_max = (peak_hours * peak_rate_multiplier) / (peak_hours * peak_rate_multiplier + off_peak_hours)
            
            # Calculate actual peak cost based on power usage
            actual_peak_cost = (peak_power * peak_hours * peak_rate_multiplier) / total_cost if total_cost > 0 else 0
            peak_cost_percentage = actual_peak_cost * 100
            
            # Store results
            analysis[scheduler][month].update({
                'total_cost': total_cost,
                'cost_per_job': cost_per_job,
                'peak_power': peak_power,
                'peak_cost_percentage': peak_cost_percentage
            })
    
    # Calculate averages across all months
    for scheduler in schedulers:
        avg_cost_per_job = np.mean([analysis[scheduler][m]['cost_per_job'] for m in analysis[scheduler]])
        avg_peak_percentage = np.mean([analysis[scheduler][m]['peak_cost_percentage'] for m in analysis[scheduler]])
        analysis[scheduler]['average'] = {
            'avg_cost_per_job': avg_cost_per_job,
            'avg_peak_percentage': avg_peak_percentage
        }
    
    # Save detailed results
    with open('cost_job_peak_analysis.txt', 'w') as f:
        f.write("Comprehensive Cost Analysis for Power-Aware HPC Scheduling\n")
        f.write("====================================================\n\n")
        
        # 0. Total Simulation Costs
        f.write("1. Total Simulation Costs (May-October 2020)\n")
        f.write("----------------------------------------\n")
        fcfs_total = total_simulation_costs['fcfs']
        for scheduler in schedulers:
            f.write(f"\n{scheduler.upper()}:")
            f.write(f"\nTotal Cost: ${total_simulation_costs[scheduler]:,.2f}")
            if scheduler != 'fcfs':
                savings = fcfs_total - total_simulation_costs[scheduler]
                savings_percent = (savings / fcfs_total) * 100
                f.write(f"\nSavings vs FCFS: ${savings:,.2f} ({savings_percent:.2f}%)")
            f.write("\n")
        
        # 2. Cost per Job Analysis
        f.write("\n2. Cost per Job Analysis\n")
        f.write("-----------------------\n")
        for scheduler in schedulers:
            f.write(f"\n{scheduler.upper()}:\n")
            f.write(f"Average Cost per Job: ${analysis[scheduler]['average']['avg_cost_per_job']:.2f}\n")
            f.write("Monthly Breakdown:\n")
            for month in sorted(analysis[scheduler].keys()):
                if month != 'average':
                    f.write(f"  {month.capitalize()}: ${analysis[scheduler][month]['cost_per_job']:.2f}\n")
        
        # 3. Peak Hour Cost Analysis
        f.write("\n3. Peak Hour Cost Analysis\n")
        f.write("-------------------------\n")
        f.write("\nPercentage of costs incurred during peak hours:\n")
        for scheduler in schedulers:
            f.write(f"\n{scheduler.upper()}:\n")
            f.write(f"Average Peak Hour Cost: {analysis[scheduler]['average']['avg_peak_percentage']:.2f}%\n")
            f.write("Monthly Breakdown:\n")
            for month in sorted(analysis[scheduler].keys()):
                if month != 'average':
                    f.write(f"  {month.capitalize()}: {analysis[scheduler][month]['peak_cost_percentage']:.2f}%\n")
    
    return analysis, total_simulation_costs

def main():
    # Load results
    results_df = load_monthly_results()
    
    if results_df.empty:
        print("No results found!")
        return
    
    # Calculate detailed analysis
    analysis, total_costs = calculate_costs_and_ratios(results_df)
    print("Analysis complete. Results saved to cost_job_peak_analysis.txt")

def calculate_detailed_costs(results_df):
    """Calculate detailed cost metrics including cost improvements and efficiency"""
    schedulers = ['fcfs', 'enhanced', 'celf', 'sjf']
    analysis = {scheduler: {} for scheduler in schedulers}
    
    # Calculate aggregate metrics across all months
    for scheduler in schedulers:
        total_cost = results_df[f'{scheduler}_total_cost'].mean()
        completed_jobs = results_df[f'{scheduler}_completed'].mean()
        peak_power = results_df[f'{scheduler}_peak_power'].mean()
        
        # Calculate base metrics
        analysis[scheduler].update({
            'total_cost': total_cost,
            'cost_per_job': total_cost / completed_jobs if completed_jobs > 0 else 0,
            'peak_power': peak_power,
            'completed_jobs': completed_jobs,
        })
        
        # Calculate efficiency metrics
        analysis[scheduler]['cost_per_watt'] = total_cost / peak_power if peak_power > 0 else 0
        
    # Calculate improvements relative to FCFS
    fcfs_baseline = analysis['fcfs']['total_cost']
    for scheduler in schedulers:
        if scheduler != 'fcfs':
            cost_improvement = ((fcfs_baseline - analysis[scheduler]['total_cost']) / fcfs_baseline * 100)
            analysis[scheduler]['cost_improvement'] = cost_improvement
    
    # Monthly trends
    monthly_trends = {}
    for month in results_df['month'].unique():
        month_data = results_df[results_df['month'] == month]
        monthly_trends[month] = {
            scheduler: {
                'total_cost': month_data[f'{scheduler}_total_cost'].iloc[0],
                'cost_improvement': month_data[f'cost_improvement_{scheduler}'].iloc[0] if scheduler != 'fcfs' else 0
            }
            for scheduler in schedulers
        }
    
    # Save detailed results
    with open('cost_analysis_results.txt', 'w') as f:
        f.write("Cost Analysis Results for Power-Aware HPC Scheduling\n")
        f.write("================================================\n\n")
        
        # 1. Overall Cost Comparison
        f.write("1. Overall Cost Comparison\n")
        f.write("--------------------------\n")
        for scheduler in schedulers:
            f.write(f"\n{scheduler.upper()}:\n")
            f.write(f"Total Cost: ${analysis[scheduler]['total_cost']:,.2f}\n")
            f.write(f"Cost per Job: ${analysis[scheduler]['cost_per_job']:.2f}\n")
            if scheduler != 'fcfs':
                f.write(f"Cost Improvement vs FCFS: {analysis[scheduler]['cost_improvement']:.2f}%\n")
        
        # 2. Efficiency Metrics
        f.write("\n2. Efficiency Metrics\n")
        f.write("--------------------\n")
        for scheduler in schedulers:
            f.write(f"\n{scheduler.upper()}:\n")
            f.write(f"Peak Power: {analysis[scheduler]['peak_power']:.2f} kW\n")
            f.write(f"Cost per kW: ${analysis[scheduler]['cost_per_watt']:.2f}\n")
        
        # 3. Monthly Trends
        f.write("\n3. Monthly Cost Trends\n")
        f.write("---------------------\n")
        for month in sorted(monthly_trends.keys()):
            f.write(f"\n{month.upper()}:\n")
            for scheduler in schedulers:
                f.write(f"{scheduler.upper()}: ${monthly_trends[month][scheduler]['total_cost']:,.2f}")
                if scheduler != 'fcfs':
                    f.write(f" (Change: {monthly_trends[month][scheduler]['cost_improvement']:.2f}%)")
                f.write("\n")
    
    return analysis, monthly_trends

def create_visualizations(results_df, analysis, monthly_trends):
    """Create publication-quality visualizations"""
    # plt.style.use('seaborn')
    
    # 1. Monthly Cost Trends
    plt.figure(figsize=(12, 6))
    months_order = ['may', 'june', 'july', 'august', 'september', 'october']
    schedulers = ['fcfs', 'enhanced', 'celf', 'sjf']
    scheduler_colors = {'fcfs': 'gray', 'enhanced': 'blue', 'celf': 'green', 'sjf': 'red'}
    
    for scheduler in schedulers:
        costs = [monthly_trends[month][scheduler]['total_cost'] for month in months_order]
        plt.plot(months_order, costs, marker='o', label=scheduler.upper(), color=scheduler_colors[scheduler])
    
    plt.xlabel('Month')
    plt.ylabel('Total Cost ($)')
    plt.title('Monthly Cost Comparison Across Schedulers')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('monthly_costs.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Cost Improvements Bar Plot
    plt.figure(figsize=(10, 6))
    schedulers = ['enhanced', 'celf', 'sjf']
    improvements = [analysis[s]['cost_improvement'] for s in schedulers]
    colors = ['blue', 'green', 'red']
    
    plt.bar(schedulers, improvements, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Scheduler')
    plt.ylabel('Cost Improvement vs FCFS (%)')
    plt.title('Overall Cost Improvement Comparison')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cost_improvements.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Efficiency Metrics Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cost per Job
    cost_per_job = [analysis[s]['cost_per_job'] for s in schedulers]
    ax1.bar(schedulers, cost_per_job, color=colors)
    ax1.set_xlabel('Scheduler')
    ax1.set_ylabel('Cost per Job ($)')
    ax1.set_title('Cost per Job Comparison')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Cost per kW
    cost_per_kw = [analysis[s]['cost_per_watt'] for s in schedulers]
    ax2.bar(schedulers, cost_per_kw, color=colors)
    ax2.set_xlabel('Scheduler')
    ax2.set_ylabel('Cost per kW ($)')
    ax2.set_title('Cost per kW Comparison')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('efficiency_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_statistical_analysis(results_df):
    """Perform statistical analysis on the results"""
    schedulers = ['enhanced', 'celf', 'sjf']
    stats_results = {}
    
    # Perform t-tests comparing each scheduler with FCFS
    fcfs_costs = results_df['fcfs_total_cost']
    
    for scheduler in schedulers:
        scheduler_costs = results_df[f'{scheduler}_total_cost']
        t_stat, p_value = stats.ttest_rel(fcfs_costs, scheduler_costs)
        
        # Calculate effect size (Cohen's d)
        d = np.mean(fcfs_costs - scheduler_costs) / np.std(fcfs_costs - scheduler_costs)
        
        stats_results[scheduler] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': d
        }
    
    # Save statistical analysis results
    with open('statistical_analysis.txt', 'w') as f:
        f.write("Statistical Analysis Results\n")
        f.write("==========================\n\n")
        
        for scheduler in schedulers:
            f.write(f"{scheduler.upper()} vs FCFS:\n")
            f.write(f"t-statistic: {stats_results[scheduler]['t_statistic']:.4f}\n")
            f.write(f"p-value: {stats_results[scheduler]['p_value']:.4f}\n")
            f.write(f"Cohen's d: {stats_results[scheduler]['cohens_d']:.4f}\n")
            f.write("\n")
    
    return stats_results

def main():
    # Load results
    results_df = load_monthly_results()
    
    if results_df.empty:
        print("No results found!")
        return
    
    # Calculate detailed analysis
    analysis = calculate_costs_and_ratios(results_df)
    print("Analysis complete. Results saved to cost_job_peak_analysis.txt")

if __name__ == "__main__":
    main()