# run_cost_analysis.py

from datetime import datetime
from multizone_generator import MultiZoneConfig, MultiZoneJobGenerator
from scheduler_comparison import SchedulerComparison
from cost_benefit_analysis import run_analysis

def main():
    # Set up configuration
    print("Initializing configuration...")
    config = MultiZoneConfig()
    
    # Create comparison framework
    print("Setting up scheduler comparison...")
    comparison = SchedulerComparison(config.zones)
    
    # Generate sample data
    print("Generating job data...")
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 30, 23, 59, 59)  # One week of data
    generator = MultiZoneJobGenerator(start_date, end_date, config)
    jobs_df = generator.generate_dataset(jobs_per_hour=10)
    
    # Run comparison
    print("Running scheduler comparison...")
    results = comparison.run_comparison(jobs_df)
    
    # Run our new cost benefit analysis
    print("\nRunning cost benefit analysis...")
    run_analysis(comparison.results)

if __name__ == "__main__":
    main()