from datetime import datetime
from multizone_generator import MultiZoneConfig, MultiZoneJobGenerator
from scheduler_comparison import SchedulerComparison
from peak_analysis import run_analysis

def main():
    # Set up configuration
    print("Initializing configuration...")
    config = MultiZoneConfig()
    
    # Create comparison framework
    print("Setting up scheduler comparison...")
    comparison = SchedulerComparison(config.zones)
    
    # Generate sample data for one week with increased job frequency
    print("Generating job data...")
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 7, 23, 59, 59)
    generator = MultiZoneJobGenerator(start_date, end_date, config)
    
    # Increase jobs per hour from default 10 to 50
    jobs_df = generator.generate_dataset(jobs_per_hour=50)
    
    # Run comparison
    print("Running scheduler comparison...")
    results = comparison.run_comparison(jobs_df)
    
    # Extract completed jobs from results for each scheduler
    completed_jobs = {
        name: data['completed_jobs'] 
        for name, data in comparison.results.items()
    }
    
    # Run peak analysis
    print("\nRunning peak hour analysis...")
    run_analysis(config.zones, completed_jobs)
    print("Analysis complete! Check peak_analysis_results.txt for results")

if __name__ == "__main__":
    main()