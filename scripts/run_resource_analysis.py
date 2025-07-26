#!/usr/bin/env python3

from datetime import datetime
from multizone_generator import MultiZoneConfig, MultiZoneJobGenerator
from scheduler_comparison import SchedulerComparison
from resource_utilization_analysis import ResourceUtilizationAnalyzer

def main():
    print("Initializing configuration...")
    config = MultiZoneConfig()
    
    # Create comparison framework
    print("Setting up scheduler comparison...")
    comparison = SchedulerComparison(config.zones)
    
    # Generate sample data - 1 week of jobs
    print("Generating job data...")
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 7, 23, 59, 59)
    generator = MultiZoneJobGenerator(start_date, end_date, config)
    jobs_df = generator.generate_dataset()
    
    # Run comparison
    print("Running scheduler comparison...")
    results = comparison.run_comparison(jobs_df)
    
    # Extract completed jobs from results for each scheduler
    completed_jobs = {
        name: data['completed_jobs'] 
        for name, data in comparison.results.items()
    }
    
    # Run resource utilization analysis
    print("\nRunning resource utilization analysis...")
    analyzer = ResourceUtilizationAnalyzer(
        completed_jobs,
        config.zones,
        total_nodes=512  # This should match your scheduler configuration
    )
    
    # Analyze temporal utilization
    print("Analyzing temporal utilization patterns...")
    temporal_results = analyzer.analyze_temporal_utilization()
    
    print("\nAnalysis complete! Results have been written to: utilization_analysis_results.txt")
    print("Review the file for detailed utilization patterns and statistics.")

if __name__ == "__main__":
    main()