from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List

class MultiZoneConfig:
    """Configuration class for multi-zone HPC setup"""
    
    def __init__(self):
        # Zone configurations with realistic pricing
        self.zones = {
            'us_east': {  # Based on New England rates
                'base_rate': 0.10,        # $/kWh (includes markup from wholesale $0.064/kWh)
                'peak_rate_multiplier': 3,  # Reduced multiplier to be more realistic
                'peak_start': 9,          
                'peak_end': 21,           
                'timezone_offset': 0,      # Reference timezone (EST)
                'power_budget': 1000      # kW
            },
            'us_west': {  # Based on Northern California rates
                'base_rate': 0.12,       # $/kWh (includes markup from wholesale $0.063/kWh)
                'peak_rate_multiplier': 2.5,
                'peak_start': 9,
                'peak_end': 21,
                'timezone_offset': -3,     # 3 hours behind EST
                'power_budget': 1000
            },
            'us_south': {  # Based on Texas rates
                'base_rate': 0.15,       # $/kWh (higher markup from very low wholesale $0.0125/kWh)
                'peak_rate_multiplier': 2,  # Lower multiplier due to more stable grid
                'peak_start': 8,
                'peak_end': 20,
                'timezone_offset': 6,     # 1 hour behind EST
                'power_budget': 1000
            }
        }
        
    def get_zone_time(self, reference_time: datetime, zone: str) -> datetime:
        """Convert reference time to zone-specific time"""
        offset = self.zones[zone]['timezone_offset']
        return reference_time + timedelta(hours=offset)
    
    def is_peak_hour(self, time: datetime, zone: str) -> bool:
        """Check if given time is peak hour in specified zone"""
        zone_time = self.get_zone_time(time, zone)
        zone_config = self.zones[zone]
        return zone_config['peak_start'] <= zone_time.hour < zone_config['peak_end']
    
    def get_power_rate(self, time: datetime, zone: str) -> float:
        """Get power rate for specified time and zone"""
        zone_config = self.zones[zone]
        if self.is_peak_hour(time, zone):
            return zone_config['base_rate'] * zone_config['peak_rate_multiplier']
        return zone_config['base_rate']

class MultiZoneJobGenerator:
    """Generate synthetic HPC jobs with multi-zone awareness"""
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        config: MultiZoneConfig,
        seed: int = 42
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        self.seed = seed
        np.random.seed(seed)
        
        # Use existing job type definitions
        self.job_types = {
            'small_fast': {
                'nodes': (1, 4),
                'runtime': (300, 1800),
                'power': (100, 200),
                'weight': 0.4
            },
            'medium': {
                'nodes': (4, 16),
                'runtime': (1800, 7200),
                'power': (200, 400),
                'weight': 0.3
            },
            'large_long': {
                'nodes': (16, 64),
                'runtime': (7200, 14400),
                'power': (400, 800),
                'weight': 0.2
            },
            'high_power': {
                'nodes': (32, 128),
                'runtime': (3600, 10800),
                'power': (800, 1200),
                'weight': 0.1
            }
        }
    
    def _generate_job(self, submit_time: datetime, job_id: int) -> Dict:
        """Generate a single job with multi-zone awareness"""
        # Select job type
        job_type = np.random.choice(
            list(self.job_types.keys()),
            p=[self.job_types[t]['weight'] for t in self.job_types]
        )
        job_params = self.job_types[job_type]
        
        # Generate base parameters
        num_nodes = np.random.randint(*job_params['nodes'])
        runtime = np.random.uniform(*job_params['runtime'])
        power_per_node = np.random.uniform(*job_params['power'])
        
        # Basic job properties
        job = {
            'job_id': job_id,
            'submit_time': submit_time,
            'num_nodes_alloc': num_nodes,
            'run_time': runtime,
            'mean_node_power': power_per_node,
            'job_type': job_type,
            'priority': np.random.randint(1, 6),
            'power_kw': power_per_node * num_nodes / 1000,
            'estimated_runtime': runtime * np.random.uniform(0.8, 1.2)
        }
        
        return job
    
    def generate_dataset(self, jobs_per_hour: int = 10) -> pd.DataFrame:
        """Generate complete dataset with multi-zone timestamps"""
        jobs = []
        job_id = 1
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Generate jobs for each hour
            submit_times = np.sort(np.random.uniform(0, 3600, jobs_per_hour))
            
            for submit_time in submit_times:
                submit_datetime = current_date + timedelta(seconds=float(submit_time))
                jobs.append(self._generate_job(submit_datetime, job_id))
                job_id += 1
            
            current_date += timedelta(hours=1)
        
        return pd.DataFrame(jobs)

# Example usage
if __name__ == "__main__":
    # Set up configuration and generator
    config = MultiZoneConfig()
    start_date = datetime(2020, 5, 1)
    end_date = datetime(2020, 10, 30)  
    
    generator = MultiZoneJobGenerator(start_date, end_date, config)
    df = generator.generate_dataset()
    
    # Save to CSV
    df.to_csv('multizone_hpc_jobs.csv', index=False)
    print(f"Generated {len(df)} jobs")
    print("\nSample jobs:")
    print(df.head())
    
    # Display rate examples
    test_time = start_date + timedelta(hours=12)  # Noon EST
    print("\nPower rates at", test_time)
    for zone in config.zones:
        rate = config.get_power_rate(test_time, zone)
        is_peak = config.is_peak_hour(test_time, zone)
        print(f"{zone}: ${rate:.3f}/kWh ({'peak' if is_peak else 'off-peak'})")