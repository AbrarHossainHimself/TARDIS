import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class HPCJobGenerator:
    def __init__(self, start_date: datetime, end_date: datetime, seed: int = 42):
        self.start_date = start_date
        self.end_date = end_date
        self.seed = seed
        np.random.seed(seed)

        # Base job types (same as before)
        self.job_types = {
            'small_fast': {
                'nodes': (1, 4),
                'runtime': (300, 1800),
                'power': (100, 200),
                'weight': 0.4,
                'priority_dist': [0.1, 0.2, 0.4, 0.2, 0.1]
            },
            'medium': {
                'nodes': (4, 16),
                'runtime': (1800, 7200),
                'power': (200, 400),
                'weight': 0.3,
                'priority_dist': [0.05, 0.15, 0.6, 0.15, 0.05]
            },
            'large_long': {
                'nodes': (16, 64),
                'runtime': (7200, 14400),
                'power': (400, 800),
                'weight': 0.2,
                'priority_dist': [0.02, 0.08, 0.3, 0.4, 0.2]
            },
            'high_power': {
                'nodes': (32, 128),
                'runtime': (3600, 10800),
                'power': (800, 1200),
                'weight': 0.1,
                'priority_dist': [0.01, 0.04, 0.15, 0.4, 0.4]
            }
        }

        # Monthly variations in job type distributions
        self.monthly_job_weights = {
            5: {'small_fast': 0.45, 'medium': 0.3, 'large_long': 0.15, 'high_power': 0.1},    # May
            6: {'small_fast': 0.35, 'medium': 0.35, 'large_long': 0.2, 'high_power': 0.1},    # June
            7: {'small_fast': 0.25, 'medium': 0.3, 'large_long': 0.25, 'high_power': 0.2},    # July
            8: {'small_fast': 0.3, 'medium': 0.25, 'large_long': 0.25, 'high_power': 0.2},    # August
            9: {'small_fast': 0.4, 'medium': 0.3, 'large_long': 0.2, 'high_power': 0.1},      # September
            10: {'small_fast': 0.45, 'medium': 0.35, 'large_long': 0.15, 'high_power': 0.05}  # October
        }

        # Monthly power scaling factors (accounting for cooling overhead)
        self.monthly_power_factors = {
            5: 1.0,    # May - baseline
            6: 1.15,   # June - increasing cooling needs
            7: 1.25,   # July - peak cooling needs
            8: 1.2,    # August - high cooling needs
            9: 1.1,    # September - decreasing cooling needs
            10: 1.0    # October - back to baseline
        }

        # Department monthly activity factors
        self.department_monthly_factors = {
            'research': {
                5: 1.2,    # May (high activity - end of semester)
                6: 0.8,    # June (lower - summer break)
                7: 0.7,    # July (summer break)
                8: 0.9,    # August (preparation)
                9: 1.3,    # September (start of semester)
                10: 1.1    # October (normal activity)
            },
            'engineering': {
                5: 1.0,    # Relatively stable throughout
                6: 1.1,
                7: 1.2,
                8: 1.1,
                9: 1.0,
                10: 0.9
            },
            'production': {
                5: 0.9,    # Production varies with business cycle
                6: 1.0,
                7: 1.2,
                8: 1.3,
                9: 1.1,
                10: 1.0
            }
        }

        # Keep original hourly patterns
        self.hourly_patterns = {
            'weekday': {
                0: 0.3, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.4,
                6: 0.6, 7: 0.8, 8: 1.0, 9: 1.2, 10: 1.3, 11: 1.2,
                12: 1.0, 13: 1.1, 14: 1.2, 15: 1.1, 16: 1.0, 17: 0.9,
                18: 0.7, 19: 0.6, 20: 0.5, 21: 0.4, 22: 0.3, 23: 0.3
            },
            'weekend': {
                0: 0.2, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,
                6: 0.3, 7: 0.4, 8: 0.5, 9: 0.6, 10: 0.7, 11: 0.7,
                12: 0.6, 13: 0.6, 14: 0.5, 15: 0.5, 16: 0.4, 17: 0.4,
                18: 0.3, 19: 0.3, 20: 0.2, 21: 0.2, 22: 0.2, 23: 0.2
            }
        }

        # Modified department base configuration
        self.departments = {
            'research': {
                'base_priority': [0.05, 0.15, 0.5, 0.2, 0.1],
                'weight': 0.4
            },
            'engineering': {
                'base_priority': [0.05, 0.1, 0.4, 0.3, 0.15],
                'weight': 0.3
            },
            'production': {
                'base_priority': [0.02, 0.08, 0.3, 0.4, 0.2],
                'weight': 0.3
            }
        }

    def _get_submission_count(self, date: datetime) -> int:
        """Calculate number of job submissions for a given date with monthly variations"""
        is_weekend = date.weekday() >= 5
        pattern = self.hourly_patterns['weekend' if is_weekend else 'weekday']
        
        # Base number of jobs varies by month
        month = date.month
        month_factor = {
            5: 1.0,    # May (baseline)
            6: 0.9,    # June (slight decrease)
            7: 0.8,    # July (summer lull)
            8: 1.1,    # August (ramp up)
            9: 1.3,    # September (peak)
            10: 1.2    # October (high)
        }[month]
        
        base_jobs = (6 if is_weekend else 10) * month_factor

        # Monthly variation (more jobs during middle of month)
        day_of_month = date.day
        monthly_factor = 1.0 + 0.2 * np.sin(np.pi * day_of_month / 30)

        # Get hourly factor
        hourly_factor = pattern[date.hour]

        # Add some randomness
        random_factor = np.random.normal(1.0, 0.1)

        return int(base_jobs * hourly_factor * monthly_factor * random_factor)

    def _generate_job(self, submit_time: datetime, job_id: int) -> Dict:
        """Generate a single job with realistic characteristics and monthly variations"""
        month = submit_time.month
        
        # Select department with monthly-adjusted weights
        dept_weights = {
            dept: self.departments[dept]['weight'] * self.department_monthly_factors[dept][month]
            for dept in self.departments
        }
        # Normalize weights
        total_weight = sum(dept_weights.values())
        dept_weights = {k: v/total_weight for k, v in dept_weights.items()}
        
        department = np.random.choice(
            list(dept_weights.keys()),
            p=list(dept_weights.values())
        )

        # Select job type based on monthly distribution
        job_type = np.random.choice(
            list(self.monthly_job_weights[month].keys()),
            p=list(self.monthly_job_weights[month].values())
        )
        job_params = self.job_types[job_type]

        # Generate base job parameters
        num_nodes = np.random.randint(*job_params['nodes'])
        runtime = np.random.uniform(*job_params['runtime'])
        
        # Apply monthly power scaling
        base_power = np.random.uniform(*job_params['power'])
        power_per_node = base_power * self.monthly_power_factors[month]

        # Rest of the job generation logic remains similar...
        cores_per_node = 16
        if job_type == 'small_fast':
            cores_per_task = np.random.choice([1, 2, 4])
        elif job_type == 'medium':
            cores_per_task = np.random.choice([2, 4, 8])
        else:
            cores_per_task = np.random.choice([8, 16])

        power_profile = {
            'startup': power_per_node * np.random.uniform(0.7, 0.9),
            'compute': power_per_node,
            'cooldown': power_per_node * np.random.uniform(0.4, 0.6)
        }

        has_dependency = np.random.choice([True, False], p=[0.15, 0.85])
        dependency_job_id = job_id - np.random.randint(1, 10) if has_dependency and job_id > 10 else None

        # Adjust priority based on department and month
        dept_info = self.departments[department]
        dept_priority = np.random.choice(range(1, 6), p=dept_info['base_priority'])
        job_priority = np.random.choice(range(1, 6), p=job_params['priority_dist'])
        final_priority = max(dept_priority, job_priority)

        return {
            'job_id': job_id,
            'submit_time': submit_time,
            'num_nodes_alloc': num_nodes,
            'run_time': runtime,
            'mean_node_power': power_per_node,
            'power_profile': power_profile,
            'cores_per_task': cores_per_task,
            'cores_per_node': cores_per_node,
            'shared': np.random.choice([0, 1], p=[0.7, 0.3]),
            'priority': final_priority,
            'job_type': job_type,
            'department': department,
            'dependency_job_id': dependency_job_id,
            'memory_gb': num_nodes * np.random.uniform(4, 32),
            'state_reason': np.random.choice(
                ['None', 'TimeLimit', 'NodeFail'],
                p=[0.92, 0.05, 0.03]
            ),
            'estimated_runtime': runtime * np.random.uniform(0.8, 1.2),
            'qos_level': np.random.choice(['standard', 'high', 'urgent'],
                                        p=[0.7, 0.2, 0.1])
        }

    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete dataset"""
        jobs = []
        job_id = 1

        current_date = self.start_date
        while current_date <= self.end_date:
            num_jobs = self._get_submission_count(current_date)

            submit_times = np.sort(np.random.uniform(
                0,
                3600,
                num_jobs
            ))

            for submit_time in submit_times:
                submit_datetime = current_date + timedelta(seconds=float(submit_time))
                jobs.append(self._generate_job(submit_datetime, job_id))
                job_id += 1

            current_date += timedelta(hours=1)

        df = pd.DataFrame(jobs)

        df['actual_runtime'] = df.apply(
            lambda x: min(x['run_time'],
                        x['run_time'] * np.random.uniform(0.8, 1.1) if x['state_reason'] == 'TimeLimit'
                        else x['run_time']),
            axis=1
        )

        return df.sort_values('submit_time')
    

# Example Usage (moved to simulator.py)
if __name__ == "__main__":
    start_date = datetime(2020, 5, 1)
    end_date = datetime(2020, 10, 31)
    generator = HPCJobGenerator(start_date, end_date)
    df = generator.generate_dataset()

    # Save dataset to file
    df.to_csv('synthetic_hpc_jobs_2020.csv', index=False)
    print(df.head())