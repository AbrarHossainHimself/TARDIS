from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from heapq import heappush, heappop

@dataclass
class ZoneEvent:
    time: datetime
    priority: int
    event_type: str
    job: Dict[str, Any]
    zone: str

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

class MultiZoneScheduler:
    def __init__(
        self,
        zone_config: Dict[str, Dict],
        total_nodes_per_zone: int = 512
    ):
        self.zone_config = zone_config
        self.total_nodes_per_zone = total_nodes_per_zone
        
        # Initialize per-zone resources
        self.available_nodes = {
            zone: total_nodes_per_zone for zone in zone_config
        }
        
        # Event queue and job tracking
        self.event_queue = []
        self.running_jobs = {zone: {} for zone in zone_config}  # zone -> {job_id -> job}
        self.queued_jobs = []
        self.current_time = None
        
        # Metrics tracking
        self.cost_by_zone = {zone: 0.0 for zone in zone_config}
        self.completed_jobs = []
        self.power_usage = []
        
        # Additional metrics
        self.peak_usage_by_zone = {zone: [] for zone in zone_config}
        self.queue_length_history = []
        self.zone_utilization_history = []
    
    def get_zone_time(self, time: datetime, zone: str) -> datetime:
        """Convert reference time to zone-specific time"""
        offset = self.zone_config[zone]['timezone_offset']
        return time + timedelta(hours=offset)
    
    def is_peak_hour(self, time: datetime, zone: str) -> bool:
        """Check if given time is peak hour in specified zone"""
        zone_time = self.get_zone_time(time, zone)
        zone_config = self.zone_config[zone]
        return zone_config['peak_start'] <= zone_time.hour < zone_config['peak_end']
    
    def get_power_rate(self, time: datetime, zone: str) -> float:
        """Get power rate for specified time and zone"""
        zone_config = self.zone_config[zone]
        if self.is_peak_hour(time, zone):
            return zone_config['base_rate'] * zone_config['peak_rate_multiplier']
        return zone_config['base_rate']
    
    def calculate_zone_score(self, job: Dict, zone: str) -> float:
        """Calculate score for running job in specified zone"""
        power_kw = job['power_kw']
        duration_hours = job['run_time'] / 3600
        
        # Calculate potential cost
        rate = self.get_power_rate(self.current_time, zone)
        potential_cost = power_kw * duration_hours * rate
        cost_factor = 1.0 / (1.0 + potential_cost)
        
        # Consider zone utilization
        utilization = (self.total_nodes_per_zone - self.available_nodes[zone]) / self.total_nodes_per_zone
        utilization_factor = 1.0 - utilization  # Prefer less utilized zones
        
        # Consider peak hours
        peak_penalty = 0.7 if self.is_peak_hour(self.current_time, zone) else 1.0
        
        # Calculate final score
        return (cost_factor * 0.6 + utilization_factor * 0.4) * peak_penalty
    
    def can_schedule_in_zone(self, job: Dict, zone: str) -> bool:
        """Check if job can be scheduled in specified zone"""
        if job['num_nodes_alloc'] > self.available_nodes[zone]:
            return False
        
        if self.is_peak_hour(self.current_time, zone):
            zone_config = self.zone_config[zone]
            if 'power_budget' in zone_config:
                current_power = sum(job['power_kw'] for job in self.running_jobs[zone].values())
                if (current_power + job['power_kw']) > zone_config['power_budget']:
                    return False
        
        return True
    
    def select_best_zone(self, job: Dict) -> str:
        """Select best zone for job based on scoring"""
        best_score = -1
        best_zone = None
        
        for zone in self.zone_config:
            if self.can_schedule_in_zone(job, zone):
                score = self.calculate_zone_score(job, zone)
                if score > best_score:
                    best_score = score
                    best_zone = zone
        
        return best_zone
    
    def start_job(self, job: Dict, zone: str) -> None:
        """Start job in specified zone"""
        self.available_nodes[zone] -= job['num_nodes_alloc']
        
        job['start_time'] = self.current_time
        job['end_time'] = self.current_time + timedelta(seconds=job['run_time'])
        job['wait_time'] = (job['start_time'] - job['submit_time']).total_seconds()
        job['assigned_zone'] = zone
        
        # Calculate power cost
        duration_hours = job['run_time'] / 3600
        rate = self.get_power_rate(self.current_time, zone)
        cost = job['power_kw'] * duration_hours * rate
        
        job['power_cost'] = cost
        job['is_peak'] = self.is_peak_hour(self.current_time, zone)
        
        self.cost_by_zone[zone] += cost
        self.running_jobs[zone][job['job_id']] = job
        
        # Schedule completion event
        completion_event = ZoneEvent(
            job['end_time'], 1, 'complete', job, zone
        )
        heappush(self.event_queue, completion_event)
    
    def handle_job_completion(self, job: Dict, zone: str) -> None:
        """Handle job completion in specified zone"""
        if job['job_id'] in self.running_jobs[zone]:
            completed_job = self.running_jobs[zone].pop(job['job_id'])
            self.available_nodes[zone] += completed_job['num_nodes_alloc']
            self.completed_jobs.append(completed_job)
    
    def schedule_queued_jobs(self) -> None:
        """Attempt to schedule queued jobs across all zones"""
        if not self.queued_jobs:
            return
        
        # Sort queued jobs by priority and wait time
        self.queued_jobs.sort(
            key=lambda j: (
                j['priority'],
                (self.current_time - j['submit_time']).total_seconds()
            ),
            reverse=True
        )
        
        scheduled_jobs = []
        for job in self.queued_jobs:
            best_zone = self.select_best_zone(job)
            if best_zone:
                self.start_job(job, best_zone)
                scheduled_jobs.append(job)
        
        # Remove scheduled jobs from queue
        for job in scheduled_jobs:
            self.queued_jobs.remove(job)
    
    def update_metrics(self) -> None:
        """Update various scheduling metrics"""
        # Update queue length history
        self.queue_length_history.append({
            'time': self.current_time,
            'queue_length': len(self.queued_jobs)
        })
        
        # Update zone utilization
        utilization = {}
        for zone in self.zone_config:
            used_nodes = self.total_nodes_per_zone - self.available_nodes[zone]
            utilization[zone] = used_nodes / self.total_nodes_per_zone
        
        self.zone_utilization_history.append({
            'time': self.current_time,
            **utilization
        })
        
        # Update peak power usage
        for zone in self.zone_config:
            current_power = sum(job['power_kw'] for job in self.running_jobs[zone].values())
            self.peak_usage_by_zone[zone].append({
                'time': self.current_time,
                'power_usage': current_power,
                'is_peak': self.is_peak_hour(self.current_time, zone)
            })
    
    def simulate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Run simulation with provided jobs"""
        self.current_time = df['submit_time'].min()
        end_time = df['submit_time'].max() + timedelta(days=1)
        
        # Initialize event queue with job submissions
        for _, job in df.iterrows():
            event = ZoneEvent(
                job['submit_time'], 0, 'submit', job.to_dict(), ''
            )
            heappush(self.event_queue, event)
        
        # Process events
        while self.event_queue and self.current_time < end_time:
            event = heappop(self.event_queue)
            self.current_time = event.time
            
            # Update metrics
            self.update_metrics()
            
            if event.event_type == 'submit':
                # Try to find best zone for the job
                best_zone = self.select_best_zone(event.job)
                if best_zone:
                    self.start_job(event.job, best_zone)
                else:
                    self.queued_jobs.append(event.job)
            
            elif event.event_type == 'complete':
                self.handle_job_completion(event.job, event.zone)
                self.schedule_queued_jobs()
        
        return self.generate_final_metrics()
    
    def generate_final_metrics(self) -> Tuple[pd.DataFrame, Dict]:
        """Generate final simulation metrics"""
        completed_jobs_df = pd.DataFrame(self.completed_jobs)
        
        metrics = {
            'total_jobs': len(self.completed_jobs),
            'total_cost': sum(self.cost_by_zone.values()),
            'cost_by_zone': self.cost_by_zone.copy(),
            'average_wait_time': completed_jobs_df['wait_time'].mean() if not completed_jobs_df.empty else 0,
            
            # Zone-specific metrics
            'zone_metrics': {
                zone: {
                    'total_jobs': len([j for j in self.completed_jobs if j['assigned_zone'] == zone]),
                    'peak_jobs': len([j for j in self.completed_jobs if j['assigned_zone'] == zone and j['is_peak']]),
                    'avg_power': np.mean([log['power_usage'] for log in self.peak_usage_by_zone[zone]]),
                    'max_power': max([log['power_usage'] for log in self.peak_usage_by_zone[zone]]),
                    'avg_utilization': np.mean([log[zone] for log in self.zone_utilization_history])
                }
                for zone in self.zone_config
            },
            
            # Queue metrics
            'queue_metrics': {
                'max_queue_length': max([log['queue_length'] for log in self.queue_length_history]),
                'avg_queue_length': np.mean([log['queue_length'] for log in self.queue_length_history])
            },
            
            # Time series data
            'queue_history': pd.DataFrame(self.queue_length_history),
            'utilization_history': pd.DataFrame(self.zone_utilization_history),
            'power_usage': {zone: pd.DataFrame(usage) for zone, usage in self.peak_usage_by_zone.items()}
        }
        
        return completed_jobs_df, metrics

# Example usage
if __name__ == "__main__":
    from multizone_generator import MultiZoneConfig, MultiZoneJobGenerator
    
    # Set up configuration
    config = MultiZoneConfig()
    
    # Create scheduler
    scheduler = MultiZoneScheduler(config.zones)
    
    # Generate sample data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 7)
    generator = MultiZoneJobGenerator(start_date, end_date, config)
    jobs_df = generator.generate_dataset()
    
    # Run simulation
    completed_jobs, metrics = scheduler.simulate(jobs_df)
    
    # Print summary
    print(f"\nSimulation completed with {len(completed_jobs)} jobs")
    print("\nCosts by zone:")
    for zone, cost in metrics['cost_by_zone'].items():
        print(f"{zone}: ${cost:,.2f}")
    
    print("\nZone utilization:")
    for zone, zone_metrics in metrics['zone_metrics'].items():
        print(f"\n{zone}:")
        print(f"Total jobs: {zone_metrics['total_jobs']}")
        print(f"Peak jobs: {zone_metrics['peak_jobs']}")
        print(f"Average utilization: {zone_metrics['avg_utilization']:.2%}")