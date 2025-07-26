from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from heapq import heappush, heappop

@dataclass
class BaseEvent:
    time: datetime
    priority: int
    event_type: str
    job: Dict[str, Any]
    zone: str

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

class BaseScheduler(ABC):
    """Base class for all scheduler implementations with improved wait time tracking"""
    
    def __init__(
        self,
        zone_config: Dict[str, Dict],
        total_nodes_per_zone: int = 128,
        scheduler_name: str = "base"
    ):
        self.zone_config = zone_config
        self.total_nodes_per_zone = total_nodes_per_zone
        self.scheduler_name = scheduler_name
        
        # Initialize tracking
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all metrics tracking"""
        self.available_nodes = {
            zone: self.total_nodes_per_zone for zone in self.zone_config
        }
        self.event_queue = []
        self.running_jobs = {zone: {} for zone in self.zone_config}
        self.queued_jobs = []
        self.current_time = None
        
        # Enhanced metrics tracking
        self.cost_by_zone = {zone: 0.0 for zone in self.zone_config}
        self.completed_jobs = []
        self.power_usage = []
        self.wait_times = []
        self.queue_length_history = []
        self.zone_utilization_history = []
        
        # Track job queue entry times
        self.job_queue_times = {}  # job_id -> queue_entry_time
    
    def queue_job(self, job: Dict) -> None:
        """Add job to queue with tracking"""
        self.queued_jobs.append(job)
        self.job_queue_times[job['job_id']] = self.current_time
        
        # Record queue state
        self.queue_length_history.append({
            'time': self.current_time,
            'queue_length': len(self.queued_jobs),
            'job_id': job['job_id'],
            'event': 'queued'
        })
    
    def calculate_wait_time(self, job: Dict) -> float:
        """Calculate accurate wait time for a job"""
        if job['job_id'] in self.job_queue_times:
            queue_time = self.job_queue_times[job['job_id']]
            wait_time = (self.current_time - queue_time).total_seconds()
            return max(0, wait_time)
        return 0.0
    
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
    
        # base_scheduler.py
    # Add these methods to the BaseScheduler class

    def debug_log_queue_state(self) -> None:
        """Log current queue state for debugging"""
        print(f"\n[{self.scheduler_name}] Queue State at {self.current_time}:")
        print(f"Queue Length: {len(self.queued_jobs)}")
        print(f"Jobs in queue: {[job['job_id'] for job in self.queued_jobs]}")
        print(f"Available nodes by zone: {self.available_nodes}")

    def debug_log_scheduling_attempt(self, job: Dict, selected_zone: str) -> None:
        """Log scheduling attempt details"""
        print(f"\n[{self.scheduler_name}] Scheduling Job {job['job_id']}:")
        print(f"Time: {self.current_time}")
        print(f"Selected Zone: {selected_zone}")
        print(f"Job Nodes Required: {job['num_nodes_alloc']}")
        if selected_zone:
            print(f"Zone Available Nodes: {self.available_nodes[selected_zone]}")
            print(f"Is Peak Hour: {self.is_peak_hour(self.current_time, selected_zone)}")
            current_power = sum(j['power_kw'] for j in self.running_jobs[selected_zone].values())
            print(f"Current Zone Power: {current_power:.2f} kW")
            print(f"Job Power Required: {job['power_kw']:.2f} kW")

    def queue_job(self, job: Dict) -> None:
        """Enhanced job queuing with debug logging"""
        print(f"\n[{self.scheduler_name}] Queuing Job {job['job_id']}:")
        print(f"Submit Time: {job['submit_time']}")
        print(f"Current Time: {self.current_time}")
        
        self.queued_jobs.append(job)
        self.job_queue_times[job['job_id']] = self.current_time
        
        self.queue_length_history.append({
            'time': self.current_time,
            'queue_length': len(self.queued_jobs),
            'job_id': job['job_id'],
            'event': 'queued'
        })
        
        self.debug_log_queue_state()

    def start_job(self, job: Dict, zone: str) -> None:
        """Enhanced job start with debug logging"""
        print(f"\n[{self.scheduler_name}] Starting Job {job['job_id']} in {zone}:")
        print(f"Queue Entry Time: {self.job_queue_times.get(job['job_id'], 'Not queued')}")
        print(f"Start Time: {self.current_time}")
        
        self.available_nodes[zone] -= job['num_nodes_alloc']
        
        job['start_time'] = self.current_time
        job['end_time'] = self.current_time + timedelta(seconds=job['run_time'])
        
        # Calculate wait time
        wait_time = self.calculate_wait_time(job)
        print(f"Calculated Wait Time: {wait_time:.2f} seconds")
        
        job['wait_time'] = wait_time
        job['assigned_zone'] = zone
        job['scheduler'] = self.scheduler_name
        
        # Calculate power cost
        duration_hours = job['run_time'] / 3600
        rate = self.get_power_rate(self.current_time, zone)
        cost = job['power_kw'] * duration_hours * rate
        
        job['power_cost'] = cost
        job['power_rate'] = rate
        job['is_peak'] = self.is_peak_hour(self.current_time, zone)
        
        # Update metrics
        self.cost_by_zone[zone] += cost
        self.running_jobs[zone][job['job_id']] = job
        self.wait_times.append(wait_time)
        
        # Clean up queue tracking
        if job['job_id'] in self.job_queue_times:
            del self.job_queue_times[job['job_id']]
        
        self.queue_length_history.append({
            'time': self.current_time,
            'queue_length': len(self.queued_jobs),
            'job_id': job['job_id'],
            'event': 'started',
            'zone': zone
        })
        
        # Schedule completion event
        completion_event = BaseEvent(
            job['end_time'], 1, 'complete', job, zone
        )
        heappush(self.event_queue, completion_event)
        
        self.debug_log_queue_state()
    
    def start_job(self, job: Dict, zone: str) -> None:
        """Start job in specified zone with accurate wait time calculation"""
        self.available_nodes[zone] -= job['num_nodes_alloc']
        
        job['start_time'] = self.current_time
        job['end_time'] = self.current_time + timedelta(seconds=job['run_time'])
        
        # Calculate and record wait time
        wait_time = self.calculate_wait_time(job)
        job['wait_time'] = wait_time
        job['assigned_zone'] = zone
        job['scheduler'] = self.scheduler_name
        
        # Calculate power cost
        duration_hours = job['run_time'] / 3600
        rate = self.get_power_rate(self.current_time, zone)
        cost = job['power_kw'] * duration_hours * rate
        
        job['power_cost'] = cost
        job['power_rate'] = rate
        job['is_peak'] = self.is_peak_hour(self.current_time, zone)
        
        # Update metrics
        self.cost_by_zone[zone] += cost
        self.running_jobs[zone][job['job_id']] = job
        self.wait_times.append(wait_time)
        
        # Clean up queue tracking
        if job['job_id'] in self.job_queue_times:
            del self.job_queue_times[job['job_id']]
        
        # Record queue state change
        self.queue_length_history.append({
            'time': self.current_time,
            'queue_length': len(self.queued_jobs),
            'job_id': job['job_id'],
            'event': 'started',
            'zone': zone
        })
        
        # Schedule completion event
        completion_event = BaseEvent(
            job['end_time'], 1, 'complete', job, zone
        )
        heappush(self.event_queue, completion_event)

    
    
    def handle_job_completion(self, job: Dict, zone: str) -> None:
        """Handle job completion in specified zone"""
        if job['job_id'] in self.running_jobs[zone]:
            completed_job = self.running_jobs[zone].pop(job['job_id'])
            self.available_nodes[zone] += completed_job['num_nodes_alloc']
            self.completed_jobs.append(completed_job)
    
    @abstractmethod
    def select_zone(self, job: Dict) -> str:
        """Select zone for job - to be implemented by specific schedulers"""
        pass
    
    def update_metrics(self) -> None:
        """Update metrics at current timestamp"""
        # Zone utilization
        utilization = {}
        power_usage = {}
        for zone in self.zone_config:
            used_nodes = self.total_nodes_per_zone - self.available_nodes[zone]
            utilization[zone] = used_nodes / self.total_nodes_per_zone
            power_usage[zone] = sum(job['power_kw'] for job in self.running_jobs[zone].values())
        
        self.zone_utilization_history.append({
            'time': self.current_time,
            **utilization
        })
        
        self.power_usage.append({
            'time': self.current_time,
            **power_usage
        })
    
    def simulate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Run simulation with provided jobs"""
        self.reset_metrics()
        self.current_time = df['submit_time'].min()
        end_time = df['submit_time'].max() + timedelta(days=1)
        
        # Initialize event queue
        for _, job in df.iterrows():
            event = BaseEvent(
                job['submit_time'], 0, 'submit', job.to_dict(), ''
            )
            heappush(self.event_queue, event)
        
        # Process events
        while self.event_queue and self.current_time < end_time:
            event = heappop(self.event_queue)
            self.current_time = event.time
            
            if event.event_type == 'submit':
                selected_zone = self.select_zone(event.job)
                if selected_zone and self.can_schedule_in_zone(event.job, selected_zone):
                    self.start_job(event.job, selected_zone)
                else:
                    self.queue_job(event.job)
            
            elif event.event_type == 'complete':
                self.handle_job_completion(event.job, event.zone)
                
                # Try to schedule queued jobs
                still_queued = []
                for job in self.queued_jobs:
                    selected_zone = self.select_zone(job)
                    if selected_zone and self.can_schedule_in_zone(job, selected_zone):
                        self.start_job(job, selected_zone)
                    else:
                        still_queued.append(job)
                self.queued_jobs = still_queued
            
            self.update_metrics()
        
        return pd.DataFrame(self.completed_jobs), self._generate_metrics()
    
    def _generate_metrics(self) -> Dict:
        """Generate final metrics for the simulation"""
        return {
            'scheduler_name': self.scheduler_name,
            'total_jobs': len(self.completed_jobs),
            'total_cost': sum(self.cost_by_zone.values()),
            'cost_by_zone': self.cost_by_zone.copy(),
            'zone_metrics': {
                zone: {
                    'total_jobs': len([j for j in self.completed_jobs if j['assigned_zone'] == zone]),
                    'peak_jobs': len([j for j in self.completed_jobs if j['assigned_zone'] == zone and j['is_peak']]),
                    'avg_power': np.mean([log[zone] for log in self.power_usage]) if self.power_usage else 0,
                    'max_power': max([log[zone] for log in self.power_usage]) if self.power_usage else 0,
                    'avg_utilization': np.mean([log[zone] for log in self.zone_utilization_history]) if self.zone_utilization_history else 0
                }
                for zone in self.zone_config
            }
        }