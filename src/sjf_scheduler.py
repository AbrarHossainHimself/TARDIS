import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from heapq import heappush, heappop
from dataclasses import dataclass
from typing import Any

@dataclass
class Event:
    time: datetime
    priority: int
    event_type: str
    job: Dict[str, Any]

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

class SJFScheduler:
    def __init__(
        self,
        total_nodes: int = 512,
        peak_power_budget: float = None,  # Added power budget parameter
        base_rate: float = 0.12,     # $/kWh
        peak_rate_multiplier: float = 3,
        peak_start: int = 6,
        peak_end: int = 22
    ):
        self.total_nodes = total_nodes
        self.available_nodes = total_nodes
        self.peak_power_budget = peak_power_budget  # Store power budget
        self.base_rate = base_rate
        self.peak_rate = base_rate * peak_rate_multiplier
        self.peak_start = peak_start
        self.peak_end = peak_end

        # Event queue for efficient time advancement
        self.event_queue = []
        self.running_jobs = {}  # job_id -> job_dict
        self.queued_jobs = []   # List to hold queued jobs
        self.current_time = None

        # Metrics tracking
        self.total_cost = 0
        self.peak_cost = 0
        self.offpeak_cost = 0
        self.completed_jobs = []
        self.power_usage = []

    def is_peak_hour(self, time: datetime) -> bool:
        return self.peak_start <= time.hour < self.peak_end

    def get_power_rate(self, time: datetime) -> float:
        return self.peak_rate if self.is_peak_hour(time) else self.base_rate

    def calculate_current_power(self) -> float:
        if not self.running_jobs:
            return 0.0
        return sum(job['power_kw'] for job in self.running_jobs.values())

    def can_schedule_job(self, job: Dict) -> bool:
        """Check if job can be scheduled considering both node and power availability"""
        # First check node availability
        if job['num_nodes_alloc'] > self.available_nodes:
            return False

        # Then check power constraints if a budget is set
        if self.peak_power_budget and self.is_peak_hour(self.current_time):
            current_power = self.calculate_current_power()
            power_kw = job['power_kw']
            return (current_power + power_kw) <= self.peak_power_budget

        return True

    def simulate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Run SJF scheduling simulation"""
        # Preprocess jobs
        jobs = df.copy()
        jobs['power_kw'] = jobs['mean_node_power'] * jobs['num_nodes_alloc'] / 1000

        # Sort jobs by submission time initially to maintain some order for event processing
        jobs = jobs.sort_values('submit_time')

        # Initialize simulation
        self.current_time = jobs['submit_time'].min()
        end_time = jobs['submit_time'].max() + timedelta(days=1)

        # Add initial job submissions to event queue
        for _, job in jobs.iterrows():
            heappush(self.event_queue, Event(job['submit_time'], 0, 'submit', job.to_dict()))

        # Main simulation loop
        while self.event_queue and self.current_time < end_time:
            event = heappop(self.event_queue)
            self.current_time = event.time

            # Log power usage
            current_power = self.calculate_current_power()
            self.power_usage.append({
                'time': self.current_time,
                'power_usage': current_power,
                'is_peak': self.is_peak_hour(self.current_time),
                'num_running': len(self.running_jobs)
            })

            if event.event_type == 'submit':
                self._handle_submission(event.job)
            elif event.event_type == 'complete':
                self._handle_completion(event.job)
                self._schedule_queued_jobs_sjf() # Schedule queued jobs after completion

        # Calculate metrics
        return self._generate_metrics()

    def _handle_submission(self, job: Dict) -> None:
        """Handle job submission with SJF scheduling"""
        self.queued_jobs.append(job) # Add job to queue
        self._schedule_queued_jobs_sjf() # Attempt to schedule immediately

    def _handle_completion(self, job: Dict) -> None:
        """Handle job completion"""
        if job['job_id'] in self.running_jobs:
            completed_job = self.running_jobs.pop(job['job_id'])
            self.available_nodes += completed_job['num_nodes_alloc']
            self.completed_jobs.append(completed_job)
            self._schedule_queued_jobs_sjf() # Attempt to schedule after completion

    def _start_job(self, job: Dict) -> None:
        """Start a job and record its metrics"""
        self.available_nodes -= job['num_nodes_alloc']

        job['start_time'] = self.current_time
        job['end_time'] = self.current_time + timedelta(seconds=job['run_time'])
        job['wait_time'] = (job['start_time'] - job['submit_time']).total_seconds()

        # Calculate power cost
        duration_hours = job['run_time'] / 3600
        rate = self.get_power_rate(self.current_time)
        cost = job['power_kw'] * duration_hours * rate

        job['power_cost'] = cost
        job['is_peak'] = self.is_peak_hour(self.current_time)

        if job['is_peak']:
            self.peak_cost += cost
        else:
            self.offpeak_cost += cost

        self.running_jobs[job['job_id']] = job
        heappush(self.event_queue, Event(job['end_time'], 2, 'complete', job))

    def _schedule_queued_jobs_sjf(self) -> None:
        """Schedule queued jobs based purely on Shortest Job First (SJF), only respecting power budget as a constraint"""
        if not self.queued_jobs:
            return

        # Sort queued jobs by runtime (shortest first) - pure SJF behavior
        self.queued_jobs.sort(key=lambda job: job['run_time'])

        scheduled_jobs = []
        remaining_jobs = []

        # Try to schedule jobs in SJF order, only checking resource constraints
        for job in self.queued_jobs:
            if self.can_schedule_job(job):  # Only checks nodes and power budget
                self._start_job(job)
                scheduled_jobs.append(job)
            else:
                remaining_jobs.append(job)

        # Update queue with jobs that couldn't be scheduled
        self.queued_jobs = remaining_jobs

    def _generate_metrics(self) -> Tuple[pd.DataFrame, Dict]:
        """Generate final metrics and summary with safe initialization"""
        completed_jobs_df = pd.DataFrame(self.completed_jobs)
        
        # Initialize default stats
        default_stats = {
            'avg': 0,
            'median': 0,
            'max': 0,
            'min': 0
        }
        
        wait_stats = {
            'overall': default_stats.copy(),
            'peak': default_stats.copy(),
            'offpeak': default_stats.copy()
        }
        
        if not completed_jobs_df.empty:
            peak_jobs = completed_jobs_df[completed_jobs_df['is_peak']]
            offpeak_jobs = completed_jobs_df[~completed_jobs_df['is_peak']]
            
            # Update wait stats only if we have completed jobs
            wait_stats.update({
                'overall': {
                    'avg': float(completed_jobs_df['wait_time'].mean()),
                    'median': float(completed_jobs_df['wait_time'].median()),
                    'max': float(completed_jobs_df['wait_time'].max()),
                    'min': float(completed_jobs_df['wait_time'].min())
                },
                'peak': {
                    'avg': float(peak_jobs['wait_time'].mean()) if not peak_jobs.empty else 0,
                    'median': float(peak_jobs['wait_time'].median()) if not peak_jobs.empty else 0,
                    'max': float(peak_jobs['wait_time'].max()) if not peak_jobs.empty else 0,
                    'min': float(peak_jobs['wait_time'].min()) if not peak_jobs.empty else 0
                },
                'offpeak': {
                    'avg': float(offpeak_jobs['wait_time'].mean()) if not offpeak_jobs.empty else 0,
                    'median': float(offpeak_jobs['wait_time'].median()) if not offpeak_jobs.empty else 0,
                    'max': float(offpeak_jobs['wait_time'].max()) if not offpeak_jobs.empty else 0,
                    'min': float(offpeak_jobs['wait_time'].min()) if not offpeak_jobs.empty else 0
                }
            })
            
            try:
                completed_jobs_df['power_category'] = pd.qcut(
                    completed_jobs_df['power_kw'],
                    q=3,
                    labels=['low', 'medium', 'high']
                )
                wait_by_power = completed_jobs_df.groupby('power_category', observed=True)['wait_time'].agg([
                    'mean', 'median', 'max', 'min', 'count'
                ]).to_dict('index')
            except Exception as e:
                print(f"Warning: Could not calculate power categories: {str(e)}")
                wait_by_power = {}
        else:
            wait_by_power = {}
        
        # Safely calculate power usage statistics
        peak_power_usage = [log['power_usage'] for log in self.power_usage if log['is_peak']]
        offpeak_power_usage = [log['power_usage'] for log in self.power_usage if not log['is_peak']]
        
        summary = {
            'total_jobs': len(self.completed_jobs),
            'peak_jobs': sum(1 for job in self.completed_jobs if job['is_peak']),
            'offpeak_jobs': sum(1 for job in self.completed_jobs if not job['is_peak']),
            'total_cost': float(self.peak_cost + self.offpeak_cost),
            'peak_cost': float(self.peak_cost),
            'offpeak_cost': float(self.offpeak_cost),
            'max_power': float(max(log['power_usage'] for log in self.power_usage)) if self.power_usage else 0,
            'avg_power': float(np.mean([log['power_usage'] for log in self.power_usage])) if self.power_usage else 0,
            'peak_max_power': float(max(peak_power_usage)) if peak_power_usage else 0,
            'peak_avg_power': float(np.mean(peak_power_usage)) if peak_power_usage else 0,
            'offpeak_max_power': float(max(offpeak_power_usage)) if offpeak_power_usage else 0,
            'offpeak_avg_power': float(np.mean(offpeak_power_usage)) if offpeak_power_usage else 0,
            'wait_stats': wait_stats,
            'wait_by_power': wait_by_power,
            'power_usage': pd.DataFrame(self.power_usage)
        }
        
        return completed_jobs_df, summary