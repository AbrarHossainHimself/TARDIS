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

class CELFPowerAwareScheduler: # Renamed class to CELF Power Aware
    def __init__(
        self,
        total_nodes: int = 512,
        peak_power_budget: float = None,  # in kW
        peak_start: int = 6,
        peak_end: int = 22,
        base_rate: float = 0.12,     # $/kWh
        peak_rate_multiplier: float = 3
    ):
        self.total_nodes = total_nodes
        self.available_nodes = total_nodes
        self.peak_power_budget = peak_power_budget
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.base_rate = base_rate
        self.peak_rate = base_rate * peak_rate_multiplier

        # Event queue for efficient time advancement
        self.event_queue = []
        self.running_jobs = {}  # job_id -> job_dict
        self.queued_jobs = []   # List of jobs waiting to be scheduled
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
        if job['num_nodes_alloc'] > self.available_nodes:
            return False

        if self.is_peak_hour(self.current_time) and self.peak_power_budget:
            current_power = self.calculate_current_power()
            power_kw = job['power_kw']
            return (current_power + power_kw) <= self.peak_power_budget

        return True

    def simulate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        jobs = df.copy()
        jobs['power_kw'] = jobs['mean_node_power'] * jobs['num_nodes_alloc'] / 1000

        jobs = jobs.sort_values('submit_time') # CELF is often used with FCFS queue order

        self.current_time = jobs['submit_time'].min()
        end_time = jobs['submit_time'].max() + timedelta(days=1)

        for _, job in jobs.iterrows():
            event = Event(job['submit_time'], 0, 'submit', job.to_dict())
            heappush(self.event_queue, event)

        while self.event_queue and self.current_time < end_time:
            event = heappop(self.event_queue)
            self.current_time = event.time

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
                self._schedule_queued_jobs_celf() # Call CELF backfill scheduler

        return self._generate_metrics()

    def _handle_submission(self, job: Dict) -> None:
        self.queued_jobs.append(job) # Just add to queue upon submission
        self._schedule_queued_jobs_celf() # Try to schedule immediately


    def _handle_completion(self, job: Dict) -> None:
        if job['job_id'] in self.running_jobs:
            completed_job = self.running_jobs.pop(job['job_id'])
            self.available_nodes += completed_job['num_nodes_alloc']
            self.completed_jobs.append(completed_job)
            self._schedule_queued_jobs_celf() # Try to schedule new jobs after completion


    def _start_job(self, job: Dict) -> None:
        self.available_nodes -= job['num_nodes_alloc']

        job['start_time'] = self.current_time
        job['end_time'] = self.current_time + timedelta(seconds=job['run_time'])
        job['wait_time'] = (job['start_time'] - job['submit_time']).total_seconds()

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
        completion_event = Event(job['end_time'], 1, 'complete', job)
        heappush(self.event_queue, completion_event)


    def _schedule_queued_jobs_celf(self) -> None:
        """Schedule jobs using pure CELF (Conservative Easy backFiLling) strategy"""
        if not self.queued_jobs:
            return

        # Sort queued jobs by submission time to maintain FCFS base order
        self.queued_jobs.sort(key=lambda job: job['submit_time'])

        # Try to schedule the head of queue job
        head_of_queue_job = self.queued_jobs[0]
        if self.can_schedule_job(head_of_queue_job):
            self._start_job(head_of_queue_job)
            self.queued_jobs.pop(0)
        else:
            return  # If HOQ job can't be scheduled, no backfilling possible

        # Attempt backfilling with remaining jobs
        remaining_jobs = list(self.queued_jobs)  # Create a copy for iteration
        scheduled_backfill = []

        for candidate_job in remaining_jobs:
            # Check if job can be backfilled
            if (self.can_schedule_job(candidate_job) and 
                self.can_backfill_job_celf(candidate_job, head_of_queue_job)):
                self._start_job(candidate_job)
                scheduled_backfill.append(candidate_job)

        # Remove scheduled backfill jobs from queue
        self.queued_jobs = [job for job in self.queued_jobs 
                            if job not in scheduled_backfill]


    def can_backfill_job_celf(self, backfill_job, hoq_job) -> bool:
        """
        Pure CELF backfilling check - only considers job size and runtime
        """
        if not hoq_job:
            return True

        # Basic CELF checks for conservative backfilling
        if backfill_job['num_nodes_alloc'] >= hoq_job['num_nodes_alloc']:
            return False  # Backfill job must be smaller than HOQ job

        # Check if backfill job will finish before HOQ job could potentially start
        backfill_runtime = backfill_job['run_time']
        hoq_runtime = hoq_job['run_time']
        
        return backfill_runtime < hoq_runtime


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