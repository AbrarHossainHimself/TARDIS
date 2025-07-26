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

class EnhancedPowerAwareScheduler:
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

    def calculate_job_score(self, job: Dict) -> float:
        power_kw = job['mean_node_power'] * job['num_nodes_alloc'] / 1000
        duration_hours = job['run_time'] / 3600
        
        potential_cost = power_kw * duration_hours * self.get_power_rate(self.current_time)
        cost_factor = 1.0 / (1.0 + potential_cost)
        
        power_efficiency = 1.0 / (job['mean_node_power'] + 1)
        
        wait_time = (self.current_time - job['submit_time']).total_seconds()
        wait_factor = min(wait_time / 3600, 24)
        
        priority_weight = job['priority'] / 5.0 if 'priority' in job else 1.0
        
        return (
            cost_factor * 0.4 +
            power_efficiency * 0.3 +
            (wait_factor / 24.0) * 0.2 +
            priority_weight * 0.1
        )

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
        
        jobs = jobs.sort_values('power_kw', ascending=False)
        
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
                self._schedule_queued_jobs()
        
        return self._generate_metrics()

    def _handle_submission(self, job: Dict) -> None:
        if self.can_schedule_job(job):
            self._start_job(job)
        else:
            self.queued_jobs.append(job)
            self._schedule_queued_jobs()

    def _handle_completion(self, job: Dict) -> None:
        if job['job_id'] in self.running_jobs:
            completed_job = self.running_jobs.pop(job['job_id'])
            self.available_nodes += completed_job['num_nodes_alloc']
            self.completed_jobs.append(completed_job)

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

    def _schedule_queued_jobs(self) -> None:
        if not self.queued_jobs:
            return
            
        is_peak = self.is_peak_hour(self.current_time)
        current_power = self.calculate_current_power()
        
        scored_jobs = []
        for job in self.queued_jobs:
            # Calculate base score regardless of power budget
            base_score = self.calculate_job_score(job)
            
            # Apply scoring penalties based on current conditions
            power_kw = job['power_kw']
            
            # Always consider peak/off-peak periods, even without power budget
            if is_peak:
                # Penalize high-power jobs during peak hours
                if power_kw > 50:
                    base_score *= 0.7
                
                # Additional penalty for very high power jobs
                if power_kw > 100:
                    base_score *= 0.8
            else:
                # Slightly boost high-power jobs during off-peak
                if power_kw > 50:
                    base_score *= 1.2
            
            # Consider current system power load
            if current_power > 500:  # High system load
                if power_kw > 50:
                    base_score *= 0.9
            
            scored_jobs.append((base_score, job))
        
        scored_jobs.sort(key=lambda x: x[0], reverse=True)
        
        scheduled_jobs = []
        for _, job in scored_jobs:
            if self.can_schedule_job(job):
                self._start_job(job)
                scheduled_jobs.append(job)
                current_power = self.calculate_current_power()
        
        for job in scheduled_jobs:
            self.queued_jobs.remove(job)
        
        # Consider delaying high-power jobs during peak hours
        if is_peak:
            next_offpeak = self.current_time.replace(hour=self.peak_end, minute=0, second=0)
            if next_offpeak <= self.current_time:
                next_offpeak += timedelta(days=1)
            
            delayed_jobs = []
            for job in self.queued_jobs:
                # Delay high-power jobs to off-peak even without strict budget
                if job['power_kw'] > 75:  # Threshold for delaying jobs
                    event = Event(next_offpeak, 0, 'submit', job)
                    heappush(self.event_queue, event)
                    delayed_jobs.append(job)
            
            for job in delayed_jobs:
                self.queued_jobs.remove(job)

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