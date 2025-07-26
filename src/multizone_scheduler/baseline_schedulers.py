# baseline_schedulers.py
# Implementation of baseline scheduling strategies for comparison

from typing import Dict
from datetime import datetime, timedelta
import numpy as np
from base_scheduler import BaseScheduler

class RandomZoneScheduler(BaseScheduler):
    """Scheduler that randomly assigns jobs to zones"""
    
    def __init__(self, zone_config: Dict[str, Dict], total_nodes_per_zone: int = 512):
        super().__init__(zone_config, total_nodes_per_zone, "random")
        self.zones = list(zone_config.keys())
    
    def select_zone(self, job: Dict) -> str:
        """Randomly select a zone"""
        np.random.shuffle(self.zones)
        for zone in self.zones:
            if self.can_schedule_in_zone(job, zone):
                return zone
        return None

class SingleZoneScheduler(BaseScheduler):
    """Scheduler that only uses one specified zone"""
    
    def __init__(
        self,
        zone_config: Dict[str, Dict],
        zone: str,
        total_nodes_per_zone: int = 512
    ):
        super().__init__(zone_config, total_nodes_per_zone, f"single_{zone}")
        self.target_zone = zone
    
    def select_zone(self, job: Dict) -> str:
        """Always select the specified zone"""
        return self.target_zone if self.can_schedule_in_zone(job, self.target_zone) else None
    
    def get_zone_time(self, time: datetime, zone: str) -> datetime:
        """Convert reference time to zone-specific time"""
        if zone != self.target_zone:
            return time
        return super().get_zone_time(time, zone)
    
    def is_peak_hour(self, time: datetime, zone: str) -> bool:
        """Check if given time is peak hour in specified zone"""
        if zone != self.target_zone:
            return False
        return super().is_peak_hour(time, zone)