from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Step:
    step_id: int
    description: str
    energy: str
    location: str
    minutes: int

@dataclass
class Example:
    user_profile: Dict[str, Any]
    goal_description: str
    plan: List[Step]
    nudges: List[str]