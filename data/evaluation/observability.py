# observability.py

import json
import os
from datetime import datetime
from typing import Dict, Any, List

class GAObservability:
    def __init__(self, run_id: str = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history = []
        
    def record_generation(self, gen_summary: Dict[str, Any]):
        """
        Records a summary of a single generation.
        """
        self.history.append(gen_summary)
        print(f"[Observability] Recorded Gen {gen_summary['gen']} summary.")

    def save_report(self, output_dir: str = "reports"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        report_path = os.path.join(output_dir, f"ga_report_{self.run_id}.json")
        with open(report_path, "w") as f:
            json.dump(self.history, f, indent=4)
        print(f"[Observability] Full GA report saved to {report_path}")
        return report_path

    def get_summary(self) -> str:
        if not self.history:
            return "No GA history recorded."
            
        last = self.history[-1]
        summary = f"GA Run {self.run_id} Summary:\n"
        summary += f"- Generations: {len(self.history)}\n"
        summary += f"- Best Final Fitness: {last['best_fitness']:.4f}\n"
        summary += f"- Top Model: {last['best_model']}\n"
        return summary
    
# In-memory short-term store
class MemoryStore:
    _storage = {}
    
    @classmethod
    def store(cls, key: str, value: Any):
        cls._storage[key] = value
        
    @classmethod
    def retrieve(cls, key: str):
        return cls._storage.get(key)
