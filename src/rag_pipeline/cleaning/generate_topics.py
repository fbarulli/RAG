
from pathlib import Path
import json
from rag_pipeline.cleaning.core.paths import Paths
from rag_pipeline.cleaning.core.logging import get_logger
from rag_pipeline.cleaning.topic_test.schemas import TopicAssignments

logger = get_logger(__name__)

class TopicPipeline:
    """Clean, end-to-end topic assignment pipeline."""
    
    def __init__(self):
        self.paths = Paths()
    
    def run(self, force: bool = False) -> TopicAssignments:
        """Main entry point - produces final topic_assignments_all.json"""
        print("🔄 Running Topic Pipeline...")
        
        
        assignments = self._merge_existing()
        assignments.save()
        
        print(f"Topic pipeline finished → {self.paths.topic_assignments()}")
        return assignments
    
    def _merge_existing(self) -> TopicAssignments:
        """Merge per-model files into final format"""
        exp_dir = self.paths.experiments_dir() / "topics"
        model_files = list(exp_dir.glob("topic_assignments_*.json"))
        
        if not model_files:
            logger.warning("No topic model files found. Creating empty skeleton.")
            return TopicAssignments(models=[], results={})
        
        results = {}
        models = []
        
        for f in model_files:
            if "all.json" in f.name:
                continue
            with open(f) as fh:
                data = json.load(fh)
            model_name = data.get("metadata", {}).get("model", f.stem)
            models.append(model_name)
            results[model_name] = data
        
        return TopicAssignments(models=models, results=results)
