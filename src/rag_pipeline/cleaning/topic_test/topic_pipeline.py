from pathlib import Path
import json
from rag_pipeline.core.paths import Paths
from rag_pipeline.logging import get_logger
from rag_pipeline.core.schemas import TopicAssignments

logger = get_logger(__name__)

class TopicTestPipeline:
    """Isolated topic pipeline for testing - does NOT touch production files."""
    
    def __init__(self):
        self.paths = Paths()
        self.test_dir = Path("src/rag_pipeline/cleaning/topic_test")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.test_dir / "topic_assignments_all_test.json"
    
    def run(self) -> TopicAssignments:
        """Run test pipeline"""
        print("🔄 Running TEST Topic Pipeline (isolated)...")
        
        assignments = self._merge_existing()
        assignments.save(self.output_file)   # Save to test location only
        
        print(f"✅ Test topic file created: {self.output_file}")
        print(f"   Models found: {len(assignments.models)}")
        return assignments
    
    def _merge_existing(self) -> TopicAssignments:
        """Safe merge - looks in original locations but writes to test folder"""
        # Look for existing topic files
        possible_dirs = [
            self.paths.experiments_dir() / "topics",
            Path("src/rag_pipeline/eda/p02_eda/experiments"),
            Path("experiments"),
        ]
        
        model_files = []
        for d in possible_dirs:
            if d.exists():
                model_files.extend(list(d.glob("topic_assignments_*.json")))
        
        model_files = [f for f in model_files if "all.json" not in f.name]
        
        if not model_files:
            logger.warning("No existing topic files found. Creating minimal test file.")
            return TopicAssignments(models=[], results={})
        
        results = {}
        models = []
        
        for f in model_files:
            with open(f) as fh:
                data = json.load(fh)
            model_name = data.get("metadata", {}).get("model", f.stem)
            models.append(model_name)
            results[model_name] = data
        
        return TopicAssignments(models=models, results=results)
