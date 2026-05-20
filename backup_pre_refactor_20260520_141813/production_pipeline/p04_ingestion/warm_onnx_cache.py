"""
production_pipeline/p04_ingestion/warm_onnx_cache.py

Utility script to pre-download and compile all cross-encoder models 
defined in rerankers.json into the ONNX cache.
"""

import logging
import sys
import traceback
from ._onnx_bench_engine import load_matrix_configs
from ._onnx_model_loader import ONNXModelLoader

# Set up clean terminal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("warm_onnx_cache")


def main():
    logger.info("Initializing ONNX Cache Warmer...")
    
    try:
        # 1. Load the models we want to target
        reranker_entries = load_matrix_configs()
        logger.info(f"Found {len(reranker_entries)} models to pre-cache.")
        
        # 2. Iterate and trigger the export process for each model
        for idx, entry in enumerate(reranker_entries, start=1):
            name = entry["name"]
            hf_path = entry["model"]
            
            logger.info(f"\n======================================== [{idx}/{len(reranker_entries)}]")
            logger.info(f"Targeting: {name}")
            logger.info(f"Hub Path:  {hf_path}")
            logger.info("========================================")
            
            # Instantiating the loader checks the cache automatically.
            # If missing, it downloads and exports to 'experiments/onnx_cache'
            ONNXModelLoader(model_name=hf_path, provider="CPUExecutionProvider")
            
        logger.info("\n🎉 Success! All models have been downloaded, compiled, and cached.")
        
    except Exception as e:
        logger.error(f"Cache warming sequence failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
