"""
rag_pipeline/ingestion/warm_onnx_cache.py
Utility script to pre-download and compile all cross-encoder models
defined in rerankers.json into the ONNX cache.
"""
import logging
import sys
import traceback

from .onnx_bench_config import load_matrix_configs
from .onnx_model_loader import ONNXModelLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('warm_onnx_cache')


def main():
    logger.info('Initializing ONNX Cache Warmer...')
    try:
        reranker_entries = load_matrix_configs()
        logger.info(f'Found {len(reranker_entries)} models to pre-cache.')
        for idx, entry in enumerate(reranker_entries, start=1):
            name = entry['name']
            hf_path = entry['model']
            logger.info(f'\n======================================== [{idx}/{len(reranker_entries)}]')
            logger.info(f'Targeting: {name}')
            logger.info(f'Hub Path:  {hf_path}')
            logger.info('========================================')
            ONNXModelLoader(model_name=hf_path, provider='CPUExecutionProvider')
        logger.info('\n🎉 All models downloaded, compiled, and cached.')
    except Exception as e:
        logger.error(f'Cache warming failed: {e}')
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
