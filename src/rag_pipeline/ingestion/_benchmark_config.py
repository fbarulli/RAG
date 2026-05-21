"""
Public Functions for Benchmark and Ingestion Configuration Management:

def from_defaults() -> BenchmarkConfig:
    Build a config from defaults.json with project-root relative path resolution.
    I/O: None -> BenchmarkConfig

def from_args(args: argparse.Namespace) -> BenchmarkConfig:
    Create config from defaults + CLI arguments.
    I/O: args (argparse.Namespace) -> BenchmarkConfig

def merge_args(args: argparse.Namespace) -> BenchmarkConfig:
    Return a new config with any non-None CLI values overlaid on self.
    I/O: args (argparse.Namespace) -> BenchmarkConfig

rag_pipeline/p04_ingestion/_benchmark_config.py
===================
Centralized configuration dataclass for all benchmark scripts.

Holds resolved, typed settings. Does not own argparse — see benchmark_cli.py
for parser factories and the CLI → config merge.

Typical usage
-------------
    # 1. Start from defaults.json
    config = BenchmarkConfig.from_defaults()

    # 2. Override with whatever the user passed on the CLI
    args = create_ingestion_parser().parse_args()
    config = config.merge_args(args)

    # 3. Use
    client = config.qdrant_client   # lazy, cached
    entries = config.get_model_entries()
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Optional, List, Tuple
from rag_pipeline.core.paths import Paths
from functools import cached_property
from rag_pipeline.core.logging import get_logger
from ._benchmark_loader import load_defaults
logger = get_logger(__name__)

@dataclass
class BenchmarkConfig:
    """Resolved configuration for benchmark and ingestion scripts."""
    test_set_path: Optional[Path] = None
    clean_path: Optional[Path] = None
    topic_path: Optional[Path] = None
    configs_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    qdrant_host: str = 'localhost'
    qdrant_port: int = 6333
    es_host: Optional[str] = None
    es_index: str = 'faqs'
    models: Optional[List[str]] = None
    config: Optional[str] = None
    top_k: int = 10
    encode_batch_size: int = 32
    batch_size: int = 100
    quick: bool = False
    force_encode: bool = False
    skip_existing: bool = True
    no_detail: bool = False
    auto_prepare: bool = False
    resume: bool = True
    reset: bool = False
    fail_fast: bool = False
    model_timeout: int = 3600

    @cached_property
    def qdrant_client(self):
        """Lazy Qdrant client."""
        from qdrant_client import QdrantClient
        client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        client.get_collections()
        return client

    @cached_property
    def es_client(self):
        """Lazy Elasticsearch client (optional)."""
        if not self.es_host:
            return None
        from elasticsearch import Elasticsearch
        try:
            es = Elasticsearch(hosts=[self.es_host])
            if es.ping():
                return es
            logger.warning('Elasticsearch not reachable')
            return None
        except Exception as e:
            logger.warning(f'Elasticsearch connection failed: {e}')
            return None

    @classmethod
    def from_defaults(cls) -> 'BenchmarkConfig':
        """
        Build a config from ``defaults.json``.

        All values fall back to the field defaults above when a key is absent
        from the file, so this never raises on a partial defaults.json.
        """
        try:
            defaults = load_defaults()
        except FileNotFoundError:
            logger.warning('defaults.json not found — using built-in defaults')
            defaults = {}
        paths = defaults.get('paths', {})
        qdrant = defaults.get('qdrant', {})
        es = defaults.get('elasticsearch', {})
        bench = defaults.get('benchmark', {})
        ingest = defaults.get('ingestion', {})

        def resolve(path_str: Optional[str]) -> Optional[Path]:
            if not path_str:
                return None
            p = Path(path_str)
            if p.is_absolute():
                return p
            return Paths.base() / path_str
        return cls(test_set_path=resolve(paths.get('test_jsonl')), clean_path=resolve(paths.get('clean_jsonl')), topic_path=resolve(paths.get('topic_assignments')), configs_path=resolve(paths.get('retrieval_configs')), output_dir=Paths.reranker_results_dir(), cache_dir=resolve(paths.get('embeddings_cache_dir')), qdrant_host=qdrant.get('host', 'localhost'), qdrant_port=qdrant.get('port', 6333), es_host=es.get('host'), es_index=es.get('index', 'faqs'), top_k=bench.get('top_k', 10), encode_batch_size=bench.get('encode_batch_size', 32), batch_size=ingest.get('batch_size', 100), quick=bench.get('quick', False))

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'BenchmarkConfig':
        """Create config from defaults + CLI arguments."""
        config = cls.from_defaults()
        return config.merge_args(args)

    def merge_args(self, args: argparse.Namespace) -> 'BenchmarkConfig':
        """
        Return a *new* config with any non-None CLI values overlaid on self.

        Only attributes that were explicitly set on the CLI (i.e. not None /
        not the argparse sentinel) override the defaults already in this
        instance.  This means callers can safely do::

            config = BenchmarkConfig.from_defaults().merge_args(parsed_args)

        and always get a fully-resolved config regardless of which arguments
        the user actually passed.

        The ``--model`` (singular) used by script-specific parsers is
        normalised to ``models`` (list) here so the rest of the codebase only
        deals with one field.
        """
        single_model: Optional[str] = getattr(args, 'model', None)
        plural_models: Optional[List[str]] = getattr(args, 'models', None)
        resolved_models = [single_model] if single_model is not None else plural_models if plural_models is not None else self.models

        def _path(attr: str, arg_attr: Optional[str]=None) -> Optional[Path]:
            key = arg_attr or attr
            v = getattr(args, key, None)
            return Path(v) if v is not None else getattr(self, attr)

        def _val(attr: str, arg_attr: Optional[str]=None) -> object:
            """Return CLI value if present, otherwise keep current value."""
            key = arg_attr or attr
            v = getattr(args, key, None)
            return v if v is not None else getattr(self, attr)

        def _bool_flag(attr: str, cli_attr: Optional[str]=None) -> bool:
            """Return CLI boolean flag if explicitly set, otherwise current value."""
            key = cli_attr or attr
            if hasattr(args, key):
                return getattr(args, key)
            return getattr(self, attr)
        return BenchmarkConfig(test_set_path=_path('test_set_path', 'test_set'), clean_path=_path('clean_path'), topic_path=_path('topic_path'), configs_path=_path('configs_path'), output_dir=_path('output_dir'), cache_dir=_path('cache_dir'), qdrant_host=_val('qdrant_host'), qdrant_port=_val('qdrant_port'), es_host=_val('es_host'), es_index=_val('es_index'), models=resolved_models, config=_val('config'), top_k=_val('top_k'), encode_batch_size=_val('encode_batch_size'), batch_size=_val('batch_size'), quick=_bool_flag('quick'), force_encode=_bool_flag('force_encode'), skip_existing=_bool_flag('skip_existing'), no_detail=_bool_flag('no_detail'), auto_prepare=_bool_flag('auto_prepare'), resume=_bool_flag('resume'), reset=_bool_flag('reset'), fail_fast=_bool_flag('fail_fast'), model_timeout=_val('model_timeout', 'timeout'))

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate config before running benchmark. Returns (is_valid, issues)."""
        issues = []
        if self.test_set_path and (not self.test_set_path.exists()):
            issues.append(f'Test set not found: {self.test_set_path}')
        if self.configs_path and (not self.configs_path.exists()):
            issues.append(f'Configs file not found: {self.configs_path}')
        if self.topic_path and (not self.topic_path.exists()):
            logger.warning(f'Topic file not found: {self.topic_path}')
        if self.clean_path and (not self.clean_path.exists()):
            logger.warning(f'Clean corpus not found: {self.clean_path}')
        return (len(issues) == 0, issues)

    def make_qdrant_client(self):
        from qdrant_client import QdrantClient
        client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        client.get_collections()
        return client

    def make_es_client(self):
        if not self.es_host:
            return None
        from elasticsearch import Elasticsearch
        try:
            es = Elasticsearch(hosts=[self.es_host])
            if es.ping():
                return es
            logger.warning('Elasticsearch not reachable — ES retrieval disabled')
            return None
        except Exception as e:
            logger.warning(f'Elasticsearch connection failed: {e}')
            return None

    def get_model_entries(self) -> list[dict]:
        """Return model registry entries for the configured model list."""
        from ._benchmark_loader import load_model_registry, get_model_entry
        if self.models:
            return [get_model_entry(m) for m in self.models]
        return load_model_registry(enabled_only=True)

    def get_configs(self) -> dict:
        """
        Return retrieval config dict, optionally filtered to a single config.

        Raises
        ------
        ValueError
            If ``self.configs_path`` is not set.
        KeyError
            If ``self.config`` names a config that does not exist in the file.
        """
        if self.configs_path is None:
            raise ValueError("configs_path is not set — pass --configs-path or add 'retrieval_configs' to defaults.json")
        from ._benchmark_loader import load_configs
        configs = load_configs(self.configs_path)
        if self.config:
            if self.config not in configs:
                raise KeyError(f"Config '{self.config}' not found. Available: {list(configs.keys())}")
            return {self.config: configs[self.config]}
        return configs

    def get_test_set(self, limit: Optional[int]=None) -> list[dict]:
        """
        Load the benchmark test set.

        If ``self.quick`` is True and no explicit limit is provided,
        automatically limits to 20 queries for fast testing.

        Raises
        ------
        ValueError
            If ``self.test_set_path`` is not set.
        """
        if self.test_set_path is None:
            raise ValueError("test_set_path is not set — pass --test-set or add 'test_jsonl' to defaults.json")
        from ._benchmark_loader import load_test_set
        test_set = load_test_set(self.test_set_path, self.clean_path)
        if self.quick and limit is None:
            limit = 20
            logger.info(f'Quick mode enabled: limiting to {limit} test queries')
        return test_set[:limit] if limit else test_set

    def get_topic_map(self, model_name: str) -> dict[str, dict]:
        """
        Load NER/topic assignments for *model_name*.

        Always returns a dict — missing topic files or an absent model entry
        are treated as a soft failure (warning logged, empty dict returned)
        because topic enrichment is optional metadata, not a hard dependency.
        """
        if self.topic_path is None:
            logger.warning('topic_path is not set — skipping NER enrichment')
            return {}
        from ._benchmark_loader import load_topic_assignments
        try:
            return load_topic_assignments(self.topic_path, model=model_name)
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Topic assignments unavailable for '{model_name}': {e}")
            return {}

    def __repr__(self) -> str:
        """Compact representation showing non-default values."""
        items = []
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            default = getattr(self.__class__, field_name).default
            if value != default:
                if isinstance(value, list) and len(value) > 3:
                    value = value[:3] + [f'... ({len(value)} total)']
                items.append(f'{field_name}={value!r}')
        return f"BenchmarkConfig({', '.join(items)})"