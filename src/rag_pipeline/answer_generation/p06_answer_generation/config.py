"""Answer generation configuration and prompt templates."""
from dataclasses import dataclass, field
from typing import Literal
import json
from rag_pipeline.core.paths import Paths

_DEFAULTS = json.load(open(Paths.base() / 'configs' / 'defaults.json'))
PromptStyle = Literal['strict', 'relaxed', 'minimal', 'verbose']

@dataclass
class PromptConfig:
    """Settings for a single prompt style."""
    description: str
    temperature: float
    max_tokens: int
    template: str
    system: str | None = None

    def format(self, context: str, query: str) -> str:
        return self.template.format(context=context, query=query)

def load_prompt_configs() -> dict[PromptStyle, PromptConfig]:
    """Load prompt configs from retrieval_prompts.json."""
    path = Paths.base() / 'configs' / 'retrieval_prompts.json'
    with open(path) as f:
        raw = json.load(f)
    return {style: PromptConfig(**cfg) for style, cfg in raw.items()}
PROMPT_CONFIGS: dict[PromptStyle, PromptConfig] = load_prompt_configs()

def get_prompt(style: PromptStyle, context: str, query: str) -> str:
    """Get formatted prompt for the specified style."""
    if style not in PROMPT_CONFIGS:
        raise ValueError(f'Unknown prompt style: {style!r}. Available: {list(PROMPT_CONFIGS)}')
    return PROMPT_CONFIGS[style].format(context=context, query=query)

def get_prompt_config(style: PromptStyle) -> PromptConfig:
    """Get the full config object for the specified style."""
    if style not in PROMPT_CONFIGS:
        raise ValueError(f'Unknown prompt style: {style!r}. Available: {list(PROMPT_CONFIGS)}')
    return PROMPT_CONFIGS[style]

@dataclass
class GenerationConfig:
    """Configuration for answer generation."""
    prompt_style: PromptStyle = 'strict'
    llm_model: str = field(default_factory=lambda: _DEFAULTS['llm_model'])
    retrieval_model: str = field(default_factory=lambda: _DEFAULTS['production_model'])
    retrieval_config: str = field(default_factory=lambda: _DEFAULTS['production_config'])
    qdrant_host: str = field(default_factory=lambda: _DEFAULTS['qdrant']['host'])
    qdrant_port: int = 6333
    top_k: int = 3
    temperature: float = field(init=False)
    max_tokens: int = field(init=False)

    def __post_init__(self):
        prompt_cfg = PROMPT_CONFIGS.get(self.prompt_style)
        if prompt_cfg is None:
            raise ValueError(f'Unknown prompt style: {self.prompt_style!r}')
        self.temperature = prompt_cfg.temperature
        self.max_tokens = prompt_cfg.max_tokens
        if self.top_k < 1:
            raise ValueError(f'top_k must be >= 1, got {self.top_k}')

    def with_top_k(self, top_k: int) -> 'GenerationConfig':
        """Return a copy of this config with a different top_k."""
        import copy
        clone = copy.copy(self)
        if top_k < 1:
            raise ValueError(f'top_k must be >= 1, got {top_k}')
        clone.top_k = top_k
        return clone