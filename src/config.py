"""
Configuration file for LFM2.5 Unified RAG Pipeline & Evaluation Framework.
"""

import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    EMBEDDING_MODEL: str = "liquid/LFM2.5-Embedding-350M"
    COLBERT_MODEL: str = "liquid/LFM2.5-ColBERT-350M"
    GENERATOR_MODEL: str = "liquid/LFM2.5-1.2B-Instruct"
    GENERATOR_THINKING_MODEL: str = "liquid/LFM2.5-1.2B-Thinking"
    
    EMBEDDING_DIM: int = 1024
    COLBERT_DIM: int = 128
    MAX_CONTEXT_LENGTH: int = 32768
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    
    QUANTIZATION: str = "INT8"
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

@dataclass
class RetrievalConfig:
    TOP_K_DENSE: int = 20
    TOP_K_BM25: int = 20
    RRF_K: int = 60
    TOP_K_RERANK: int = 5
    ENABLE_HYBRID: bool = True
    ENABLE_RERANKING: bool = True

@dataclass
class GenerationConfig:
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: int = 0.2
    TOP_P: int = 0.9
    REPETITION_PENALTY: int = 1.1
    SYSTEM_PROMPT: str = (
        "You are an accurate, grounded AI research assistant powered by LFM2.5-1.2B.\n"
        "Answer the user query strictly using the provided reference context.\n"
        "If the context does not contain enough information, state that clearly.\n"
        "Always cite source chunk numbers [Doc N] in your response."
    )

@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
    CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(__file__), "chroma_db")
    RESULTS_DIR: str = os.path.join(os.path.dirname(__file__), "results")

cfg = PipelineConfig()
