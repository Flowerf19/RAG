"""
Embedder Helper
Utilities for embedder type parsing and configuration.
"""

from embedders.embedder_type import EmbedderType
from embedders.embedder_factory import EmbedderFactory


class EmbedderHelper:
    """Helper class for embedder configuration and parsing."""

    @staticmethod
    def parse_embedder_type(embedder_type: str):
        """Parse embedder type string to enum and API flag."""
        if embedder_type.lower() in ["e5_large_instruct", "e5_base", "gte_multilingual_base",
                                   "paraphrase_mpnet_base_v2", "paraphrase_minilm_l12_v2"]:
            return EmbedderType.HUGGINGFACE, False
        elif embedder_type == "huggingface_api":
            return EmbedderType.HUGGINGFACE, True
        elif embedder_type == "huggingface_local":
            return EmbedderType.HUGGINGFACE, False
        else:  # ollama and others
            return EmbedderType.OLLAMA, False

    @staticmethod
    def configure_pipeline_embedder(pipeline, embedder_choice: str):
        """Configure pipeline embedder for specific multilingual models."""
        if embedder_choice.lower() in ["e5_large_instruct", "e5_base", "gte_multilingual_base",
                                     "paraphrase_mpnet_base_v2", "paraphrase_minilm_l12_v2"]:
            factory = EmbedderFactory()

            if embedder_choice.lower() == "e5_large_instruct":
                pipeline.embedder = factory.create_e5_large_instruct(device="cpu")
            elif embedder_choice.lower() == "e5_base":
                pipeline.embedder = factory.create_e5_base(device="cpu")
            elif embedder_choice.lower() == "gte_multilingual_base":
                pipeline.embedder = factory.create_gte_multilingual_base(device="cpu")
            elif embedder_choice.lower() == "paraphrase_mpnet_base_v2":
                pipeline.embedder = factory.create_paraphrase_mpnet_base_v2(device="cpu")
            elif embedder_choice.lower() == "paraphrase_minilm_l12_v2":
                pipeline.embedder = factory.create_paraphrase_minilm_l12_v2(device="cpu")