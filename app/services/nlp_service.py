"""
NLP Service – Transformer-based Schema Recognition for column header mapping.

Architecture
------------
Uses a **3-tier escalation** strategy to map arbitrary column headers
(e.g. "Pedestrians Clockwise") to standard movement names ("Peds CW"):

    Tier 1 – Deterministic alias lookup          (instant, 0 cost)
    Tier 2 – Fuzzy string matching (thefuzz)      (fast, no GPU)
    Tier 3 – Transformer bi-encoder similarity    (accurate, ML-based)

Transformer Model
-----------------
**`all-MiniLM-L6-v2`** — a 6-layer MiniLM (Microsoft) distilled from
BERT-base, fine-tuned on 1B+ sentence pairs for semantic similarity.

Architecture details:
  • Base: Microsoft MiniLM-L6-H384 (6 Transformer encoder layers)
  • Each layer: Multi-Head Self-Attention (12 heads) + Feed-Forward
  • Hidden size: 384, Intermediate: 1536
  • Total parameters: 22.7 million
  • Tokenizer: WordPiece (same as BERT)
  • Pooling: Mean pooling over token embeddings
  • Output: 384-dimensional dense vector per sentence

Why this model:
  • True transformer architecture (multi-head self-attention, layer norm)
  • Trained via knowledge distillation from larger transformers
  • Fine-tuned on STS-Benchmark, NLI, and 1B+ sentence pairs
  • Very small: ~80 MB on disk, 22.7M params
  • Fast CPU inference: <50ms per sentence
  • Free & open-source (Apache 2.0 license)
  • Robust: works on Windows/Linux/Mac without GPU

Alternative models (set via NLP_MODEL_NAME env var):
  • "all-MiniLM-L6-v2"              — default (22M params, balanced)
  • "all-MiniLM-L12-v2"             — deeper (33M params, more accurate)
  • "paraphrase-MiniLM-L6-v2"       — optimized for paraphrase detection
  • "all-mpnet-base-v2"             — most accurate (110M params, slower)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard movement vocabulary
# ---------------------------------------------------------------------------

STANDARD_MOVEMENTS = [
    "Right",
    "Thru",
    "Left",
    "U-Turn",
    "Peds CW",
    "Peds CCW",
]

# Extended natural-language descriptions for better transformer matching.
# The model computes semantic similarity between the raw header and these
# rich descriptions, so more descriptive = better matching accuracy.
_MOVEMENT_DESCRIPTIONS: dict[str, str] = {
    "Right":    "right turn vehicle turning right at intersection",
    "Thru":     "through straight ahead vehicle going forward passing through",
    "Left":     "left turn vehicle turning left at intersection",
    "U-Turn":   "u-turn vehicle reversing direction turning around",
    "Peds CW":  "pedestrians walking clockwise on crosswalk",
    "Peds CCW": "pedestrians walking counter-clockwise counterclockwise on crosswalk",
}

# Deterministic aliases for instant resolution (Tier 1)
_ALIASES: dict[str, str] = {
    # Right
    "right": "Right", "right turn": "Right", "rt": "Right", "r": "Right",
    "right-turn": "Right", "rightturn": "Right", "vehicles right": "Right",
    # Thru
    "thru": "Thru", "through": "Thru", "thr": "Thru", "t": "Thru",
    "straight": "Thru", "str": "Thru", "ahead": "Thru",
    "going straight": "Thru", "go straight": "Thru",
    "passing through": "Thru", "straight ahead": "Thru",
    "forward": "Thru",
    # Left
    "left": "Left", "left turn": "Left", "lt": "Left", "l": "Left",
    "left-turn": "Left", "leftturn": "Left", "vehicles left": "Left",
    # U-Turn
    "u-turn": "U-Turn", "uturn": "U-Turn", "u turn": "U-Turn", "u": "U-Turn",
    "u-trn": "U-Turn", "utrn": "U-Turn",
    # Peds CW
    "peds cw": "Peds CW", "pedestrians clockwise": "Peds CW",
    "ped cw": "Peds CW", "pedestrian cw": "Peds CW",
    "peds clockwise": "Peds CW", "pedestrian clockwise": "Peds CW",
    "crosswalk cw": "Peds CW", "xwalk cw": "Peds CW",
    # Peds CCW
    "peds ccw": "Peds CCW", "pedestrians counterclockwise": "Peds CCW",
    "pedestrians counter clockwise": "Peds CCW",
    "ped ccw": "Peds CCW", "pedestrian ccw": "Peds CCW",
    "peds counterclockwise": "Peds CCW", "peds counter clockwise": "Peds CCW",
    "pedestrian counterclockwise": "Peds CCW",
    "crosswalk ccw": "Peds CCW", "xwalk ccw": "Peds CCW",
}


class NLPService:
    """
    Transformer-based schema recognition engine.

    Uses a HuggingFace sentence-transformers model (MiniLM-L6 — a 6-layer
    transformer distilled from BERT) to semantically match column headers
    to standard traffic movement names.

    The model architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Input: "walkers on crosswalk clockwise"                │
    │  ↓ WordPiece Tokenization                               │
    │  [CLS] walkers on cross ##walk clock ##wise [SEP]       │
    │  ↓ Token Embeddings + Position Embeddings               │
    │  ↓ 6 × Transformer Encoder Layers                       │
    │    ├─ Multi-Head Self-Attention (12 heads × 32 dim)     │
    │    ├─ Layer Normalization                                │
    │    ├─ Feed-Forward Network (384 → 1536 → 384)           │
    │    └─ Residual Connection + Layer Norm                   │
    │  ↓ Mean Pooling (over all tokens)                       │
    │  Output: 384-dim dense sentence embedding               │
    │  ↓ Cosine Similarity vs pre-computed movement embeddings│
    │  Result: "Peds CW" (score: 0.847)                       │
    └─────────────────────────────────────────────────────────┘
    """

    SUPPORTED_MODELS = {
        "all-MiniLM-L6-v2":          "MiniLM 6-layer (22M params, ~80MB, fastest)",
        "all-MiniLM-L12-v2":         "MiniLM 12-layer (33M params, ~120MB, balanced)",
        "paraphrase-MiniLM-L6-v2":   "MiniLM 6-layer paraphrase-tuned (22M params)",
        "all-mpnet-base-v2":         "MPNet-base (110M params, ~420MB, most accurate)",
    }

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        confidence_threshold: float = 0.55,
    ):
        self._model_name = model_name or os.getenv("NLP_MODEL_NAME", self.DEFAULT_MODEL)
        self._confidence_threshold = confidence_threshold

        # Lazy-loaded transformer resources
        self._model = None          # SentenceTransformer instance
        self._desc_embeddings = None  # Pre-computed description embeddings
        self._desc_names: list[str] = list(_MOVEMENT_DESCRIPTIONS.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_headers(self, raw_headers: list[str]) -> dict[str, str]:
        """
        Map raw column headers to standard movement names.

        Escalation:
          1. Alias lookup (instant)
          2. Fuzzy matching (fast, ≥80 score → accept)
          3. Transformer semantic similarity (≥threshold → accept)
          4. Fallback: keep original name + log warning

        Returns
        -------
        dict mapping each raw header to its best-matching standard name.
        """
        mapping: dict[str, str] = {}
        transformer_candidates: list[str] = []

        for hdr in raw_headers:
            clean = hdr.strip()

            # Tier 1: Deterministic alias lookup
            alias_result = self._alias_lookup(clean)
            if alias_result is not None:
                mapping[clean] = alias_result
                continue

            # Tier 2: Fuzzy string matching
            fuzzy_result, fuzzy_score = self._fuzzy_match(clean)
            if fuzzy_score >= 80:
                mapping[clean] = fuzzy_result
                continue

            # Tier 3: Queue for transformer batch inference
            transformer_candidates.append(clean)

        # --- Batch transformer inference ---
        if transformer_candidates:
            results = self._batch_semantic_match(transformer_candidates)
            for hdr, (best_match, score) in zip(transformer_candidates, results):
                if score >= self._confidence_threshold:
                    mapping[hdr] = best_match
                    logger.info(
                        "Transformer mapped '%s' → '%s' (cosine=%.3f, model=%s)",
                        hdr, best_match, score, self._model_name,
                    )
                else:
                    mapping[hdr] = hdr
                    logger.warning(
                        "No confident match for '%s' (best='%s' @ %.3f)",
                        hdr, best_match, score,
                    )

        return mapping

    def get_confidence_scores(self, raw_headers: list[str]) -> dict[str, float]:
        """Return transformer cosine similarity scores for each header."""
        results = self._batch_semantic_match(raw_headers)
        return {hdr: round(score, 4) for hdr, (_, score) in zip(raw_headers, results)}

    def get_model_info(self) -> dict:
        """Return metadata about the loaded transformer model."""
        param_count = None
        if self._model is not None:
            try:
                param_count = sum(
                    p.numel() for p in self._model[0].auto_model.parameters()
                )
            except Exception:
                pass

        return {
            "model_name": self._model_name,
            "architecture": "Transformer Bi-Encoder (MiniLM / DistilBERT variant)",
            "base_model": "Microsoft MiniLM-L6-H384 (distilled from BERT-base)",
            "layers": 6,
            "hidden_size": 384,
            "attention_heads": 12,
            "embedding_dim": 384,
            "parameters": f"{param_count:,}" if param_count else "not loaded yet",
            "description": self.SUPPORTED_MODELS.get(self._model_name, "Custom"),
            "confidence_threshold": self._confidence_threshold,
            "is_loaded": self._model is not None,
        }

    # ------------------------------------------------------------------
    # Tier 1: Alias lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _alias_lookup(header: str) -> str | None:
        return _ALIASES.get(header.lower().strip())

    # ------------------------------------------------------------------
    # Tier 2: Fuzzy matching
    # ------------------------------------------------------------------

    @staticmethod
    def _fuzzy_match(header: str) -> tuple[str, int]:
        """Token-sort-ratio fuzzy matching. Returns (best_match, score 0-100)."""
        try:
            from thefuzz import fuzz
        except ImportError:
            return ("", 0)

        best_match, best_score = "", 0
        for std in STANDARD_MOVEMENTS:
            score = fuzz.token_sort_ratio(header.lower(), std.lower())
            if score > best_score:
                best_score = score
                best_match = std
        return best_match, best_score

    # ------------------------------------------------------------------
    # Tier 3: Transformer Semantic Similarity
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> bool:
        """
        Lazy-load the sentence-transformers model.

        Under the hood, sentence-transformers wraps HuggingFace transformers:
          1. AutoTokenizer  — WordPiece tokenization (from transformers lib)
          2. AutoModel      — MiniLM transformer encoder (from transformers lib)
          3. Pooling layer  — Mean pooling over token outputs
          4. Normalization  — L2 normalization of the final embedding

        The model weights are downloaded from HuggingFace Hub and cached
        locally (~/.cache/huggingface/ or TRANSFORMERS_CACHE).
        """
        if self._model is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading transformer model: %s ...", self._model_name)
            self._model = SentenceTransformer(self._model_name)

            # Pre-compute embeddings for all movement descriptions
            descriptions = list(_MOVEMENT_DESCRIPTIONS.values())
            self._desc_embeddings = self._model.encode(
                descriptions, convert_to_numpy=True, normalize_embeddings=True
            )

            param_count = sum(
                p.numel() for p in self._model[0].auto_model.parameters()
            )
            logger.info(
                "✅ Transformer loaded: %s (%s params, %d-dim embeddings)",
                self._model_name, f"{param_count:,}",
                self._desc_embeddings.shape[1],
            )
            return True

        except Exception as e:
            logger.warning(
                "⚠️ Could not load transformer '%s': %s. "
                "Falling back to fuzzy matching.",
                self._model_name, e,
            )
            self._model = None
            return False

    def _batch_semantic_match(
        self, headers: list[str]
    ) -> list[tuple[str, float]]:
        """
        For each header, compute cosine similarity against all movement
        descriptions using the transformer encoder.

        Pipeline per header:
          1. Tokenize (WordPiece): "walkers clockwise" → [101, 5765, 14161, 102]
          2. Forward through 6 transformer layers (self-attention + FFN)
          3. Mean-pool token outputs → 384-dim embedding
          4. L2-normalize
          5. Cosine similarity vs pre-computed movement embeddings

        Returns list of (best_match_name, cosine_score) tuples.
        """
        if not self._ensure_model_loaded():
            return [self._fuzzy_fallback(h) for h in headers]

        # Encode all headers in a single batch (efficient GPU/CPU utilization)
        query_embeddings = self._model.encode(
            headers, convert_to_numpy=True, normalize_embeddings=True
        )

        results: list[tuple[str, float]] = []
        for q_emb in query_embeddings:
            # Cosine similarity (embeddings are already L2-normalized)
            scores = np.dot(self._desc_embeddings, q_emb)
            best_idx = int(np.argmax(scores))
            results.append((self._desc_names[best_idx], float(scores[best_idx])))

        return results

    def _fuzzy_fallback(self, header: str) -> tuple[str, float]:
        """Fallback when transformer unavailable: fuzzy match, return 0-1 score."""
        try:
            from thefuzz import fuzz
        except ImportError:
            return ("", 0.0)

        best_match, best_score = "", 0
        for name, desc in _MOVEMENT_DESCRIPTIONS.items():
            score_name = fuzz.token_sort_ratio(header.lower(), name.lower())
            score_desc = fuzz.token_sort_ratio(header.lower(), desc.lower())
            score = max(score_name, score_desc)
            if score > best_score:
                best_score = score
                best_match = name

        return best_match, best_score / 100.0
