"""
Tests for NLPService — Transformer-based schema recognition.

Tests the 3-tier matching:
  Tier 1: Alias lookup (deterministic)
  Tier 2: Fuzzy matching (thefuzz)
  Tier 3: Transformer semantic similarity (MiniLM-L6, 6-layer DistilBERT)
"""

import pytest

from app.services.nlp_service import NLPService


@pytest.fixture
def nlp():
    return NLPService()


class TestModelInfo:
    """Verify the model metadata is reported correctly."""

    def test_model_name_default(self, nlp):
        info = nlp.get_model_info()
        assert "MiniLM" in info["model_name"]

    def test_architecture_is_transformer(self, nlp):
        info = nlp.get_model_info()
        assert "Transformer" in info["architecture"]

    def test_has_6_layers(self, nlp):
        info = nlp.get_model_info()
        assert info["layers"] == 6

    def test_embedding_dim_384(self, nlp):
        info = nlp.get_model_info()
        assert info["embedding_dim"] == 384


class TestTier1AliasLookup:
    """Tier 1: deterministic alias lookup — instant, no model required."""

    def test_exact_standard_names(self, nlp):
        result = nlp.map_headers(["Right", "Thru", "Left", "U-Turn", "Peds CW", "Peds CCW"])
        assert result["Right"] == "Right"
        assert result["Thru"] == "Thru"
        assert result["Left"] == "Left"
        assert result["U-Turn"] == "U-Turn"
        assert result["Peds CW"] == "Peds CW"
        assert result["Peds CCW"] == "Peds CCW"

    def test_common_aliases(self, nlp):
        result = nlp.map_headers(["Through", "Right Turn", "Left Turn", "Going Straight"])
        assert result["Through"] == "Thru"
        assert result["Right Turn"] == "Right"
        assert result["Left Turn"] == "Left"
        assert result["Going Straight"] == "Thru"


class TestTier2FuzzyMatching:
    """Tier 2: fuzzy string matching for close variations."""

    def test_pedestrians_clockwise(self, nlp):
        result = nlp.map_headers(["Pedestrians Clockwise"])
        assert result["Pedestrians Clockwise"] == "Peds CW"

    def test_pedestrians_counterclockwise(self, nlp):
        result = nlp.map_headers(["Pedestrians Counterclockwise"])
        assert result["Pedestrians Counterclockwise"] == "Peds CCW"


class TestTier3TransformerSemantic:
    """
    Tier 3: transformer semantic similarity (MiniLM-L6).

    These headers are NOT in the alias table and have low fuzzy scores,
    so they MUST go through the transformer to be resolved.
    """

    def test_u_turn_variant(self, nlp):
        """'U Turn' (no hyphen) should fuzzy-match to 'U-Turn'."""
        result = nlp.map_headers(["U Turn"])
        assert result["U Turn"] == "U-Turn"

    def test_confidence_scores_range(self, nlp):
        """All scores should be valid floats between 0 and 1."""
        scores = nlp.get_confidence_scores(["Right turn", "Going through", "Walking on crosswalk"])
        for hdr, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Score for '{hdr}' out of range: {score}"

    def test_batch_mapping(self, nlp):
        """Mapping multiple headers should return results for all."""
        headers = ["Right", "Through", "Pedestrians Clockwise", "U Turn"]
        result = nlp.map_headers(headers)
        assert len(result) == 4
        assert result["Right"] == "Right"       # Tier 1
        assert result["Through"] == "Thru"      # Tier 1
        assert result["U Turn"] == "U-Turn"     # Tier 2 fuzzy

    def test_model_loads_on_demand(self, nlp):
        """After running get_confidence_scores, model should be loaded."""
        nlp.get_confidence_scores(["test"])
        info = nlp.get_model_info()
        assert info["is_loaded"] is True
