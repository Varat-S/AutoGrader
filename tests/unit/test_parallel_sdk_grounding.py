import pytest
from unittest.mock import MagicMock
from parallel.types.web_search_result import WebSearchResult
from app.tools.research import research_cinematography_principles, synthesize_creative_specification
from app.models.analysis import CinematographyResearchResult, SearchCitation

def test_real_parallel_sdk_web_search_result_parsing():
    mock_client = MagicMock()
    
    # Real SDK-shaped WebSearchResult instance
    item = WebSearchResult(
        title="American Cinematographer: Dune Analysis",
        url="https://theasc.com/articles/dune-part-one-color",
        excerpts=["Greig Fraser ASC ACS selected muted saturation with warm amber highlights for desert exteriors.", "Shadows were kept cool and soft with negative fill."],
        publish_date="2021-11-01"
    )
    
    mock_search_res = MagicMock()
    mock_search_res.results = [item]
    mock_client.search.return_value = mock_search_res
    
    res: CinematographyResearchResult = research_cinematography_principles(
        creative_prompt="desert sci-fi",
        parallel_client=mock_client
    )
    
    assert res.is_grounded is True
    assert len(res.sources) == 1
    assert res.sources[0].title == "American Cinematographer: Dune Analysis"
    assert res.sources[0].url == "https://theasc.com/articles/dune-part-one-color"
    assert "Greig Fraser" in res.sources[0].excerpt
    assert "Shadows were kept cool" in res.sources[0].excerpt

def test_empty_excerpts_does_not_substitute_title_or_claim_grounding():
    mock_client = MagicMock()
    
    # Result with empty excerpts list
    item = WebSearchResult(
        title="Some Page Without Content",
        url="https://example.com/empty",
        excerpts=[],
        publish_date=None
    )
    
    mock_search_res = MagicMock()
    mock_search_res.results = [item]
    mock_client.search.return_value = mock_search_res
    
    res: CinematographyResearchResult = research_cinematography_principles(
        creative_prompt="desert sci-fi",
        parallel_client=mock_client
    )
    
    # Must NOT substitute title as fake evidence, and must NOT be marked grounded
    assert res.is_grounded is False
    assert len(res.sources) == 0

def test_parallel_exception_returns_clean_ungrounded_state():
    mock_client = MagicMock()
    mock_client.search.side_effect = ConnectionError("Parallel API unreachable")
    
    res: CinematographyResearchResult = research_cinematography_principles(
        creative_prompt="desert sci-fi",
        parallel_client=mock_client
    )
    
    assert res.is_grounded is False
    assert len(res.sources) == 0