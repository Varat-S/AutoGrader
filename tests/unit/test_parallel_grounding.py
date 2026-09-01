import pytest
from unittest.mock import MagicMock
from app.tools.research import research_cinematography_principles
from app.models.analysis import CinematographyResearchResult

def test_parallel_genuine_excerpt_parsing():
    mock_client = MagicMock()
    mock_result_1 = MagicMock()
    mock_result_1.title = "Cinematography of Dune"
    mock_result_1.url = "https://theasc.com/dune"
    mock_result_1.snippet = "Greig Fraser used soft negative fill and custom amber highlight rolloff for Arrakis exterior scenes."
    
    mock_search_res = MagicMock()
    mock_search_res.results = [mock_result_1]
    mock_client.search.return_value = mock_search_res
    
    res: CinematographyResearchResult = research_cinematography_principles(
        creative_prompt="desert sci-fi",
        parallel_client=mock_client
    )
    
    assert res.is_grounded is True
    assert len(res.sources) == 1
    assert res.sources[0].title == "Cinematography of Dune"
    assert "Greig Fraser" in res.sources[0].excerpt

def test_parallel_error_fallback_never_fabricates():
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("Parallel 429 Rate Limit")
    
    res: CinematographyResearchResult = research_cinematography_principles(
        creative_prompt="desert sci-fi",
        parallel_client=mock_client
    )
    
    # Must NOT fabricate any citations
    assert res.is_grounded is False
    assert len(res.sources) == 0