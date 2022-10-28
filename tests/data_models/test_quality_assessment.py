# pylint: disable=no-name-in-module,import-error
# make python-format
# make python lint
"""
Unit tests for QualityAssessment
"""

import pytest
from ska_sdp_datamodels.science_data_model.qa_model import QualityAssessment


@pytest.fixture(scope="module", name="result_quality_assessment")
def fixture_quality_assessment():
    """
    Generate a Quality Assessment object using __init__
    """
    quality_assessment = QualityAssessment(
        "Test_origin", {"test_data_name": "test_data"}, "Test_context"
    )
    return quality_assessment


def test_quality_assessment_str(result_quality_assessment):
    s = "Quality assessment:\n"
    s += f"\tOrigin: Test_origin\n"
    s += f"\tContext: Test_context\n"
    s += "\tData:\n"
    s += f"\t\ttest_data_name: test_data\n"
    assert str(result_quality_assessment) == s
