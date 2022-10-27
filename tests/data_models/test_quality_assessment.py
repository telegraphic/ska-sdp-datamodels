# pylint: disable=no-name-in-module,import-error

"""
Unit tests for QualityAssessment
"""

import pytest

from ska_sdp_datamodels.science_data_model.qa_model import QualityAssessment


@pytest.fixture(scope="module", name="result_qualityAssessment")
def fixture_qualityAssessment():
    """
    Generate a Quality Assessment object using __init__
    """
    qualityAssessment = QualityAssessment("Test_origin", {"test_data_name": "test_data"}, "Test_context")
    return qualityAssessment


def test_qualityAssessment_str(result_qualityAssessment):
    s = "Quality assessment:\n"
    s += f"\tOrigin: Test_origin\n"
    s += f"\tContext: Test_context\n"
    s += "\tData:\n"
    s += f"\t\ttest_data_name: test_data\n"
    assert str(result_qualityAssessment) == s
