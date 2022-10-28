include .make/base.mk
include .make/python.mk
include .make/oci.mk

PROJECT_NAME = ska-sdp-datamodels

# W503: line break before binary operator
PYTHON_SWITCHES_FOR_FLAKE8 = --ignore=W503

# tmp, until we get all tests working
PYTHON_TEST_FILE = tests/configuration tests/image tests/visibility tests/science_data_model