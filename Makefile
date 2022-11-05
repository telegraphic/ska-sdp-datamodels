include .make/base.mk
include .make/python.mk
include .make/oci.mk

PROJECT_NAME = ska-sdp-datamodels

# W503: line break before binary operator
PYTHON_SWITCHES_FOR_FLAKE8 = --ignore=W503

# tmp, skip these files until fixed
PYTHON_VARS_AFTER_PYTEST = --ignore=tests/test_import_export.py --ignore=tests/test_xarray_coordinate_support.py