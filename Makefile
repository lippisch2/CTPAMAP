# ctpamap — Lung subtraction CTPA perfusion mapper
#
# The system Python already ships the heavy scientific stack (numpy, scipy,
# matplotlib, scikit-image, nibabel), so `make setup` creates a venv with
# --system-site-packages to reuse it and only installs what's missing
# (e.g. SimpleITK) plus the package itself in editable mode.

PYTHON ?= python3
VENV   := venv

.PHONY: setup test run clean help slicer-sync slicer-test

setup: ## Create venv (reusing system site-packages) and install the package
	$(PYTHON) -m venv --system-site-packages $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

test: ## Run the test suite
	$(VENV)/bin/pytest -q

run: ## ctpamap PRE=$P CTPA=$C [-o OUT] ...  e.g. make run P=pre.nii C=ctpa.nii
	$(VENV)/bin/ctpamap $(P) $(C) $(EXTRA)

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

clean: ## Remove venv and build/cache artifacts
	rm -rf $(VENV) .pytest_cache build dist *.egg-info
	rm -rf ctpamap/__pycache__ tests/__pycache__

slicer-sync: ## Refresh the vendored ctpamap core snapshot used by SlicerCTPAMap
	cp ctpamap/preprocessing.py ctpamap/segmentation.py ctpamap/registration.py \
	   ctpamap/registration_vessel.py ctpamap/subtraction.py ctpamap/overlay.py \
	   ctpamap/qc.py ctpamap/jacobian.py ctpamap/IodineHeat256.txt \
	   SlicerCTPAMap/SlicerCTPAMap/SlicerCTPAMapLib/ctpamap/
	@echo "Vendored core refreshed (SlicerCTPAMap/.../SlicerCTPAMapLib/ctpamap/)."

slicer-test: ## Run the Slicer extension headless tests against the vendored copy
	$(VENV)/bin/python -m pytest -q SlicerCTPAMap/SlicerCTPAMap/SlicerCTPAMapLib/tests