PYTHON ?= python

.PHONY: install test foundation

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

foundation:
	$(PYTHON) -m pipeline.foundation
