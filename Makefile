PYTHON ?= python

.PHONY: all data pipeline test test-r notebook clean

all: data pipeline test

data:
	$(PYTHON) data/download_data.py

pipeline:
	$(PYTHON) -m src.pipeline

test:
	$(PYTHON) -m pytest -q

test-r:
	Rscript -e "testthat::test_dir('r/tests', stop_on_failure = TRUE)"

notebook:
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace walkthrough.ipynb

clean:
	rm -rf reports/figures/*.png reports/tables/*
