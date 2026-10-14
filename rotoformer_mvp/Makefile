.PHONY: install test demo compile

install:
	python3 -m venv .venv
	source .venv/bin/activate
	python -m pip install -e '.[hf,dev]'

test:
	PYTHONPATH=. pytest

demo:
	PYTHONPATH=. python examples/toy_rotoformer.py

compile:
	python -m compileall -q rotoformer tests examples scripts
