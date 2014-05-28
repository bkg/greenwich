PYTHON ?= python

check:
	$(PYTHON) setup.py test

clean:
	rm -rf build dist *.egg-info
	find . -iname '*.py[co]' -delete

dist: clean
	$(PYTHON) setup.py sdist
