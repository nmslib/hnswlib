pypi: dist
	twine upload dist/*

dist:
	-rm dist/*
	pip install build
	python3 -m build --sdist

test:
	python3 -m unittest discover --start-directory tests/python --pattern "bindings_test*.py"

clean:
	rm -rf *.egg-info build dist tmp var tests/__pycache__ hnswlib.cpython*.so

.PHONY: dist
