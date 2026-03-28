pypi: dist
	twine upload dist/*

dist:
	-rm dist/*
	pip install build
	python3 -m build --sdist

test:
	python3 -m unittest discover --start-directory tests/python --pattern "bindings_test*.py"

examples:
	@for f in examples/python/example*.py; do \
		echo "=== $$f ==="; \
		python3 "$$f" || exit 1; \
	done

install:
	pip install .

clean:
	rm -rf *.egg-info build dist tmp var tests/__pycache__ hnswlib.cpython*.so

.PHONY: dist test examples install clean
