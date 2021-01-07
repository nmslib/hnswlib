pypi: dist
	twine upload dist/*

dist:
	-rm dist/*
	python3 setup.py sdist

test:
	python3 setup.py test

clean:
	rm -rf *.egg-info build dist tmp var tests/__pycache__ hnswlib.cpython*.so

.PHONY: dist