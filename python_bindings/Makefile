pypi: dist
	twine upload dist/*

dist:
	-rm dist/*
	python3 setup.py sdist

test:
	python3 setup.py test

clean:
	rm -rf *.egg-info build dist var first_half.bin tests/__pycache__ hnswlib.cpython-36m-darwin.so

.PHONY: dist