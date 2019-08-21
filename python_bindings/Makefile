pypi: dist
	twine upload dist/*

dist:
	cp -r ../hnswlib .
	-rm dist/*
	python3 setup.py sdist bdist_wheel

clean:
	rm -rf *.egg-info build dist

.PHONY: dist