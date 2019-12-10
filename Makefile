# This must match the Python version in the Docker image.
PYTHON=python

dummy:

export:
	conda env export > conda.yml

install:
	conda env create -f conda.yml

black:
	black code

test:
	./test.sh

clean:
	for dir in code ; \
	do \
	    find "$$dir" -name '*.pyc' -print0 \
	        -or -name '*.egg-info' -print0 \
	        -or -name '__pycache__' -print0 | \
	        xargs -0 rm -vrf ; \
	done
	rm -f *.log
	rm -rf *.egg-info
	rm -rf logs
	rm -f first_run

.PHONY: dummy install black test clean
