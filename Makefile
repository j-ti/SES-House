# This must match the Python version in the Docker image.
PYTHON=python3.6

dummy:

pipenv:
	pipenv shell
	. env

install: ./requirements.txt
	# @note: Have pipenv installed, e.g. sudo -H pip3 install pipenv
	pipenv install

black:
	black code

test:
	$(PYTHON) -m unittest discover -s code
	gurobi.sh code/simple-model.py
	$(PYTHON) -m flake8 code

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

distclean:
	git clean -fxd

.PHONY: dummy pipenv install black test clean distclean
