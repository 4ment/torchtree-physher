install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall physherpy

lint: FORCE
	flake8 --exit-zero physherpy tests
	isort --check .
	black --check .
	cpplint physherpy/physher.cpp

format: license FORCE
	isort .
	black .
	clang-format -i physherpy/physher.cpp --style=Google

test: FORCE
	pytest

clean: FORCE
	rm -fr physherpy/__pycache__ physherpy/physher.cpython-* build var

nuke: FORCE
	git clean -dfx -e physherpy.egg-info

	done

FORCE: