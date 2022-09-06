install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall torchtree_physher

lint: FORCE
	flake8 --exit-zero torchtree_physher tests
	isort --check .
	black --check .
	cpplint torchtree_physher/physher.cpp

format: license FORCE
	isort .
	black .
	clang-format -i torchtree_physher/physher.cpp --style=Google

test: FORCE
	pytest

clean: FORCE
	rm -fr torchtree_physher/__pycache__ torchtree_physher/physher.cpython-* build var

nuke: FORCE
	git clean -dfx -e torchtree_physher.egg-info

	done

FORCE: