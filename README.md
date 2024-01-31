# torchtree-physher

[![Testing](https://github.com/4ment/torchtree-physher/actions/workflows/test_linux.yml/badge.svg)](https://github.com/4ment/torchtree-physher/actions/workflows/test_linux.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## About torchtree-physher
torchtree-physher is a python package providing fast gradient calculation implemented in [physher] for [torchtree].

## Getting Started

A C++ compiler such as ``g++`` or ``clang++`` is required.
On Debian-based systems, this can be installed via ``apt``:

```bash
sudo apt install g++
```

On MacOS, it is recommended to use the latest version of ``clang++``:
```bash
brew install llvm
```

The [pybind11] library is also used for binding the
C++ functionality to Python.

### Dependencies
 - [physher]
 - [pybind11]
 - [PyTorch]
 - [torchtree]

[physher] is a phylogenetic program written in C that provides C++ wrappers to compute the tree and coalescent likelihoods and their gradients under different models.

To build physher from source you can run
```bash
git clone https://github.com/4ment/physher
cmake -S physher/ -B physher/build -DBUILD_CPP_WRAPPER=on -DBUILD_TESTING=on
cmake --build physher/build/ --target install
```

Check it works (optional)
```bash
ctest --test-dir physher/build/
```


### Installation
To build `torchtree-physher` from source you can run
```bash
git clone https://github.com/4ment/torchtree-physher
pip install torchtree-physher/
```

### Check install
If the installation was successful, this command should print the version of the `torchtree_physher` library
```bash
python -c "import torchtree_physher;print(torchtree_physher.__version__)"
```

## Command line arguments
The torchtree-physher plugin adds these arguments to the torchtree CLI:

```bash
torchtree-cli advi --help
  ...
  --physher             use physher
  --physher_include_jacobian
                        include Jacobian of the node height transform in the node height gradient
  --physher_disable_sse
                        disable SSE in physher
  --physher_disable_coalescent
                        disable coalescent calculation by physher
  --physher_site {weibull,gamma}
                        distribution for rate heterogeneity across sites
```

## Features
### Tree likelihood
Some types in the JSON configuration file have to be replaced in order to use the tree likelihood implementation of physher. You simply need to add `torchtree_physher.` before a model type. Here is a list of models implemented in this plugin:

- `TreeLikelihoodModel`
- Substitution models:
  - `JC69`
  - `HKY`
  - `GTR`
  - `GeneralNonSymmetricSubstitutionModel`
- Tree models:
  - `UnRootedTreeModel`
  - `ReparameterizedTimeTreeModel`
- Clock models (optional):
  - `StrictClockModel`
  - `SimpleClockModel`
- Site models:
  - `ConstantSiteModel`
  - `GammaSiteModel`
  - `InvariantSiteModel`
  - `WeibullSiteModel`

Note that the type of every sub-model of the tree likelihood object (e.g. site, tree models...) has to be replaced.

For example if we want to use ADVI with an unrooted tree and a Weibull site model:

```bash
torchtree-cli advi -i data.fa -t data.tree -C 4 > data.json
sed -i -E 's/TreeLikelihoodModel/torchtree_physher.TreeLikelihoodModel/; s/UnRootedTreeModel/torchtree_physher.UnRootedTreeModel/; s/WeibullSiteModel/torchtree_physher.WeibullSiteModel/' data.json
torchtree data.json
```

The JSON file can be created directly using the `--physher` option:
```bash
torchtree-cli advi -i data.fa -t data.tree -C 4 --physher > data.json
```

### Coalescent models
Here is a list of coalescent models implemented in this plugin:

- `ConstantCoalescentModel`
- `PiecewiseConstantCoalescentGridModel` (aka skygrid)
- `PiecewiseConstantCoalescentModel` (aka skyride)
- `PiecewiseLinearCoalescentGridModel`

## License

Distributed under the GPLv3 License. See [LICENSE](LICENSE) for more information.

## Acknowledgements

torchtree-physher makes use of the following libraries and tools, which are under their own respective licenses:

 - [physher]
 - [pybind11]
 - [PyTorch]
 - [torchtree]

[physher]: https://github.com/4ment/physher
[pybind11]: https://pybind11.readthedocs.io/en/stable
[PyTorch]: https://pytorch.org
[torchtree]: https://github.com/4ment/torchtree