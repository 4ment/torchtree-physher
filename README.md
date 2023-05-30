# torchtree-physher

[![Testing (Linux)](https://github.com/4ment/torchtree-physher/actions/workflows/test_linux.yml/badge.svg)](https://github.com/4ment/torchtree-physher/actions/workflows/test_linux.yml)

torchtree-physher is a package providing fast gradient calculation implemented in [physher] for [torchtree]

## Dependencies
 - [torchtree]
 - [physher]

## Installation

```bash
git clone https://github.com/4ment/physher
cmake -S . -B physher/build -DBUILD_CPP_WRAPPER=on -DBUILD_TESTING=on
cmake --build physher/build/ --target install
```

Check it works (optional)
```bash
ctest --test-dir physher/build/
```

### Get the source code
```bash
git clone https://github.com/4ment/torchtree-physher
cd torchtree-physher
```

### Install using pip
```bash
pip install .
```

## Check install

```bash
torchtree --help
```

```bash
python -c "import torchtree_physher"
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
- Tree models:
  - `UnRootedTreeModel`
  - `ReparameterizedTimeTreeModel`
- Clock models (optional):
  - `StrictClockModel`
  - `SimpleClockModel`
- Site models:
  - `ConstantSiteModel`
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

[torchtree]: https://github.com/4ment/torchtree
[physher]: https://github.com/4ment/physher