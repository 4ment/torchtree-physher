# torchtree-physher

torchtree-physher is a package providing fast gradient calculation implemented in [physher] for [torchtree]

## Dependencies
 - [torchtree]
 - [physher]

## Installation

```bash
git clone https://github.com/4ment/physher
cmake -B physher/build -DBUILD_CPP_WRAPPER=on -DBUILD_TESTING=on
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

### Coalescent models
Here is a list of coalescent models implemented in this plugin:

- `ConstantCoalescentModel`
- `PiecewiseConstantCoalescentGridModel` (aka skygrid)
- `PiecewiseConstantCoalescentModel` (aka skyride)

[torchtree]: https://github.com/4ment/torchtree
[physher]: https://github.com/4ment/physher