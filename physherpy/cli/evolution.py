def process_tree_likelihood(arg, json_tree_likelihood):
    json_tree_likelihood['type'] = 'physherpy.' + json_tree_likelihood['type']

    for model in ('tree_model', 'site_model', 'substitution_model'):
        json_tree_likelihood[model]['type'] = (
            'physherpy.' + json_tree_likelihood[model]['type']
        )
    if arg.clock is not None:
        json_tree_likelihood['branch_model']['type'] = (
            'physherpy.' + json_tree_likelihood['branch_model']['type']
        )
    if 'include_jacobian' in arg and arg.include_jacobian:
        json_tree_likelihood['include_jacobian'] = True


def process_coalescent(arg, json_coalescent):
    if json_coalescent['type'] in (
        'ConstantCoalescentModel',
        'PiecewiseConstantCoalescentGridModel',
        'PiecewiseConstantCoalescentModel',
    ):
        json_coalescent['type'] = 'physherpy.' + json_coalescent['type']
