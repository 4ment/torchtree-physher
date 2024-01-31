from torchtree.cli.plugin import Plugin


class Physher(Plugin):
    def load_arguments(self, subparsers):
        for name, parser in subparsers._name_parser_map.items():
            parser.add_argument(
                "--physher",
                action="store_true",
                help="""use physher""",
            )
            parser.add_argument(
                "--physher_include_jacobian",
                action="store_true",
                help="include Jacobian of the node height transform in "
                "the node height gradient",
            )
            parser.add_argument(
                "--physher_disable_sse",
                action="store_true",
                help="disable SSE in physher",
            )
            parser.add_argument(
                "--physher_disable_coalescent",
                action="store_true",
                help="disable coalescent calculation by physher",
            )
            parser.add_argument(
                "--physher_site",
                choices=["weibull", "gamma"],
                help="""distribution for rate heterogeneity across sites""",
            )

    def process_tree_likelihood(self, arg, json_tree_likelihood):
        if arg.physher and isinstance(json_tree_likelihood, dict):
            json_tree_likelihood["type"] = (
                "torchtree_physher." + json_tree_likelihood["type"]
            )

            for model in ("tree_model", "substitution_model"):
                if isinstance(json_tree_likelihood[model], dict):
                    json_tree_likelihood[model]["type"] = (
                        "torchtree_physher." + json_tree_likelihood[model]["type"]
                    )
            if isinstance(json_tree_likelihood["site_model"], dict):
                if arg.physher_site == "gamma":
                    json_tree_likelihood["site_model"][
                        "type"
                    ] = "torchtree_physher.GammaSiteModel"
                else:
                    json_tree_likelihood["site_model"]["type"] = (
                        "torchtree_physher."
                        + json_tree_likelihood["site_model"]["type"]
                    )
            if arg.clock is not None and isinstance(
                json_tree_likelihood["branch_model"], dict
            ):
                json_tree_likelihood["branch_model"]["type"] = (
                    "torchtree_physher." + json_tree_likelihood["branch_model"]["type"]
                )
            if arg.physher_include_jacobian:
                json_tree_likelihood["include_jacobian"] = True

            if arg.physher_disable_sse:
                json_tree_likelihood["use_sse"] = False

    def process_coalescent(self, arg, json_coalescent):
        if (
            arg.physher
            and isinstance(json_coalescent, dict)
            and not arg.physher_disable_coalescent
        ):
            if (
                json_coalescent["type"]
                in (
                    "ConstantCoalescentModel",
                    "PiecewiseConstantCoalescentGridModel",
                    "PiecewiseConstantCoalescentModel",
                    "PiecewiseLinearCoalescentGridModel",
                )
                and arg.coalescent_temperature is None
            ):
                json_coalescent["type"] = "torchtree_physher." + json_coalescent["type"]
