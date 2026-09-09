from clrs._src.specs import Stage, Location, Type, SPECS

SPECS = dict(SPECS)
SPECS.update(
    {
        "tsp": {
            "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
            "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
            "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
            "xc": (Stage.INPUT, Location.NODE, Type.SCALAR),
            "yc": (Stage.INPUT, Location.NODE, Type.SCALAR),
        },
        "mvc": {
            "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
            "nw": (Stage.INPUT, Location.NODE, Type.SCALAR),  # node weights
        },
        "robust_construction": {
            "initial_edges": (Stage.INPUT, Location.EDGE, Type.MASK),
            "tau": (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        },
    }
)
