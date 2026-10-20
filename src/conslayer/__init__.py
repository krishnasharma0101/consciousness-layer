"""conslayer: experimental implementation of the 'Consciousness Layer' memory graph.

Nothing in this package measures or demonstrates consciousness. It implements the
paper's equations (with the MATH_AUDIT fixes as options) so they can be tested as an
LLM memory / retrieval mechanism.
"""
from .graph import MemoryGraph, WeightConfig
from .astar import astar_path
from .random_walk import transition_matrix, random_walk, personalized_stationary
from .convergence import (convergence, convergence_normalized, phi_paper, phi_dynamic,
                          phi_mincut, update_activation, should_refine)

__all__ = ["MemoryGraph", "WeightConfig", "astar_path", "transition_matrix", "random_walk",
           "personalized_stationary", "convergence", "convergence_normalized", "phi_paper",
           "phi_dynamic", "phi_mincut", "update_activation", "should_refine"]
