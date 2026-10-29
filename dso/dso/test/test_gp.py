"""Test cases for the GP meld step's eval_tree and eval_seeds switches."""

import random
from pkg_resources import resource_filename

import numpy as np
import pytest

from dso import DeepSymbolicOptimizer
from dso.config import load_config
from dso.gp import utils as U


N = 100


def _model(**gp_meld):
    """Set up Nguyen-1 with GP meld on."""
    config = load_config(resource_filename("dso", "config/config_regression_gp.json"))
    config["experiment"]["logdir"] = None # Turn off saving results
    config["training"]["batch_size"] = N
    config["gp_meld"].update({"population_size" : N, "generations" : 5,
                              "parallel_eval" : False})
    config["gp_meld"].update(gp_meld)
    model = DeepSymbolicOptimizer(config)
    model.setup()
    np.random.seed(0)
    random.seed(0)
    return model


def _individuals(gpc, actions):
    return [gpc.creator.Individual(a, gpc.pset, None, '', None, gpc.eval_tree)
            for a in actions]


def _meld_step(model):
    """Run one meld step on a policy batch. Return the seeds and the hall of
    fame, each as token tuples."""
    gpc = model.gp_controller
    actions, _, _ = model.policy.sample(N)
    seeds = [tuple(U.DEAP_to_tokens(ind)) for ind in _individuals(gpc, actions)]
    gpc(actions)
    return seeds, [tuple(U.DEAP_to_tokens(ind)) for ind in gpc.hof]


@pytest.mark.parametrize("eval_tree", [False, True])
def test_only_eval_tree_scores_and_copies_the_edited_tree(eval_tree):
    model = _model(eval_tree=eval_tree)
    gpc = model.gp_controller
    ind, = _individuals(gpc, model.policy.sample(1)[0])
    seed = list(U.DEAP_to_tokens(ind))
    while list(U.DEAP_to_tokens(ind)) == seed:
        ind, = U.multi_mutate(ind, gpc.toolbox.expr_mut, gpc.pset, 1.0)
    edited = list(U.DEAP_to_tokens(ind))

    expected = edited if eval_tree else seed
    assert list(U._finish_tokens(ind.tokenized_repr)) == expected
    assert list(U.DEAP_to_tokens(gpc.toolbox.clone(ind))) == expected


def test_without_eval_tree_the_hall_of_fame_holds_only_seeds():
    """The uDSR release: GP's edits are never scored, so it only resamples."""
    seeds, hof = _meld_step(_model(eval_tree=False))
    assert set(hof) <= set(seeds)


def test_with_eval_tree_the_hall_of_fame_holds_new_unique_expressions():
    seeds, hof = _meld_step(_model(eval_tree=True))
    assert len(set(hof)) == len(hof)
    assert set(hof) - set(seeds)


def test_with_eval_seeds_the_best_seed_reaches_the_hall_of_fame_for_free():
    model = _model(eval_tree=True, eval_seeds=True)
    gpc = model.gp_controller
    actions, _, _ = model.policy.sample(N)
    seeds, _, _ = gpc.algorithm._eval(_individuals(gpc, actions))
    best_seed = min(ind.fitness.values[0] for ind in seeds)

    gpc(actions)

    assert gpc.hof[0].fitness.values[0] <= best_seed
    # The seeds are the policy batch, already charged to the budget.
    assert gpc.nevals == gpc.generations * N
