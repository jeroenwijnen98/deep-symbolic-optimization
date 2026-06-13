"""Constant optimizer used for deep symbolic optimization."""

from functools import partial

import numpy as np
from scipy.optimize import minimize


def _map_gurobi_status(status_code):
    """Map a gurobipy GRB.Status code to a unified status string."""
    from gurobipy import GRB
    # Build defensively: some constants (e.g. WORK_LIMIT, MEM_LIMIT) do not
    # exist in Gurobi 9.1.2.  Use getattr with a sentinel so missing attrs are
    # silently skipped rather than raising AttributeError.
    _SENTINEL = object()
    raw_mapping = {
        "OPTIMAL": "optimal",
        "INFEASIBLE": "infeasible",
        "INF_OR_UNBD": "unbounded",
        "UNBOUNDED": "unbounded",
        "TIME_LIMIT": "limit_reached",
        "ITERATION_LIMIT": "limit_reached",
        "NODE_LIMIT": "limit_reached",
        "SOLUTION_LIMIT": "limit_reached",
        "WORK_LIMIT": "limit_reached",
        "MEM_LIMIT": "limit_reached",
        "NUMERIC": "numeric",
        "SUBOPTIMAL": "numeric",
        "INTERRUPTED": "interrupted",
    }
    mapping = {}
    for attr_name, unified in raw_mapping.items():
        code = getattr(GRB, attr_name, _SENTINEL)
        if code is not _SENTINEL:
            mapping[code] = unified
    return mapping.get(status_code, "other")


_SCIPY_STATUS_MAP = {0: "optimal", 1: "limit_reached", 2: "numeric"}


def _map_scipy_status(status_code):
    return _SCIPY_STATUS_MAP.get(status_code, "other")


def make_const_optimizer(name, **kwargs):
    """Returns a ConstOptimizer given a name and keyword arguments"""

    const_optimizers = {
        None : Dummy,
        "dummy" : Dummy,
        "scipy" : ScipyMinimize,
        "gurobi" : GurobiConstOptimizer,
    }

    return const_optimizers[name](**kwargs)


class ConstOptimizer(object):
    """Base class for constant optimizer"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs


    def __call__(self, f, x0, program=None):
        """
        Optimizes an objective function from an initial guess.

        The objective function is the negative of the base reward (reward
        without penalty) used for training. Optimization excludes any penalties
        because they are constant w.r.t. to the constants being optimized.

        Parameters
        ----------
        f : function mapping np.ndarray to float
            Objective function (negative base reward).

        x0 : np.ndarray
            Initial guess for constant placeholders.

        program : Program, optional
            The Program instance being optimized. Accepted for interface
            compatibility; subclasses may use it if needed.

        Returns
        -------
        x : np.ndarray
            Vector of optimized constants.
        """
        raise NotImplementedError


class Dummy(ConstOptimizer):
    """Dummy class that selects the initial guess for each constant"""

    def __init__(self, **kwargs):
        super(Dummy, self).__init__(**kwargs)

    
    def __call__(self, f, x0, program=None):
        if program is not None:
            program.hard_status = None
            program.hard_has_solution = None
            program.elastic_status = None
            program.elastic_has_solution = None
        return x0


class ScipyMinimize(ConstOptimizer):
    """SciPy's non-linear optimizer"""

    def __init__(self, **kwargs):
        super(ScipyMinimize, self).__init__(**kwargs)


    def __call__(self, f, x0, program=None):
        try:
            with np.errstate(divide='ignore'):
                opt_result = partial(minimize, **self.kwargs)(f, x0)
            if program is not None:
                program.hard_status = _map_scipy_status(opt_result.status)
                program.hard_has_solution = True
                program.elastic_status = None
                program.elastic_has_solution = None
            return opt_result['x']
        except Exception:
            if program is not None:
                program.hard_status = "error"
                program.hard_has_solution = False
                program.elastic_status = None
                program.elastic_has_solution = None
            return x0


class GurobiConstOptimizer(ConstOptimizer):
    """Gurobi-based inner constant optimizer for DSR trees.

    Solves the constrained inner problem: minimize positive-residual welfare
    loss subject to an aggregate budget constraint, by compiling the DSR
    expression tree into an algebraic Gurobi model.

    For trees that are affine in their constants the inner problem is a convex
    LP (pos_resid) or QP (pos_squared) solved fast and globally.  For trees
    where constants appear nonlinearly, NonConvex=2 is enabled and an optional
    time_limit (seconds) may be set.

    Budget status contract (written to program after each call with constants):
      program.budget_status   : "feasible" | "infeasible" | "unknown"
      program.budget_violation: float >= 0 for "infeasible", 0.0 for "feasible",
                                None for "unknown"

    Feasibility is determined by trust-but-verify: after extracting candidate
    constants from any solve (hard or elastic), the program is executed in numpy
    with those constants and the budget shortfall is measured directly.  The
    solver's own feasibility claim is never trusted, because on numerically
    ill-conditioned non-convex bilinear models Gurobi 9.1.2 can return
    incumbents whose true constraint violation is orders of magnitude larger
    than the reported tolerance.

    When program is None or has no constants, budget_status is left as-is (None)
    so the reward function can handle those programs separately.

    On any unrecoverable error (import, Gurobi crash) budget_status is set to
    "unknown" and x0 is returned so the RL search always continues.
    """

    def __init__(self, budget_slack=0.0, loss_function="pos_squared",
                 lower_bound=-1e5, upper_bound=1e5, time_limit=None,
                 **kwargs):
        super(GurobiConstOptimizer, self).__init__(**kwargs)
        self.budget_slack = budget_slack
        self.loss_function = loss_function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.time_limit = time_limit

    @staticmethod
    def _verify_feasibility(program, X, y, consts, budget_slack):
        """Ground-truth feasibility check: execute the program in numpy with the
        candidate constants and measure the budget shortfall directly, rather
        than trusting the solver's feasibility claim (which can be violated by
        large margins on numerically difficult non-convex models).

        Returns (status, violation): ("feasible", 0.0), ("infeasible", v>0),
        or ("unknown", None) if the program's output is not finite.
        """
        program.set_constants(consts)
        y_hat = program.execute(X)
        if y_hat is None or not np.all(np.isfinite(y_hat)):
            return ("unknown", None)
        shortfall = float(np.sum(y) - np.sum(y_hat))
        tol = 1e-9 * max(1.0, abs(float(np.sum(y))))
        if shortfall <= budget_slack + tol:
            return ("feasible", 0.0)
        else:
            return ("infeasible", float(shortfall - budget_slack))

    def __call__(self, f, x0, program=None):
        # Rule 1: no constants → stamp status and return x0.
        if program is None or len(program.const_pos) == 0:
            if program is not None:
                program.hard_status = "no_constants"
                program.hard_has_solution = None
                program.elastic_status = None
                program.elastic_has_solution = None
            return x0

        try:
            # Lazy imports: keeps dso importable without Gurobi / the tax repo.
            from solvers.tax_model import TaxModel
            from solvers.tree_to_gurobi import classify_tree, build_preds

            # Initialise elastic attrs; they stay None unless we reach the
            # elastic phase.
            program.elastic_status = None
            program.elastic_has_solution = None

            # Data from the program's task.
            X = program.task.X_train
            y = program.task.y_train

            # Helper: apply nonlinear settings to a freshly built model.
            def _apply_nonlinear_settings(m, kind):
                if kind == "nonlinear":
                    m.Params.NonConvex = 2
                    if self.time_limit is not None:
                        m.Params.TimeLimit = self.time_limit

            # ----------------------------------------------------------------
            # Rule 2: build and solve the HARD-constrained model.
            # ----------------------------------------------------------------
            tm = TaxModel("gurobipy")
            m = tm.solver
            m.Params.OutputFlag = 0

            kind = classify_tree(program.traversal, program.library)
            _apply_nonlinear_settings(m, kind)

            n_consts = len(program.const_pos)
            const_vars = [
                tm.add_variable(f"const_{j}", self.lower_bound, self.upper_bound)
                for j in range(n_consts)
            ]
            m.update()

            preds = build_preds(m, program.traversal, X, const_vars,
                                program.library)

            tm.add_residuals_budget_objective(
                preds, y, self.budget_slack, self.loss_function
            )
            tm.optimize()

            # Stamp hard-solve status immediately after optimize returns.
            program.hard_status = _map_gurobi_status(m.Status)
            program.hard_has_solution = m.SolCount > 0

            # Rule 3: hard solve found an incumbent → verify in numpy.
            if m.SolCount > 0:
                consts = np.array([v.X for v in const_vars])
                status, violation = self._verify_feasibility(
                    program, X, y, consts, self.budget_slack
                )
                program.budget_status = status
                program.budget_violation = violation
                # elastic attrs remain None (elastic phase was skipped).
                return consts

            # ----------------------------------------------------------------
            # Rule 4: no incumbent from hard solve → run ELASTIC solve.
            # ----------------------------------------------------------------
            tm2 = TaxModel("gurobipy")
            m2 = tm2.solver
            m2.Params.OutputFlag = 0
            _apply_nonlinear_settings(m2, kind)

            const_vars2 = [
                tm2.add_variable(f"const_{j}", self.lower_bound, self.upper_bound)
                for j in range(n_consts)
            ]
            m2.update()

            preds2 = build_preds(m2, program.traversal, X, const_vars2,
                                 program.library)

            # The returned violation variable is intentionally unused: the call
            # is needed for its side effect (elastic constraint + objective),
            # but feasibility is decided by _verify_feasibility in numpy, never
            # by the solver's claimed violation.
            tm2.add_budget_violation_objective(preds2, y, self.budget_slack)
            tm2.optimize()

            # Stamp elastic-solve status (hard_status was already stamped above).
            program.elastic_status = _map_gurobi_status(m2.Status)
            program.elastic_has_solution = m2.SolCount > 0

            if m2.SolCount > 0:
                consts2 = np.array([cv.X for cv in const_vars2])
                status2, violation2 = self._verify_feasibility(
                    program, X, y, consts2, self.budget_slack
                )
                program.budget_status = status2
                program.budget_violation = violation2
                return consts2
            else:
                # Elastic solve also has no incumbent.
                program.budget_status = "unknown"
                program.budget_violation = None
                return x0

        except Exception:
            # Catches ImportError, gurobipy.GurobiError, and anything else so
            # the RL search always continues.
            if program is not None:
                program.hard_status = "error"
                program.hard_has_solution = False
                program.elastic_status = None
                program.elastic_has_solution = None
                program.budget_status = "unknown"
                program.budget_violation = None
            return x0
