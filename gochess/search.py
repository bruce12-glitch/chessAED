"""Reference search agents and a bounded exact solver."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Protocol

from gochess.core import Action, GoChessState


class Agent(Protocol):
    def select_action(self, state: GoChessState) -> Action: ...


@dataclass(slots=True)
class RandomAgent:
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select_action(self, state: GoChessState) -> Action:
        return self._rng.choice(state.legal_actions())


@dataclass(slots=True)
class GreedyAgent:
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select_action(self, state: GoChessState) -> Action:
        actions = list(state.legal_actions())
        self._rng.shuffle(actions)
        sign = 1.0 if state.current_player == 0 else -1.0
        return max(actions, key=lambda action: sign * state.apply(action).heuristic_value())


@dataclass(slots=True)
class _Node:
    state: GoChessState
    parent: "_Node | None" = None
    action: Action | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: list["_Node"] = field(default_factory=list)
    untried: list[Action] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class UCTAgent:
    simulations: int = 128
    exploration: float = math.sqrt(2.0)
    rollout_depth: int = 80
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        self._rng = random.Random(self.seed)

    def select_action(self, state: GoChessState) -> Action:
        root = self.search(state)
        if not root.children:
            raise ValueError("terminal state has no action")
        best_visits = max(child.visits for child in root.children)
        finalists = [child for child in root.children if child.visits == best_visits]
        action = self._rng.choice(finalists).action
        if action is None:
            raise RuntimeError("search child is missing its action")
        return action

    def action_statistics(self, state: GoChessState) -> dict[Action, tuple[int, float]]:
        root = self.search(state)
        return {
            child.action: (child.visits, child.mean_value)
            for child in root.children
            if child.action is not None
        }

    def search(self, state: GoChessState) -> _Node:
        root = _Node(state=state, untried=list(state.legal_actions()))
        for _ in range(self.simulations):
            node = root
            while not node.state.is_terminal() and not node.untried and node.children:
                node = self._select_child(node)
            if not node.state.is_terminal() and node.untried:
                action = node.untried.pop(self._rng.randrange(len(node.untried)))
                child_state = node.state.apply(action)
                node = _Node(
                    state=child_state,
                    parent=node,
                    action=action,
                    untried=list(child_state.legal_actions()),
                )
                node.parent.children.append(node)
            value = self._rollout(node.state)
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent
        return root

    def _select_child(self, node: _Node) -> _Node:
        log_parent = math.log(max(1, node.visits))
        sign = 1.0 if node.state.current_player == 0 else -1.0

        def score(child: _Node) -> float:
            exploit = sign * child.mean_value
            explore = self.exploration * math.sqrt(log_parent / child.visits)
            return exploit + explore

        scored = [(score(child), child) for child in node.children]
        best = max(value for value, _ in scored)
        choices = [child for value, child in scored if math.isclose(value, best)]
        return self._rng.choice(choices)

    def _rollout(self, state: GoChessState) -> float:
        current = state
        for _ in range(self.rollout_depth):
            if current.is_terminal():
                return current.returns()[0]
            current = current.apply(self._rng.choice(current.legal_actions()))
        return current.heuristic_value()


class SearchBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SolveResult:
    value: float
    best_action: Action | None
    nodes: int


def solve_exact(state: GoChessState, max_nodes: int = 250_000) -> SolveResult:
    """Solve a short-horizon state by alpha-beta search."""
    nodes = 0
    cache: dict[tuple[object, ...], tuple[float, Action | None]] = {}

    def visit(position: GoChessState, alpha: float, beta: float) -> tuple[float, Action | None]:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise SearchBudgetExceeded(f"exact search exceeded {max_nodes} nodes")
        if position.is_terminal():
            return position.returns()[0], None
        key = position.state_key()
        if key in cache:
            return cache[key]

        actions = list(position.legal_actions())
        actions.sort(
            key=lambda action: abs(position.apply(action).heuristic_value()),
            reverse=True,
        )
        maximizing = position.current_player == 0
        best_value = -math.inf if maximizing else math.inf
        best_action = None
        for action in actions:
            value, _ = visit(position.apply(action), alpha, beta)
            if (maximizing and value > best_value) or (not maximizing and value < best_value):
                best_value, best_action = value, action
            if maximizing:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)
            if alpha >= beta:
                break
        result = (best_value, best_action)
        cache[key] = result
        return result

    value, action = visit(state, -math.inf, math.inf)
    return SolveResult(value=value, best_action=action, nodes=nodes)
