"""Optional OpenSpiel adapter for GoChess."""

from __future__ import annotations

try:
    import numpy as np
    import pyspiel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OpenSpiel adapter requires the openspiel extra: pip install -e '.[openspiel]'"
    ) from exc

from gochess.core import Action, GoChessState, Rules


_GAME_TYPE = pyspiel.GameType(
    short_name="python_gochess",
    long_name="GoChess: Go capture with chess movement",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "size": 5,
        "stone_supply": 6,
        "max_plies": 100,
        "komi": 1.5,
    },
)


class GoChessGame(pyspiel.Game):
    def __init__(self, params=None):
        params = params or {}
        self.rules = Rules(
            size=int(params.get("size", 5)),
            stone_supply=int(params.get("stone_supply", 6)),
            max_plies=int(params.get("max_plies", 100)),
            komi=float(params.get("komi", 1.5)),
        )
        game_info = pyspiel.GameInfo(
            num_distinct_actions=GoChessState.initial(self.rules).num_distinct_actions,
            max_chance_outcomes=0,
            num_players=2,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=self.rules.max_plies,
        )
        super().__init__(_GAME_TYPE, game_info, params)

    def new_initial_state(self):
        return GoChessSpielState(self, GoChessState.initial(self.rules))

    def make_py_observer(self, iig_obs_type=None, params=None):
        return GoChessObserver(self)


class GoChessSpielState(pyspiel.State):
    def __init__(self, game: GoChessGame, state: GoChessState):
        super().__init__(game)
        self._state = state

    def current_player(self):
        return pyspiel.PlayerId.TERMINAL if self._state.is_terminal() else self._state.current_player

    def _legal_actions(self, player):
        if player != self._state.current_player or self._state.is_terminal():
            return []
        return sorted(self._state.legal_action_ids())

    def _apply_action(self, action):
        self._state = self._state.apply(action)

    def _action_to_string(self, player, action):
        return self._state.format_action(Action.decode(action, self._state.size))

    def is_terminal(self):
        return self._state.is_terminal()

    def returns(self):
        return list(self._state.returns())

    def clone(self):
        return GoChessSpielState(self.get_game(), self._state)

    def __str__(self):
        return str(self._state)

    def information_state_string(self, player):
        return f"p{player}|to_move={self._state.current_player}|{self._state.to_ascii()}"

    def information_state_tensor(self, player):
        return list(self._state.observation_tensor(player))

    def observation_string(self, player):
        return self.information_state_string(player)

    def observation_tensor(self, player):
        return self.information_state_tensor(player)


class GoChessObserver:
    def __init__(self, game: GoChessGame):
        shape = (11, game.rules.size, game.rules.size)
        self.tensor = np.zeros(shape, dtype=np.float32)
        self.dict = {"observation": self.tensor}

    def set_from(self, state: GoChessSpielState, player):
        values = state._state.observation_tensor(player)
        self.tensor[:] = np.asarray(values, dtype=np.float32).reshape(self.tensor.shape)

    def string_from(self, state: GoChessSpielState, player):
        return state.observation_string(player)


pyspiel.register_game(_GAME_TYPE, GoChessGame)
