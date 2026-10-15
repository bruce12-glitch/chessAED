"""Rules engine for a deterministic Go/Chess hybrid.

The engine is dependency-free and immutable: applying an action returns a new
state. This keeps simulations reproducible and makes states safe as search keys.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, IntEnum
import math
from typing import Iterator, Sequence


class IllegalAction(ValueError):
    """Raised when an action violates the game rules."""


class Piece(IntEnum):
    EMPTY = 0
    P0_STONE = 1
    P0_KING = 2
    P0_ROOK = 3
    P0_KNIGHT = 4
    P1_STONE = -1
    P1_KING = -2
    P1_ROOK = -3
    P1_KNIGHT = -4


class ActionKind(str, Enum):
    PLACE = "place"
    MOVE = "move"
    PASS = "pass"


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    source: int | None = None
    target: int | None = None

    @classmethod
    def place(cls, target: int) -> "Action":
        return cls(ActionKind.PLACE, target=target)

    @classmethod
    def move(cls, source: int, target: int) -> "Action":
        return cls(ActionKind.MOVE, source=source, target=target)

    @classmethod
    def pass_turn(cls) -> "Action":
        return cls(ActionKind.PASS)

    def encode(self, size: int) -> int:
        cells = size * size
        if self.kind is ActionKind.PLACE and self.target is not None:
            return self.target
        if self.kind is ActionKind.MOVE and self.source is not None and self.target is not None:
            return cells + self.source * cells + self.target
        if self.kind is ActionKind.PASS:
            return cells + cells * cells
        raise ValueError(f"malformed action: {self}")

    @classmethod
    def decode(cls, value: int, size: int) -> "Action":
        cells = size * size
        pass_id = cells + cells * cells
        if not 0 <= value <= pass_id:
            raise ValueError(f"action id {value} outside [0, {pass_id}]")
        if value < cells:
            return cls.place(value)
        if value == pass_id:
            return cls.pass_turn()
        value -= cells
        return cls.move(value // cells, value % cells)

    @staticmethod
    def action_space_size(size: int) -> int:
        cells = size * size
        return cells + cells * cells + 1


@dataclass(frozen=True, slots=True)
class Rules:
    size: int = 5
    stone_supply: int = 6
    max_plies: int = 100
    komi: float = 1.5
    rook_bonus: float = 2.0
    knight_bonus: float = 1.0

    def __post_init__(self) -> None:
        if self.size < 5:
            raise ValueError("size must be at least 5")
        if self.stone_supply < 0:
            raise ValueError("stone_supply must be non-negative")
        if self.max_plies < 1:
            raise ValueError("max_plies must be positive")


_PIECE_TO_CHAR = {
    Piece.EMPTY: ".",
    Piece.P0_STONE: "X",
    Piece.P0_KING: "K",
    Piece.P0_ROOK: "R",
    Piece.P0_KNIGHT: "N",
    Piece.P1_STONE: "o",
    Piece.P1_KING: "k",
    Piece.P1_ROOK: "r",
    Piece.P1_KNIGHT: "n",
}
_CHAR_TO_PIECE = {value: key for key, value in _PIECE_TO_CHAR.items()}


def owner(piece: Piece | int) -> int | None:
    value = int(piece)
    if value > 0:
        return 0
    if value < 0:
        return 1
    return None


def piece_for(player: int, magnitude: int) -> Piece:
    return Piece(magnitude if player == 0 else -magnitude)


@dataclass(frozen=True, slots=True)
class GoChessState:
    rules: Rules
    board: tuple[Piece, ...]
    current_player: int
    reserves: tuple[int, int]
    consecutive_passes: int = 0
    ply: int = 0
    history: frozenset[bytes] = frozenset()
    terminal_winner: int | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        cells = self.rules.size * self.rules.size
        if len(self.board) != cells:
            raise ValueError(f"board has {len(self.board)} cells, expected {cells}")
        if self.current_player not in (0, 1):
            raise ValueError("current_player must be 0 or 1")
        if len(self.reserves) != 2 or min(self.reserves) < 0:
            raise ValueError("reserves must contain two non-negative counts")

    @classmethod
    def initial(cls, rules: Rules | None = None) -> "GoChessState":
        rules = rules or Rules()
        size = rules.size
        board = [Piece.EMPTY] * (size * size)
        center = size // 2
        board[0] = Piece.P0_ROOK
        board[center] = Piece.P0_KING
        board[size - 1] = Piece.P0_KNIGHT
        bottom = (size - 1) * size
        board[bottom] = Piece.P1_KNIGHT
        board[bottom + center] = Piece.P1_KING
        board[bottom + size - 1] = Piece.P1_ROOK
        state = cls(
            rules=rules,
            board=tuple(board),
            current_player=0,
            reserves=(rules.stone_supply, rules.stone_supply),
        )
        return replace(state, history=frozenset({state.board_signature()}))

    @classmethod
    def from_ascii(
        cls,
        rows: Sequence[str],
        *,
        current_player: int = 0,
        reserves: tuple[int, int] = (0, 0),
        max_plies: int = 100,
        ply: int = 0,
        consecutive_passes: int = 0,
        komi: float = 1.5,
    ) -> "GoChessState":
        cleaned = tuple(row.replace(" ", "") for row in rows)
        if not cleaned or any(len(row) != len(cleaned) for row in cleaned):
            raise ValueError("ASCII board must be non-empty and square")
        try:
            board = tuple(_CHAR_TO_PIECE[ch] for row in cleaned for ch in row)
        except KeyError as exc:
            raise ValueError(f"unknown board character: {exc.args[0]!r}") from exc
        rules = Rules(
            size=len(cleaned),
            stone_supply=max(reserves),
            max_plies=max_plies,
            komi=komi,
        )
        state = cls(
            rules=rules,
            board=board,
            current_player=current_player,
            reserves=reserves,
            consecutive_passes=consecutive_passes,
            ply=ply,
        )
        state._validate_kings()
        return replace(state, history=frozenset({state.board_signature()}))

    @property
    def size(self) -> int:
        return self.rules.size

    @property
    def num_distinct_actions(self) -> int:
        return Action.action_space_size(self.size)

    def board_signature(self, board: Sequence[Piece] | None = None) -> bytes:
        source = self.board if board is None else board
        return bytes(int(piece) + 4 for piece in source)

    def state_key(self) -> tuple[object, ...]:
        return (
            self.board_signature(),
            self.current_player,
            self.reserves,
            self.consecutive_passes,
            self.rules.max_plies - self.ply,
            self.history,
        )

    def is_terminal(self) -> bool:
        return self.terminal_reason is not None or self.ply >= self.rules.max_plies

    def returns(self) -> tuple[float, float]:
        if not self.is_terminal():
            return (0.0, 0.0)
        winner = self.terminal_winner
        if winner is None:
            margin = self.score_margin()
            if math.isclose(margin, 0.0, abs_tol=1e-12):
                return (0.0, 0.0)
            winner = 0 if margin > 0 else 1
        return (1.0, -1.0) if winner == 0 else (-1.0, 1.0)

    def score_margin(self) -> float:
        area = [0.0, self.rules.komi]
        visited: set[int] = set()
        for index, piece in enumerate(self.board):
            player = owner(piece)
            if player is not None:
                area[player] += 1.0
                magnitude = abs(int(piece))
                if magnitude == 3:
                    area[player] += self.rules.rook_bonus
                elif magnitude == 4:
                    area[player] += self.rules.knight_bonus
                continue
            if index in visited:
                continue
            region, borders = self._empty_region(index)
            visited.update(region)
            if len(borders) == 1:
                area[next(iter(borders))] += len(region)
        return area[0] - area[1]

    def heuristic_value(self) -> float:
        if self.is_terminal():
            return self.returns()[0]
        scale = max(1.0, self.size * self.size / 3)
        return math.tanh(self.score_margin() / scale)

    def legal_actions(self) -> tuple[Action, ...]:
        if self.is_terminal():
            return ()
        candidates: list[Action] = []
        if self.reserves[self.current_player] > 0:
            candidates.extend(
                Action.place(index)
                for index, piece in enumerate(self.board)
                if piece is Piece.EMPTY
            )
        for source, piece in enumerate(self.board):
            if owner(piece) == self.current_player and abs(int(piece)) != 1:
                candidates.extend(self._pseudo_piece_moves(source, piece))
        legal = [action for action in candidates if self._try_apply(action) is not None]
        legal.append(Action.pass_turn())
        return tuple(legal)

    def legal_action_ids(self) -> tuple[int, ...]:
        return tuple(action.encode(self.size) for action in self.legal_actions())

    def apply(self, action: Action | int) -> "GoChessState":
        if isinstance(action, int):
            action = Action.decode(action, self.size)
        if self.is_terminal():
            raise IllegalAction("cannot act in a terminal state")
        if action.kind is ActionKind.PASS:
            return self._apply_pass()
        result = self._try_apply(action)
        if result is None:
            raise IllegalAction(f"illegal action: {self.format_action(action)}")
        return result

    def _try_apply(self, action: Action) -> "GoChessState | None":
        try:
            return self._apply_non_pass(action)
        except IllegalAction:
            return None

    def _apply_pass(self) -> "GoChessState":
        passes = self.consecutive_passes + 1
        reason = "two_passes" if passes >= 2 else None
        next_state = replace(
            self,
            current_player=1 - self.current_player,
            consecutive_passes=passes,
            ply=self.ply + 1,
            terminal_reason=reason,
        )
        return next_state._with_horizon_terminal()

    def _apply_non_pass(self, action: Action) -> "GoChessState":
        board = list(self.board)
        reserves = list(self.reserves)
        direct_king_capture = False

        if action.kind is ActionKind.PLACE:
            if action.target is None or not 0 <= action.target < len(board):
                raise IllegalAction("placement target outside board")
            target = action.target
            if reserves[self.current_player] <= 0:
                raise IllegalAction("no stones left in reserve")
            if board[target] is not Piece.EMPTY:
                raise IllegalAction("placement target is occupied")
            board[target] = piece_for(self.current_player, 1)
            reserves[self.current_player] -= 1
        elif action.kind is ActionKind.MOVE:
            if action.source is None or action.target is None:
                raise IllegalAction("move requires source and target")
            if not 0 <= action.source < len(board) or not 0 <= action.target < len(board):
                raise IllegalAction("move endpoint outside board")
            source, target = action.source, action.target
            moving = board[source]
            if owner(moving) != self.current_player or abs(int(moving)) == 1:
                raise IllegalAction("source is not a movable friendly piece")
            if action not in self._pseudo_piece_moves(source, moving):
                raise IllegalAction("piece cannot move to target")
            captured = board[target]
            direct_king_capture = (
                owner(captured) == 1 - self.current_player and abs(int(captured)) == 2
            )
            board[source] = Piece.EMPTY
            board[target] = moving
        else:
            raise IllegalAction("use pass handling for pass action")

        captured_king = self._remove_surrounded_groups(board, 1 - self.current_player)
        own_group, own_liberties = self._group_and_liberties(board, target)
        if not own_group or not own_liberties:
            raise IllegalAction("suicide is forbidden")

        signature = self.board_signature(board)
        if signature in self.history:
            raise IllegalAction("positional superko violation")

        winner = self.current_player if direct_king_capture or captured_king else None
        reason = "king_capture" if winner is not None else None
        next_state = replace(
            self,
            board=tuple(board),
            current_player=1 - self.current_player,
            reserves=tuple(reserves),
            consecutive_passes=0,
            ply=self.ply + 1,
            history=self.history | {signature},
            terminal_winner=winner,
            terminal_reason=reason,
        )
        return next_state._with_horizon_terminal()

    def _with_horizon_terminal(self) -> "GoChessState":
        if self.terminal_reason is None and self.ply >= self.rules.max_plies:
            return replace(self, terminal_reason="move_limit")
        return self

    def _remove_surrounded_groups(self, board: list[Piece], player: int) -> bool:
        visited: set[int] = set()
        captured_king = False
        for index, piece in enumerate(board):
            if index in visited or owner(piece) != player:
                continue
            group, liberties = self._group_and_liberties(board, index)
            visited.update(group)
            if not liberties:
                for member in group:
                    captured_king |= abs(int(board[member])) == 2
                    board[member] = Piece.EMPTY
        return captured_king

    def _group_and_liberties(
        self, board: Sequence[Piece], start: int
    ) -> tuple[set[int], set[int]]:
        player = owner(board[start])
        if player is None:
            return set(), set()
        group: set[int] = set()
        liberties: set[int] = set()
        stack = [start]
        while stack:
            index = stack.pop()
            if index in group:
                continue
            group.add(index)
            for neighbor in self._orthogonal_neighbors(index):
                neighbor_owner = owner(board[neighbor])
                if neighbor_owner is None:
                    liberties.add(neighbor)
                elif neighbor_owner == player and neighbor not in group:
                    stack.append(neighbor)
        return group, liberties

    def _empty_region(self, start: int) -> tuple[set[int], set[int]]:
        region: set[int] = set()
        borders: set[int] = set()
        stack = [start]
        while stack:
            index = stack.pop()
            if index in region:
                continue
            region.add(index)
            for neighbor in self._orthogonal_neighbors(index):
                player = owner(self.board[neighbor])
                if player is None and neighbor not in region:
                    stack.append(neighbor)
                elif player is not None:
                    borders.add(player)
        return region, borders

    def _pseudo_piece_moves(self, source: int, piece: Piece) -> tuple[Action, ...]:
        row, col = divmod(source, self.size)
        magnitude = abs(int(piece))
        destinations: list[int] = []
        if magnitude == 2:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        self._append_destination(destinations, row + dr, col + dc)
        elif magnitude == 3:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                while self._in_bounds(nr, nc):
                    target = nr * self.size + nc
                    if owner(self.board[target]) == self.current_player:
                        break
                    destinations.append(target)
                    if self.board[target] is not Piece.EMPTY:
                        break
                    nr += dr
                    nc += dc
        elif magnitude == 4:
            for dr, dc in (
                (-2, -1),
                (-2, 1),
                (-1, -2),
                (-1, 2),
                (1, -2),
                (1, 2),
                (2, -1),
                (2, 1),
            ):
                self._append_destination(destinations, row + dr, col + dc)
        return tuple(Action.move(source, target) for target in destinations)

    def _append_destination(self, destinations: list[int], row: int, col: int) -> None:
        if not self._in_bounds(row, col):
            return
        target = row * self.size + col
        if owner(self.board[target]) != self.current_player:
            destinations.append(target)

    def _orthogonal_neighbors(self, index: int) -> Iterator[int]:
        row, col = divmod(index, self.size)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if self._in_bounds(nr, nc):
                yield nr * self.size + nc

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def _validate_kings(self) -> None:
        if self.board.count(Piece.P0_KING) != 1 or self.board.count(Piece.P1_KING) != 1:
            raise ValueError("a non-terminal position must contain exactly one king per player")

    def coord(self, index: int) -> str:
        row, col = divmod(index, self.size)
        return f"{chr(ord('a') + col)}{row + 1}"

    def format_action(self, action: Action) -> str:
        if action.kind is ActionKind.PASS:
            return "pass"
        if action.kind is ActionKind.PLACE and action.target is not None:
            return f"stone@{self.coord(action.target)}"
        if action.source is not None and action.target is not None:
            return f"{self.coord(action.source)}-{self.coord(action.target)}"
        return repr(action)

    def to_ascii(self) -> str:
        rows = []
        for row in range(self.size):
            start = row * self.size
            rows.append("".join(_PIECE_TO_CHAR[p] for p in self.board[start : start + self.size]))
        return "\n".join(rows)

    def observation_tensor(self, player: int | None = None) -> tuple[float, ...]:
        """Return 11 planes flattened in channel-first order."""
        perspective = self.current_player if player is None else player
        cells = self.size * self.size
        planes = [[0.0] * cells for _ in range(11)]
        for index, piece in enumerate(self.board):
            piece_owner = owner(piece)
            if piece_owner is None:
                continue
            side_offset = 0 if piece_owner == perspective else 5
            planes[side_offset + abs(int(piece)) - 1][index] = 1.0
        planes[10] = [1.0 if self.current_player == perspective else 0.0] * cells
        return tuple(value for plane in planes for value in plane)

    def __str__(self) -> str:
        return self.to_ascii()
