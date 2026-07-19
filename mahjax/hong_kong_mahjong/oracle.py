"""Privileged deterministic policy and reward features for HKOS training.

The oracle deliberately reads ``state.round_state.deck`` and every concealed
hand.  It is therefore suitable as a teacher/reward model, never as a player
observation policy.  All calculations use JAX primitives so the public
functions can be transformed with :func:`jax.jit` and :func:`jax.vmap`.
"""

import jax
import jax.numpy as jnp

from mahjax._src.types import Array
from mahjax.hong_kong_mahjong.action import Action
from mahjax.hong_kong_mahjong.env import _score_result
from mahjax.hong_kong_mahjong.hand import THIRTEEN_ORPHAN_IDX, Hand
from mahjax.hong_kong_mahjong.scoring import HongKongScoring
from mahjax.hong_kong_mahjong.tile import Tile
from mahjax.no_red_mahjong.shanten import Shanten

# oracle_action_features columns.  Keeping this an array (rather than a dict)
# makes batches of features cheap to jit/vmap in RL data generation.
TERMINAL_VALUE = 0
SHANTEN_PROGRESS = 1
LIVE_UKEIRE = 2
HAND_VALUE = 3
SAFETY = 4
NUM_FEATURES = 5

# Offline reward contract. Terminal wins are handled lexicographically below;
# these weights order nonterminal decisions by safety, shanten, exact ukeire,
# and HK hand-value potential. Keep them stable across a frozen dataset.
EXACT_FEATURE_WEIGHTS = jnp.array([0.0, 8.0, 6.0, 2.0, 20.0], dtype=jnp.float32)
EXACT_TERMINAL_PRIORITY = jnp.float32(1_000_000.0)


def hk_shanten(hand: Array) -> Array:
    """Return HKOS shanten: standard or thirteen-orphans, never seven-pairs.

    The result uses conventional notation (``-1`` complete, ``0`` tenpai).
    ``Shanten.normal`` also handles shortened concealed hands after calls.
    """

    hand = jnp.asarray(hand, dtype=jnp.int8)
    standard = Shanten.normal(hand)
    orphans = Shanten.thirteen_orphan(hand)
    return (jnp.minimum(standard, orphans) - 1).astype(jnp.int32)


def _shanten_with_melds(hand: Array, meld_count: Array) -> Array:
    # Thirteen orphans is impossible once a hand has opened or declared a kong.
    standard = Shanten.normal(hand) - 1
    return jnp.where(meld_count > 0, standard, hk_shanten(hand)).astype(jnp.int32)


def _next_live_standard(state) -> tuple[Array, Array, Array]:
    indices = jnp.arange(Tile.NUM_TILE_ID, dtype=jnp.int32)
    live = indices >= state.round_state.wall_index.astype(jnp.int32)
    standard = state.round_state.deck < Tile.NUM_STANDARD_TILE_TYPES
    candidates = live & standard
    index = jnp.min(jnp.where(candidates, indices, Tile.NUM_TILE_ID))
    available = index < Tile.NUM_TILE_ID
    tile = state.round_state.deck[jnp.minimum(index, Tile.NUM_TILE_ID - 1)]
    consumed = jnp.where(
        available,
        index - state.round_state.wall_index.astype(jnp.int32) + 1,
        0,
    )
    return tile.astype(jnp.int32), available, consumed.astype(jnp.int32)


def _post_action_hand(state, action: Array, actor: Array) -> tuple[Array, Array, Array]:
    """Return (concealed hand, meld count, tiles consumed from live wall)."""

    action = action.astype(jnp.int32)
    actor = actor.astype(jnp.int32)
    hand = state.players.hand[actor]
    meld_count = state.players.meld_counts[actor].astype(jnp.int32)
    target = jnp.clip(state.round_state.target.astype(jnp.int32), 0, 33)
    last_draw = jnp.clip(state.round_state.last_draw.astype(jnp.int32), 0, 33)

    is_plain_discard = action < 34
    is_tsumogiri = action == Action.TSUMOGIRI
    discard_tile = jnp.where(is_tsumogiri, last_draw, jnp.clip(action, 0, 33))
    hand = hand.at[discard_tile].add(-(is_plain_discard | is_tsumogiri).astype(jnp.int8))

    is_self_kong = (action >= 34) & (action < 68)
    kong_tile = jnp.clip(action - 34, 0, 33)
    is_added = state.players.pon[actor, kong_tile] > 0
    kong_remove = jnp.where(is_added, 1, 4).astype(jnp.int8) * is_self_kong.astype(jnp.int8)
    hand = hand.at[kong_tile].add(-kong_remove)
    meld_count = meld_count + (is_self_kong & ~is_added).astype(jnp.int32)

    is_pon = action == Action.PON
    is_open_kong = action == Action.OPEN_KAN
    hand = hand.at[target].add(-(2 * is_pon.astype(jnp.int8) + 3 * is_open_kong.astype(jnp.int8)))
    meld_count = meld_count + is_pon.astype(jnp.int32) + is_open_kong.astype(jnp.int32)

    is_chi = (action >= Action.CHI_L) & (action <= Action.CHI_R)
    chi_start = jnp.clip(target - (action - Action.CHI_L), 0, 26)
    chi_tiles = jnp.clip(chi_start + jnp.arange(3), 0, 33)
    chi_delta = jnp.zeros(34, dtype=jnp.int8).at[chi_tiles].add(-is_chi.astype(jnp.int8))
    chi_delta = chi_delta.at[target].add(is_chi.astype(jnp.int8))
    hand = hand + chi_delta
    meld_count = meld_count + is_chi.astype(jnp.int32)

    replacement = is_self_kong | is_open_kong
    replacement_tile, replacement_available, replacement_consumed = _next_live_standard(state)
    add_replacement = replacement & replacement_available
    hand = hand.at[replacement_tile].add(add_replacement.astype(jnp.int8))
    consumed = jnp.where(add_replacement, replacement_consumed, 0)
    return hand, meld_count, consumed.astype(jnp.int32)


def _best_discard_view(hand: Array, meld_count: Array) -> tuple[Array, Array]:
    """Convert a draw/call hand to its best deterministic post-discard view."""

    shanten_now = _shanten_with_melds(hand, meld_count)

    def discard_shanten(tile):
        candidate = hand.at[tile].add(-1)
        return jnp.where(hand[tile] > 0, _shanten_with_melds(candidate, meld_count), 99)

    candidates = jax.vmap(discard_shanten)(jnp.arange(34, dtype=jnp.int32))
    best_tile = jnp.argmin(candidates)
    needs_discard = (hand.sum(dtype=jnp.int32) % 3) == 2
    best_hand = jnp.where(needs_discard, hand.at[best_tile].add(-1), hand)
    best_shanten = jnp.where(needs_discard, candidates[best_tile], shanten_now)
    return best_hand, best_shanten.astype(jnp.int32)


def _live_counts(state, consumed: Array) -> Array:
    indices = jnp.arange(Tile.NUM_TILE_ID, dtype=jnp.int32)
    live = indices >= (state.round_state.wall_index.astype(jnp.int32) + consumed)
    deck = state.round_state.deck.astype(jnp.int32)
    return jax.vmap(lambda tile: jnp.sum(live & (deck == tile)))(jnp.arange(34, dtype=jnp.int32))


def _ukeire(state, hand: Array, meld_count: Array, shanten: Array, consumed: Array) -> Array:
    live_counts = _live_counts(state, consumed)

    def improves(tile):
        next_hand = hand.at[tile].add(1)
        return (_shanten_with_melds(next_hand, meld_count) < shanten) & (hand[tile] < 4)

    useful = jax.vmap(improves)(jnp.arange(34, dtype=jnp.int32))
    return jnp.sum(jnp.where(useful, live_counts, 0), dtype=jnp.float32)


def _hand_value_potential(state, hand: Array, actor: Array, meld_count: Array) -> Array:
    """Smooth proxy for reaching the HK three-faan threshold."""

    suited = hand[:27].reshape(3, 9).sum(axis=1).astype(jnp.float32)
    honors = hand[27:].sum(dtype=jnp.float32)
    flush_focus = (jnp.max(suited) + honors - (suited.sum() - jnp.max(suited))) / 14.0
    triplet_mass = jnp.sum(jnp.minimum(hand, 3) // 2, dtype=jnp.float32) / 7.0

    seat = state.round_state.seat_wind[actor].astype(jnp.int32)
    prevalent = state.round_state.prevalent_wind.astype(jnp.int32)
    honor_weights = jnp.ones(7, dtype=jnp.float32)
    honor_weights = honor_weights.at[seat].add(1.0).at[prevalent].add(1.0)
    honor_weights = honor_weights.at[4:].add(1.0)  # dragons
    valuable_honors = jnp.dot((hand[27:] >= 2).astype(jnp.float32), honor_weights) / 9.0

    orphan_kind = jnp.sum(hand[THIRTEEN_ORPHAN_IDX] > 0, dtype=jnp.float32) / 13.0
    orphan_pair = jnp.any(hand[THIRTEEN_ORPHAN_IDX] >= 2).astype(jnp.float32)
    orphan_value = (orphan_kind + 0.15 * orphan_pair) * (meld_count == 0)
    concealed = (meld_count == 0).astype(jnp.float32)
    flowers = state.players.flowers[actor]
    flower_value = HongKongScoring.flower_faan(flowers, seat).astype(jnp.float32) / 4.0
    return 0.75 * flush_focus + 0.45 * triplet_mass + valuable_honors + orphan_value + 0.2 * concealed + 0.2 * flower_value


def _terminal_value(state, action: Array, actor: Array) -> Array:
    self_draw = action == Action.TSUMO
    ron = action == Action.RON
    tile = jnp.where(self_draw, state.round_state.last_draw, state.round_state.target).astype(jnp.int32)
    tile = jnp.clip(tile, 0, 33)
    result = _score_result(state, actor, tile, self_draw)
    rewards, _ = HongKongScoring.settle(
        actor,
        state.round_state.last_player,
        result.faan,
        self_draw,
        state.round_state.dealer,
    )
    return jnp.where(self_draw | ron, rewards[actor].astype(jnp.float32) / 128.0, 0.0)


def _discard_danger(state, action: Array, actor: Array) -> Array:
    is_discard = (action < 34) | (action == Action.TSUMOGIRI)
    tile = jnp.where(action == Action.TSUMOGIRI, state.round_state.last_draw, action)
    tile = jnp.clip(tile.astype(jnp.int32), 0, 33)
    opponents = jnp.arange(4, dtype=jnp.int32)

    def opponent_loss(player):
        shape = Hand.can_ron(state.players.hand[player], tile)
        result = _score_result(state, player, tile, jnp.bool_(False))
        legal_win = (player != actor) & shape & (result.faan >= 3)
        amount = HongKongScoring.payout(result.faan).astype(jnp.float32)
        dealer_double = (player == state.round_state.dealer) | (actor == state.round_state.dealer)
        loss = amount * jnp.where(dealer_double, 2.0, 1.0)
        return legal_win, jnp.where(legal_win, loss, 0.0)

    eligible, losses = jax.vmap(opponent_loss)(opponents)
    distance = (opponents - actor) % 4
    claimant = jnp.argmin(jnp.where(eligible, distance, 5))
    return jnp.where(is_discard & eligible.any(), losses[claimant] / 256.0, 0.0)


def _added_kong_rob_danger(state, action: Array, actor: Array) -> Array:
    """Exact nearest-winner loss if an added kong can be robbed."""

    is_self_kong = (action >= 34) & (action < 68)
    tile = jnp.clip(action.astype(jnp.int32) - 34, 0, 33)
    is_added = is_self_kong & (state.players.pon[actor, tile] > 0)
    robbing_state = state.replace(
        round_state=state.round_state.replace(robbing_kong=jnp.bool_(True))
    )
    opponents = jnp.arange(4, dtype=jnp.int32)

    def opponent_loss(player):
        shape = Hand.can_ron(state.players.hand[player], tile)
        result = _score_result(robbing_state, player, tile, jnp.bool_(False))
        legal_win = (player != actor) & shape & (result.faan >= 3)
        amount = HongKongScoring.payout(result.faan).astype(jnp.float32)
        dealer_double = (player == state.round_state.dealer) | (actor == state.round_state.dealer)
        loss = amount * jnp.where(dealer_double, 2.0, 1.0)
        return legal_win, jnp.where(legal_win, loss, 0.0)

    eligible, losses = jax.vmap(opponent_loss)(opponents)
    distance = (opponents - actor) % 4
    claimant = jnp.argmin(jnp.where(eligible, distance, 5))
    return jnp.where(is_added & eligible.any(), losses[claimant] / 256.0, 0.0)


def oracle_action_features(state, action: Array, actor: Array) -> Array:
    """Return privileged features for one action from ``actor``'s viewpoint.

    Columns are terminal settlement, shanten progress, exact live-wall ukeire,
    HK hand-value potential, and immediate safety.  Higher is always better.
    """

    action = jnp.asarray(action, dtype=jnp.int32)
    actor = jnp.asarray(actor, dtype=jnp.int32)
    before = _shanten_with_melds(state.players.hand[actor], state.players.meld_counts[actor])
    post_hand, meld_count, consumed = _post_action_hand(state, action, actor)
    effective_hand, after = _best_discard_view(post_hand, meld_count)
    ukeire = _ukeire(state, effective_hand, meld_count, after, consumed)
    potential = _hand_value_potential(state, effective_hand, actor, meld_count)
    danger = jnp.maximum(
        _discard_danger(state, action, actor),
        _added_kong_rob_danger(state, action, actor),
    )
    terminal = _terminal_value(state, action, actor)
    progress = (before - after).astype(jnp.float32)
    safety = 1.0 - danger
    return jnp.array([terminal, progress, ukeire / 144.0, potential, safety], dtype=jnp.float32)


def score_legal_actions(state, actor=None) -> Array:
    """Score every action, normalized to [0, 1] over the legal subset.

    Illegal actions are exactly ``-inf``.  Legal scores are deterministic and
    finite, including positions with only one legal action.
    """

    actor = state.current_player if actor is None else jnp.asarray(actor, dtype=jnp.int32)
    hand = state.players.hand[actor]
    meld_count = state.players.meld_counts[actor]
    before = _shanten_with_melds(hand, meld_count)
    live_counts = _live_counts(state, jnp.int32(0))

    # One exact hidden-wall ukeire calculation is used to break the best
    # shanten/value discard.  Computing 34 draws for every one of 34 discards
    # creates a 1,156-way shanten graph and is too expensive for RL rollouts.
    # The full per-action exact value remains available through
    # ``oracle_action_features`` when constructing reward labels.
    opponent_hands = state.players.hand
    wait_matrix = jax.vmap(
        lambda opponent_hand: jax.vmap(
            lambda tile: Hand.can_ron(opponent_hand, tile)
        )(jnp.arange(34, dtype=jnp.int32))
    )(opponent_hands)

    def discard_metrics(tile):
        candidate = hand.at[tile].add(-1)
        shanten = _shanten_with_melds(candidate, meld_count)

        potential = _hand_value_potential(state, candidate, actor, meld_count)
        # Exact hidden shape-wait danger.  The feature API additionally checks
        # the HK faan floor and settlement amount; this conservative rollout
        # signal may avoid a shape wait that could not yet claim three faan.
        danger = jnp.any(wait_matrix[:, tile] & (jnp.arange(4) != actor)).astype(jnp.float32)
        progress = (before - shanten).astype(jnp.float32)
        return 7.0 * progress + 1.5 * potential + 18.0 * (1.0 - danger)

    discard_raw = jax.vmap(discard_metrics)(jnp.arange(34, dtype=jnp.int32))
    provisional = jnp.argmax(jnp.where(hand > 0, discard_raw, -jnp.inf))
    provisional_hand = hand.at[provisional].add(-1)
    provisional_shanten = _shanten_with_melds(provisional_hand, meld_count)

    def improves(draw):
        next_hand = provisional_hand.at[draw].add(1)
        return (
            _shanten_with_melds(next_hand, meld_count) < provisional_shanten
        ) & (provisional_hand[draw] < 4)

    useful = jax.vmap(improves)(jnp.arange(34, dtype=jnp.int32))
    exact_ukeire = jnp.sum(jnp.where(useful, live_counts, 0), dtype=jnp.float32) / 144.0
    discard_raw = discard_raw.at[provisional].add(5.0 * exact_ukeire)
    raw = jnp.zeros(Action.NUM_ACTION, dtype=jnp.float32).at[:34].set(discard_raw)
    last_draw = jnp.clip(state.round_state.last_draw.astype(jnp.int32), 0, 33)
    raw = raw.at[Action.TSUMOGIRI].set(discard_raw[last_draw] + 1.0e-4)

    # Kongs preserve the basic hand plan and buy a replacement draw.  The
    # hidden next tile gives a small oracle bonus when it advances the hand.
    # Score only legal kongs, from the true post-kong concealed hand after
    # removing kong tiles and consuming the exact replacement draw. A loop
    # keeps the expensive feature body single in the compiled graph and means
    # empty-tile counterfactuals can never influence a legal kong score.
    kong_actions = jnp.concatenate(
        [jnp.arange(34, 68, dtype=jnp.int32), jnp.array([Action.OPEN_KAN], dtype=jnp.int32)]
    )

    def score_one_kong(index, values):
        action = kong_actions[index]

        def legal_score(current):
            post_hand, post_melds, consumed = _post_action_hand(state, action, actor)
            effective, shanten = _best_discard_view(post_hand, post_melds)
            ukeire = _ukeire(state, effective, post_melds, shanten, consumed) / 144.0
            potential = _hand_value_potential(state, effective, actor, post_melds)
            danger = _added_kong_rob_danger(state, action, actor)
            score = (
                7.0 * (before - shanten).astype(jnp.float32)
                + 5.0 * ukeire
                + 1.5 * potential
                + 18.0 * (1.0 - danger)
                + 0.25
            )
            return current.at[action].set(score)

        return jax.lax.cond(state.legal_action_mask[action], legal_score, lambda current: current, values)

    raw = jax.lax.fori_loop(0, kong_actions.shape[0], score_one_kong, raw)

    # Opening is costly under a three-faan minimum.  Honor pungs are the main
    # exception; exact call features remain available to an RL reward model.
    target = jnp.clip(state.round_state.target.astype(jnp.int32), 0, 33)
    target_is_value_honor = (target >= 31) | (target == state.round_state.seat_wind[actor] + 27) | (
        target == state.round_state.prevalent_wind + 27
    )
    pass_base = 18.0 + 1.5 * _hand_value_potential(state, hand, actor, meld_count)
    raw = raw.at[Action.PON].set(pass_base - 1.2 + 5.0 * target_is_value_honor.astype(jnp.float32))
    raw = raw.at[Action.CHI_L].set(pass_base - 1.5)
    raw = raw.at[Action.CHI_M].set(pass_base - 1.5)
    raw = raw.at[Action.CHI_R].set(pass_base - 1.5)
    # Passing retains flexibility and concealment.
    raw = raw.at[Action.PASS].set(pass_base)
    # Legal win masks already enforce complete shape and the three-faan floor.
    raw = raw.at[Action.RON].set(1_000.0).at[Action.TSUMO].set(2_000.0)
    mask = state.legal_action_mask
    legal_min = jnp.min(jnp.where(mask, raw, jnp.inf))
    legal_max = jnp.max(jnp.where(mask, raw, -jnp.inf))
    span = jnp.maximum(legal_max - legal_min, 1.0e-6)
    normalized = (raw - legal_min) / span
    normalized = jnp.where(legal_max == legal_min, 0.0, normalized)
    return jnp.where(mask, normalized, -jnp.inf).astype(jnp.float32)


def score_legal_actions_exact(state, actor=None) -> Array:
    """Offline privileged reward scorer for dataset-label generation.

    Every legal action is evaluated with :func:`oracle_action_features`, hence
    using true post-call/kong shape, exact physical-wall ukeire, exact terminal
    settlement, and faan-aware immediate deal-in/rob danger. Nonterminal raw
    utility is ``dot(EXACT_FEATURE_WEIGHTS, features)``. A legal RON/TSUMO is
    lexicographically dominant, with exact settlement breaking terminal ties.
    Legal utilities are min-max normalized to ``[0, 1]``; illegal entries are
    exactly ``-inf``.
    """

    actor = state.current_player if actor is None else jnp.asarray(actor, dtype=jnp.int32)
    mask = state.legal_action_mask
    initial = jnp.full(Action.NUM_ACTION, -jnp.inf, dtype=jnp.float32)

    def score_one(action, raw):
        def legal_score(values):
            features = oracle_action_features(state, action, actor)
            utility = jnp.dot(EXACT_FEATURE_WEIGHTS, features)
            is_terminal = (action == Action.RON) | (action == Action.TSUMO)
            utility = jnp.where(
                is_terminal,
                EXACT_TERMINAL_PRIORITY + features[TERMINAL_VALUE],
                utility,
            )
            return values.at[action].set(utility)

        return jax.lax.cond(mask[action], legal_score, lambda values: values, raw)

    raw = jax.lax.fori_loop(0, Action.NUM_ACTION, score_one, initial)
    legal_min = jnp.min(jnp.where(mask, raw, jnp.inf))
    legal_max = jnp.max(jnp.where(mask, raw, -jnp.inf))
    span = jnp.maximum(legal_max - legal_min, 1.0e-6)
    normalized = jnp.where(legal_max == legal_min, 0.0, (raw - legal_min) / span)
    return jnp.where(mask, normalized, -jnp.inf).astype(jnp.float32)


def oracle_policy(state) -> Array:
    """Return the highest-scoring legal action with deterministic tie-breaks."""

    return jnp.argmax(score_legal_actions(state)).astype(jnp.int32)


__all__ = [
    "LIVE_UKEIRE",
    "EXACT_FEATURE_WEIGHTS",
    "EXACT_TERMINAL_PRIORITY",
    "NUM_FEATURES",
    "SAFETY",
    "SHANTEN_PROGRESS",
    "HAND_VALUE",
    "TERMINAL_VALUE",
    "hk_shanten",
    "oracle_action_features",
    "oracle_policy",
    "score_legal_actions",
    "score_legal_actions_exact",
]
