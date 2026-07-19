import jax
import jax.numpy as jnp

import mahjax
from mahjax.hong_kong_mahjong.action import Action
from mahjax.hong_kong_mahjong.env import _replace_state
from mahjax.hong_kong_mahjong.meld import Meld
from mahjax.hong_kong_mahjong.oracle import (
    EXACT_FEATURE_WEIGHTS,
    NUM_FEATURES,
    _added_kong_rob_danger,
    _post_action_hand,
    hk_shanten,
    oracle_action_features,
    oracle_policy,
    score_legal_actions,
    score_legal_actions_exact,
)


def _hand(indices):
    return jnp.zeros(34, dtype=jnp.int8).at[jnp.asarray(indices)].add(1)


def test_hk_shanten_excludes_seven_pairs_and_allows_thirteen_orphans():
    seven_pairs = jnp.zeros(34, dtype=jnp.int8).at[jnp.arange(27, 34)].set(2)
    orphans = _hand([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 33])
    assert hk_shanten(seven_pairs) > -1
    assert hk_shanten(orphans) == -1
    assert jax.jit(hk_shanten)(orphans) == -1


def test_scores_are_normalized_finite_for_legal_and_negative_infinity_for_illegal():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = env.init(jax.random.PRNGKey(7))
    scores = score_legal_actions(state)
    mask = state.legal_action_mask
    assert scores.shape == (Action.NUM_ACTION,)
    assert jnp.isfinite(scores[mask]).all()
    assert ((scores[mask] >= 0) & (scores[mask] <= 1)).all()
    assert jnp.isneginf(scores[~mask]).all()


def test_oracle_is_deterministic_jittable_and_never_selects_an_illegal_action():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    policy = jax.jit(oracle_policy)
    step = jax.jit(env.step)
    state = env.init(jax.random.PRNGKey(19))
    for _ in range(40):
        if bool(state.terminated | state.truncated):
            break
        action = policy(state)
        assert state.legal_action_mask[action]
        assert action == policy(state)
        state = step(state, action)


def test_oracle_takes_terminal_win_and_reports_features():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = env.init(jax.random.PRNGKey(24))
    winner = jnp.int8(1)
    pre_win = _hand([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 22])
    complete = pre_win.at[22].add(1)
    mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_).at[winner, Action.TSUMO].set(True)
    state = _replace_state(
        state,
        current_player=winner,
        dealer=jnp.int8(3),
        hand=jnp.zeros_like(state.players.hand).at[winner].set(complete),
        last_draw=jnp.int8(22),
        discard_counts=state.players.discard_counts.at[0].set(2),
        legal_action_mask=mask,
    )
    features = oracle_action_features(state, jnp.int32(Action.TSUMO), winner)
    assert features.shape == (NUM_FEATURES,)
    assert jnp.isfinite(features).all()
    assert oracle_policy(state) == Action.TSUMO


def test_true_post_kong_counts_and_physical_replacement_consumption():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = env.init(jax.random.PRNGKey(31))
    actor = jnp.int8(0)
    # Two flowers followed by tile 6 means the replacement consumes 3 wall slots.
    ix = state.round_state.wall_index
    deck = state.round_state.deck.at[ix].set(34).at[ix + 1].set(39).at[ix + 2].set(6)
    closed = jnp.zeros(34, dtype=jnp.int8).at[0].set(4).at[9].set(2).at[10].set(2).at[11].set(2).at[18].set(2).at[19].set(2)
    state = _replace_state(state, current_player=actor, hand=state.players.hand.at[actor].set(closed), deck=deck)
    post, melds, consumed = _post_action_hand(state, jnp.int32(34), actor)
    assert post[0] == 0
    assert post[6] == 1
    assert post.sum() == closed.sum() - 3
    assert melds == 1
    assert consumed == 3

    # Added kong removes one tile, adds one replacement, and does not add a meld.
    added_state = _replace_state(
        state,
        hand=state.players.hand.at[actor].set(closed.at[0].set(1)),
        meld_counts=state.players.meld_counts.at[actor].set(1),
        pon=state.players.pon.at[actor, 0].set(1),
    )
    post, melds, consumed = _post_action_hand(added_state, jnp.int32(34), actor)
    assert post[0] == 0 and post[6] == 1
    assert post.sum() == added_state.players.hand[actor].sum()
    assert melds == 1 and consumed == 3

    # Open kong removes three concealed copies and gets the replacement.
    open_state = _replace_state(state, target=jnp.int8(0))
    post, melds, consumed = _post_action_hand(open_state, jnp.int32(Action.OPEN_KAN), actor)
    assert post[0] == 1 and post[6] == 1
    assert post.sum() == closed.sum() - 2
    assert melds == 1 and consumed == 3


def test_added_kong_rob_danger_is_faan_aware_and_fast_scores_stay_legal():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = env.init(jax.random.PRNGKey(32))
    actor, winner = jnp.int8(0), jnp.int8(1)
    waiting = jnp.zeros(34, dtype=jnp.int8).at[0].set(1).at[jnp.array([1, 11, 12, 22])].set(3)
    actor_hand = jnp.zeros(34, dtype=jnp.int8).at[0].set(1).at[jnp.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22])].set(1)
    melds = state.players.melds.at[actor, 0].set(Meld.init(Action.PON, jnp.int8(0), jnp.int8(3)))
    mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_).at[actor, 34].set(True).at[actor, 3].set(True)
    state = _replace_state(
        state,
        current_player=actor,
        hand=state.players.hand.at[actor].set(actor_hand).at[winner].set(waiting),
        melds=melds,
        meld_counts=state.players.meld_counts.at[actor].set(1),
        pon=state.players.pon.at[actor, 0].set(1),
        discard_counts=state.players.discard_counts.at[2].set(1),
        legal_action_mask=mask,
    )
    assert _added_kong_rob_danger(state, jnp.int32(34), actor) > 0
    fast = jax.jit(score_legal_actions)(state)
    assert jnp.isfinite(fast[mask[actor]]).all()
    assert jnp.isneginf(fast[~mask[actor]]).all()
    # Empty tiles and every other illegal counterfactual are excluded entirely.
    assert jnp.isneginf(fast[1]) and actor_hand[1] == 0


def test_exact_scorer_terminal_dominance_and_feature_weight_alignment():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = env.init(jax.random.PRNGKey(33))
    actor = state.current_player
    legal_discards = jnp.flatnonzero(state.players.hand[actor] > 0)
    first, second = legal_discards[0], legal_discards[1]
    mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_).at[actor, first].set(True).at[actor, second].set(True)
    state = _replace_state(state, legal_action_mask=mask)
    exact = jax.jit(score_legal_actions_exact)(state)
    features = jnp.stack([
        oracle_action_features(state, first, actor),
        oracle_action_features(state, second, actor),
    ])
    raw = features @ EXACT_FEATURE_WEIGHTS
    expected = (raw - raw.min()) / jnp.maximum(raw.max() - raw.min(), 1.0e-6)
    assert jnp.allclose(exact[jnp.array([first, second])], expected)
    assert jnp.isneginf(exact[~state.legal_action_mask]).all()

    win_state = env.init(jax.random.PRNGKey(24))
    winner = jnp.int8(1)
    pre_win = _hand([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 22])
    complete = pre_win.at[22].add(1)
    win_mask = (
        jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        .at[winner, Action.TSUMO].set(True)
        .at[winner, 0].set(True)
    )
    win_state = _replace_state(
        win_state,
        current_player=winner,
        dealer=jnp.int8(3),
        hand=jnp.zeros_like(win_state.players.hand).at[winner].set(complete),
        last_draw=jnp.int8(22),
        discard_counts=win_state.players.discard_counts.at[0].set(2),
        legal_action_mask=win_mask,
    )
    win_scores = jax.jit(score_legal_actions_exact)(win_state)
    assert win_scores[Action.TSUMO] == 1.0
    assert win_scores[Action.TSUMO] > win_scores[0]
