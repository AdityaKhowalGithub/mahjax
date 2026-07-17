"""Basic agents for the Hong Kong environment."""

import jax
import jax.numpy as jnp

from mahjax._src.types import Array, PRNGKey
from mahjax.hong_kong_mahjong.action import Action


def rule_based_player(state, rng: PRNGKey) -> Array:
    """Choose wins first, then kongs/calls, otherwise a random legal discard."""
    mask = state.legal_action_mask
    priority = jnp.zeros(Action.NUM_ACTION, dtype=jnp.float32)
    priority = priority.at[Action.RON].set(100)
    priority = priority.at[Action.TSUMO].set(100)
    priority = priority.at[34:68].set(30)
    priority = priority.at[Action.OPEN_KAN].set(30)
    priority = priority.at[Action.PON].set(15)
    priority = priority.at[Action.CHI_L : Action.CHI_R + 1].set(10)
    priority = priority.at[Action.PASS].set(1)
    noise = jax.random.uniform(rng, (Action.NUM_ACTION,), maxval=0.5)
    logits = jnp.where(mask, priority + noise, -jnp.inf)
    return jnp.argmax(logits).astype(jnp.int32)


__all__ = ["rule_based_player"]
