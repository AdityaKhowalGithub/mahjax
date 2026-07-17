import asyncio
from typing import Any

import jax
import jax.numpy as jnp
from fastapi import FastAPI

from mahjax.hong_kong_mahjong.action import Action
from mahjax.hong_kong_mahjong.env import HongKongMahjong, _replace_state
from mahjax.ui.app import ActionRequest, CreateGameRequest, create_app
from mahjax.ui.game_manager import build_legal_actions_view_hk


def _endpoint(app: FastAPI, path: str, method: str) -> Any:
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set())
    )


def _create_game(app: FastAPI) -> dict:
    create_game = _endpoint(app, "/api/game", "POST")
    return asyncio.run(
        create_game(
            CreateGameRequest(
                env_id="hong_kong_mahjong",
                agent_id="rule_based",
                mode="single",
                seed=7,
                human_seat=0,
                ai_delay_ms=0,
            )
        )
    )


def test_index_exposes_hong_kong_rules_and_flower_area() -> None:
    app = create_app()
    index = _endpoint(app, "/", "GET")
    response = asyncio.run(index())

    assert 'value="hong_kong_mahjong"' in response
    assert 'id="flowerTiles"' in response


def test_create_hong_kong_game_returns_native_ui_contract() -> None:
    state = _create_game(create_app())

    assert state["envId"] == "hong_kong_mahjong"
    assert state["phase"] == "awaiting_human"
    assert state["rules"] == {
        "id": "hkos-v1",
        "name": "Hong Kong Old Style",
        "minimumFaan": 3,
        "maximumFaan": 10,
        "flowers": True,
    }
    assert state["wallRemaining"] < 144
    assert 'data-ruleset="hkos-v1"' in state["svg"]
    assert isinstance(state["hand"]["flowers"], list)
    assert state["legalActions"]["riichi"] is None
    assert state["legalActions"]["tsumogiri"] is None
    assert state["legalActions"]["tsumo"] is None
    assert state["legalActions"]["ron"] is None
    assert state["legalActions"]["pass"] is None
    assert state["legalActions"]["call"] == {}


def test_hong_kong_game_accepts_a_legal_discard() -> None:
    app = create_app()
    state = _create_game(app)
    legal = state["legalActions"]
    action = next(item["action"] for item in legal["discardTiles"] if item["enabled"])
    post_action = _endpoint(app, "/api/game/{game_id}/action", "POST")
    next_state = asyncio.run(post_action(state["gameId"], ActionRequest(action=action)))

    assert next_state["envId"] == "hong_kong_mahjong"
    assert next_state["step"] > state["step"]
    assert next_state["scores"] == state["scores"]


def test_hong_kong_rule_based_selection_uses_hk_agent() -> None:
    app = create_app()
    state = _create_game(app)
    session = app.state.manager.get(state["gameId"])

    assert session.agent.agent_id == "rule_based_hk"


def test_hong_kong_claim_view_only_exposes_current_legal_actions() -> None:
    env = HongKongMahjong(round_mode="single")
    state = env.init(jax.random.PRNGKey(31))
    player = 0
    mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    mask = mask.at[player, Action.CHI_M].set(True)
    mask = mask.at[player, Action.PASS].set(True)
    state = _replace_state(
        state,
        current_player=jnp.int8(player),
        target=jnp.int8(13),
        last_player=jnp.int8(3),
        legal_action_mask=mask,
    )

    view = build_legal_actions_view_hk(state, player)

    assert view["tsumogiri"] is None
    assert view["tsumo"] is None
    assert view["ron"] is None
    assert view["pass"] == {"enabled": True, "action": Action.PASS}
    assert view["call"] == {
        "chi": [
            {
                "action": Action.CHI_M,
                "tiles": [12, 13, 14],
                "labels": ["4p", "5p", "6p"],
            }
        ]
    }
