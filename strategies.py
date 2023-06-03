"""
Some example strategies for people who want to create a custom, homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""

from __future__ import annotations
import os
import json
import logging
import chess
from chess.pgn import Game
from chess.engine import PlayResult
import random
from engine_wrapper import MinimalEngine
from typing import Any
import requests
from conversation import Conversation
import model
import lichess

logger = logging.getLogger(__name__)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

    pass


# Strategy names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_RUNSYNC_ENDPOINT = os.environ.get("RUNPOD_RUNSYNC_ENDPOINT")
assert RUNPOD_API_KEY is not None, "Please set the RUNPOD_API_KEY environment variable"
assert RUNPOD_RUNSYNC_ENDPOINT is not None, "Please set the RUNPOD_RUNSYNC_ENDPOINT environment variable"


class LlamaEngine(MinimalEngine):
    """Communicates with the serverless GPU endpoint for the DeepLLaMA bot."""

    def search(self, lichess_game: model.Game, board: chess.Board, li: lichess.Lichess, *args: Any) -> PlayResult:
        """Makes a POST request to the given endpoint and returns the move from the response."""
        game = Game.from_board(board)
        pgn = str(game.mainline_moves())
        white_elo = lichess_game.white.rating
        black_elo = lichess_game.black.rating
        pgn = f'[WhiteElo "{white_elo}"]\n[BlackElo "{black_elo}"]\n\n' + pgn.strip()
        color = lichess_game.my_color  # The side to move
        move_count = board.fullmove_number  # The fullmove number of the move to be made by the endpoint

        conversation = Conversation(lichess_game, self, li, "", None)

        # The request data
        data = {
            "input": {
                "pgn": pgn,
                "color": color,
                "move_count": move_count,

                # For controlling inference params from the bot, so the docker image doesn't have to be rebuilt
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 1000,
                "num_beams": 1,
                "do_sample": True,
                "use_cache": True,
            }
        }

        logger.debug(f"Sending request to {RUNPOD_RUNSYNC_ENDPOINT} with data {data}")

        # The headers for the request
        headers = {
            "Authorization": "Bearer " + RUNPOD_API_KEY,
            "Content-Type": "application/json"
        }

        # Make the HTTP request
        response = requests.post(RUNPOD_RUNSYNC_ENDPOINT, headers=headers,
                                 json=data, timeout=300)  # A large timeout

        # Check the status code of the response
        response.raise_for_status()

        # Extract the move from the response
        result = response.json().get('output')
        result = json.loads(result)
        logger.debug(f"Inference response: {result}")
        move_str = result.get("move", None)
        illegal_moves = result.get("illegal_moves", None)
        resign = result.get("resign", None)
        if illegal_moves:
            illegal_moves_message = f"[Move {move_count}] tried {len(illegal_moves)} illegal move{'s' if len(illegal_moves) > 1 else ''}:  {', '.join(illegal_moves)}"
            conversation.send_message("player", illegal_moves_message)
            conversation.send_message("spectator", illegal_moves_message)

        if resign:
            return PlayResult(None, None, resigned=True)

        # Convert the move string to a move
        move = board.parse_san(move_str)

        return PlayResult(move, None)
