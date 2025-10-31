import sys
from dotenv import load_dotenv
import os
from pydantic import BaseModel

import random
import xml.etree.ElementTree as ET
from typing import Literal, TypedDict


from dotenv import load_dotenv
import art
from art.local import LocalBackend
random.seed(42)

import math

import openai
import weave
from openai import AsyncOpenAI
from pydantic import BaseModel

import art

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is required for RULER functionality when using openai/o4-mini."
    )

if not os.environ.get("WANDB_API_KEY"):
    print("WANDB_API_KEY is not set. We'll skip logging metrics to Weights & Biases.")

class TicTacToeGame(TypedDict):
    board: list[list[str]]
    agent_symbol: Literal["x", "o"]
    opponent_symbol: Literal["x", "o"]


def generate_game(board_length: int = 3) -> TicTacToeGame:
    board = [["_" for _ in range(board_length)] for _ in range(board_length)]
    agent_symbol = random.choice(["x", "o"])
    opponent_symbol = "x" if agent_symbol == "o" else "o"
    return {
        "board": board,
        "agent_symbol": agent_symbol,
        "opponent_symbol": opponent_symbol,
    }


def render_board(game: TicTacToeGame) -> str:
    board = game["board"]
    board_length = len(board)
    # print something like this:
    #    1   2   3
    # A  _ | x | x
    # B  o | _ | _
    # C  _ | o | _
    # where _ is an empty cell

    board_str = "   " + "   ".join([str(i + 1) for i in range(board_length)]) + "\n"
    for i in range(board_length):
        board_str += f"{chr(65 + i)}  {board[i][0]} | {board[i][1]} | {board[i][2]}\n"
    return board_str


def get_opponent_move(game: TicTacToeGame) -> tuple[int, int]:
    # get a random empty cell
    empty_cells = [
        (i, j) for i in range(3) for j in range(3) if game["board"][i][j] == "_"
    ]
    return random.choice(empty_cells)


def apply_agent_move(game: TicTacToeGame, move: str) -> None:
    board_length = len(game["board"])

    try:
        root = ET.fromstring(move)
        square = root.text
    except Exception:
        raise ValueError("Invalid xml")

    try:
        row_index = ord(square[0]) - 65
        col_index = int(square[1]) - 1
    except Exception as e:
        print(e)
        raise ValueError("Unable to parse square")

    if (
        row_index < 0
        or row_index >= board_length
        or col_index < 0
        or col_index >= board_length
    ):
        raise ValueError(
            f"Invalid move, row or column out of bounds: {row_index}, {col_index}"
        )

    # check if the move is valid
    if game["board"][row_index][col_index] != "_":
        raise ValueError("Square already occupied")

    game["board"][row_index][col_index] = game["agent_symbol"]


def check_winner(board: list[list[str]]) -> Literal["x", "o", "draw", None]:
    board_length = len(board)
    # check rows
    for row in board:
        if row.count(row[0]) == board_length and row[0] != "_":
            return row[0]
    # check columns
    for col in range(board_length):
        if [board[row][col] for row in range(board_length)].count(
            board[0][col]
        ) == board_length and board[0][col] != "_":
            return board[0][col]

    # top right to bottom left
    upward_diagonal = [board[i][board_length - i - 1] for i in range(board_length)]
    if (
        upward_diagonal.count(upward_diagonal[0]) == board_length
        and upward_diagonal[0] != "_"
    ):
        return upward_diagonal[0]

    # top left to bottom right
    downward_diagonal = [board[i][i] for i in range(board_length)]
    if (
        downward_diagonal.count(downward_diagonal[0]) == board_length
        and downward_diagonal[0] != "_"
    ):
        return downward_diagonal[0]

    # check for draw
    if all(cell != "_" for row in board for cell in row):
        return "draw"
    return None

# Projects can define whatever they need to inside their config objects. In this
# case, we're storing a `litellm_model_name`, which we can use to point
# inference to a specific model on litellm, as well as a `use_thinking` flag,
# which lets us compare models trained (or prompted) to use CoT to models
# trained to just output the answer directly in the same project. Our rollout
# and reward functions will use this config to adjust their behavior, but to ART
# itself it's completely opaque.
class MyConfig(BaseModel):
    use_thinking: bool = False
    # When using LightLLM / LiteLLM gateways you may want to override the
    # underlying model name that is sent to the backend.
    litellm_model_name: str | None = None

    
class TicTacToeScenario(BaseModel):
    step: int

@weave.op
@art.retry(exceptions=(openai.LengthFinishReasonError,))
async def rollout(model: art.Model, scenario: TicTacToeScenario) -> art.Trajectory:
    game = generate_game()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": f"You are a tic-tac-toe player. You are playing against an opponent. Always choose the move most likely to lead to an eventual win. Return your move as an XML object with a single property 'move', like so: <move>A1</move>. Optional moves are 'A1', 'B3', 'C2', etc. You are the {game['agent_symbol']} symbol.",
            }
        ],
        metadata={
            "notebook-id": "tic-tac-toe",
            "step": scenario.step,
        },
        reward=0,
    )

    move_number = 0

    if game["agent_symbol"] == "o":
        starting_opponent_move = get_opponent_move(game)
        game["board"][starting_opponent_move[0]][starting_opponent_move[1]] = game[
            "opponent_symbol"
        ]

    while check_winner(game["board"]) is None:
        trajectory.messages_and_choices.append(
            {"role": "user", "content": render_board(game)}
        )

        messages = trajectory.messages()

        try:
            client = AsyncOpenAI(
                base_url=model.inference_base_url,
                api_key=model.inference_api_key,
            )

            chat_completion = await client.chat.completions.create(
                model=model.get_inference_name(),
                messages=messages,
                max_completion_tokens=1024,
            )
        except openai.LengthFinishReasonError as e:
            raise e
        except Exception as e:
            print("caught exception generating chat completion")
            print(e)
            global failing_trajectory
            failing_trajectory = trajectory
            raise e

        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)

        try:
            apply_agent_move(game, content)
        except ValueError:
            trajectory.reward = -1 + (math.log(move_number + 1) / math.log(100))
            break

        move_number += 1
        if check_winner(game["board"]) is not None:
            break

        opponent_move = get_opponent_move(game)
        game["board"][opponent_move[0]][opponent_move[1]] = game["opponent_symbol"]

    winner = check_winner(game["board"])

    if winner == game["agent_symbol"]:
        trajectory.reward = 1
        trajectory.metrics["win"] = 1
    elif winner == game["opponent_symbol"]:
        trajectory.reward = 0
        trajectory.metrics["win"] = 0
    elif winner == "draw":
        trajectory.reward = 0.5
        trajectory.metrics["win"] = 0.5

    trajectory.metrics["num_moves"] = move_number

    return trajectory

async def benchmark_model(model: art.Model, test_tasks):
    trajectories = await art.gather_trajectories(
        (rollout(model, scenario) for scenario in test_tasks),
        pbar_desc="benchmark",
        max_exceptions=100,
    )
    valid_trajectories = [t for t in trajectories if isinstance(t, art.Trajectory)]
    await model.log(valid_trajectories)


async def main():
    backend = LocalBackend(path="./.art")

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model = art.Model(
        name=f"o3-{timestamp}",
        project="tic-tac-toe",
        config=MyConfig(),
        inference_base_url="https://api.openai.com/v1/",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_model_name="o3",  # This is what actually gets sent to OpenAI
    )

    await model.register(backend)

    if os.getenv("WANDB_API_KEY", ""):
        print("initializing weave")
        weave.init(model.project, settings={"print_call_link": False})

    test_tasks = [
        TicTacToeScenario(step=100),
        TicTacToeScenario(step=101),
        TicTacToeScenario(step=102),
        TicTacToeScenario(step=103),
        TicTacToeScenario(step=104),
        TicTacToeScenario(step=105),
        TicTacToeScenario(step=106),
        TicTacToeScenario(step=107),
    ]

    await benchmark_model(model, test_tasks=test_tasks)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

    print("done")