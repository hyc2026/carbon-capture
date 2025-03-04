# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import math
import numpy as np
from os import path
from random import choice, randint, randrange, sample, seed
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from .helpers import board_agent, Board, WorkerAction, RecrtCenterAction, Occupation, ConnectedField4
from .idgen import new_worker_id, new_recrtCenter_id, new_tree_id, reset as reset_ids
from zerosum_env import utils


def get_col_row(size, pos):
    return pos % size, pos // size


def get_to_pos(size, pos, direction):
    col, row = get_col_row(size, pos)
    if direction == "UP":
        return pos - size if pos >= size else size ** 2 - size + col
    elif direction == "DOWN":
        return col if pos + size >= size ** 2 else pos + size
    elif direction == "RIGHT":
        return pos + 1 if col < size - 1 else row * size
    elif direction == "LEFT":
        return pos - 1 if col > 0 else (row + 1) * size - 1


@board_agent
def random_agent(board):
    me = board.current_player
    remaining_carbon = me.cash
    workers = me.workers
    # randomize worker order
    workers = sample(workers, len(workers))
    for worker in workers:
        if worker.cell.carbon > worker.carbon and randint(0, 1) == 0:
            # 50% chance to mine
            continue
        if worker.cell.recrtCenter is None and remaining_carbon > board.configuration.plant_cost:
            # 5% chance to convert at any time
            if randint(0, 19) == 0:
                # remaining_carbon -= board.configuration.plant_cost
                # worker.next_action = WorkerAction.CONVERT
                continue
            # 50% chance to convert if there are no recrtCenters
            if randint(0, 1) == 0 and len(me.recrtCenters) == 0:
                # remaining_carbon -= board.configuration.plant_cost
                # worker.next_action = WorkerAction.CONVERT
                continue
        # None represents the chance to do nothing
        worker.next_action = choice(WorkerAction.moves())
    recrtCenters = me.recrtCenters
    # randomize recrtCenter order
    recrtCenters = sample(recrtCenters, len(recrtCenters))
    worker_count = len(board.next().current_player.workers)
    for recrtCenter in recrtCenters:
        # If there are no workers, always spawn if possible
        if worker_count == 0 and remaining_carbon > board.configuration.rec_collector_cost:
            remaining_carbon -= board.configuration.rec_collector_cost
            recrtCenter.next_action = RecrtCenterAction.RECCOLLECTOR
        # 20% chance to spawn if no workers
        elif randint(0, 4) == 0 and remaining_carbon > board.configuration.rec_planter_cost:
            remaining_carbon -= board.configuration.rec_planter_cost
            recrtCenter.next_action = RecrtCenterAction.RECPLANTER


agents = {"random": random_agent}


def populate_board(state, env):
    full_obs = state[0].observation
    config = env.configuration
    size = env.configuration.size

    # Set seed for random number generators
    if not hasattr(config, "randomSeed"):
        max_int_32 = (1 << 31) - 1
        config.randomSeed = randrange(max_int_32)

    np_rs = RandomState(MT19937(SeedSequence(config.randomSeed)))
    # np.random.seed(config.randomSeed)
    # seed(config.randomSeed)

    # Distribute Carbon evenly into quartiles.
    half = math.ceil(size / 2)

    min_carbon_cells = int(math.ceil(config.startingCarbon / config.startingCellCarbon / 4))
    n_carbon_cells = int(math.ceil(half * half * config.carbonCoverage / 100))
    n_carbon_cells = max(n_carbon_cells, min_carbon_cells)
    carbon_mean = min(config.startingCarbon / 4 / n_carbon_cells, config.startingCellCarbon)
    assert carbon_mean > 0
    half_grid_carbon = np.random.uniform(2 * carbon_mean, size=n_carbon_cells)
    half_grid_indices = np.random.choice((half * half), size=n_carbon_cells, replace=False)
    half_grid = np.zeros(half*half, np.float16)
    half_grid[half_grid_indices] = half_grid_carbon
    half_grid = half_grid.reshape((half, half))

    # Normalize the available carbon against the defined configuration starting carbon.
    full_obs.carbon = [0] * (size ** 2)
    for r, row in enumerate(half_grid):
        for c, val in enumerate(row):
            val = min(int(val), config.startingCellCarbon)
            full_obs.carbon[size * r + c] = val
            full_obs.carbon[size * r + (size - c - 1)] = val
            full_obs.carbon[size * (size - 1) - (size * r) + c] = val
            full_obs.carbon[size * (size - 1) - (size * r) + (size - c - 1)] = val

    # Distribute the starting workers evenly.
    num_agents = len(state)
    num_bases = config.numberOfBases
    starting_positions = [0] * num_agents * num_bases
    starting_positions[0] = size * (size // 4 + config.startPosOffset) + size // 4 + config.startPosOffset
    starting_positions[1] = size * (3 * size // 4 - config.startPosOffset) + 3 * size // 4 - config.startPosOffset
    starting_positions[2] = size * (size // 4 + config.startPosOffset) + 3 * size // 4 - config.startPosOffset
    starting_positions[3] = size * (3 * size // 4 - config.startPosOffset) + size // 4 + config.startPosOffset

    for position in starting_positions:
        full_obs.carbon[position] = 0

    # Initialize the players.
    reset_ids()
    full_obs.players = []
    for i in range(num_agents):
        # workers = {new_worker_id(i): [starting_positions[i], 0, '']}
        base_start_index = i * num_bases
        base_dict = {new_recrtCenter_id(i): starting_positions[base_start_index+j] for j in range(num_bases)}
        # tree = {new_tree_id(i): starting_positions[i]+3}
        full_obs.players.append([state[0].reward, base_dict, {}, {}])

    full_obs.trees = {}

    full_obs.full_carbon = full_obs.carbon
    for agent_state in state:
        agent_state.observation.carbon = [0] * (size ** 2)
    return state


def compute_next_board(state, env):
    full_obs = state[0].observation
    config = env.configuration

    full_obs.carbon = full_obs.full_carbon  # hack here for assuming carbon is fully observable!

    # Interpreter invoked here
    actions = [agent.action for agent in state]
    board = Board(full_obs, config, actions)
    board = board.next()
    state[0].observation = full_obs = utils.structify(board.observation)
    full_obs.full_carbon = full_obs.carbon

    # Remove players with invalid status or insufficient potential.
    for index, agent in enumerate(state):
        player_cash, recrtCenters, workers, trees = full_obs.players[index]
        if agent.status == "ACTIVE":
            collector, planter = [], []
            for worker in workers.values():
                _, _, worker_type = worker
                if worker_type == str(Occupation.COLLECTOR):
                    collector.append(worker)
                if worker_type == str(Occupation.PLANTER):
                    planter.append(worker)
            # 无捕碳员 且无种树员 且无树 且（无转化中心 或（玩家金额不足以 招募捕碳员 或 招募种树员且种一棵树 ））
            if len(workers) == 0 and len(trees) == 0 and (
                    len(recrtCenters) == 0 or player_cash < min(config.recCollectorCost, config.recPlanterCost)):
                # Agent can no longer gather any cash
                agent.status = "DONE"
                agent.reward = board.step - board.configuration.episode_steps - 1

        if agent.status != "ACTIVE" and agent.status != "DONE":
            full_obs.players[index] = [0, recrtCenters, {}, {}]

    # Check if done (< 2 players and num_agents > 1)
    if len(state) > 1 and sum(1 for agent in state if agent.status == "ACTIVE") < 2:
        for agent in state:
            if agent.status == "ACTIVE":
                agent.status = "DONE"

    # Update Rewards.
    for index, agent in enumerate(state):
        if agent.status == "ACTIVE":
            agent.reward = full_obs.players[index][0]
        elif agent.status != "DONE":
            agent.reward = 0

    # Filter unseen carbon(s) on map
    for index, agent in enumerate(state):
        agent.observation.carbon = [0] * (config.size ** 2)
        visible_positions = []
        for worker in board.players[index].workers:
            if worker.is_collector:
                visible_positions.append(worker.position)
            elif worker.is_planter:
                for direction in ConnectedField4:
                    position = worker.position.translate(direction, config.size)
                    visible_positions.append(position)

        for visible_position in visible_positions:
            index = visible_position.to_index(config.size)
            agent.observation.carbon[index] = full_obs.full_carbon[index]
    return state


def interpreter(state, env):
    # Initialize the board (place cell carbon and starting workers).
    if env.done:
        new_state = populate_board(state, env)
    else:
        new_state = compute_next_board(state, env)

    return new_state


def renderer(state, env):
    config = env.configuration
    size = config.size
    obs = state[0].observation

    board = [[h, -1, -1, -1, "", -1, -1] for h in obs.carbon]
    for index, player in enumerate(obs.players):
        _, recrtCenters, workers, trees = player
        for recrtCenter_pos in recrtCenters.values():
            board[recrtCenter_pos][1] = index
        for worker in workers.values():
            worker_pos, worker_carbon, worker_type = worker
            board[worker_pos][2] = index
            board[worker_pos][3] = worker_carbon
            board[worker_pos][4] = worker_type
        for tree_pos, tree_lifecycle in trees.values():
            board[tree_pos][5] = index
            board[tree_pos][6] = tree_lifecycle

    col_divider = "|"
    row_divider = "+" + "+".join(["----"] * size) + "+\n"

    out = row_divider
    for row in range(size):

        for col in range(size):
            carbon, recrtCenter, _, _, _, _, _ = board[col + row * size]
            if recrtCenter > -1:
                out += col_divider + f"R{recrtCenter}".rjust(4)
            else:
                out += col_divider + str(min(int(carbon), 9999)).rjust(4)
        out += col_divider + "\n"
        for col in range(size):
            _, _, worker, worker_carbon, worker_type, _, _ = board[col + row * size]
            out += col_divider + (
                f"{min(int(worker_carbon), 99)}{worker_type[0]}{worker}" if worker > -1 else ""
            ).rjust(4)
        out += col_divider + "\n"
        for col in range(size):
            _, _, _, _, _, tree, tree_lifecycle = board[col + row * size]
            out += col_divider + (
                f"{tree_lifecycle}T{tree}" if tree > -1 else ""
            ).rjust(4)
        out += col_divider + "\n" + row_divider
    print(out)
    return out


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "carbon.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    js_path = path.abspath(path.join(dir_path, "carbon.js"))
    with open(js_path, encoding="utf-8") as js_file:
        return js_file.read()
