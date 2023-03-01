from einops import asnumpy
import torch
import numpy as np
from habitat.config import Config as CN
from planner import (
    AStarPlannerVector,
    AStarPlannerSequential,
)
import matplotlib
import matplotlib.pyplot as plt

import pickle as pk
_C = CN()
_C.RL = CN()
# =============================================================================
# Planner
# =============================================================================
_C.RL.PLANNER = CN()
_C.RL.PLANNER.nplanners = 18  # Same as the number of processes
_C.RL.PLANNER.allow_diagonal = True  # planning diagonally
# local region around the agent / goal that is set to free space when either
# are classified occupied
_C.RL.PLANNER.local_free_size = 0.25
# Assign weights to graph based on proximity to obstacles?
_C.RL.PLANNER.use_weighted_graph = False
# Weight factors
_C.RL.PLANNER.weight_scale = 4.0
_C.RL.PLANNER.weight_niters = 1

config = _C.clone()
# # Planner
# if config.RL.PLANNER.nplanners > 1:
#     planner = AStarPlannerVector(config.RL.PLANNER)
# else:
#     planner = AStarPlannerSequential(config.RL.PLANNER)
# with open('/mnt/cephfs/dataset/soundspaces_data/data/datasets/multiON/oracle_maps/map3000.pickle','rb') as file:
#     map = pk.load(file)

# input_map = map["data/scene_datasets/mp3d/D7N2EKCX4Sj/D7N2EKCX4Sj.glb"][:,:,0:2]
# input_map[:,:,1]=1
# input_map = input_map.transpose(2,0,1)
# input_map = np.expand_dims(input_map,0)

# agent = np.array([150,50])
# agent = torch.tensor(np.expand_dims(agent,0))

# goal = np.array([152,52])
# goal = torch.tensor(np.expand_dims(goal,0))

# sample_flags = [1]

def _process_maps(config, collision_map, maps, goals=None):
    """
    Inputs:
        maps - (bs, 2, M, M) --- 1st channel is prob of obstacle present
                                --- 2nd channel is prob of being explored
    """
    map_scale = 0.08
    # Compute a map with ones for obstacles and zeros for the rest

    obstacle_mask = (maps[:, 0] > 0.6) & (
        maps[:, 1] > 0.6
    )

    final_maps = obstacle_mask.astype(np.float32)  # (bs, M, M)

    # Post-process map based on previously visited locations
    final_maps[maps[:, 0] == 1] = 0
    final_maps[maps[:, 0] == 2] = 1
    # Post-process map based on previously collided regions
    final_maps[maps[:, 0] == 3] = 0

    final_maps[collision_map == 1] = 1
    # Set small regions around the goal location to be zeros
    if goals is not None:
        lfs = int(config.PLANNER.local_free_size / map_scale)
        for i in range(final_maps.shape[0]):
            goal_x = int(goals[i, 0].item())
            goal_y = int(goals[i, 1].item())
            final_maps[
                i,
                (goal_y - lfs) : (goal_y + lfs + 1),
                (goal_x - lfs) : (goal_x + lfs + 1),
            ] = 0.0

    return final_maps

# final_maps = _process_maps(config.RL,input_map)



def _compute_plans(
    planner,
    config,
    collision_map,
    global_map,
    agent_map_xy,
    goal_map_xy,
    sample_goal_flags,
):
    """
    global_map - (bs, 2, V, V) tensor
    agent_map_xy - (bs, 2) agent's current position on the map
    goal_map_xy - (bs, 2) goal's current position on the map
    sample_goal_flags - list of zeros and ones should a new goal be sampled?
    """
    
    s = 0.08
    # ==================== Process the map to get planner map =====================
    # Processed map has zeros for free-space and ones for obstacles
    global_map_proc = _process_maps(
        config.RL, collision_map, global_map, goal_map_xy,
    )  # (bs, M, M) tensor
    # =================== Crop a local region around agent, goal ==================
 

    cropped_global_map = global_map_proc  # (bs, M, M)
    old_center_xy = (agent_map_xy + goal_map_xy) / 2
    new_center_xy = old_center_xy
    new_agent_map_xy = agent_map_xy
    new_goal_map_xy = goal_map_xy
    S = cropped_global_map.shape[1]
    

    # Clip points to ensure they are within map limits
    new_agent_map_xy = torch.clamp(new_agent_map_xy, 0, S - 1)
    new_goal_map_xy = torch.clamp(new_goal_map_xy, 0, S - 1)
    # Convert to numpy
    agent_map_xy_np = asnumpy(new_agent_map_xy).astype(np.int32)
    goal_map_xy_np = asnumpy(new_goal_map_xy).astype(np.int32)
    global_map_np = asnumpy(cropped_global_map)
    # =================== Plan path from agent to goal positions ==================
    plans = planner.plan(
        global_map_np, agent_map_xy_np, goal_map_xy_np, sample_goal_flags
    )  # List of tuple of lists
    # Convert plans back to original coordinates

    final_plans = []
    for i in range(len(plans)):
        plan_x, plan_y = plans[i]
        # Planning failure
        if plan_x is None:
            final_plans.append((plan_x, plan_y))
            continue
        offset_x = int((old_center_xy[i, 0] - new_center_xy[i, 0]).item())
        offset_y = int((old_center_xy[i, 1] - new_center_xy[i, 1]).item())
        final_plan_x, final_plan_y = [], []
        for px, py in zip(plan_x, plan_y):
            final_plan_x.append(px + offset_x)
            final_plan_y.append(py + offset_y)
        final_plans.append((final_plan_x, final_plan_y))
    return final_plans

