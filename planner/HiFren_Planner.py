import random
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

# from HPs.my_args import get_args
from Model.hifren_planner import HiFren_planner
from data_process.data_processor import DataProcessor
from data_process.utils import local_to_global_SE2
from data_process.reference_process import Frent_to_relative
from omegaconf import OmegaConf
OPTIONS = ['stop','follow','left','right','left_U','right_U','back']
class HiFrenPlanner(AbstractPlanner):
    def __init__(
            self,
            ckpt_path: str,
            args_path: str,
            noise_lib_X_path: str,
            noise_lib_Y_path: str,
            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,
            device: str = "cpu",
        ):
        
        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        args = OmegaConf.load(args_path)
        self._ckpt_path = ckpt_path
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._device = device

        self.data_processor = DataProcessor(args)

        self._planner = HiFren_planner(
            args=args,
            noise_lib_X_path=noise_lib_X_path,
            noise_lib_Y_path=noise_lib_Y_path,
        )
        self.count = 0
        self._last_global_trajectory = None
        self._last_option = torch.tensor([0], dtype=torch.long).to(self._device)

    def name(self) -> str:
        """
        Inherited.
        """
        return "HiFren_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._mission_goal = initialization.mission_goal

        if self._ckpt_path is not None:
            self._planner.load_state_dict(torch.load(self._ckpt_path))

        self._planner.eval()
        self._planner.to(self._device)
        
        
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput,_last_global_trajectory) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs, reference_line, start_point = self.data_processor.process_data(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._mission_goal, _last_global_trajectory, self._device)

        return model_inputs, reference_line, start_point
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        ego_global_state = current_input.history.current_state[0]
        ego_global_state_se2 = np.array([
            ego_global_state.rear_axle.x,
            ego_global_state.rear_axle.y,
            ego_global_state.rear_axle.heading,
        ], dtype=np.float64)
        inputs, reference_line, start_point = self.planner_input_to_model_inputs(current_input,self._last_global_trajectory)
        outputs, option = self._planner(inputs, self._last_option)
        self._last_option = option
        predictions_frenet = outputs[0].detach().cpu().numpy().astype(np.float64)
        reference_line_std = np.array([20, 20])
        reference_line_mean = np.array([10, 0])
        reference_line = reference_line * reference_line_std + reference_line_mean

        predictions_relative = Frent_to_relative(predictions_frenet, reference_line, start_point)

        dx = np.gradient(predictions_relative[:, 0])
        dx[0] = predictions_relative[2, 0] - 0
        dy = np.gradient(predictions_relative[:, 1])
        dy[0] = predictions_relative[2, 1] - 0
        dtheta = np.arctan2(dy, dx)[..., None]

        predictions = np.concatenate([predictions_relative, dtheta], axis=-1)
        global_trajectory = local_to_global_SE2(predictions, ego_global_state_se2)
        self._last_global_trajectory = global_trajectory
        states = transform_predictions_to_states(predictions, current_input.history.ego_states, self._future_horizon, self._step_interval)
        
        trajectory = InterpolatedTrajectory(
            trajectory=states
        )

        return trajectory
    