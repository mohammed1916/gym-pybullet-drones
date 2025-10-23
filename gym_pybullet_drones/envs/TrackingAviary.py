import os
import numpy as np
import csv
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TrackingAviary(BaseRLAviary):
    """RL environment for drone trajectory tracking."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 trajectory_file: str=os.path.join(os.path.dirname(__file__), '../assets/beta-traj.csv'),  # Path to CSV with trajectory
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a trajectory tracking RL environment.

        Parameters
        ----------
        trajectory_file : str, optional
            Path to CSV file containing trajectory data (columns: p_x, p_y, p_z, v_x, v_y, v_z).
        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

        # Load trajectory
        self.trajectory = []
        with open(trajectory_file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                self.trajectory.append({
                    'pos': np.array([float(row['p_x']), float(row['p_y']), float(row['p_z'])]),
                    'vel': np.array([float(row['v_x']), float(row['v_y']), float(row['v_z'])])
                })
        self.trajectory_index = 0
        self.EPISODE_LEN_SEC = len(self.trajectory) / self.CTRL_FREQ  # Adjust episode length based on trajectory

    ################################################################################

    def _computeObs(self):
        """Returns the current observation, including target info for tracking."""
        obs = super()._computeObs()
        # For KIN obs, append target position and velocity
        if self.OBS_TYPE == ObservationType.KIN:
            target_pos = self.trajectory[min(self.trajectory_index, len(self.trajectory)-1)]['pos']
            target_vel = self.trajectory[min(self.trajectory_index, len(self.trajectory)-1)]['vel']
            target_obs = np.array([target_pos, target_vel]).flatten()  # 6 values
            # Repeat for each drone if multi-agent
            target_obs = np.tile(target_obs, (self.NUM_DRONES, 1))
            obs = np.hstack([obs, target_obs])
        return obs

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space, expanded for target info."""
        space = super()._observationSpace()
        if self.OBS_TYPE == ObservationType.KIN:
            # Add 6 for target pos/vel
            low = np.full((self.NUM_DRONES, 6), -np.inf)
            high = np.full((self.NUM_DRONES, 6), np.inf)
            space = spaces.Box(low=np.hstack([space.low, low]), high=np.hstack([space.high, high]), dtype=np.float32)
        return space

    ################################################################################

    def _computeReward(self):
        """Computes reward based on tracking error."""
        ret = 0
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            target = self.trajectory[min(self.trajectory_index, len(self.trajectory)-1)]
            pos_error = np.linalg.norm(target['pos'] - state[0:3])
            vel_error = np.linalg.norm(target['vel'] - state[10:13])
            ret -= (pos_error + vel_error)  # Negative reward for error
        return ret

    ################################################################################

    def _computeTerminated(self):
        """Terminates at end of trajectory."""
        return self.trajectory_index >= len(self.trajectory)

    ################################################################################

    def _computeTruncated(self):
        """Truncates on timeout or excessive error."""
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos_error = np.linalg.norm(self.trajectory[min(self.trajectory_index, len(self.trajectory)-1)]['pos'] - state[0:3])
            if pos_error > 5.0:
                return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s)."""
        return {"trajectory_index": self.trajectory_index}