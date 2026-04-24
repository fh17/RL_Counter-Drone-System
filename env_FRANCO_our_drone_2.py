# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_apply
##
# Pre-defined configs
##
from .assets_FRANCO import OUR_DRONE_2_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip



class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_space = 8
    observation_space = 50
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)

    # robot
    robot:  ArticulationCfg = OUR_DRONE_2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    payload_vel_reward_scale = -0.07
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    tilt_angle_reward_scale = -1.5
    progress_to_goal_reward_scale = 0.0
    success_reward_scale = 50.0

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust_drone_1 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment_drone_1 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust_drone_2 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment_drone_2 = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel_body_1",
                "ang_vel_body_1",
                "lin_vel_body_2",
                "ang_vel_body_2",
                "distance_to_goal",
                "lin_payload_vel",
                "tilt_angle_1",
                "tilt_angle_2",
                "progress_to_goal",
                "success",
                
            ]
        }

        # Get specific body indices
        print("ALL BODY NAMES:", self._robot.body_names)
        
        body_ids = self._robot.find_bodies("Cube")[0]
        self._body_id_1 = body_ids[0]
        self._body_id_2 = body_ids[1]
        self._payload_payload_id = self._robot.find_bodies("Payload")[0][0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self._z_body = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)
        self._g_world = torch.tensor(self.sim.cfg.gravity, device=self.device)
        self._g_world = self._g_world / torch.linalg.norm(self._g_world)    
        self._g_world = self._g_world.unsqueeze(0)

        #Prev step
        self._prev_payload_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._payload_acc = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_distance_to_goal = torch.zeros(self.num_envs, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust_drone_1[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 4.0 # less power per drone, weight per drone = _robot_weight / 2.0
        self._moment_drone_1[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:4]
        self._thrust_drone_2[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 4] + 1.0) / 4.0 # less power per drone, weight per drone = _robot_weight / 2.0
        self._moment_drone_2[:, 0, :] = self.cfg.moment_scale * self._actions[:, 5:8]


    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust_drone_1, self._moment_drone_1, body_ids=self._body_id_1)
        self._robot.set_external_force_and_torque(self._thrust_drone_2, self._moment_drone_2, body_ids=self._body_id_2)


    def _get_observations(self) -> dict:
        body_1_state = self._robot.data.body_state_w[:, self._body_id_1, :]
        body_2_state = self._robot.data.body_state_w[:, self._body_id_2, :]
        payload_state = self._robot.data.body_state_w[:, self._payload_payload_id, :]
        

        drone1_to_payload = payload_state[:, :3] - body_1_state[:, :3]
        drone2_to_payload = payload_state[:, :3] - body_2_state[:, :3]
        drone_to_drone = body_2_state[:, :3] - body_1_state[:, :3]
        payload_vel_w = payload_state[:, 7:10]
        payload_pos_w = payload_state[:, :3]
        payload_to_goal_w = self._desired_pos_w - payload_pos_w
        self._payload_acc = (payload_vel_w - self._prev_payload_vel) / self.step_dt
        self._prev_payload_vel = payload_vel_w.clone()

        obs = torch.cat(
            [
                body_1_state[:, :3] - self._terrain.env_origins[:, :3], #position drone 1 in env
                body_2_state[:, :3] - self._terrain.env_origins[:, :3], #position drone 2 in env
                payload_state[:, :3] - self._terrain.env_origins[:, :3], #position payload in env
                body_1_state[:, 3:7],  # quat1
                body_2_state[:, 3:7],  # quat2
                body_1_state[:, 7:10], #drone 1 lin vel
                body_1_state[:, 10:13], # dreone 1 ang vel
                body_2_state[:, 7:10], #drone 2 lin vel
                body_2_state[:, 10:13], # drone 2 ang vel
                self._robot.data.projected_gravity_b,
                drone1_to_payload,
                drone2_to_payload,
                drone_to_drone,
                payload_vel_w,
                self._payload_acc,
                payload_to_goal_w,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
    
    
    def _get_rewards(self) -> torch.Tensor:

        body_1_state = self._robot.data.body_state_w[:, self._body_id_1, :]
        body_2_state = self._robot.data.body_state_w[:, self._body_id_2, :]
        payload_state = self._robot.data.body_state_w[:, self._payload_payload_id, :]

        quat_1 = self._robot.data.body_state_w[:, self._body_id_1, 3:7]
        quat_2 = self._robot.data.body_state_w[:, self._body_id_2, 3:7]
        z_body = self._z_body.repeat(self.num_envs, 1)
        z_world_1 = quat_apply(quat_1, z_body)
        z_world_2 = quat_apply(quat_2, z_body)
        g_world = self._g_world.repeat(self.num_envs, 1)

        
        lin_vel_body_1 = torch.sum(torch.square(body_1_state[:, 7:10]), dim=1)
        ang_vel_body_1 = torch.sum(torch.square(body_1_state[:, 10:13]), dim=1)
        lin_vel_body_2 = torch.sum(torch.square(body_2_state[:, 7:10]), dim=1)
        ang_vel_body_2 = torch.sum(torch.square(body_2_state[:, 10:13]), dim=1)
        payload_vel_w = torch.sum(torch.square(payload_state[:, 7:10]), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.body_state_w[:, self._payload_payload_id, :3], dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        cos_angle_1 = -torch.sum(z_world_1 * g_world, dim=1)
        cos_angle_2 = -torch.sum(z_world_2 * g_world, dim=1)
        progress_to_goal = self._prev_distance_to_goal - distance_to_goal
        self._prev_distance_to_goal = distance_to_goal.clone()
        success = (distance_to_goal < 0.05) & (payload_vel_w < 0.2)

        rewards = {
            "lin_vel_body_1": lin_vel_body_1 * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel_body_1": ang_vel_body_1 * self.cfg.ang_vel_reward_scale * self.step_dt,
            "lin_vel_body_2": lin_vel_body_2 * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel_body_2": ang_vel_body_2 * self.cfg.ang_vel_reward_scale * self.step_dt,
            "lin_payload_vel": payload_vel_w * self.cfg.payload_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "tilt_angle_1": cos_angle_1 * self.cfg.tilt_angle_reward_scale * self.step_dt,
            "tilt_angle_2": cos_angle_1 * self.cfg.tilt_angle_reward_scale * self.step_dt,
            "progress_to_goal": progress_to_goal * self.cfg.progress_to_goal_reward_scale * self.step_dt,
            "success": success.float() * self.cfg.success_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        body_1_state = self._robot.data.body_state_w[:, self._body_id_1, :]
        body_2_state = self._robot.data.body_state_w[:, self._body_id_2, :]

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        quat_1 = self._robot.data.body_state_w[:, self._body_id_1, 3:7]
        quat_2 = self._robot.data.body_state_w[:, self._body_id_2, 3:7]

        z_body = self._z_body.repeat(self.num_envs, 1)
        z_world_1 = quat_apply(quat_1, z_body)
        z_world_2 = quat_apply(quat_2, z_body)
        g_world = self._g_world.repeat(self.num_envs, 1)


        root_pos = self._robot.data.root_pos_w
        lin_vel_1 = torch.linalg.norm(body_1_state[:, 7:10], dim=1)
        ang_vel_1 = torch.linalg.norm(body_1_state[:, 10:13], dim=1)
        lin_vel_2 = torch.linalg.norm(body_2_state[:, 7:10], dim=1)
        ang_vel_2 = torch.linalg.norm(body_2_state[:, 10:13], dim=1)
        drone_distance = torch.linalg.norm(body_1_state[:, :3] - body_2_state[:, :3], dim=1)
        #payload_acc_norm = torch.linalg.norm(self._payload_acc, dim=1)
        cos_angle_1 = -torch.sum(z_world_1 * g_world, dim=1)
        cos_angle_2 = -torch.sum(z_world_2 * g_world, dim=1)


        bad_height = torch.logical_or(root_pos[:, 2] < 0.1, root_pos[:, 2] > 5.0)
        bad_lin_vel_1 = lin_vel_1 > 10.0
        bad_ang_vel_1 = ang_vel_1 > 20.0
        bad_lin_vel_2 = lin_vel_2 > 10.0
        bad_ang_vel_2 = ang_vel_2 > 20.0
        bad_state = ~torch.isfinite(body_1_state).all(dim=1) | ~torch.isfinite(body_2_state).all(dim=1) | ~torch.isfinite(root_pos).all(dim=1)
        too_close = drone_distance < 0.3  #drones width 0.5
        #bad_payload_acc = payload_acc_norm > 15.0
        bad_tilt_1 = cos_angle_1 < 0.4  # n degrees      n= -->   0.4  ~66, 0.7 ~45, 0.85 ~30
        bad_tilt_2 = cos_angle_2 < 0.4  # n degrees

        self._termination_reasons = {
            "bad_height": bad_height,
            "bad_lin_vel_1": bad_lin_vel_1,
            "bad_ang_vel_1": bad_ang_vel_1,
            "bad_lin_vel_2": bad_lin_vel_2,
            "bad_ang_vel_2": bad_ang_vel_2,
            "bad_state": bad_state,
            "bad_tilt_1": bad_tilt_1,
            "bad_tilt_2": bad_tilt_2,
            "too_close": too_close,
        }   

        died = bad_height | bad_ang_vel_1 | bad_ang_vel_2 | bad_lin_vel_1 | bad_lin_vel_2 | bad_state | bad_tilt_1 | bad_tilt_2 | too_close
        return died, time_out

   # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        # Couter for training stats
        if not hasattr(self, "_termination_counts_total"):
            # define keys manually (safe)
            self._termination_counts_total = {
                "bad_height": 0,
                "bad_lin_vel_1": 0,
                "bad_ang_vel_1": 0,
                "bad_lin_vel_2": 0,
                "bad_ang_vel_2": 0,
                "bad_state": 0,
                "bad_tilt_1": 0,
                "bad_tilt_2": 0,
                "too_close": 0,
            }
            self._num_terminated_total = 0
            self._num_timeouts_total = 0
            self._num_resets_total = 0
        if not hasattr(self, "_reward_sums_total"):
            self._reward_sums_total = {key: 0.0 for key in self._episode_sums.keys()}
            self._episodes_count_total = 0

        # Logging
        payload_state = self._robot.data.body_state_w[env_ids, self._payload_payload_id, :]
        payload_pos_w = payload_state[:, :3]

        #self._prev_distance_to_goal[env_ids] = torch.linalg.norm(
        #    self._desired_pos_w[env_ids] - payload_pos_w,
        #    dim=1
        #)

        #self._prev_payload_vel[env_ids] = 0.0
        #self._payload_acc[env_ids] = 0.0

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - payload_pos_w, dim=1
        ).mean()
        extras = dict()

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids]).item()       
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._reward_sums_total[key] += episodic_sum_avg * len(env_ids)          
            self._episode_sums[key][env_ids] = 0.0

        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Fixed target pos 
        #self._desired_pos_w[env_ids, 0] = self._terrain.env_origins[env_ids, 0] + 0.0
        #self._desired_pos_w[env_ids, 1] = self._terrain.env_origins[env_ids, 1] + 0.0
        #self._desired_pos_w[env_ids, 2] = 1.0
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Update counters for stats
        num_died = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        num_time_out = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self._num_terminated_total += num_died
        self._num_timeouts_total += num_time_out
        self._num_resets_total += len(env_ids)
        self._episodes_count_total += len(env_ids)
        if hasattr(self, "_termination_reasons"):
            for key in self._termination_reasons.keys():
                count = torch.count_nonzero(self._termination_reasons[key][env_ids]).item()
                self._termination_counts_total[key] += count

        # Print rewards
        if self._num_resets_total % 1000 < len(env_ids):
            print("\n---- Termination STATS ---")
            print(f"Total resets: {self._num_resets_total}")
            print(f"Total died: {self._num_terminated_total}")
            print(f"Total timeouts: {self._num_timeouts_total}")
            for key, count in self._termination_counts_total.items():
                print(f"{key}: {count}")
            print("\n---- Rewards STATS ----")
            print(f"Total finished episodes: {self._episodes_count_total}")
            for key, total in self._reward_sums_total.items():
                print(f"{key}: {total / self._episodes_count_total:.4f}")

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

