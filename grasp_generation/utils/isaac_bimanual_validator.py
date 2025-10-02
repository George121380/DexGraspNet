"""
Last modified date: 2025.10.01
Author: GPT-5 Codex Assistant
Description: Bimanual (dual-hand) grasp validator based on Isaac Gym
"""

from isaacgym import gymapi
from isaacgym import gymutil
import math
import os
from pathlib import Path
from time import sleep


gym = gymapi.acquire_gym()


class BimanualIsaacValidator:
    """Bimanual validator that manages two ShadowHand assets in the same env."""

    def __init__(
        self,
        hand_asset_left_path,
        hand_asset_right_path,
        mode="direct",
        hand_friction=3.0,
        obj_friction=3.0,
        threshold_dis=0.1,
        env_batch=1,
        sim_step=100,
        gpu=0,
        debug_interval=0.05,
        viewer_width=1280,
        viewer_height=960,
    ):
        self.hand_friction = hand_friction
        self.obj_friction = obj_friction
        self.debug_interval = debug_interval
        self.threshold_dis = threshold_dis
        self.env_batch = env_batch
        self.gpu = gpu
        self.sim_step = sim_step

        self.envs = []
        self.left_hand_handles = []
        self.right_hand_handles = []
        self.obj_handles = []
        self.left_rigid_body_sets = []
        self.right_rigid_body_sets = []
        self.obj_rigid_body_sets = []

        # ShadowHand joint order (identical to single-hand validator).
        self.joint_names = [
            "robot0:FFJ3",
            "robot0:FFJ2",
            "robot0:FFJ1",
            "robot0:FFJ0",
            "robot0:MFJ3",
            "robot0:MFJ2",
            "robot0:MFJ1",
            "robot0:MFJ0",
            "robot0:RFJ3",
            "robot0:RFJ2",
            "robot0:RFJ1",
            "robot0:RFJ0",
            "robot0:LFJ4",
            "robot0:LFJ3",
            "robot0:LFJ2",
            "robot0:LFJ1",
            "robot0:LFJ0",
            "robot0:THJ4",
            "robot0:THJ3",
            "robot0:THJ2",
            "robot0:THJ1",
            "robot0:THJ0",
        ]

        self.hand_asset_left_path = hand_asset_left_path
        self.hand_asset_right_path = hand_asset_right_path
        self.hand_asset_left = None
        self.hand_asset_right = None
        self.obj_asset = None

        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1 / 60
        self.sim_params.substeps = 2
        self.sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)
        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.01
        self.sim_params.physx.rest_offset = 0.0
        self.sim_params.use_gpu_pipeline = False

        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        self.viewer_width = viewer_width
        self.viewer_height = viewer_height

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = viewer_width
        self.camera_props.height = viewer_height
        self.camera_props.use_collision_geometry = True

        self.viewer = None
        if mode == "gui":
            self.has_viewer = True
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
            gym.viewer_camera_look_at(
                self.viewer,
                None,
                gymapi.Vec3(0, 0, 1),
                gymapi.Vec3(0, 0, 0),
            )
        else:
            self.has_viewer = False

        self.hand_asset_options = gymapi.AssetOptions()
        self.hand_asset_options.disable_gravity = True
        self.hand_asset_options.fix_base_link = True
        self.hand_asset_options.collapse_fixed_joints = True

        self.obj_asset_options = gymapi.AssetOptions()
        self.obj_asset_options.override_com = True
        self.obj_asset_options.override_inertia = True
        self.obj_asset_options.density = 500

        self.test_rotations = [
            gymapi.Transform(gymapi.Vec3(0, 0, 0), gymapi.Quat(0, 0, 0, 1)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.5 * math.pi),
            ),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -0.5 * math.pi),
            ),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi),
            ),
        ]

    def _parse_asset_path(self, asset_path):
        """Return (asset_root, asset_file) usable by gym.load_asset."""
        if os.path.isabs(asset_path):
            root = os.path.dirname(asset_path)
            fname = os.path.basename(asset_path)
            return root, fname
        if '/' in asset_path:
            first, rest = asset_path.split('/', 1)
            return first, rest
        return '.', asset_path

    def _load_hand_asset(self, asset_path):
        root, file_name = self._parse_asset_path(asset_path)
        if not os.path.exists(os.path.join(root, file_name)) and not os.path.isabs(asset_path):
            # Try interpreting root relative to project base (one level up from grasp_generation)
            project_root = Path(__file__).resolve().parents[2]
            candidate_root = project_root / root
            candidate_file = candidate_root / file_name
            if candidate_file.exists():
                root = str(candidate_root)
                file_name = os.path.basename(file_name)
        return gym.load_asset(self.sim, root, file_name, self.hand_asset_options)

    def set_assets(self, obj_root, obj_file):
        """Load hand and object assets."""
        self.hand_asset_left = self._load_hand_asset(self.hand_asset_left_path)
        self.hand_asset_right = self._load_hand_asset(self.hand_asset_right_path)
        self.obj_asset = gym.load_asset(
            self.sim, obj_root, obj_file, self.obj_asset_options
        )

    def _create_hand_actor(self, env, asset, pose, qpos, rigid_body_sets_list):
        hand_actor_handle = gym.create_actor(env, asset, pose, "hand", 0, -1)
        props = gym.get_actor_dof_properties(env, hand_actor_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(1000)
        props["damping"].fill(0.0)
        gym.set_actor_dof_properties(env, hand_actor_handle, props)

        dof_states = gym.get_actor_dof_states(env, hand_actor_handle, gymapi.STATE_ALL)
        for i, joint in enumerate(self.joint_names):
            joint_idx = gym.find_actor_dof_index(env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR)
            dof_states["pos"][joint_idx] = qpos[i]
        gym.set_actor_dof_states(env, hand_actor_handle, dof_states, gymapi.STATE_ALL)
        gym.set_actor_dof_position_targets(env, hand_actor_handle, dof_states["pos"])

        shape_props = gym.get_actor_rigid_shape_properties(env, hand_actor_handle)
        rigid_body_set = set()
        for body_id in range(gym.get_actor_rigid_body_count(env, hand_actor_handle)):
            rigid_body_set.add(
                gym.get_actor_rigid_body_index(env, hand_actor_handle, body_id, gymapi.DOMAIN_ENV)
            )
        rig_count = len(shape_props)
        for idx in range(rig_count):
            shape_props[idx].friction = self.hand_friction
        gym.set_actor_rigid_shape_properties(env, hand_actor_handle, shape_props)
        rigid_body_sets_list.append(rigid_body_set)
        return hand_actor_handle

    def add_env(self, handL_rot, handL_trans, handL_qpos, handR_rot, handR_trans, handR_qpos, obj_scale):
        for test_rot in self.test_rotations:
            env = gym.create_env(self.sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 6)
            self.envs.append(env)

            # Left hand pose
            pose_left = gymapi.Transform()
            pose_left.r = gymapi.Quat(handL_rot[1], handL_rot[2], handL_rot[3], handL_rot[0])
            pose_left.p = gymapi.Vec3(*handL_trans)
            pose_left = test_rot * pose_left

            left_handle = self._create_hand_actor(env, self.hand_asset_left, pose_left, handL_qpos, self.left_rigid_body_sets)
            self.left_hand_handles.append(left_handle)

            # Right hand pose
            pose_right = gymapi.Transform()
            pose_right.r = gymapi.Quat(handR_rot[1], handR_rot[2], handR_rot[3], handR_rot[0])
            pose_right.p = gymapi.Vec3(*handR_trans)
            pose_right = test_rot * pose_right

            right_handle = self._create_hand_actor(env, self.hand_asset_right, pose_right, handR_qpos, self.right_rigid_body_sets)
            self.right_hand_handles.append(right_handle)

            # Object actor
            pose_obj = gymapi.Transform()
            pose_obj.p = gymapi.Vec3(0, 0, 0)
            pose_obj.r = gymapi.Quat(0, 0, 0, 1)
            pose_obj = test_rot * pose_obj

            obj_handle = gym.create_actor(env, self.obj_asset, pose_obj, "obj", 0, 1)
            self.obj_handles.append(obj_handle)
            gym.set_actor_scale(env, obj_handle, obj_scale)

            obj_shape_props = gym.get_actor_rigid_shape_properties(env, obj_handle)
            obj_rigid_body_set = set()
            for body_id in range(gym.get_actor_rigid_body_count(env, obj_handle)):
                obj_rigid_body_set.add(
                    gym.get_actor_rigid_body_index(env, obj_handle, body_id, gymapi.DOMAIN_ENV)
                )
            for idx in range(len(obj_shape_props)):
                obj_shape_props[idx].friction = self.obj_friction
            gym.set_actor_rigid_shape_properties(env, obj_handle, obj_shape_props)
            self.obj_rigid_body_sets.append(obj_rigid_body_set)

    def run_sim(self):
        for _ in range(self.sim_step):
            gym.simulate(self.sim)
            if self.has_viewer:
                sleep(self.debug_interval)
                if gym.query_viewer_has_closed(self.viewer):
                    break
                gym.step_graphics(self.sim)
                gym.draw_viewer(self.viewer, self.sim, False)

        success = []
        for idx, env in enumerate(self.envs):
            contacts = gym.get_env_rigid_contacts(env)
            left_touch = False
            right_touch = False
            for contact in contacts:
                a = contact[2]
                b = contact[3]
                if (a in self.left_rigid_body_sets[idx] and b in self.obj_rigid_body_sets[idx]) or (
                    b in self.left_rigid_body_sets[idx] and a in self.obj_rigid_body_sets[idx]
                ):
                    left_touch = True
                if (a in self.right_rigid_body_sets[idx] and b in self.obj_rigid_body_sets[idx]) or (
                    b in self.right_rigid_body_sets[idx] and a in self.obj_rigid_body_sets[idx]
                ):
                    right_touch = True
                if left_touch and right_touch:
                    break
            success.append(left_touch and right_touch)
        return success

    def reset_simulator(self):
        for env in self.envs:
            gym.destroy_env(env)
        self.envs = []
        self.left_hand_handles = []
        self.right_hand_handles = []
        self.obj_handles = []
        self.left_rigid_body_sets = []
        self.right_rigid_body_sets = []
        self.obj_rigid_body_sets = []
        if self.has_viewer and self.viewer is not None:
            gym.destroy_viewer(self.viewer)
            self.viewer = None
        gym.destroy_sim(self.sim)
        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        if self.has_viewer:
            self.viewer = gym.create_viewer(self.sim, self.camera_props)

    def destroy(self):
        if self.has_viewer and self.viewer is not None:
            gym.destroy_viewer(self.viewer)
            self.viewer = None
        gym.destroy_sim(self.sim)
