import setup_path
import airsim
import time
import numpy as np
from config import STATE_DIM, ACTION_DIM

class AirSimEnv:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Single agent parameters
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.drone_name = "Drone1"
        
        # Goal position
        self.goal_position = np.array([15.0, 0.0, -0.5])
        self.goal_achieved = False
        self.last_distance = None  # Track previous distance to goal
        self.last_action = None  # Track previous action for smoothness
        self.drone_collided = False  # Single collision flag
        self.collision_threshold = 0.5
        self.kick_strength = 2.0
        
        # Reward parameters
        self.R_GOAL = 300.0
        self.eta = 0.1
        self.R_COLLISION = 3.0
        self.R_NEAR = 0.05
        self.d_safe = 0.6
        self.R_STEP = 0.1
        self.R_TIMEOUT = 1.0
        self.R_SMOOTH = 0.005
        self.max_episode_steps = 120
        self.current_step = 0
        
        self.collision_counter = 0
        self.episode_start_time = 0
        self.completion_time = None
        
        self.dist_to_goal = -1
        # Initialize drone
        self.client.enableApiControl(True, vehicle_name=self.drone_name)
        self.client.armDisarm(True, vehicle_name=self.drone_name)

    def reset(self):
        self.client.reset()
        time.sleep(1)
        self.current_step = 0
        self.goal_achieved = False
        self.last_distance = None
        self.last_action = None
        
        self.collision_counter = 0
        self.episode_start_time = time.time()
        self.completion_time = None
        
        self.dist_to_goal = -1
        
        # Re-enable API control after reset
        self.client.enableApiControl(True, vehicle_name=self.drone_name)
        self.client.armDisarm(True, vehicle_name=self.drone_name)
    
        # Single drone initialization
        init_x = np.random.uniform(-0.5, 0.5)
        init_y = np.random.uniform(-2.0, 2.0)
        init_z = -0.5
        
        self.client.moveToPositionAsync(init_x, init_y, init_z, velocity=2.0, 
                                      vehicle_name=self.drone_name).join()
        
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        
        # Execute action
        vx = float(action[0]) * 1.5
        vy = float(action[1]) * 1.5
        self.client.moveByVelocityZAsync(
            vx=vx,
            vy=vy,
            z=-0.5,
            duration=0.5,
            vehicle_name=self.drone_name
        ).join()

        time.sleep(0.1)
        
        # Get new state
        next_obs = self._get_observation()
        reward = self._get_reward(action)
        done, details = self._check_done()
        
        return next_obs, reward, done, details

    def _apply_velocity_kick(self):
        name = self.drone_name
        state = self.client.getMultirotorState(vehicle_name=name)
        
        # Get current velocity
        current_vel = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            0  # Ignore vertical velocity
        ])
    
        # Get collision direction
        collision_info = self.client.simGetCollisionInfo(vehicle_name=name)
        collision_normal = np.array([
            collision_info.normal.x_val,
            collision_info.normal.y_val,
            collision_info.normal.z_val
        ])
        collision_normal[2] = 0  # Keep horizontal
        collision_normal /= np.linalg.norm(collision_normal) + 1e-6

        # Calculate reflection direction (modified for wall skimming)
        tangent_component = np.dot(current_vel, collision_normal) * collision_normal
        reflection_dir = current_vel - 2 * tangent_component  # Mirror velocity
        
        # Normalize and combine with collision normal kick
        reflection_dir /= np.linalg.norm(reflection_dir) + 1e-6
        combined_dir = collision_normal + 0.5 * reflection_dir  # Adjust weights
        combined_dir /= np.linalg.norm(combined_dir) + 1e-6
        
        # Apply stronger kick for parallel cases
        kick_velocity = combined_dir * self.kick_strength * 1.5  # 50% boost
        
        # Apply kick velocity
        self.client.moveByVelocityZAsync(
            vx=kick_velocity[0],
            vy=kick_velocity[1],
            z=-0.5,
            duration=0.3,
            vehicle_name=name
        ).join()

        
    def _get_observation(self):
        name = self.drone_name
        
        state = self.client.getMultirotorState(vehicle_name=name)
        pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
            ])  # Current position
        
        goal = self.goal_position  # Absolute goal position
        rel_goal = goal - pos  # Relative vector to goal
        
        # Get sensor distances (4 values: front, back, left, right)
        sensor_dists = self._get_sensor_distances()
        
        obs = np.array([
        # Position (3)
        pos[0], pos[1], pos[2],
        
        # Velocity (3)
        state.kinematics_estimated.linear_velocity.x_val,
        state.kinematics_estimated.linear_velocity.y_val,
        state.kinematics_estimated.linear_velocity.z_val,
        
        # Relative goal (3)
        (self.goal_position[0] - pos[0]),
        (self.goal_position[1] - pos[1]),
        (self.goal_position[2] - pos[2]),
        
        # Sensor distances (4)
        *self._get_sensor_distances()
        ])
        
        return obs

    def _get_nearest_obstacle_distance(self):
        name = self.drone_name
        # Fetch full Lidar point cloud
        data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name=name)
        pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
        if pts.size:
            dists = np.linalg.norm(pts, axis=1)
            return float(dists.min())
        return np.inf
    
    def _get_sensor_distances(self):
        name = self.drone_name
        sensors = ["FrontSensor", "BackSensor", "LeftSensor", "RightSensor"]
        distances = []
        for sensor in sensors:
            data = self.client.getDistanceSensorData(distance_sensor_name=sensor, vehicle_name=name)
            distances.append(data.distance)
        return distances

    def _get_reward(self, action):
        name = self.drone_name
        
        state = self.client.getMultirotorState(vehicle_name=name)
        pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])
        goal = self.goal_position

        # 1) Goal progress reward
        dist = np.linalg.norm(pos - goal)
        prev_dist = self.last_distance if self.last_distance is not None else dist
        reward = self.eta * (prev_dist - dist)
        reward += 10 / (dist + 1e-3) #newly added to exponentially reward drone as it gets near goal
        if dist <= 0.5:
            reward += self.R_GOAL
            self.goal_achieved = True
        self.last_distance = dist
        self.dist_to_goal = dist

        # Forward velocity bonus
        dir_vec = (goal - pos) / (np.linalg.norm(goal - pos) + 1e-6)
        vel = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val
        ])
        reward += 0.05 * np.dot(vel, dir_vec)

        # 2) Collision & near-miss penalty
        collided = self._check_collision()
        if collided:
                reward -= self.R_COLLISION
                self._apply_velocity_kick()
                return reward
        else:
            d_obs = self._get_nearest_obstacle_distance()
            if d_obs < self.d_safe:
                reward -= self.R_NEAR * (1 - d_obs / self.d_safe)

        # 3) Efficiency penalty
        reward -= self.R_STEP

        # 4) Timeout penalty
        if self.current_step >= self.max_episode_steps:
            reward -= self.R_TIMEOUT

        # 5) Smoothness reward
        if self.last_action is not None:
            smooth_term = 1 - np.linalg.norm(action - self.last_action)
            reward += self.R_SMOOTH * smooth_term
        self.last_action= action.copy()

        return reward

    def _check_collision(self):
        name = self.drone_name
        state = self.client.getMultirotorState(vehicle_name=name)
        pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])

        # Existing collision checks
        info = self.client.simGetCollisionInfo(vehicle_name=name)
        if "floor" in info.object_name.lower():
            return False
    
        api_collided = info.has_collided or "arm" in info.object_name.lower()
        
        d_obs = self._get_nearest_obstacle_distance()
        lidar_collided = d_obs < self.collision_threshold
        
        distances = self._get_sensor_distances()
        collision_threshold = 0.5
        sensor_collided = any(d < collision_threshold for d in distances)
        
        collided = (api_collided or lidar_collided 
                    or sensor_collided)
        
        if collided:
            self.collision_counter += 1
            
        return collided

    def _check_done(self):
        done = False
        details = {}
        name = self.drone_name
        succeeded = self.goal_achieved
        timeout = self.current_step >= self.max_episode_steps
        # print(f"Current step: {self.current_step}, Timeout: {timeout}")

        # Initialize details
        if self.drone_collided == False and self.goal_achieved == False:
            details[name] = "Ongoing"
        
        # Check collision
        if self._check_collision() or self.drone_collided == True:
            details[name] = "Collided"
            self.drone_collided = True       
        
        # Termination conditions (Order matters!)
        if succeeded:
            details[name] = "Goal Reached"
            done = True 
        elif timeout:
            done = True
            details[name] = "Timeout"
        
        if done:
            self.completion_time = time.time() - self.episode_start_time
            
        return done, details
    
        