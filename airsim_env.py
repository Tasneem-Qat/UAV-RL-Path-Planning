import setup_path
import airsim
import time
import numpy as np
from config import NUM_AGENTS, STATE_DIM, ACTION_DIM

# Custom environment class to handle multiple drone agents
class AirSimMultiAgentEnv:
    def __init__(self):
        # Creates a client for communicating with AirSim and confirms the connection
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Stores the number of agents and dimensions as defined in our config.py file
        self.num_agents = NUM_AGENTS
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        
        # Constructs a list of drone names (“Drone1”, “Drone2”, …)
        self.drone_names = [f"Drone{i+1}" for i in range(self.num_agents)]
        
        # Sets goal coordinates for each drone as a dictionary: drone name -> respective target
        self.goal_positions = {
            "Drone1": np.array([15.0, 0.0, -0.5])
        }
        
        # Reward & termination parameters
        self.R_GOAL = 700.0       # bonus for reaching goal
        self.eta = 1           # shaping factor for goal progress
        self.R_COLLISION = 50.0   # collision penalty
        self.R_NEAR = 0.3         # near-miss penalty factor
        self.d_safe = 0.5        # safe distance for near-miss
        self.R_STEP = 0.0001         # per-step penalty
        self.R_TIMEOUT = 30.0     # penalty for exceeding max steps
        self.R_SMOOTH = 0.1      # smoothness reward
        self.max_episode_steps = 200
        self.current_step = 0
        
        # collision threshold fallback (using Lidar)
        self.collision_threshold = 0.5  # meters
        
        # Track last distance and last action per drone
        self.last_distance = {}
        self.last_action = {}
        
        # Initializes drones, enabling API control and arming them
        for name in self.drone_names:
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)

    def reset(self):
        # Reset simulation and clear state
        self.client.reset()
        time.sleep(1)
        self.current_step = 0
        self.last_distance.clear()
        self.last_action.clear()
        
        # Randomly reposition each drone
        for name in self.drone_names:
            init_x = np.random.uniform(-2, 2)
            init_y = np.random.uniform(-2, 2)
            init_z = -0.5

            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)
            # Directly move to altitude instead of takeoff to avoid freeze
            self.client.moveToPositionAsync(init_x, init_y, init_z, velocity=2.0, vehicle_name=name).join()

        # Return initial observations
        return self._get_observation()

    def step(self, actions):
        # Increment time step
        self.current_step += 1
        rewards = []
        dones = []
        next_obs = []
        
        # Issue velocity commands
        for i, action in enumerate(actions):
            vx = float(action[0]) * 1.5
            vy = float(action[1]) * 1.5
            self.client.moveByVelocityZAsync(
                vx=vx,
                vy=vy,
                z=-0.5,  # maintain 0.5m altitude
                duration=0.5,
                vehicle_name=self.drone_names[i]
            ).join()

        time.sleep(0.1)
        
        # Gather next state, reward, done
        for i, name in enumerate(self.drone_names):
            state = self.client.getMultirotorState(vehicle_name=name)
            pos = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val
            ])

            # Compute reward
            reward = self._get_reward(i, np.array(actions[i]))
            rewards.append(reward)

            done, info = self._check_done()
            dones.append(done)

            if dones[0]:
                goal = self.goal_positions[name]
                dist = np.linalg.norm(pos - goal)
                print(f"Final distance to goal: {dist:.2f}m")
        
            # Next observation (position + velocity)
            next_obs.append(np.array([
                pos[0], pos[1], pos[2],
                state.kinematics_estimated.linear_velocity.x_val,
                state.kinematics_estimated.linear_velocity.y_val,
                state.kinematics_estimated.linear_velocity.z_val
            ]))

        # Collapse into single done flag
        return np.array(next_obs), rewards, dones, info

    def _get_observation(self):
        obs = []
        for name in self.drone_names:
            state = self.client.getMultirotorState(vehicle_name=name)
            obs.append([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val,
                state.kinematics_estimated.linear_velocity.x_val,
                state.kinematics_estimated.linear_velocity.y_val,
                state.kinematics_estimated.linear_velocity.z_val,
            ])
        return np.array(obs)

    def _get_nearest_obstacle_distance(self, agent_index):
        name = self.drone_names[agent_index]
        # fetch full Lidar point cloud
        data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name=name)
        pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
        if pts.size:
            dists = np.linalg.norm(pts, axis=1)
            return float(dists.min())
        return np.inf
    
    def _get_sensor_distances(self, agent_index):
        name = self.drone_names[agent_index]
        sensors = ["FrontSensor", "BackSensor", "LeftSensor", "RightSensor"]
        distances = []
        for sensor in sensors:
            data = self.client.getDistanceSensorData(distance_sensor_name=sensor, vehicle_name=name)
            distances.append(data.distance)
        return distances

    def _get_reward(self, agent_index, action):
        reward = 0
        name = self.drone_names[agent_index]
        state = self.client.getMultirotorState(vehicle_name=name)
        pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])
        goal = self.goal_positions[name]

        # 1) Goal progress reward and penalty
        dist = np.linalg.norm(pos - goal)
        prev_dist = self.last_distance.get(name, dist)
        reward = self.eta * (prev_dist - dist)
        # if prev_dist > dist:
        #     reward = self.eta * (prev_dist - dist)
        # elif prev_dist < dist:
        #     reward = self.eta * (prev_dist - dist)
            
        if dist <= 0.5:
            reward += self.R_GOAL
        self.last_distance[name] = dist

        # Straight line to goal velocity bonus
        dir_vec = (goal - pos) / (np.linalg.norm(goal - pos) + 1e-6)
        vel = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val
        ])
        reward += 0.05 * np.dot(vel, dir_vec)

        # 2) Collision & near-miss penalty
        collided = self._check_collision(agent_index)
        if collided:
            reward -= self.R_COLLISION
        else:
            d_obs = self._get_nearest_obstacle_distance(agent_index)
            if d_obs < self.d_safe:
                reward -= self.R_NEAR * (1 - d_obs / self.d_safe)

        # 3) Efficiency penalty
        reward -= self.R_STEP

        # 4) Timeout penalty
        if self.current_step >= self.max_episode_steps:
            reward -= self.R_TIMEOUT

        # 5) Smoothness reward
        last_a = self.last_action.get(name, np.zeros_like(action))
        smooth_term = 1 - np.linalg.norm(action - last_a)
        reward += self.R_SMOOTH * smooth_term
        self.last_action[name] = action.copy()

        return reward

    def _check_collision(self, agent_index):
        name = self.drone_names[agent_index]
        info = self.client.simGetCollisionInfo(vehicle_name=name)
        
        # Add explicit check for "is_arm collision" (common in AirSim)
        api_collided = info.has_collided or "arm" in info.object_name.lower()
        
        # Improve LIDAR reliability (average over last 5 readings)
        d_obs = self._get_nearest_obstacle_distance(agent_index)
        lidar_collided = d_obs < self.collision_threshold
        
        distances = self._get_sensor_distances(agent_index)
        collision_threshold = 0.5  # In meters
        sensor_collided = any(d < collision_threshold for d in distances)

        return api_collided or lidar_collided or sensor_collided

    def _check_done(self):
        done = False
        details = {}
        
        for i, name in enumerate(self.drone_names):
            # Check collision
            if self._check_collision(i):
                done = True
                details[name] = "collision"
                break
            # Check timeout
            timeout = (self.current_step >= self.max_episode_steps)
            if timeout:
                done = True
                details[name] = "timeout"
                break

        # Check goal
        if not done:
            for name in self.drone_names:
                state = self.client.getMultirotorState(vehicle_name=name)
                pos = np.array([state.kinematics_estimated.position.x_val,
                                state.kinematics_estimated.position.y_val,
                                state.kinematics_estimated.position.z_val])
                goal = self.goal_positions[name]
                if np.linalg.norm(pos - goal) <= 0.5:
                    done = True
                    details[name] = "goal reached"
                    break
        return done, details
