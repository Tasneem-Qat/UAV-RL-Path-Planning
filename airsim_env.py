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
            "Drone1": np.array([15.0, 2.0, -0.5]),
            "Drone2": np.array([15.0, -2.0, -0.5])
        }
        
        # Reward & termination parameters
        self.R_GOAL = 700.0       # bonus for reaching goal
        self.eta = 2.0           # shaping factor for goal progress
        self.R_COLLISION = 30.0   # collision penalty
        self.R_NEAR = 0.3         # near-miss penalty factor
        self.d_safe = 0.5        # safe distance for near-miss
        self.R_STEP = 0.1         # per-step penalty
        self.R_TIMEOUT = 10.0     # penalty for exceeding max steps
        self.R_SMOOTH = 0.1      # smoothness reward
        
        self.max_episode_steps = 200
        self.current_step = 0
        
        # collision parmeters
        self.collision_threshold = 0.5  # meters
        self.success_threshold = 0.5
        self.collision_flags = {name: False for name in self.drone_names}
        self.kick_strength = 3.0  # Adjust this value based on testing
        self.collision_cooldown = {name: 0 for name in self.drone_names}
        self.cooldown_duration = 10  # Steps to ignore after kick
        
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
        init_x = np.random.uniform(-0.5,0.5)
        init_y = [np.random.uniform(1.5,2.0), np.random.uniform(-1.5,-2.0)]
        init_z = -0.5
        
        # Randomly reposition each drone
        for i, name in enumerate(self.drone_names):
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)
            # Directly move to altitude instead of takeoff to avoid freeze
            self.client.moveToPositionAsync(init_x, init_y[i], init_z, velocity=2.0, vehicle_name=name).join()

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

            goal = self.goal_positions[name]
            rel_goal = goal - pos
    
            # Compute reward
            reward = self._get_reward(i, np.array(actions[i]))
            rewards.append(reward)

            # Next observation (position + velocity)
            next_obs.append(np.array([
                pos[0], pos[1], pos[2],
                state.kinematics_estimated.linear_velocity.x_val,
                state.kinematics_estimated.linear_velocity.y_val,
                state.kinematics_estimated.linear_velocity.z_val,
                rel_goal[0], rel_goal[1], rel_goal[2]
            ]))
            
        done, info = self._check_done()
        dones = [done] * self.num_agents
        
        return np.array(next_obs), rewards, dones, info

    def _apply_velocity_kick(self, agent_index):
        name = self.drone_names[agent_index]
        state = self.client.getMultirotorState(vehicle_name=name)
        
        # Get current velocity
        current_vel = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            0  # Ignore vertical velocity
        ])
    
        # Get collision direction (simplified example)
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
        # kick_velocity = collision_normal * self.kick_strength
        
        self.client.moveByVelocityZAsync(
            vx=kick_velocity[0],
            vy=kick_velocity[1],
            z=-0.5,
            duration=0.3,
            vehicle_name=name
        ).join()
        
    def _get_observation(self):
        obs = []
        for name in self.drone_names:
            state = self.client.getMultirotorState(vehicle_name=name)
            pos = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val
                ])  # Current position
            goal = self.goal_positions[name]  # Absolute goal position
            rel_goal = goal - pos  # Relative vector to goal
            
            obs.append([
                # Current state
                pos[0], pos[1], pos[2],
                state.kinematics_estimated.linear_velocity.x_val,
                state.kinematics_estimated.linear_velocity.y_val,
                state.kinematics_estimated.linear_velocity.z_val,
                # Goal information
                rel_goal[0], rel_goal[1], rel_goal[2]
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
        name = self.drone_names[agent_index]
        state = self.client.getMultirotorState(vehicle_name=name)
        pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])
        goal = self.goal_positions[name]

        # 1) Goal progress reward
        dist = np.linalg.norm(pos - goal)
        prev_dist = self.last_distance.get(name, dist)
        reward = self.eta * (prev_dist - dist)
        if dist <= 0.5:
            reward += self.R_GOAL
        self.last_distance[name] = dist

        # Forward velocity bonus
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
            if not self.collision_flags[name]:
                # First timestep of collision
                reward -= self.R_COLLISION
                self._apply_velocity_kick(agent_index)
                self.collision_flags[name] = True
        else:
            self.collision_flags[name] = False
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
        state = self.client.getMultirotorState(vehicle_name=name)
        pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])

        # Existing collision checks
        info = self.client.simGetCollisionInfo(vehicle_name=name)
        api_collided = info.has_collided or "arm" in info.object_name.lower()
        
        d_obs = self._get_nearest_obstacle_distance(agent_index)
        lidar_collided = d_obs < self.collision_threshold
        
        distances = self._get_sensor_distances(agent_index)
        collision_threshold = 0.5
        sensor_collided = any(d < collision_threshold for d in distances)
        
        # Check for collisions with other drones
        inter_drone_collision = False
        for other_name in self.drone_names:
            if other_name == name:
                continue
            other_state = self.client.getMultirotorState(vehicle_name=other_name)
            other_pos = np.array([
                other_state.kinematics_estimated.position.x_val,
                other_state.kinematics_estimated.position.y_val,
                other_state.kinematics_estimated.position.z_val
            ])
            distance = np.linalg.norm(pos - other_pos)
            if distance < self.collision_threshold:
                inter_drone_collision = True
                break
        
        collided = (api_collided or lidar_collided 
                    or sensor_collided or inter_drone_collision)

        return collided

    def _check_done(self):
        done = False
        details = {}
        all_goals_reached = True
        any_collision = False
        num_of_collided_agents = 0
        all_collided = False
        timeout = self.current_step >= self.max_episode_steps
        # print(f"Current step: {self.current_step}, Timeout: {timeout}")

        # Check status for all drones
        for i, name in enumerate(self.drone_names):
            # Initialize per-drone details
            details[name] = "Ongoing"
            
            # Check collision
            if self._check_collision(i):
                num_of_collided_agents+=1
                details[name] = "Collision"
                 
            # Check goal status
            state = self.client.getMultirotorState(vehicle_name=name)
            pos = np.array([state.kinematics_estimated.position.x_val,
                            state.kinematics_estimated.position.y_val,
                            state.kinematics_estimated.position.z_val])
            goal = self.goal_positions[name]
            distance_to_goal = np.linalg.norm(pos - goal)
            
            if distance_to_goal > 0.5:
                all_goals_reached = False
            else:
                details[name] = "Goal Reached"
                
            success_count = sum(1 for v in details.values() if v == "Goal Reached")
    
        # Termination conditions (order matters!)
        if all_goals_reached:
            done = True
            details["global"] = "All Goals Reached"
        elif success_count/self.num_agents >= self.success_threshold:
            done = True
            details["global"] = "Partial Success"
        # elif any_collision:
        #     done = True
        #     details["global"] = "collision"
        elif num_of_collided_agents == NUM_AGENTS:
            done = True
            details["global"] = "Collisions"
        elif timeout:
            done = True
            details["global"] = "Timeout"
        # elif both_collided:
        #     done = True
        #     details["global"] = "Both_Collided"
        
        return done, details