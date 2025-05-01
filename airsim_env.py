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
        self.goal_achieved = {name: False for name in self.drone_names}  # Track per-drone success
        
        # Reward & termination parameters
        self.R_GOAL = 70.0       # bonus for reaching goal 700
        self.eta = 0.1           # shaping factor for goal progress 2.0
        self.R_COLLISION = 3.0   # collision penalty 30.0
        self.R_NEAR = 0.05         # near-miss penalty factor 0.3
        self.d_safe = 0.6        # safe distance for near-miss 0.5
        self.R_STEP = 0.01         # per-step penalty 0.1
        self.R_TIMEOUT = 1.0     # penalty for exceeding max steps 10.0
        self.R_SMOOTH = 0.005
        self.max_episode_steps = 120
        self.current_step = 0
        
        # Collision parmeters
        self.collision_threshold = 0.5  # meters
        self.success_threshold = 0.5
        self.kick_strength = 2.0  # We can adjust this value based on testing
        self.drone_collided = {name: False for name in self.drone_names}
        
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
        self.goal_achieved = {name: False for name in self.drone_names}
        self.drone_collided = {name: False for name in self.drone_names}
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
            name = self.drone_names[i]
            if self.goal_achieved[self.drone_names[i]]:
                continue
            vx = float(action[0]) * 1.5
            vy = float(action[1]) * 1.5
            self.client.moveByVelocityZAsync(
                vx=vx,
                vy=vy,
                z=-0.5,  # Maintain 0.5m altitude
                duration=0.5,
                vehicle_name=self.drone_names[i]
            ).join()

        time.sleep(0.1)
        
        # Gather next state, reward, done
        for i, name in enumerate(self.drone_names):
            # Freeze succeeded agents
            if self.goal_achieved[name]:
                rewards.append(0.0)  # Explicitly append 0 reward
                next_obs.append(np.zeros(STATE_DIM))  # Masked observation
                continue  # Skip further processing
            else:
                state = self.client.getMultirotorState(vehicle_name=name)
                pos = np.array([
                    state.kinematics_estimated.position.x_val,
                    state.kinematics_estimated.position.y_val,
                    state.kinematics_estimated.position.z_val
                ])

                goal = self.goal_positions[name]
                rel_goal = goal - pos
        
                # Get sensor distances (4 values all around drone)
                sensor_dists = self._get_sensor_distances(i)

                # Compute reward
                reward = self._get_reward(i, np.array(actions[i]))
                rewards.append(reward)

                # Next observation (position + velocity)
                next_obs.append(np.array([
                    pos[0], pos[1], pos[2],
                    state.kinematics_estimated.linear_velocity.x_val,
                    state.kinematics_estimated.linear_velocity.y_val,
                    state.kinematics_estimated.linear_velocity.z_val,
                    rel_goal[0], rel_goal[1], rel_goal[2],
                    *sensor_dists
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
        obs = []
        for i, name in enumerate(self.drone_names):
            if self.goal_achieved[name]:
                obs.append(np.zeros(STATE_DIM))
                continue
            state = self.client.getMultirotorState(vehicle_name=name)
            pos = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val
                ])  # Current position
            goal = self.goal_positions[name]  # Absolute goal position
            rel_goal = goal - pos  # Relative vector to goal
            
            # Get sensor distances (4 values: front, back, left, right)
            sensor_dists = self._get_sensor_distances(i)
            
            obs.append([
                # Current state
                pos[0], pos[1], pos[2],
                state.kinematics_estimated.linear_velocity.x_val,
                state.kinematics_estimated.linear_velocity.y_val,
                state.kinematics_estimated.linear_velocity.z_val,
                # Goal information
                rel_goal[0], rel_goal[1], rel_goal[2],
                *sensor_dists
            ])
        return np.array(obs)

    def _get_nearest_obstacle_distance(self, agent_index):
        name = self.drone_names[agent_index]
        # Fetch full Lidar point cloud
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
        if self.goal_achieved[name]:
            return 0.0
        
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
        reward = self.eta * (prev_dist**2 - dist**2)
        if dist <= 0.5:
            reward += self.R_GOAL
            self.goal_achieved[name] = True
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
                reward -= self.R_COLLISION
                self._apply_velocity_kick(agent_index)
                return reward
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
        all_succeeded = all(self.goal_achieved.values())
        num_of_collided_agents = 0
        timeout = self.current_step >= self.max_episode_steps
        # print(f"Current step: {self.current_step}, Timeout: {timeout}")

        # Check status for all drones
        for i, name in enumerate(self.drone_names):
            # Initialize per-drone details
            if self.drone_collided[name] == False and self.goal_achieved[name] == False:
                details[name] = "Ongoing"
            
            # Check collision
            if self._check_collision(i) or self.drone_collided[name] == True:
                num_of_collided_agents+=1
                details[name] = "Collision"
                self.drone_collided[name] = True
                
            if self.goal_achieved[name] == True:
                details[name] = "Goal Reached"
                
            all_collided = (num_of_collided_agents == NUM_AGENTS)
            
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
        if all_collided:
            done = True
            details["Global"] = "Collisions"
            if success_count/self.num_agents >= self.success_threshold:
                details["Global"] = "Collisions with Partial Success"
        elif all_succeeded or all_goals_reached:
            done = True
            details["Global"] = "All Goals Reached"
        elif timeout:
            done = True
            details["Global"] = "Timeout"
            if success_count/self.num_agents >= self.success_threshold:
                details["Global"] = "Timeout with Partial Success"
        
        return done, details
    
        