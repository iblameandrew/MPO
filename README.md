# CAPO

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/69828e58-26ca-4b04-8ea0-6f4786239885" />


The CAPO Algorithm

### The Meta-Controller: Dynamic Switching Mechanism

The "context" is determined by the agent's recent performance and data statistics. The Meta-Controller is a high-level module that decides which of the 12 functions to use. A practical way to implement this is with a multi-armed bandit algorithm, such as Upper Confidence Bound (UCB1).

Arms: Each of the 12 objective functions is an "arm" of the bandit.

Action: At the beginning of each major training iteration (e.g., every 20,480 timesteps), the Meta-Controller "pulls an arm," selecting one objective function to be used for the next round of data collection and policy updates.

Reward: The "reward" for the bandit is the improvement in the agent's performance (e.g., the change in average extrinsic reward) after using the selected objective.
This allows CAPO to learn which objective is most effective at different stages of training. For instance, it might favor exploration functions early on and switch to exploitation or fine-tuning functions later.

### The Reference Experience Buffer (R)

This buffer stores the top k trajectories (e.g., the 5% with the highest cumulative rewards) seen so far. After each training iteration, new trajectories that outperform the worst ones in the buffer are added, and the worst are discarded. This gives the algorithm a dynamic, high-quality "baseline" of its own best experiences to draw from.

### The 12 Surrogate Objective Functions

Each function calculates an intrinsic reward r_int which is added to the environment's extrinsic reward r_ext to form the total reward used for learning: r_total = r_ext + β * r_int, where β is a hyperparameter balancing the two. The policy is then updated using the standard PPO clipped surrogate objective, but the advantage Â is calculated based on r_total.

Here's a breakdown of how each of the 12 functions could be implemented:

### Objective 1: Exploration (Novelty-Seeking)

Concept: Reward the agent for visiting states it has not seen before.

Implementation (inspired by RND): Use two neural networks: a fixed, randomly initialized "target network" and a "predictor network." The intrinsic reward is the prediction error between them.

Intrinsic Reward r_1: ||predictor(state) - target(state)||^2. This reward is high for novel states and low for familiar ones.

### Objective 2: Exploitation (Reward-Seeking)

Concept: Focus solely on maximizing the extrinsic reward from the environment.
Implementation: This is the standard PPO behavior.
Intrinsic Reward r_2: 0. The agent optimizes for the standard advantage function based only on environmental rewards.

### Objective 3: Extrapolation from Baseline

Concept: Reward the agent for finding new trajectories that are "plausible" extensions of its best past experiences.
Implementation: Train a forward dynamics model on the trajectories in the Reference Buffer R. This model learns to predict s_{t+1} from (s_t, a_t) for successful trajectories.
Intrinsic Reward r_3: The reward is the negative prediction error of this dynamics model on the agent's current trajectory. This encourages the agent to follow paths that "look like" they could belong in the high-performance baseline.

### Objective 4: Spreading Data on a Dimension of Known Baselines

Concept: Encourage diversity within the high-performance state space.
Implementation: Use an embedding of the states from the Reference Buffer R (e.g., from the critic's penultimate layer). Compute the covariance matrix of these state embeddings.
Intrinsic Reward r_4: A reward for visiting a state that increases the determinant of this covariance matrix. This pushes the agent to visit states that expand the volume of the known "good" state distribution.

### Objective 5: Data Gathering and Bootstrapping

Concept: Reward the agent for collecting data that is most informative for updating its policy.
Implementation (Information Gain): The reward is proportional to how much the new data changes the policy.
Intrinsic Reward r_5: The KL-divergence between the policy π_old before the update and the policy π_new after the update on the new data. KL(π_new || π_old). A large divergence implies the data was highly surprising and informative.

### Objective 6: Minimizing Distance over Pairs in the Baseline

Concept: Reward the agent for finding paths that connect different points within its successful past experiences.
Implementation: For each state s_t the agent visits, sample two random states s_a and s_b from the Reference Buffer R.
Intrinsic Reward r_6: Reward for being "on the path" between s_a and s_b. This can be formulated as d(s_a, s_b) - (d(s_t, s_a) + d(s_t, s_b)), where d is a distance metric in the embedding space. The reward is highest when s_t lies on the geodesic between s_a and s_b.

### Objective 7: Metric Similarity to Baseline

Concept: Reward the agent for staying close to states that are known to be part of successful trajectories.
Implementation: For each state s_t the agent visits, find its nearest neighbor s_b in the Reference Buffer R.
Intrinsic Reward r_7: exp(-||embed(s_t) - embed(s_b)||^2). The reward is high when the agent is in a state metrically similar to a known good state.

### Objective 8: Interpolating Data for Exploration

Concept: Create "imaginary" goals by interpolating between known good states and rewarding the agent for reaching them.
Implementation (inspired by Hindsight Experience Replay): At the end of an episode, create an imaginary goal g_interp by interpolating between two states from the Reference Buffer R. Replay the episode, giving the agent a large reward at the timestep where its state was closest to g_interp.

### Objective 9: Integrated Exploration and Learning

Concept: Reward exploration that also makes the agent's internal model of the world more accurate.
Implementation: Combine a novelty reward with a world model improvement reward. Use the RND reward from Objective 1, and add a term for the reduction in prediction error of a learned forward dynamics model.
Intrinsic Reward r_9: r_1 + (error_old(s_t) - error_new(s_t)). This encourages visiting novel states that also reduce uncertainty in the agent's world model.

### Objective 10: Metric Matching of Baseline Data

Concept: A form of goal-conditioned imitation. Pick a state from the baseline and try to reach it.
Implementation: At the start of each episode, sample a state s_g from the Reference Buffer R to serve as a goal for that episode.
Intrinsic Reward r_{10}: A distance-based reward: exp(-||embed(s_t) - embed(s_g)||^2).

### Objective 11: Spreading from Contrasting Baseline Data

Concept: Identify distinct behaviors in the baseline and explore away from them to find new strategies.
Implementation: Cluster the states in the Reference Buffer R into k clusters. Calculate the centroid c_i for each cluster.
Intrinsic Reward r_{11}: Reward for visiting states that are far from all cluster centers: min_i ||embed(s_t) - c_i||^2. This encourages the agent to explore the boundaries of its known successful behaviors.

### Objective 12: Extrapolating and Expanding the Baseline

Concept: Reward the agent for pushing the boundaries of what is considered a "good" trajectory.
Implementation: Use the dynamics model from Objective 3.
Intrinsic Reward r_{12}: A combination of being plausible under the baseline dynamics model, but also being far from the existing baseline states. r_{12} = -||dynamics_model(s_t, a_t) - s_{t+1}||^2 + min_{s_b in R} ||embed(s_t) - embed(s_b)||^2.
Summary of the CAPO Algorithm Flow
Initialize policy π, critic V, Reference Buffer R, and Meta-Controller (bandit).

### Loop for N iterations:

a. Select Objective: The Meta-Controller selects an objective function J_i.
b. Collect Data: The agent collects a batch of trajectories. For each step, calculate r_ext and the intrinsic reward r_i from the selected objective module.
c. Compute Rewards & Advantages: Calculate r_total = r_ext + β * r_i. Compute advantages Â using GAE.
d. Update Policy: For K epochs, update π and V using the PPO clipped surrogate objective on the collected data.
e. Update Meta-Controller: Evaluate the change in agent performance (e.g., Δ avg(r_ext)). Use this value to update the statistics of the chosen bandit arm i.
f. Update Reference Buffer: Add new, high-performing trajectories to R.

This CAPO framework provides a principled way to combine multiple learning strategies within a single PPO-like algorithm, allowing the agent to dynamically adapt its learning process to the specific challenges it faces at any given moment.
