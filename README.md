# Pursuit-Evasion problem

## Abstract
Pursuit and evasion conflicts represent challenging problems with important applications in aerospace and robotics. In pursuit-evasion problems, synthesis of intelligent actions must consider the adversary’s potential strategies. Differential game theory provides an adequate framework to analyze possible outcomes of the conflict without assuming particular behaviors by the opponent. This work is concerned with a Minimum-Time Intercept Problem (MTIP), for which a Dubins vehicle is guided from a position with a prescribed initial orientation angle to intercept a moving target in minimum time. Moreover, the Deep Deterministic Policy Gradient (DDPG) algorithm  (source: https://keras.io/examples/rl/ddpg_pendulum/) is applied. 

The task of intercepting a target moving along a rectilinear or circular trajectory by a Dubins machine is formulated as an optimal control problem according to the speed criterion with an arbitrary direction of the machine speed during interception. To solve this problem and synthesize interception trajectories, neural network methods of teaching without a teacher based on the Deep Deterministic Policy Gradient algorithm were used. The analysis of the obtained control laws and interception trajectories in comparison with the analytical solutions of the interception problem was carried out, modeling was carried out for the parameters of the target movement that the neural network did not see during training. Model experiments were carried out to test the stability of the solution. The effectiveness of the use of neural network methods for the synthesis of intercept trajectories for given classes is shown.

The main purpose of the work is to study the behavior of a neural network for the pursuit-evasion problem and to check whether the DDPG algorithm can be used to find solutions to differential games. To evaluate the operation of the algorithm, a comparison is made with the analytical solution obtained in this article https://link.springer.com/article/10.1134/S0005117921050015

## Formulation of the neural network interception problem

On the plane, the problem of the fastest $\delta$-interception by a Dubins machine (pursuer) of a moving object (target) moving along two given trajectories at a constant speed is considered.  The dynamics for the pursuer was chosen:

$$\dot{x_P}=\cos{\varphi}$$
$$\dot{y_P}=\sin{\varphi}$$
$$\dot{\varphi}=u,~ |u(t)|\leqslant1$$

The initial conditions of the system are fixed:
$$x_P(0) = 0, \quad y_P(0) = 0, \quad \varphi(0) = \frac{\pi}{2}$$

The terminal condition of $\delta$-interception for a neural network solution has the following:
$$(x_P(T) - x_E(T))^2 + (y_P(T) - y_E(T))^2 \leqslant \delta^2$$

According to the condition of the task, the target moves at a constant speed in a straight line. Then the parametrized coordinate equations will have the following form:
$$x_E(t)=v_x t + x_0$$
$$y_E(t)=v_y t + y_0$$

To take into account the relative position of the pursuer and the target, we introduce a formula for finding the angle between the abscissa axis and the straight line connecting the coordinate points of the target and the pursuer. Let $(x_P, y_P)$ and $(x_E, y_E)$ be the coordinates of the pursuer and the target, respectively, at some point in time $t$. Then the desired value of the angle is found by the formula:
$$\psi = \arctan{ \left( \frac{y_E - y_P}{x_E - x_P} \right)}$$

We will also introduce a formula for calculating the distance $L$ between agents:
$$L=\sqrt{(x_P-x_E)^2 + (y_P-y_E)^2}$$

## Deep Deterministic Policy Gradient Algorithm

DDPG is an Actor-Critic algorithm based on a deterministic policy gradient. The DPG (Deterministic Policy Gradient) algorithm consists of a parameterized function Actor $\mu\left(s\mid\theta^{\mu}\right)$, which sets control at the current time by deterministic matching of states with a specific action. The function Critic $Q(s,a)$ is updated using the Bellman equation in the same way as with $Q$-training.  The Actor is updated by applying a chain rule to the expected reward from the initial distribution of $J$ in relation to the parameters of the Actor.

DDPG combines the advantages of its predecessors, which makes it more stable and effective in training. 
Since different trajectories can be very different from each other, DDPG uses the idea of a DQN, called a playback buffer. The playback buffer is a finite—size buffer into which media data is stored at any given time. It is necessary to achieve a uniform distribution of the transition sample and discrete control of neural network training. Actor and Critic are updated by evenly sampling the mini-batch from the playback buffer. Another addition to DDPG was the concept of updating program targets instead of directly copying weights to the target network. Network being updated $Q\left(s, a\mid\theta^{Q}\right)$ is also used to calculate the target value, so updating $Q$ is subject to divergence. This is possible if you make a copy of the Actor and Critic networks, $Q^{\prime}\left(s, a\mid\theta^{Q^{\prime}}\right)$ and $\mu^{\prime}\left(s, a \mid\theta^{\mu^{\prime}}\right)$. The weights of these networks are as follows: $\theta^{\prime} \leftarrow \tau\theta+(1-\tau)\theta^{\prime}$ with $\tau\ll 1$.  The research problem is solved by adding the noise received from the noise process $N$ to the control of the actor. In our study, the Ornstein-Uhlenbeck process was chosen.

![DDPG](/images/ddpg_algorithm.png "DDPG")
<!-- <img src="/images/ex_roti_map.jpeg" alt="ROTI map" width="400"/> -->

## Neural network

To implement the Deep Deterministic Policy Gradient algorithm, two neural networks were written for each method: Critic and Actor.
    
    
The Actor network has four fully connected hidden layers with 256 neurons, with the activation function $SELU$. Since the possible actions are in the range [-1,1], it is convenient to take the activation function for the output layer as $tanh$. The Critic network has five fully connected hidden layers with 16, 32, 32 and two layers with 512 neurons, with the activation function $SELU$.

The Critic and Actor networks are made up of fully connected $Dance$ layers, for the output values of which the normalization operation and the $Dropout$ method are used, which is effective in combating the problem of retraining neural networks. To calculate the output of the Actor network from the last layer, the hyperbolic tangent activation function is selected.

<img src="/images/actor_model.png" alt="Actor model" width="300"/> <img src="/images/critic_model.png" alt="Critic model" width="410"/> 

The Critic network has a complex structure because it takes two input values: the state of the environment and the actions of the pursuer. Next, the layers are connected using the $Concatenate$ method and the values pass through the fully connected layers of the network to the output, which is a layer of unit dimension.

## The training process

The initial coordinates of the target movement are randomly selected so that the network trains on different examples and works effectively after the training process.

The figure below shows a graph of the average remuneration for the entire training period. During the training of the model, there is a sharp increase in the value of the agent's reward in the first 100-150 episodes. Filling of the playback buffer $R$ corresponds to this process. Further, the training examples are randomly taken from $R$, the network training process takes place and the resulting tuple of states replaces the old data sample in $R$. At this stage, there is a slow increase in average remuneration.

<img src="/images/avg_reward_500.png" alt="Avg reward" width="600"/>
<!-- ![Avg reward](/images/avg_reward_500.png "Avg reward") -->

Graphs of dependencies of the error function of the Actor and Critic neural networks were also obtained. They are shown in the figures below, respectively:

<img src="/images/actor_loss.png" alt="Actor loss" width="400"/> <img src="/images/critic_loss.png" alt="Critic loss" width="400"/>

The graphs show a gradual decrease in the value of the loss function with an increase in training episodes, which indicates the correct choice of training coefficients.

These figures show the evolution of the neural network learning process.

<img src="/images/train_1.png" alt="train" width="250"/> <img src="/images/train_5.png" alt="train" width="250"/> <img src="/images/train_3.png" alt="train" width="250"/>

## Learning result

The figures show the trajectories obtained using a neural network and an analytical solution (The trajectory of the neural network is highlighted in orange, and the analytical trajectory is green. The intercept radius is highlighted in red):

<img src="/images/result_2.png" alt="Result" width="400"/> <img src="/images/result_3.png" alt="Result" width="400"/>

## Conclusion

In the work, two neural network algorithms based on DDPG for the synthesis of trajectories of interception by the Dubins machine of targets moving along a rectilinear trajectory were proposed. The features of the proposed algorithms are their ability to work with the space of continuous actions, the guarantee of learning and working with different relative initial positions of goals and the Dubins machine. Moreover, it is shown that the network in some situations offers the best solution to the interception problem in terms of speed.

The research on this work has been completed. The article was written and submitted to the journal. There is a process of waiting for the second review and preparing for publication.