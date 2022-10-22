# pursuit-evasion-problem

## Abstract
Pursuit and evasion conflicts represent challenging problems with important applications in aerospace and robotics. In pursuit-evasion problems, synthesis of intelligent actions must consider the adversary’s potential strategies. Differential game theory provides an adequate framework to analyze possible outcomes of the conflict without assuming particular behaviors by the opponent. This work is concerned with a Minimum-Time Intercept Problem (MTIP), for which a Dubins vehicle is guided from a position with a prescribed initial orientation angle to intercept a moving target in minimum time. Moreover, the Deep Deterministic Policy Gradient (DDPG) algorithm  (source: https://keras.io/examples/rl/ddpg_pendulum/) is applied. 

The main purpose of the work is to study the behavior of a neural network for the pursuit-evasion problem and to check whether the DDPG algorithm can be used to find solutions to differential games. To evaluate the operation of the algorithm, a comparison is made with the analytical solution obtained in this article https://link.springer.com/article/10.1134/S0005117921050015
