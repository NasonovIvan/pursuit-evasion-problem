# pursuit-evasion-problem

# Introduction
Pursuit and evasion conflicts represent challenging problems with important applications in aerospace and robotics. In pursuit-evasion problems, synthesis of intelligent actions must consider the adversary’s potential strategies. Differential game theory provides an adequate framework to analyze possible outcomes of the conflict without assuming particular behaviors by the opponent. This work is concerned with a Minimum-Time Intercept Problem (MTIP), for which a Dubins vehicle is guided from a position with a prescribed initial orientation angle to intercept a moving target in minimum time. Moreover, the Deep Deterministic Policy Gradient (DDPG) algorithm  (source: https://keras.io/examples/rl/ddpg_pendulum/) is applied. 

The main purpose of the work is to study the behavior of a neural network for the pursuit-evasion problem and to check whether the DDPG algorithm can be used to find solutions to differential games. To evaluate the operation of the algorithm, a comparison is made with the analytical solution obtained in this article https://link.springer.com/article/10.1134/S0005117921050015

# Dynamic

The dynamics of the pursuer was chosen in the form of
    \[
    \begin{cases}
    \dot{x_P}=\cos{\varphi}, & \dot{\varphi}=u,
    \\
    \dot{y_P}=\sin{\varphi}, & |u(t)|\leqslant1
    \end{cases}.
    \]
    Here $x_P(t)$ and $y_P(t)$ coordinates of the Dubins machine on the Cartesian plane, $\varphi(t)$ the angle between the direction of the pursuer's speed and the abscissa, and $u(t)$ time-dependent control.
    
    The initial conditions of the system are fixed:
    $$x_P(0) = 0 \quad y_P(0) = 0 \quad \varphi(0) = \frac{\pi}{2}.$$
    
    Continuous vector function $E(t)=(x_E(t),y_E(t))$ defines the trajectory of the target on the Cartesian plane. The terminal intercept condition for the analytical solution has the following form:
    $$x_T \stackrel{def}{=} x_P(T) = x_E(T) \quad y_T \stackrel{def}{=} y_P(T) = y_E(T).$$
    
    Here $T\in\mathbb{R}^{+}_{0}$ the time of movement from the starting point to the intercept point. Let's set the task of intercepting the target in minimal time as an optimal control problem in the class of piecewise constant functions:
    $$\displaystyl J[u] \stackrel{def}{=} \int\limits_0^T dt \rightarrow \underset{u}{\min}.$$
    
    Now let's describe the dynamics of the goal. Since, according to the condition of the problem, the target moves rectilinearly at a constant speed, the parameterized coordinate equations will have the following form:
    \[
    \begin{cases}
    x_E(t)=v_x t + x_0
    \\
    y_E(t)=v_y t + y_0
    \end{cases},
    \]
    where $x_0$ and $y_0$ are arbitrarily chosen constants.

# Additional conditions

Now we introduce a formula for calculating the angle between the abscissa axis and the straight line connecting the coordinate points of the target and the pursuer. Let $(x_P, y_P)$ and $(x_E, y_E)$ be the coordinates of the pursuer and the target, respectively, at some point in time $t$. Then the desired value: $$\psi = \arctan{ \left( \frac{y_E - y_P}{x_E - x_P} \right)}.$$

We will also introduce a formula for calculating the distance $L$ between agents: $$L=\sqrt{(x_P-x_E)^2 + (y_P-y_E)^2}.$$

Next, to simplify the task, we will make the transition to reduced coordinates. To do this, you need the current state of the agents $S=(x_P, y_P, \varphi, x_E, y_E)$ and the state predicted by the neural network $S'=(x' _P, y' _P, \varphi', x'_E, y'_E)$.

We get the values for the functions of the angles $\psi$ and $\psi'$ from the states $S$ and $S'$, respectively, and also calculate the distance $L'$ when the agents are in the state $S'$. Next, we introduce the angle between the direction of the speed of the pursuer and the line connecting the coordinate points of the agents: $$\Theta=\varphi' - \psi'.$$

Now let's introduce the rotation speed as a quotient of the difference $\psi' - \psi$ and the time $\Delta t$ during which the transition from the state $S$ to the state $S'$ occurred:
$$\omega =\frac{\psi' - \psi}{\Delta t}.$$
    
The set of coordinates $(L', \omega, \Theta)$ and there are the desired reduced coordinates.
At the initial moment of time, when the result of the neural network has not yet been received, the reduced coordinates are calculated as follows:
\
    \begin{cases}
    L'=L(S)
    \\
    \omega = 0
    \\
    \Theta = \varphi(0) - \psi
    \end{cases}
    ,
    where $\psi = \arctan{ \left( \frac{y_E(0) - y_P(0)}{x_E(0) - x_P(0)} \right)}$.

It should be noted that when solving the problem, we will adhere to the following conditions:

1) The speed of the target does not exceed the modulus of the speed of the Dubins machine. Indeed, if the target has a speed significantly exceeding the value of the speed of the pursuer, then it is impossible to intercept with straight-line movement.

2) The presence of a predetermined interception radius $R_0$ - the maximum allowable distance between the pursuer and the target, at which the interception can be considered perfect. This parameter is introduced to define the concept of interception.
