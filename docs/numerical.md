# Numerical Analysis of Degeneration in IK Solving

As mentioned in `docs/degenerate_analyzer.md`, we represent the general degenerate condition as a set of equations. 
In some cases, we can solve these equations and transform the degeneration into solved variable degeneration record.
However, these equations might be too hard to solve.

In this case, we perform numerical analysis to determine whether a set of equations can take zero value simultaneously.
We first generate a set of robot configurations q, and use forward kinematics to generate the end-effector pose. These 
should be all the parameters that appear in the equations. Then, we substitute the equations with the values to check 
how many of them degenerates. When generating the q, we take some special values (0, pi, pi/2) such that it is more
likely to yield degeneration.

Another idea is to find zeros of the degenerate equations using non-linear optimization. Which is not implemented yet.