# Degenerate Analyze in YAIK

As mentioned in the `docs/degenerate.md`, in some situation a given set of equations might degenerate, and we must 
resort to some other equations to find the solution or conclude no solution exists. This document describes how 
it is implemented.

## The General Representation

In the most general form, the degenerate situation is represented as a set of equations (all) equals zero. In other 
words, if a given set of equations all equal zero, then we resort to other equations. This is an **AND** relation,
there is no *(expr_A = 0 **Or** expr_B = 0)* implemented yet, and you might represent this **OR** relationship 
using their product (*expr_A * expr_B = 0*, refer to tangent solver as an example).


## The Solved-Variable Representation

In this case, degeneration happens when a set of variables equal some value. This is a special case of the general 
"equation all zero" degeneration, if each equation is in the form of *var_A - var_A_value = 0*. Again, this is an 
**AND** relation. We have not implemented *var_A = var_A_value **OR** var_B = var_B_value* yet. However, 
*(var_A, var_B) = (value_A_1, value_B_1) **OR** (value_A_2, value_B_2)* is implemented. This implies we might get 
multiple solutions for a given set of variables from the General Representation.

Obviously, the solved-variable representation is obtained from the general representation: we need to solve the 
equations in the general representation. Currently, we restrict ourselves to
1) Unary solved-variable: no *(variable_A, variable_B) = ...*, only ONE variable appears.
2) The variable must be an 1) input parameter; or 2) already solved variable. It cannot be a variable that 
has not been solved yet. (TODO(wei): check this)


## Constant Processing in Degenerate Analysis

For a general "equation all zero" degenerate in the form of *equ_A = 0 **AND** equ_B = 0*, if we can conclude 
*equ_A* is a non-zero constant, then the degeneration never happens. Please refer to 
`solver/degenerate_analyse/analyser_step.py:CheckConstantAnalyserStep` for the implementation.

## Numerical Analysis

Please refer to `docs/numerical.md` for more details.
