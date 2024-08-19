# Skeleton Tree in YAIK

The skeleton tree captures the dependency between different variables in the solution procedure, as shown in `codegen/skeleton_tree.py`. For example, the solving of `theta_0` happens before `theta_1`, (which usually implies the solution of `theta_1` depends on `theta_0`), then there is a parent-child relation between `theta_0` and `theta_1`.

So far, this is an array data structure instead of a tree structure. However, there might be degenerate solutions. Given the solution of `theta_0`, we might choose different solving orders/solutions according to the value of `theta_0`, in this case we would have a tree structure. The skeleton tree captures this tree structure.

#### Skeleton Tree v1:

For the version 1.0 of the skeleton tree, we do not use the full tree structure. Instead, we 
assume **the branch can only happen on the main branch (non-degenerate branch)**.


#### Skeleton Tree v2:

For the version 2.0 of the skeleton tree, it is a complete tree with arbitrary branch consist of two types of
nodes: the solution node and dispatcher node. The solution node and dispatcher node must appear one by one from the 
root of the tree to the leaf of the tree. The leaf node must be solution node.

The version 2.0 of the skeleton tree implements the standard visitor mode to support 1) numerical solving; 2) python
code generation. The C++ code generation would be the next major feature.