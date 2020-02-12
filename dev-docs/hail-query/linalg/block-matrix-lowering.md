# Overview

This document describes the process for computations represented by BlockMatrixIR 
nodes into computations represented by value IR nodes using CollectDistributedArray.

# BlockMatrixStage

BlockMatrixStage is the intermediate representation for a lowered BlockMatrix. It 
represents the block structure of a BlockMatrixIR node as well as all information 
necessary for the block-level computation.

- ctxRef: a handle for referring to the context
- blockContexts: sparse map of contexts for each block
- broadcastVals: map of (named) values that are shared across each context
- body: IR: tranformation of the block context that yields the matrix value of each block

## Broadcast values

If a value is needed in the body of the computation, such as with a relational 
Let, it should generally be included as a broadcast value.

Broadcast values are named for ease of reference; all names should be unique.

Since the broadcast values are stored as an unordered map, there is no way to 
reference other previously defined broadcast values in a new broadcast 
definition. The current structure of the BlockMatrixIR means that it is 
unlikely to be necessary.

When creating a BlockMatrixStage from other BlockMatrixStages, broadcast values
should be preserved from all child BlockMatrixStages.

### Referencing broadcast values from contexts and body

Broadcast values can be referenced from both block contexts and the body of the 
computation using `Ref(name, value.typ)`.

## Contexts

Each block in a BlockMatrixIR node will need to define its own context as 
a value IR node.

Since any given block context could be used multiple times in downstream lowering 
transformations (e.g. multiply), the actual IR constructed for any given context
should be minimal.

If a context requires non-trivial computation, such as in `ValueToBlockMatrix`
which creates an arbitrary NDArray, the computation itself should be stored as 
a broadcast value and a reference to that value used in the context IR itself.

# lowering value IRs with BlockMatrixStages

BlockMatrixStage generates a value IR node which executes all block computations and
returns the results as a (sparse) array. The primary way for doing this is with the
function:

```
toIR(bodyTranform: IR => IR, ordering: Option[Array[(Int, Int)]]): IR
```

The bodyTransform function should return the (minimal) value from each block 
necessary to evaluate that node; a `BlockMatrixCollect`, for example, will need 
the entire NDArray, but a `BlockMatrixWrite` operation might only need to know 
the filename where the data got written, and perhaps the block indices.

The function returns an array of results for each block.

The (optional) block ordering allows the result array to be computed in a 
specific order; the function otherwise makes no guarantee about the order
in which blocks will appear in the result array.