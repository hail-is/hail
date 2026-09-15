---
status: proposed
---

# Explicit scope structure in the IR: Frame nodes with per-operation Signatures

The binding structure of the IR lived in one case analysis, `Bindings.get` in
`Binds.scala`, plus parallel tables in `Pretty`, LICM, and every pass that hand-matched
name-binding operations. We decided to make scope explicit in the tree. A `Frame` node
opens a scope: it holds the parameter names, a reference to a per-operation `Signature`
singleton that computes the Frame's `Bindings[Type]` from the parent's other children, and
a body that is a plain `IR`. `Block` stays an ordered list of lets with a result and never
opens a scope. Environment extension then reads off the child without knowing the parent,
and the static facts about each operation live in one generated place.

## Considered options

**Parameters on `Block` rather than a new node.** Chosen at one point, then reversed once
the migration was costed. The argument for it is that under weak A-normal form a Block
occurs only at a scope-opening edge, so Frame and Block coincide and the second node is
redundant. The arguments against are concrete and present: every lowering that re-binds a
body under another operation becomes a splice of `bindings` and `result` instead of
`Let(bindings, frame.body)`, the same shape as inlining an `ApplyIR`; all thirty
`case Block(bindings, body)` sites change arity; every companion `apply` must absorb an
inline Block; a runtime guard is needed against a parameterised Block nested where an
expression belongs; and the Block rewrite rules and WeakANF invariant must preserve
parameters. A separate Frame leaves all of that untouched. The one wrapper node per lambda
that A-normal form makes redundant can be merged later, once every site already goes
through the companions.

**Storing `Bindings[Type]` on the Frame.** Attractive because environment extension then
needs nothing from the parent. Rejected for a fact about construction order: `Parser`
builds trees with untyped `Ref`s and types them afterwards in `annotateTypes`, which
computes each child's environment from siblings annotated moments earlier
(`BaseIR.mapChildrenWithEnv` reads `Bindings.get(res, i)` on the progressively rewritten
node). A companion `apply` cannot compute `elementType(a.typ)` while `a` is still untyped,
so stored types would need either an environment-aware parser or a recompute-from-parent
path, which is the signature again plus a cache to keep coherent. Stored types would also
need refreshing wherever a sibling's type changes and would fix the domain to `Type`. A
signature computes on demand, stores only names per instance, and shares everything else.
Every caller of `Bindings.get` already holds the parent; what the signature removes is
knowledge of the parent's kind.

**Signature as a generated virtual method on the parent, Frame holding names only.**
Rejected because zero-parameter Frames on `ApplyAggOp` init args would then carry nothing,
and the Frame is where "opens a scope" should be readable.

**Making the IR generic in the value domain.** Rejected: abstract values belong in side
tables keyed by node, as `Requiredness` and `PruneDeadFields` already do. One tree per
analysis is not viable.

**Naming.** `Lambda` is reserved for a future bindable, separately emitted function, which
this node is not. `Region` collides with the memory package in the files `Emit` lives in.
`Frame` is already the LICM pass's internal name for this concept; the glossary states
that a Frame is a scope, not an allocation.

## Consequences

- Every scope-changing child, including those that bind no names but change the agg or
  scan environment, is a Frame. `If`, `Switch`, `Coalesce` branches are not: they bind
  nothing and change no scope, and making them Frames would force every rule that
  collapses a conditional to re-wrap the surviving branch. They become Frames only when
  special forms are separated under A-normal form.
- Strictness is a property of an operation on a child position, not of the Frame. ir-gen
  marks non-strict children in the node declaration and emits a per-node method, so
  `Pretty` and LICM ask the node rather than keep tables. `Signature` does not carry it.
- The Frame's parameter names are the field `params`. `TailLoop`'s loop-variable names
  move into its body Frame's `params`; the values that stay on `TailLoop` are renamed
  `init`, since they seed the loop once, while `Recur.args` supplies each iteration.
- Operation companions keep their current `apply`/`unapply` arity: `apply(a, name, body)`
  wraps the body in a Frame and `unapply` yields `(a, name, frame.body)`, so all existing
  pattern matches and lowerings compile and behave unchanged. Code that needs the Frame
  itself reads the field.
- Signature implementations are private to their operation's companion. Only the trait is
  public, and only `Bindings.get`, `TypeCheck`, and the companions use it.
- Two generic `Simplify` rules replace the per-operator `Block` rules and land before the
  Frame change. Hoist: a strict child that is a Block moves its lets above the parent.
  Absorb: a Block in a Block's result or a let's value is spliced in place.
- Detached Frames have free parameters: the parent supplies their bindings through
  `Bindings.get(parent, i)`, so analyses started at a Frame must bind them first, as they
  do today for a detached body.
- The text wire format between Python and the JVM is unchanged. Parser and Pretty go
  through the companions.
- A Frame cannot state its parameter types without its parent. Passes that need them have
  the parent in hand; `Ref._typ` covers use sites.
- Signature also carries the readable labels the SSA pretty printer uses for parameters
  (`elt`, `accum`, `loopvar`), as a function of the parameter count so variable-arity
  operations repeat theirs. The generator has the labels from the declaration, so this
  costs nothing and deletes Pretty's hand-written table.
- `Frame` is an `IR` typed by its body. `childrenSeq` is `IndexedSeq[BaseIR]`, so a
  non-`IR` Frame would be structurally fine, but 34 sites dispatch on `BaseIR` kind with
  `case x: IR` and 66 cast children with `asInstanceOf[IR]`; `BaseIR` is unsealed, so each
  missing Frame arm is a runtime `MatchError`. The compile-time guarantee that a Frame
  never sits in an operand slot is already provided in practice by the companions.
  Making Frame its own `BaseIR` kind is deferred until special forms are separated from
  operations under A-normal form, when those dispatch sites are restructured anyway.
- Longer term the IR is to be parsed into weak A-normal form and optimised in that form,
  with special forms (Frame, Block, If, Switch, ...) separated from operations. This
  decision is the first step and must not preclude it.
