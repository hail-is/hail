package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir._
import is.hail.expr.ir.Scope._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.invariant.UniquelyNamed
import is.hail.types.virtual.{TStream, TVoid}
import is.hail.utils.TimedBlock

import scala.annotation.tailrec
import scala.collection.immutable.HashMap
import scala.collection.mutable
import scala.util.Random

// Rewrites an IR into a weak A-normal form (ANF), deduplicating repeated
// pure computations and hoisting bindings as far out as is sound:
//
//   - naming: each liftable child of a strict edge is bound to a fresh
//     name in an enclosing Block, making intermediate values explicit
//     ("weak" ANF: atoms stay inline and non-strict children keep their
//     shape);
//   - common-subexpression elimination: an expression structurally equal
//     to one already bound and in scope (an "available expression")
//     collapses to a reference to that binding;
//   - code motion: a binding lands at the shallowest frame at which its
//     free names all resolve, so values invariant in a stream or loop
//     body float out of it (loop-invariant code motion);
//   - partial redundancy elimination: an expression pinned under a
//     conditional still rises to a frame on whose strict spine it is
//     anticipated (certain to be evaluated later), since hoisting it
//     there moves an evaluation earlier instead of adding one.
//
// One recursive pass over the tree, tracking three structures:
//
//   - Region: the span within which names may be shared, delimited by
//     dropEval edges (eg TableAggregate's query), across which the child
//     cannot reference the parent's bindings;
//   - Frame: one per non-strict edge within a region; the landing spots
//     where new bindings may be placed;
//   - AvailableExprs: per frame, the expressions already bound to a name
//     and eligible for reuse.
object LiftAvailableExprs {

  def apply(ctx: ExecuteContext, ir0: BaseIR): BaseIR =
    TimedBlock.enter {
      UniquelyNamed.verify(ctx, ir0)
      new Impl()(ir0)
    }

  // is `ir` a CSE candidate that can be lifted to a let?
  //   - naming an Atom gains nothing: the ref replacing it is no smaller;
  //     UUID4 is impure by design (each occurrence is distinct);
  //   - a TVoid child has no value to name; streams cannot be let-bound;
  //   - anything containing a read of mutable aggregation state (ResultOp,
  //     AggStateValue, ...): an InitOp or SeqOp may intervene between two
  //     structurally equal occurrences, so they are never available
  //     expressions — the node itself and every wrapper around it.
  private def isLiftable(ir: IR): Boolean =
    ir match {
      case _: Atom | _: UUID4 => false
      case _ =>
        !(ir.typ == TVoid || ir.typ.isInstanceOf[TStream]) &&
        !ContainsAggIntermediate(ir)
    }

  // Set of operations that may cross a non-strict edge.
  // Evaluating one speculatively (the edge may evaluate its child zero times)
  // can change performance but not semantics. Everything else is pinned to its
  // own frame, preserving evaluation count and order.
  private def isSpeculatable(ir: IR): Boolean =
    ir match {
      case _: Literal | _: EncodedLiteral | _: Str // non-Atom constants
          | _: MakeArray | _: ArrayLen | _: ArraySlice | _: ArraySort | _: ArrayZeros
          | _: MakeStruct | _: GetField | _: SelectFields | _: InsertFields
          | _: MakeTuple | _: GetTupleElement
          | _: IsNA
          | _: Cast | _: CastToArray | _: CastRename
          | _: ToArray | _: ToSet | _: ToDict
          | _: ApplyUnaryPrimOp | _: ApplyComparisonOp => true
      case ApplyBinaryPrimOp(Add() | Subtract() | Multiply(), _, _) => true
      case _ => false
    }

  // Children the parent may evaluate zero times. Such frames set the
  // floor: placing a binding above one would evaluate it on paths that
  // never need it.
  private def isConditionallyEvaluated(parent: IR, childIdx: Int): Boolean =
    parent match {
      case _: If => childIdx > 0
      case _: Switch => childIdx > 0
      case _: Coalesce => true
      case ApplySpecial("lor" | "land", _, _, _, _) => childIdx > 0
      case _ => false
    }

  // Edges that need a lookup wall: those across which the aggregable
  // elements change while agg- (scan-) dependent expressions stay legal,
  // so an equal expression bound outside is in scope yet aggregates
  // different elements:
  //   - the transformers' (filter, explode, group-by, per-element) aggIR
  //     children reinterpret the enclosing context's elements;
  //   - StreamAgg(Scan) bodies create a context, but outer eval entries
  //     (including named aggregation results) survive into the body.
  // Edges that remove or replace the scope need no wall:
  //   - AggFold's Promote/Drop children swap the buffers holding any
  //     stale entries out of reach, and no agg op typechecks inside;
  //   - TableAggregate and MatrixAggregate create contexts behind
  //     dropEval walls, which no entry crosses.
  private def changesAggContext(ir: IR, i: Int): Option[Scope] =
    ir match {
      case AggFilter(_, _, isScan) if i == 1 => Some(if (isScan) SCAN else AGG)
      case AggExplode(_, _, _, isScan) if i == 1 => Some(if (isScan) SCAN else AGG)
      case AggGroupBy(_, _, isScan) if i == 1 => Some(if (isScan) SCAN else AGG)
      case AggArrayPerElement(_, _, _, _, _, isScan) if i == 1 =>
        Some(if (isScan) SCAN else AGG)
      case _: StreamAgg if i == 1 => Some(AGG)
      case _: StreamAggScan if i == 1 => Some(SCAN)
      case _ => None
    }

  private def noTransition(env: AggEnv): Boolean =
    env match {
      case AggEnv.NoOp | AggEnv.Bind(_) => true
      case _ => false
    }

  // A landing spot for bindings: one frame per non-strict edge on the
  // path from the region root to the current expression. The floor and
  // the walls guard different operations:
  //   - `floor` bounds placement: a binding hoisted above a
  //     conditionally-evaluated edge would be computed on paths that
  //     never use it (speculation);
  //   - the walls bound lookup: an agg- (scan-) dependent binding made
  //     outside a context change is in scope and structurally equal, yet
  //     aggregates different elements (see Region.lookup).
  private class Frame(
    // the deepest conditionally-evaluated frame at or
    // above this one; nothing may be placed shallower than it.
    val floor: Int,

    // the deepest agg context change at or above this one;
    // lookup of an agg-dependent expression sees nothing shallower than it.
    val aggWall: Int,

    // the deepest scan context change at or above this one;
    // lookup of a scan-dependent expression sees nothing shallower than it.
    val scanWall: Int,
  ) {
    val builder: IRBuilder = new IRBuilder

    // The anticipated suffixes of this frame's strict spine: one slot per
    // enclosing spine position (nested Blocks push a slot, replace it as
    // their position advances, and pop it before their body). An expression
    // found in any slot is certain to be evaluated whenever this frame's
    // block runs, so placing it here adds no evaluation on any path.
    val anticipated: mutable.ArrayBuffer[Anticipated] =
      mutable.ArrayBuffer.empty

    // the max level (below this frame's own) of any name occurring free within
    // this frame's block.
    var escapes: Int = 0
  }

  // The expressions certain to be evaluated later on a frame's strict spine,
  // seen from one spine position. The scan is deferred to the first query:
  // most positions are never asked, since queries only come from speculatable
  // expressions pinned under a conditional floor.
  // Each position scans its own suffix, so a block in which many positions
  // are queried rescans the same tail — quadratic in block length in the
  // worst case. The linear alternative (per-binding cells, scanned once and
  // shared by every position) was rejected: a cell caches its resolutions as
  // of the first query that forces it, so an alias recorded later in the
  // block would be missed and the hoists it licenses silently lost.
  // Rescanning per position always resolves through the current alias map.
  final private class Anticipated(scan: () => collection.Set[IR]) {
    private[this] lazy val exprs = scan()
    def contains(ir: IR): Boolean = exprs.contains(ir)
  }

  private object Anticipated {
    val empty: Anticipated =
      new Anticipated(() => collection.Set.empty)
  }

  // A frame's available expressions, tagged with the scope (EVAL, AGG or SCAN)
  // in which a binding placed at that frame must be declared.
  // Placement happens at a distance: a lifted value may land many frames
  // above its use, and the scope of the binding it becomes depends on where
  // the landing frame sits relative to scope transitions. The buffer
  // swaps are what track those transitions, so each slot carries the tag
  // for bindings landing at its level (read back via `scopeAt`).
  private case class ScopedEnv(scope: Scope, env: Map[IR, Atom] = HashMap.empty) {
    def bind(ir: IR, atom: Atom): ScopedEnv =
      copy(env = env.updated(ir, atom))
  }

  // The expressions available at every frame of a region, in buffers
  // mirroring Hail's three binding environments:
  //   - eval holds ordinary value bindings and is always the buffer that
  //     lookups and insertions target (the "current" environment);
  //   - agg and scan hold bindings usable per element of the enclosing
  //     aggregation (scan), present only where such a context exists.
  // Each buffer has one env per frame, indexed by frame level. Buffer
  // presence and position are uniform across a region's frames
  // (transitions apply to all frames in lockstep), so an edge's
  // transition is interpreted once and applied to every frame at once by
  // swapping whole buffers; per-frame contents and scope tags live in the
  // buffers' slots.
  private class AvailableExprs(
    private var eval: mutable.ArrayBuffer[ScopedEnv] =
      mutable.ArrayBuffer(ScopedEnv(EVAL)),
    private var agg: Option[mutable.ArrayBuffer[ScopedEnv]] =
      Some(mutable.ArrayBuffer(ScopedEnv(AGG))),
    private var scan: Option[mutable.ArrayBuffer[ScopedEnv]] =
      Some(mutable.ArrayBuffer(ScopedEnv(SCAN))),
  ) {
    // A freshly-pushed frame has no available expressions of its own; bindings
    // placed there from the current compartment are plain (EVAL) lets.
    def push(): Unit = {
      eval += ScopedEnv(EVAL)
      agg.foreach(_ += ScopedEnv(AGG))
      scan.foreach(_ += ScopedEnv(SCAN))
    }

    def pop(): Unit = {
      val end = eval.length - 1
      eval.remove(end): Unit
      agg.foreach(_.remove(end))
      scan.foreach(_.remove(end))
    }

    def scopeAt(level: Int): Scope =
      eval(level).scope

    // Both aggregation compartments intact: the current eval buffer is the
    // region's ordinary evaluation environment rather than an aggregation's
    // element environment (after a Promote) or a dying one (after a Drop).
    def full: Boolean = agg.isDefined && scan.isDefined

    def bind(level: Int, ir: IR, atom: Atom): Unit =
      eval(level) = eval(level).bind(ir, atom)

    // Search from the deepest frame outward; levels shallower than
    // `wall` are invisible (see Region.lookup).
    def lookup(ir: IR, wall: Int): Option[(Atom, Int)] = {
      @tailrec def go(level: Int): Option[(Atom, Int)] =
        if (level < wall) None
        else eval(level).env.get(ir) match {
          case Some(atom) => Some(atom -> level)
          case None => go(level - 1)
        }

      go(eval.length - 1)
    }

    // Transitions swap compartments without copying, so a snapshot is three
    // references. Additions made while a compartment is swapped in mutate its
    // slots in place and ride along when `restore` swaps it back: additions to
    // surviving compartments are kept, Create/Drop additions are discarded with
    // the compartment holding them. Frames pushed after a snapshot are popped
    // before its restore, keeping lengths consistent.
    def save: AvailableExprs =
      new AvailableExprs(eval, agg, scan)

    def promote(s: Scope): Unit =
      s match {
        case EVAL =>
        case AGG =>
          eval = agg.get
          agg = None
        case SCAN =>
          eval = scan.get
          scan = None
      }

    def restore(s: Scope, saved: AvailableExprs): Unit =
      s match {
        case EVAL =>
        case AGG =>
          agg = Some(eval)
          eval = saved.eval
        case SCAN =>
          scan = Some(eval)
          eval = saved.eval
      }

    // Apply an edge's environment transition to every frame at once:
    //   - Drop ends a context: its buffer and its entries die with it;
    //   - Promote evaluates the child in the aggregation's element
    //     environment: that buffer becomes current, and the old eval
    //     buffer (with any agg-dependent entries) is unreachable until
    //     `restore`;
    //   - Create opens a fresh context whose element environment extends
    //     the current eval environment, so eval entries remain available
    //     inside; pre-existing agg/scan entries do not.
    def extend(bindings: Bindings[_]): Unit = {
      if (bindings.dropEval)
        eval = eval.map(_ => ScopedEnv(EVAL))

      val created =
        bindings.agg.isInstanceOf[AggEnv.Create] || bindings.scan.isInstanceOf[AggEnv.Create]

      if (created) {
        agg = agg.map(_.map(a => ScopedEnv(a.scope)))
        scan = scan.map(_.map(s => ScopedEnv(s.scope)))
      }

      bindings.agg match {
        case AggEnv.Drop =>
          agg = None
        case AggEnv.Promote =>
          eval = agg.get
          agg = None
        case AggEnv.Create(_) =>
          // Behind a dropEval wall the region root itself sits inside the
          // created scope, so bindings at every level must be scope-tagged;
          // otherwise levels shallower than this edge materialize outside the
          // created scope and keep eval's tags.
          agg = Some(
            if (bindings.dropEval) eval.map(_ => ScopedEnv(AGG))
            else eval.clone()
          )
        case _ =>
      }

      bindings.scan match {
        case AggEnv.Drop =>
          scan = None
        case AggEnv.Promote =>
          eval = scan.get
          scan = None
        case AggEnv.Create(_) =>
          scan = Some(
            if (bindings.dropEval) eval.map(_ => ScopedEnv(SCAN))
            else eval.clone()
          )
        case _ =>
      }
    }

    // Inverse of `extend`, where `saved` was taken before the edge: reinstate
    // the compartment structure while keeping additions to compartments that
    // survive the edge.
    def restore(bindings: Bindings[_], saved: AvailableExprs): Unit = {
      val created =
        bindings.agg.isInstanceOf[AggEnv.Create] || bindings.scan.isInstanceOf[AggEnv.Create]

      bindings.scan match {
        case AggEnv.Promote =>
          scan = Some(eval)
          eval = saved.eval
        case AggEnv.Drop =>
          scan = saved.scan
        case _ if created =>
          scan = saved.scan
        case _ =>
      }

      bindings.agg match {
        case AggEnv.Promote =>
          agg = Some(eval)
          eval = saved.eval
        case AggEnv.Drop =>
          agg = saved.agg
        case _ if created =>
          agg = saved.agg
        case _ =>
      }
    }

    // Seed for a child region across a dropEval wall: the deepest frame's envs
    // become the child's root envs, with the edge's transition applied once.
    def seed(bindings: Bindings[_]): AvailableExprs = {
      val child = new AvailableExprs(
        eval = mutable.ArrayBuffer(eval.last),
        agg = agg.map(a => mutable.ArrayBuffer(a.last)),
        scan = scan.map(s => mutable.ArrayBuffer(s.last)),
      )
      child.extend(bindings)
      child
    }
  }

  // A stack of frames delimited by dropEval walls, across which no name may be
  // referenced; names bound outside the region contribute level 0 (at or above
  // the region root).
  private class Region(val avail: AvailableExprs) {
    val frames: mutable.ArrayBuffer[Frame] =
      mutable.ArrayBuffer(new Frame(0, 0, 0))

    // the frame level at which each name is bound; the levels of an
    // expression's free names cap how shallow it may be placed.
    val nameLevel: mutable.HashMap[Name, Int] =
      mutable.HashMap.empty

    // names whose bound value collapsed to another ref (see `forward`)
    private[this] val aliases =
      mutable.HashMap.empty[Name, LeafRef]

    def deepest: Frame = frames.last
    def depth: Int = frames.length - 1

    def memoize(name: Name, value: IR): (IR, Int) =
      memoizeAt(name, value, depth)

    // Bind `value` to `name` in the block at `level` and, if liftable,
    // publish it for reuse by structurally equal expressions.
    def memoizeAt(name: Name, value: IR, level: Int): (IR, Int) = {
      val atom = frames(level).builder.strictMemoize(value, name, avail.scopeAt(level))
      if (isLiftable(value)) avail.bind(level, value, atom)
      nameLevel(name) = level
      returning(atom.ir, level)
    }

    // A binding whose value collapses to a ref aliases the referent: uses are
    // forwarded to the referent and the binding is dropped, so expressions
    // built on the alias keep the referent's level and may hoist past the frame
    // the alias was declared in.
    def forward(name: Name, referent: LeafRef, level: Int): (IR, Int) = {
      aliases(name) = referent
      nameLevel(name) = level
      (referent, level)
    }

    def resolve(ref: LeafRef): (IR, Int) = {
      val ir = aliases.get(ref.name) match {
        case Some(referent) => referent.ir
        case None => ref
      }
      returning(ir, nameLevel.getOrElse(ref.name, 0))
    }

    // Rewrite `ir`'s direct LeafRef children through the alias map: scanned
    // source expressions then compare equal to lifted forms, whose refs have
    // already been resolved to their referents. One level is exactly the
    // granularity at which anticipation can match: deeper compound children
    // are renamed by the ANF step.
    def resolveShallow(ir: IR): IR =
      ir.mapChildren {
        case ref: LeafRef => aliases.getOrElse(ref.name, ref)
        case child => child
      }

    // Record that a value available at `level` flows through every frame
    // deeper than it: each such frame's block references a name bound at
    // `level`, so when the frame closes its block cannot rise above it
    // (see `inFrame`).
    def returning(ir: IR, level: Int): (IR, Int) = {
      var i = depth

      while (i > level) {
        val frame = frames(i)
        if (level > frame.escapes) frame.escapes = level
        i -= 1
      }

      (ir, level)
    }

    def lookup(ir: IR): Option[(Atom, Int)] = {
      // An agg (scan) expression is a value of the aggregable elements as well
      // as of its free names: entries bound outside the deepest context change
      // are different values even when structurally equal, so lookup must not
      // cross the wall.
      val wall = math.max(
        if (deepest.aggWall > 0 && ContainsAgg(ir)) deepest.aggWall else 0,
        if (deepest.scanWall > 0 && ContainsScan(ir)) deepest.scanWall else 0,
      )
      avail.lookup(ir, wall)
    }

    // Is `ir` certain to be evaluated later on frame `level`'s strict spine?
    // Meaningful only from the region's ordinary evaluation environment: a
    // promoted compartment evaluates per aggregated element, which the
    // scanned spine does not describe.
    def anticipatedAt(level: Int, ir: IR): Boolean =
      avail.full && frames(level).anticipated.exists(_.contains(ir))

    // Lift an agg (scan) binding's value with the matching buffer as the
    // current environment: the value is evaluated once per aggregated
    // element, so only entries visible per-element may be used or made.
    def withTransitions(s: Scope)(f: => (IR, Int)): (IR, Int) =
      if (s == EVAL) f
      else {
        assert(
          avail.scopeAt(depth) == EVAL,
          s"Found nested $s bindings in ${avail.scopeAt(depth)}.",
        )
        val saved = avail.save
        avail.promote(s)
        val result = f
        avail.restore(s, saved)
        result
      }

    // As above, but driven by the environment transition the edge into a
    // child declares; NoOp and Bind edges leave the buffers untouched.
    def withTransitions(bindings: Bindings[_])(f: => (IR, Int)): (IR, Int) = {
      assert(!bindings.dropEval)

      if (noTransition(bindings.agg) && noTransition(bindings.scan)) f
      else {
        val saved = avail.save
        avail.extend(bindings)
        val result = f
        avail.restore(bindings, saved)
        result
      }
    }

    // Visit a non-strict child in a fresh frame. Bindings placed in the
    // frame wrap the child in a Block on the way out; the returned level
    // (`escapes`) is the deepest enclosing frame the wrapped child still
    // references, ie its availability level in the parent.
    def inFrame(barrier: Boolean, wall: Option[Scope])(f: Int => IR): (IR, Int) = {
      val level = frames.length
      val frame = new Frame(
        floor = if (barrier) level else deepest.floor,
        aggWall = if (wall.contains(AGG)) level else deepest.aggWall,
        scanWall = if (wall.contains(SCAN)) level else deepest.scanWall,
      )
      frames += frame
      avail.push()
      val result = f(level)
      avail.pop()
      frames.remove(level): Unit
      val bindings = frame.builder.getBindings
      val wrapped = if (bindings.nonEmpty) Block(bindings, result) else result
      (wrapped, frame.escapes)
    }
  }

  private class Impl {
    private[this] val namespace = f"__lift_${Random.alphanumeric.take(5).mkString}_"
    private[this] var uidCounter = 0L

    // use a unique namespace for lifted names
    private def freshName: Name =
      Name {
        val n = uidCounter
        uidCounter += 1L
        f"$namespace$n"
      }

    def apply(ir: BaseIR): BaseIR =
      ir match {
        case ir: IR => liftRegion(ir, new AvailableExprs())
        case _: BlockMatrixIR => ir.mapChildren(apply)
        case _: MatrixIR => ir.mapChildren(apply)
        case _: TableIR => ir.mapChildren(apply)
      }

    private def liftRegion(ir: IR, availIn: AvailableExprs): IR = {
      val r = new Region(availIn)
      val (result, _) = lift(r, ir)
      val bindings = r.frames.head.builder.getBindings
      if (bindings.nonEmpty) Block(bindings, result) else result
    }

    // Returns the transformed expression and its availability level: the max
    // level among the names occurring free in it, ie the shallowest frame at
    // which it could be placed.
    private def lift(r: Region, ir: IR): (IR, Int) =
      ir match {
        case Block(bindings, body) =>
          // Existing bindings stay in place: each value is re-lifted, then
          // republished under its name; a value that collapses to a ref
          // turns the binding into an alias, which is dropped. Before each
          // value, the frame's spine position advances: the remaining EVAL
          // values and the body become the anticipated suffix, licensing
          // hoists out of conditionals within this value (see the placement
          // decision below). The slot is popped before the body, whose own
          // continuation is whatever the enclosing positions anticipate.
          val frame = r.deepest
          val slot = frame.anticipated.length
          frame.anticipated += Anticipated.empty

          bindings.view.zipWithIndex.foreach { case (Binding(name, value, scope), i) =>
            frame.anticipated(slot) = new Anticipated({ () =>
              val roots = ArraySeq.newBuilder[IR]
              for (b <- bindings.view.drop(i)) if (b.scope == EVAL) roots += b.value
              roots += body
              spineExprs(roots.result(), r.resolveShallow)
            })

            r.withTransitions(scope) {
              lift(r, value) match {
                case (ref: LeafRef, level) => r.forward(name, ref, level)
                case (newValue, _) => r.memoize(name, newValue)
              }
            }
          }

          frame.anticipated.remove(slot): Unit
          lift(r, body)

        case ref: LeafRef =>
          r.resolve(ref)

        case _ =>
          val (newIR, lvl) =
            ir.foldChildrenWithIndex(0) {
              case (child: IR, i, maxLvl) =>
                val bindings = Bindings.get(ir, i)

                if (bindings.dropEval) {
                  // No name may be referenced across a dropEval wall: the child
                  // is its own region.
                  (liftRegion(child, r.avail.seed(bindings)), maxLvl)
                } else if (IsStrict(ir, i) && child.typ != TVoid) {
                  // A strict child is evaluated exactly once, with its
                  // parent, so naming it in an enclosing frame preserves
                  // semantics: this is the ANF step, and memoize publishes
                  // the value for reuse.
                  val (result, childLvl) =
                    r.withTransitions(bindings) {
                      val (newChild, valueLvl) = lift(r, child)
                      if (!isLiftable(newChild)) (newChild, valueLvl)
                      else r.memoize(freshName, newChild)
                    }

                  (result, math.max(maxLvl, childLvl))
                } else {
                  // A non-strict child (branch arm, stream or loop body,
                  // lazy operand) may run zero or many times, so it gets a
                  // frame of its own where dependent bindings land:
                  //   - a barrier prevents placement above frames that may
                  //     never be evaluated (no speculation);
                  //   - a wall prevents agg- (scan-) dependent lookups from
                  //     reusing entries bound over different elements.
                  val barrier = IsStrict(ir, i) || isConditionallyEvaluated(ir, i)

                  val (newChild, escapes) =
                    r.withTransitions(bindings) {
                      r.inFrame(barrier, changesAggContext(ir, i)) { level =>
                        bindings.all.foreach { case (name, _) => r.nameLevel(name) = level }
                        lift(r, child)._1
                      }
                    }

                  (newChild, math.max(maxLvl, escapes))
                }

              case (child, _, maxLvl) =>
                (apply(child), maxLvl)
            }

          // All strict compound children are now atoms: reuse an equal
          // available expression if one is visible, else decide where this
          // one may be placed.
          r.lookup(newIR) match {
            case Some((atom, level)) =>
              r.returning(atom.ir, level)

            case None =>
              // A conditional floor yields to anticipation: every run of the
              // target frame's block evaluates the expression later anyway, so
              // hoisting moves that evaluation earlier rather than adding one
              // (partial redundancy elimination), and the later occurrence
              // collapses onto the hoisted binding by ordinary availability.
              //
              // An expression already at its placement level evaluates to
              // itself; its parent decides whether to memoize it. A speculatable
              // head may hide an agg-dependent expression in an unliftable
              // child, eg ToArray(StreamMap(_, _, count)), which pins no name,
              // so hoisting additionally requires aggregation-context
              // independence — checked once, last, since it walks the tree.
              if (!isSpeculatable(newIR)) r.returning(newIR, lvl)
              else {
                val floor = r.deepest.floor

                val place =
                  if (lvl < floor && r.anticipatedAt(lvl, newIR)) lvl
                  else math.max(lvl, floor)

                if (place < r.depth && !AggContextDependent(newIR))
                  r.memoizeAt(freshName, newIR, place)
                else r.returning(newIR, lvl)
              }
          }
      }

    // The liftable expressions certain to be evaluated by `roots`: those
    // reachable through strict, non-dropEval edges that leave the aggregation
    // environment untouched. Blocks are spine: their EVAL binding values and
    // body evaluate unconditionally, in order; AGG and SCAN values run per
    // aggregated element and are skipped. Each expression is recorded through
    // `resolve` so it compares equal to the lifted forms queried against it.
    private def spineExprs(roots: Seq[IR], resolve: IR => IR): mutable.HashSet[IR] = {
      val acc = mutable.HashSet.empty[IR]

      def go(ir: IR): Unit =
        ir match {
          case Block(bindings, body) =>
            bindings.foreach(b => if (b.scope == EVAL) go(b.value))
            go(body)

          case _ =>
            if (isLiftable(ir)) acc.add(resolve(ir)): Unit
            ir.children.view.zipWithIndex.foreach {
              case (child: IR, i) if IsStrict(ir, i) =>
                val bindings = Bindings.get(ir, i)
                if (!bindings.dropEval && noTransition(bindings.agg) && noTransition(bindings.scan))
                  go(child)

              case _ =>
            }
        }

      roots.foreach(go)
      acc
    }
  }
}

object IsStrict {
  def apply(ir: IR, i: Int): Boolean =
    !NonStrict(ir, i)
}

object NonStrict {
  def apply(ir: IR, i: Int): Boolean =
    ir match {
      case _: AggArrayPerElement => i == 1
      case _: AggExplode => i == 1
      case _: AggFilter => i == 1
      case _: AggFold => i > 0
      case _: AggGroupBy => i == 1
      case ApplySpecial("lor" | "land", _, _, _, _) => i > 0
      case _: ArrayMaximalIndependentSet => i == 1
      case _: ArraySort => i == 1
      case _: Coalesce => true
      case _: CollectDistributedArray => i == 2
      case _: ConsoleLog => i == 1
      case _: If => i > 0
      case _: MatrixAggregate => i == 1
      case _: NDArrayMap => i == 1
      case _: NDArrayMap2 => i == 2
      case _: RelationalLet => true
      case _: RunAgg => i == 1
      case _: RunAggScan => i > 1
      case _: ResultOp => i == 1
      case _: StreamAgg => i == 1
      case _: StreamAggScan => i == 1
      case _: StreamBufferedAggregate => i > 1
      case _: StreamDropWhile => i == 1
      case _: StreamFlatMap => i == 1
      case _: StreamFilter => i == 1
      case _: StreamFold => i == 2
      case StreamFold2(_, accum, _, _, _) => i > accum.length
      case _: StreamFor => i == 1
      case _: StreamJoinRightDistinct => i == 2
      case _: StreamLeftIntervalJoin => i == 2
      case _: StreamMap => i == 1
      case _: StreamScan => i == 2
      case _: StreamTakeWhile => i == 1
      case StreamZip(as, _, _, _, _) => i == as.length
      case StreamZipJoin(as, _, _, _, _) => i == as.length
      case _: StreamZipJoinProducers => i > 0
      case _: Switch => i > 0
      case _: TableAggregate => i == 1
      case TailLoop(_, args, _, _) => i == args.length
      case _: WriteMetadata => true
      case _ => false
    }
}
