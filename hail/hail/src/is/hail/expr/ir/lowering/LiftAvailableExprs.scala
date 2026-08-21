package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{RefEquality => _, _}
import is.hail.expr.ir.AggEnv.Promote
import is.hail.expr.ir.Scope._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.invariant.{Invariant, LowerableIR, UniquelyNamed}
import is.hail.types.virtual.{TStream, TVoid}
import is.hail.utils.StackSafe._
import is.hail.utils.TimedBlock

import scala.annotation.tailrec
import scala.collection.immutable.IntMap
import scala.collection.mutable

import java.util

// Rewrites an IR into weak A-normal form: each liftable child of a strict edge
// is bound to a fresh name in an enclosing Block (atoms stay inline; non-strict
// children keep their shape). On top of the naming:
//   - Common-Subexpression Elimination (CSE): an expression whose value number
//     matches an in-scope binding collapses to a reference to it;
//   - Loop-Invariant Code Motion (LICM): a total expression lands at the
//     shallowest frame at which its free names all resolve;
//   - Partial Redundancy Elimination (PRE): an expression pinned under a
//     conditional still rises to a frame certain to evaluate it later, moving
//     an evaluation earlier rather than adding one.
// Two phases per region (GVN-PRE, after VanDrunen & Hosking 2004):
//   1. Analyze: one post-order traversal hash-conses every node to a value
//      number (Vn) — aliases and lifted forms of one value share a number by
//      construction — and records each node's anticipated set: the Vns certain
//      to be evaluated whenever the node evaluates.
//   2. Rebuild: one recursive rewrite tracking a Region (frame stack, name
//      levels, aliases) whose Frames (one per non-strict edge) hold the landing
//      spots and the Vn-keyed AvailableExprs eligible for reuse; anticipation
//      probes are Vn-set membership, so motion cannot hide a match behind
//      renaming.
object LiftAvailableExprs {

  def apply(ctx: ExecuteContext, in: BaseIR): BaseIR =
    TimedBlock.enter {
      UniquelyNamed.verify(ctx, in)
      val out = new Impl(new Memo(ctx.irMetadata.nextFlags(16)))(in)
      WeakANF.verify(ctx, out)
      out
    }

  // CSE candidate operations, provided
  //   - the operation does not contain side effects,
  //   - reuse doesn't change aggregation semantics.
  private def isLiftable(ir: IR): Boolean =
    !(ir.isInstanceOf[Atom] || ir.typ.isInstanceOf[TStream])

  // Operations that cannot fail: evaluating one on a path that never needs its
  // value is unobservable, so it may cross an edge the source evaluated zero
  // times (a possibly-empty loop body) — speculation.
  private def isTotal(ir: IR): Boolean =
    ir match {
      case _: Literal | _: EncodedLiteral | _: Str // non-Atom constants
          | _: MakeArray | _: ArrayLen
          | _: MakeStruct | _: GetField | _: SelectFields | _: InsertFields
          | _: MakeTuple | _: GetTupleElement
          | _: IsNA
          | _: Cast | _: CastToArray | _: CastRename
          | _: ApplyUnaryPrimOp | _: ApplyComparisonOp => true
      case ApplyBinaryPrimOp(Add() | Subtract() | Multiply(), _, _) => true
      case _ => false
    }

  // Effectful operations may not be speculated along a conditional edge.
  private def isEffect(ir: IR): Boolean =
    ir match {
      case _: UUID4 | Apply("index_bgen", _, _, _, _) => true
      case _ => !IsPure(ir)
    }

  // Children the parent may evaluate zero times. Such frames set the floor:
  // placing a binding above one would evaluate it on paths that never need it.
  private def isConditionallyEvaluated(parent: IR, childIdx: Int): Boolean =
    parent match {
      case _: If => childIdx > 0
      case _: Switch => childIdx > 0
      case _: Coalesce => true
      case ApplySpecial("lor" | "land", _, _, _, _) => childIdx > 0
      case _ => false
    }

  // Edges that need a lookup wall: the aggregable elements change while agg-
  // (scan-) dependent expressions stay legal, so an equal expression bound
  // outside is in scope yet aggregates different elements. Edges that remove or
  // replace the scope (AggFold, Table/MatrixAggregate) need none: stale entries
  // leave scope with the buffers holding them.
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

  private def noTransitions(bindings: Bindings[_]): Boolean = {
    def noTransition(env: AggEnv): Boolean =
      env match {
        case AggEnv.NoOp | AggEnv.Bind(_) => true
        case _ => false
      }

    !bindings.dropEval && noTransition(bindings.agg) && noTransition(bindings.scan)
  }

  // A Value number
  private[LiftAvailableExprs] type Vn <: Int

  // A landing spot for bindings: one frame per non-strict edge on the path from
  // the region root. The floors bound placement (no speculation); the walls
  // bound lookup (see Region.lookup).
  final private class Frame(
    // the deepest conditionally-evaluated frame at or above this one
    val floor: Int,

    // the deepest dropEval frame at or above this one; a hard floor that not
    // even anticipation may cross
    val regionFloor: Int,

    // the deepest agg context change at or above this one
    val aggWall: Int,

    // the deepest scan context change at or above this one
    val scanWall: Int,

    // The Vns of this frame's site: those certain to be evaluated on its strict
    // spine whenever the frame runs. A probe hit licenses a down-safe hoist to
    // this frame; one already evaluated on the spine is found available instead
    // (lookup precedes placement), so a hit never duplicates an evaluation.
    val anticipated: VnSet,
  ) {
    val builder: IRBuilder = new IRBuilder

    // the max level (below this frame's own) of any name occurring free within
    // this frame's block.
    var escapes: Int = 0
  }

  // A frame's available expressions, keyed by value number and tagged with the
  // scope (EVAL, AGG or SCAN) a binding placed at that frame must be declared
  // in: placement happens at a distance, and the scope depends on where the
  // landing frame sits relative to scope transitions.
  private def ScopedEnv(s: Scope): ScopedEnv =
    ScopedEnv(s, IntMap.empty)

  final private case class ScopedEnv(scope: Scope, env: IntMap[Atom]) {
    def bind(vn: Vn, atom: Atom): ScopedEnv =
      copy(env = env.updated(vn, atom))
  }

  // The expressions available at every frame of a region, one buffer per
  // binding environment (eval — always the lookup/insert target — agg, and
  // scan), each holding one env per frame. Transitions apply to all frames in
  // lockstep, so an edge's transition swaps whole buffers.
  private def AvailableExprs: AvailableExprs =
    new AvailableExprs(
      mutable.ArrayBuffer(ScopedEnv(EVAL)),
      Some(mutable.ArrayBuffer(ScopedEnv(AGG))),
      Some(mutable.ArrayBuffer(ScopedEnv(SCAN))),
    )

  final private class AvailableExprs(
    private var eval: mutable.ArrayBuffer[ScopedEnv],
    private var agg: Option[mutable.ArrayBuffer[ScopedEnv]],
    private var scan: Option[mutable.ArrayBuffer[ScopedEnv]],
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

    // Both aggregation compartments intact: the eval buffer is the region's
    // ordinary evaluation environment, not a promoted one.
    def full: Boolean = agg.isDefined && scan.isDefined

    def bind(level: Int, vn: Vn, atom: Atom): Unit =
      eval(level) = eval(level).bind(vn, atom)

    // Search from the deepest frame outward; levels shallower than `wall` are
    // invisible (see Region.lookup).
    def lookup(vn: Vn, wall: Int): Option[(Atom, Int)] = {
      @tailrec def go(level: Int): Option[(Atom, Int)] =
        if (level < wall) None
        else eval(level).env.get(vn) match {
          case Some(atom) => Some(atom -> level)
          case None => go(level - 1)
        }

      go(eval.length - 1)
    }

    def save: AvailableExprs =
      new AvailableExprs(eval, agg, scan)

    // Apply an edge's transition to every frame at once.
    def extend(bindings: Bindings[_]): Unit = {
      if (bindings.dropEval)
        eval = eval.map(_ => ScopedEnv(EVAL))

      bindings.agg match {
        case AggEnv.Drop =>
          agg = None
        case AggEnv.Promote =>
          eval = agg.get
          agg = None
        case AggEnv.Create(_) =>
          agg = Some(eval.clone())
        case _ =>
      }

      bindings.scan match {
        case AggEnv.Drop =>
          scan = None
        case AggEnv.Promote =>
          eval = scan.get
          scan = None
        case AggEnv.Create(_) =>
          scan = Some(eval.clone())
        case _ =>
      }
    }

    // Inverse of `extend`, where `saved` was taken before the edge: reinstate
    // the compartment structure while keeping additions to compartments that
    // survive the edge.
    def restore(bindings: Bindings[_], saved: AvailableExprs): Unit = {
      bindings.scan match {
        case AggEnv.Promote =>
          scan = Some(eval)
          eval = saved.eval
        case AggEnv.Drop | AggEnv.Create(_) =>
          scan = saved.scan
        case _ =>
      }

      bindings.agg match {
        case AggEnv.Promote =>
          agg = Some(eval)
          eval = saved.eval
        case AggEnv.Drop | AggEnv.Create(_) =>
          agg = saved.agg
        case _ =>
      }

      // `extend` replaced the eval buffer on dropEval; the saved one, with its
      // entries, was untouched. (After a Promote this is a no-op.)
      if (bindings.dropEval)
        eval = saved.eval
    }
  }

  // The traversal state of one region: its analysis (Vns and anticipated sets),
  // its stack of frames, the levels its names are bound at, and its aliases.
  // Names bound outside contribute level 0 (at or above the root frame).
  final private class Region(
    private[this] val analysis: Analysis,
    private[this] val avail: AvailableExprs,
    rootAnticipated: VnSet,
  ) {
    val frames: mutable.ArrayBuffer[Frame] =
      mutable.ArrayBuffer(new Frame(0, 0, 0, 0, rootAnticipated))

    // each in-scope name's frame level and, when its value collapsed to an
    // atom, the referent uses are forwarded to. The levels of an expression's
    // free names cap how shallow it may be placed; names bound outside the
    // region root are absent and resolve to level 0.
    private[this] val bound = new util.HashMap[Name, (Int, Option[Atom])]

    def vn(ir: IR): Vn = analysis.vn(ir)
    def ant(ir: IR): VnSet = analysis.ant(ir)

    def deepest: Frame = frames.last
    def depth: Int = frames.length - 1

    // Record `name` as bound at `level`. An `alias` (a value that collapsed to
    // an atom) forwards uses to the referent and drops the binding: dependents
    // keep the referent's level and may hoist past this frame.
    def declare(name: Name, level: Int, alias: Option[Atom] = None): Unit =
      bound.put(name, (level, alias))

    // Bind `value` to `name` in the block at `level`. Publishing makes the
    // value available to equal-valued expressions under its Vn, and the name
    // carries the number, keeping rebuilt forms numbered like their sources.
    def memoize(name: Name, value: IR, level: Int, publish: Boolean): (IR, Int) = {
      val atom = frames(level).builder.strictMemoize(value, name, avail.scopeAt(level))
      if (publish) avail.bind(level, analysis.publish(name, value), atom)
      declare(name, level)
      returning(atom.ir, level)
    }

    def resolve(ref: LeafRef): (IR, Int) = {
      val (level, alias) = bound.getOrDefault(ref.name, (0, None))
      returning(alias.fold(ref: IR)(_.ir), level)
    }

    // Record that a value available at `level` flows through every deeper
    // frame: their blocks cannot rise above it when they close (inFrame).
    def returning(ir: IR, level: Int): (IR, Int) = {
      var i = depth

      while (i > level) {
        val frame = frames(i)
        if (level > frame.escapes) frame.escapes = level
        i -= 1
      }

      (ir, level)
    }

    def lookup(vn: Vn, agg: Boolean, scan: Boolean): Option[(Atom, Int)] = {
      // An agg (scan) expression is a value of the aggregable elements too: an
      // equal entry bound outside the deepest context change is a different
      // value, so lookup must not cross the wall.
      val wall = math.max(
        if (agg) deepest.aggWall else 0,
        if (scan) deepest.scanWall else 0,
      )
      avail.lookup(vn, wall)
    }

    // Is the value numbered `Vn` certain to be evaluated on frame `level`'s
    // strict spine? Only meaningful from the ordinary evaluation environment
    // (avail.full), and never across a dropEval wall: a binding placed outside
    // one cannot be referenced within.
    def anticipatedAt(level: Int, vn: Vn): Boolean =
      level >= deepest.regionFloor &&
        avail.full &&
        frames(level).anticipated.contains(vn)

    // Run `f` with an edge's environment transition applied to the available
    // expressions, restoring the compartment structure afterwards. A dropEval
    // edge clears eval availability for the duration of `f`: nothing bound
    // outside the edge may be found within.
    def withTransitions(bindings: Bindings[_])(f: => (IR, Int)): (IR, Int) =
      if (noTransitions(bindings)) f
      else {
        val saved = avail.save
        avail.extend(bindings)
        val result = f
        avail.restore(bindings, saved)
        result
      }

    // Visit a non-strict child in a fresh frame: bindings placed there wrap it
    // in a Block on the way out, and the returned `escapes` is its availability
    // level in the parent.
    def inFrame(
      barrier: Boolean,
      region: Boolean,
      wall: Option[Scope],
      anticipated: VnSet,
    )(
      f: Int => IR
    ): (IR, Int) = {
      val level = frames.length
      val frame = new Frame(
        floor = if (barrier || region) level else deepest.floor,
        regionFloor = if (region) level else deepest.regionFloor,
        aggWall = if (wall.contains(AGG)) level else deepest.aggWall,
        scanWall = if (wall.contains(SCAN)) level else deepest.scanWall,
        anticipated = anticipated,
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

  private object Memo {
    final private val I = 1 // Contains agg intermediate
    final private val A = 2 // Contains agg
    final private val S = 4 // Contains scan
    final private val E = 8 // Contains an effect or nondeterminism

    class Result(private[Memo] val bits: Int) extends AnyVal {
      def isClean: Boolean = bits == 0
      def containsAggIntermediate: Boolean = (bits & Memo.I) != 0
      def containsAgg: Boolean = (bits & Memo.A) != 0
      def containsScan: Boolean = (bits & Memo.S) != 0
      def containsEffect: Boolean = (bits & Memo.E) != 0
    }
  }

  // This pass's memo of four per-node facts, encoded into the shared
  // BaseIR.mark value as `base + bits`, where [base, base + 16) is reserved for
  // this run alone.
  final private class Memo(val base: Int) extends AnyVal {
    import Memo._

    def apply(ir: BaseIR): Result =
      new Result({
        val bits = ir.mark - base
        if (0 <= bits && bits < 16) bits else 0
      })

    // Derive and record `ir`'s bits. The only write path: bits cannot be
    // computed without being recorded, nor recorded unless derived.
    def stamp(ir: BaseIR): Result = {
      val r = derive(ir)
      update(ir, r)
      r
    }

    private def update(ir: BaseIR, r: Result): Unit =
      ir.mark = base + r.bits

    // The bits of a rebuilt node, from its head and its direct children's marks
    // — the mark-based mirror of ContainsAggIntermediate, ContainsAgg and
    // ContainsScan + deep effect scan.
    private def derive(ir: BaseIR): Result =
      ir match {
        case _: Atom | Constant(_) =>
          new Result(0)

        case Block(bindings, body) =>
          var bits = 0
          bindings.foreach { case Binding(_, value, scope) =>
            val x = apply(value).bits
            bits |= x & (I | E)
            bits |= (scope match {
              case EVAL => x & (A | S)
              case AGG => A
              case SCAN => S
            })
          }
          new Result(bits | apply(body).bits)

        case _: TableAggregate | _: MatrixAggregate | _: TableIR | _: MatrixIR | _: BlockMatrixIR =>
          var bits = 0
          ir.children.foreach(child => bits |= apply(child).bits)
          new Result(bits & E)

        case ir: IR =>
          var bits = 0
          if (IsAggIntermediate(ir)) bits |= I
          if (IsAggResult(ir)) bits |= A
          if (IsScanResult(ir)) bits |= S
          if (isEffect(ir)) bits |= E

          var fromChildren = 0
          ir.children.foreach(child => fromChildren |= apply(child).bits)

          // A write inside an agg body still writes: the cuts do not apply to
          // E.
          val cut = ir match {
            case _: StreamAgg => A
            case _: StreamAggScan => S
            case _ => 0
          }
          new Result(bits | (fromChildren & ~cut))
      }
  }

  // IntMap (a patricia trie branching on key bits) beats HashSet[Int] here:
  // probes hash nothing and box nothing, and union/intersection walk only where
  // the trees overlap — the anticipated-set algebra unions small sets into
  // large ones, where HashSet's concat paid for the large side.
  private object VnSet {
    import scala.collection.immutable.{IntMap => _Impl}
    private[VnSet] type Impl = _Impl[Unit]

    val empty: VnSet = new VnSet(_Impl.empty)
    def apply(vn: Vn): VnSet = new VnSet(_Impl((vn, ())))
  }

  final private class VnSet(private val vs: VnSet.Impl) extends AnyVal {
    def contains(vn: Vn): Boolean = vs.contains(vn)

    def filter(pred: Vn => Boolean): VnSet =
      new VnSet(vs.filter { case (vn, _) => pred(vn.asInstanceOf[Vn]) })

    def |(that: VnSet): VnSet = new VnSet(vs.unionWith(that.vs, (_, _, _) => ()))
    def &(that: VnSet): VnSet = new VnSet(vs.intersection(that.vs))
  }

  // ---- Phase 1: analyze ------------------------------------------------

  // One post-order traversal per region root stamps marks, assigns Vns and
  // records anticipated sets, which the rebuild reads through Region.
  final private class Analysis(
    val memo: Memo,
    private[this] val vnNames: mutable.ArrayBuffer[Name],
  ) {

    private def vnName(vn: Vn): Name = {
      while (vnNames.size <= vn) vnNames += Name(s"__Vn_${vnNames.size}")
      vnNames(vn)
    }

    // canonical form -> Vn. A canonical form is the node with each IR child
    // replaced by a Ref naming the child's Vn, so structurally distinct forms
    // of one value share a number. A java HashMap caches each key's hash, so
    // growing the table never recomputes the keys' structural hashCodes.
    private[this] val vnTable = new util.HashMap[IR, Vn]

    // per-occurrence Vn memo for compound nodes, over source and rebuilt nodes
    // alike; leaf refs resolve through nameVn instead
    private[this] val vns = new util.IdentityHashMap[IR, Vn]

    // each name's Vn: a name bound to a liftable value free of effects and agg
    // intermediates shares the value's number, so aliases and lifted nests
    // vanish by construction; every other name (params, free names, effectful
    // or agg-intermediate bindings) is opaque
    private[this] val nameVn = new util.HashMap[Name, Vn]

    // each source node's anticipated set: the Vns certain to be evaluated
    // whenever the node is
    private[this] val ants = new util.IdentityHashMap[IR, VnSet]

    // the table Vns with total heads, for busy-arm filtering. Totality is a
    // per-Vn constant — a canonical form keeps its node's head, so every node
    // sharing a Vn shares the head; opaque Vns (names, effects) are never
    // members.
    private[this] val totals = mutable.BitSet.empty

    private[this] var vnCount = 0

    private def freshVn(): Vn = {
      val vn = vnCount
      vnCount += 1
      vn.asInstanceOf[Vn]
    }

    // wipe for reuse by the next region; the tables keep their capacity
    def reset(): Unit = {
      vnTable.clear()
      vns.clear()
      nameVn.clear()
      ants.clear()
      totals.clear()
      vnCount = 0
    }

    def vn(ir: IR): Vn =
      ir match {
        case ref: LeafRef =>
          nameVn.computeIfAbsent(ref.name, _ => freshVn())

        case _ =>
          vns.computeIfAbsent(ir, computeVn)
      }

    private def computeVn(ir: IR): Vn =
      ir match {
        // a block evaluates to its body: bindings free of effects and agg
        // intermediates are transparent, and their names carry the values' Vns
        case Block(bindings, body) =>
          val transparent =
            bindings.forall { b =>
              val bits = memo(b.value)
              !bits.containsAggIntermediate && !bits.containsEffect
            }

          if (transparent) vn(body) else freshVn()

        // equal effects and reads of mutable aggregator state never share a
        // number: one per occurrence
        case _ if isEffect(ir) || IsAggIntermediate(ir) =>
          freshVn()

        case _ =>
          val key =
            ir.mapChildren {
              case child: IR => Ref(vnName(vn(child)), child.typ)
              case child => child // relational children compare verbatim
            }

          vnTable.computeIfAbsent(
            key,
            _ => {
              val vn = freshVn()
              if (isTotal(ir)) totals += vn
              vn
            },
          )
      }

    def ant(ir: IR): VnSet =
      ants.get(ir)

    // Publish a rebuilt binding: the name carries the value's number, keeping
    // rebuilt forms numbered like their sources.
    def publish(name: Name, value: IR): Vn = {
      val v = vn(value)
      nameVn.put(name, v)
      v
    }

    // One post-order traversal per region: stamp marks, assign Vns and record
    // anticipated sets. Relational children are separate regions, analyzed when
    // the rebuild reaches them.
    def analyze(ir: IR): Unit = {
      ir match {
        case Block(bindings, body) =>
          bindings.foreach { case Binding(name, value, _) =>
            analyze(value)
            val bits = memo(value)
            val transparent =
              value.isInstanceOf[Atom] ||
                (isLiftable(value) && !bits.containsAggIntermediate && !bits.containsEffect)

            nameVn.put(name, if (transparent) vn(value) else freshVn())
          }

          analyze(body)

        case _ =>
          ir.children.foreach {
            case child: IR => analyze(child)
            case _ =>
          }
      }

      val bits = memo.stamp(ir)

      // the node's own Vn, plus those of its strict spine: the edges walked
      // mirror the rebuild's naming step (strict, non-dropEval,
      // transition-free; Blocks' EVAL values and body included)
      val own =
        if (isLiftable(ir) && bits.isClean) VnSet(vn(ir))
        else VnSet.empty

      val spine =
        ir match {
          case Block(bindings, body) =>
            bindings.foldLeft(ant(body)) { (acc, b) =>
              if (b.scope == EVAL) acc | ant(b.value) else acc
            }

          case _ =>
            ir.children.view.zipWithIndex.foldLeft(VnSet.empty) {
              case (acc, (child: IR, i)) if IsStrict(ir, i) && noTransitions(Bindings.get(ir, i)) =>
                acc | ant(child)
              case (acc, _) => acc
            }
        }

      // The busy arm: a value on the spine of EVERY branch is evaluated
      // whenever the conditional is — provided its head is total: a missing
      // predicate skips all branches, and a total value hoisted onto those rows
      // is unobservable. Coalesce/lor/land get no busy arm: their arms are not
      // exhaustive alternatives.
      val busy =
        ir match {
          case If(_, cnsq, altr) =>
            ant(cnsq) & ant(altr)
          case Switch(_, default, cases) =>
            cases.foldLeft(ant(default))((acc, c) => acc & ant(c))
          case _ =>
            VnSet.empty
        }

      ants.put(ir, own | spine | busy.filter(totals))
    }
  }

  // ---- Phase 2: rebuild --------------------------------------------------

  final private class Impl(memo: Memo) {

    // The canonical Name for each Vn, shared by every region's Analysis.
    // Surprisingly, caching these is worth ~10% on canonical-key-heavy shapes
    // (many compound nodes, so many vnTable probes): a shared Name caches its
    // string's hash and compares by reference, where a fresh one re-hashes its
    // characters on every probe.
    private[this] val vnNames = mutable.ArrayBuffer.empty[Name]

    // Finished Analyses park here for the next region, keeping their tables'
    // capacity (growth-rehashing of vnTable dominated this pass's profile). A
    // pool rather than one shared instance because relational children
    // re-enter liftRegion mid-rebuild and need tables of their own.
    private[this] val pool = mutable.ArrayBuffer.empty[Analysis]

    def apply(ir0: BaseIR): BaseIR =
      recur(ir0).run()

    // Relational chains grow with the number of operations in a user's
    // pipeline, so the walk between regions is trampolined. Within a value
    // region `lift` recurses natively (depth is bounded by the region's own
    // expression); a relational child there re-enters `apply` with a nested
    // trampoline.
    private def recur(ir0: BaseIR): StackFrame[BaseIR] =
      ir0 match {
        case ir: IR => done(stamped(liftRegion(ir)))
        case _ => ir0.mapChildrenStackSafe(recur).map(stamped)
      }

    // liftRegion's result may be a fresh Block wrapping the region's bindings;
    // like every rebuilt node, it needs marks
    private def stamped(ir: BaseIR): BaseIR = {
      memo.stamp(ir): Unit
      ir
    }

    private def liftRegion(ir: IR): IR = {
      val analysis =
        if (pool.isEmpty) new Analysis(memo, vnNames)
        else pool.remove(pool.length - 1)

      analysis.analyze(ir)
      val r = new Region(analysis, AvailableExprs, analysis.ant(ir))
      val (result, _) = lift(r, ir)
      val bindings = r.frames.head.builder.getBindings
      val out = if (bindings.nonEmpty) Block(bindings, result) else result
      analysis.reset()
      pool += analysis
      out
    }

    // the environment transition under which a Block binding's value is visited
    private[this] val evalValue = Bindings.empty
    private[this] val aggValue = Bindings.empty.copy(agg = Promote)
    private[this] val scanValue = Bindings.empty.copy(scan = Promote)

    // Returns the transformed expression and its availability level: the max
    // level among the names occurring free in it, ie the shallowest frame at
    // which it could be placed.
    private def lift(r: Region, ir: IR): (IR, Int) =
      ir match {
        case Block(bindings, body) =>
          // Each value is re-lifted and republished under its name (or turned
          // into an alias if it collapsed to an atom). Hoists out of
          // conditionals within a value are licensed by the frame's anticipated
          // set: the value's own spine is in it, so a later occurrence anywhere
          // on the frame's spine counts.
          bindings.foreach { case Binding(name, value, scope) =>
            val transition =
              scope match {
                case EVAL => evalValue
                case AGG => aggValue
                case SCAN => scanValue
              }

            r.withTransitions(transition) {
              lift(r, value) match {
                case (atom: Atom, level) =>
                  r.declare(name, level, alias = Some(atom))
                  (atom, level)
                case (newValue, _) =>
                  val bits = memo(newValue)
                  val isAvailable =
                    isLiftable(newValue) && !bits.containsAggIntermediate && !bits.containsEffect
                  r.memoize(name, newValue, r.depth, publish = isAvailable)
              }
            }
          }

          lift(r, body)

        case ref: LeafRef =>
          r.resolve(ref)

        case a: Atom =>
          r.returning(a, 0)

        case _ =>
          val (newIR, lvl) =
            ir.foldChildrenWithIndex(0) {
              case (child: IR, i, maxLvl) =>
                val bindings = Bindings.get(ir, i)

                if (IsStrict(ir, i) && child.typ != TVoid && !bindings.dropEval) {
                  // A strict child is evaluated exactly once, with its parent,
                  // so naming it in an enclosing frame preserves semantics:
                  // this is the ANF step, and memoize publishes the value for
                  // reuse — unless an effect or UUID4 hides within, in which
                  // case the naming stands but no equal expression may collapse
                  // onto it.
                  val (result, childLvl) =
                    r.withTransitions(bindings) {
                      val (newChild, valueLvl) = lift(r, child)
                      val bits = memo(newChild)
                      if (isLiftable(newChild) && !bits.containsAggIntermediate)
                        r.memoize(freshName(), newChild, r.depth, publish = !bits.containsEffect)
                      else (newChild, valueLvl)
                    }

                  (result, math.max(maxLvl, childLvl))
                } else {
                  // A non-strict child may run zero or many times, and a
                  // dropEval child shares no names with its parent; either way
                  // it gets a frame of its own where dependent bindings land,
                  // guarded by the floors and walls described on Frame.

                  val (newChild, escapes) =
                    r.withTransitions(bindings) {
                      r.inFrame(
                        IsStrict(ir, i) || isConditionallyEvaluated(ir, i),
                        bindings.dropEval,
                        changesAggContext(ir, i),
                        r.ant(child),
                      ) {
                        level =>
                          bindings.all.foreach { case (name, _) => r.declare(name, level) }
                          lift(r, child)._1
                      }
                    }

                  // the wrapping Block `inFrame` may have built is not itself
                  // visited: stamp it here from its rebuilt parts
                  memo.stamp(newChild): Unit
                  (newChild, math.max(maxLvl, escapes))
                }

              case (child, _, maxLvl) =>
                (apply(child), maxLvl)
            }

          // All strict compound children are now atoms:
          //   - reuse an available expression with the same value number, else
          //   - decide where this one may be placed.
          val bits = memo.stamp(newIR)

          r.lookup(r.vn(newIR), agg = bits.containsAgg, scan = bits.containsScan) match {
            case Some((atom, level)) =>
              r.returning(atom.ir, level)

            case None =>
              // Clean bits are the movability requirement for a down-safe
              // hoist: deeply pure, deterministic, and free of agg dependence.
              // Speculation past the floor additionally demands totality.
              if (!bits.isClean) r.returning(newIR, lvl)
              else {
                val floor = r.deepest.floor

                val place =
                  if (lvl < floor && r.anticipatedAt(lvl, r.vn(newIR))) lvl
                  else if (isTotal(newIR)) math.max(lvl, floor)
                  else r.depth

                if (place < r.depth)
                  r.memoize(freshName(), newIR, place, publish = true)
                else r.returning(newIR, lvl)
              }
          }
      }
  }

  private lazy val WeakANF: Invariant =
    LowerableIR and Invariant {
      case Block(bindings, body) =>
        // A block asserts only its own shape: no value is an alias or a block,
        // and the body is not a block. Values and body are checked as nodes
        // themselves when visited.
        bindings.forall(b => !b.value.isInstanceOf[Atom] && !b.value.isInstanceOf[Block]) &&
        !body.isInstanceOf[Block]

      case ir: IR =>
        // Every liftable strict child has been named; what remains is an atom
        // or unliftable. dropEval children are lifted in a frame of their own
        // and never named into the parent. The subtree walk here is the spec
        // Impl's mark-based detection must agree with.
        ir.children.view.zipWithIndex.forall {
          case (c: IR, i) =>
            NonStrict(ir, i) ||
            Bindings.get(ir, i).dropEval ||
            !isLiftable(c) ||
            isEffect(c) ||
            ContainsAggIntermediate(c)
          case _ => true
        }

      case _ => true // relational nodes: their IR children are checked when visited
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
