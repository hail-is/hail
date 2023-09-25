package is.hail.expr.ir

import is.hail.annotations.{ExtendedOrdering, IntervalEndpointOrdering}
import is.hail.backend.ExecuteContext
import is.hail.rvd.PartitionBoundOrdering
import is.hail.types.virtual._
import is.hail.utils.{Interval, IntervalEndpoint, _}
import is.hail.variant.{Locus, ReferenceGenome}

import scala.Option.option2Iterable
import org.apache.spark.sql.Row

import scala.collection.{GenTraversableOnce, mutable}

trait JoinLattice {
  type Value <: AnyRef
  def top: Value
  def combine(l: Value, r: Value): Value
}

trait AbstractLattice extends JoinLattice {
  def bottom: Value
  def meet(l: Value, r: Value): Value
}

object ExtractIntervalFilters {

  val MAX_LITERAL_SIZE = 4096

  def apply(ctx: ExecuteContext, ir0: BaseIR): BaseIR = {
    MapIR.mapBaseIR(
      ir0,
      (ir: BaseIR) => {
        (
          ir match {
            case TableFilter(child, pred) =>
              extractPartitionFilters(ctx, pred, Ref("row", child.typ.rowType), child.typ.key)
                .map { case (newCond, intervals) =>
                  log.info(
                    s"generated TableFilterIntervals node with ${intervals.length} intervals:\n  " +
                      s"Intervals: ${intervals.mkString(", ")}\n  " +
                      s"Predicate: ${Pretty(ctx, pred)}\n " + s"Post: ${Pretty(ctx, newCond)}")
                  TableFilter(TableFilterIntervals(child, intervals, keep = true), newCond)
                }
            case MatrixFilterRows(child, pred) => extractPartitionFilters(
                ctx,
                pred,
                Ref("va", child.typ.rowType),
                child.typ.rowKey).map { case (newCond, intervals) =>
                log.info(
                  s"generated MatrixFilterIntervals node with ${intervals.length} intervals:\n  " +
                    s"Intervals: ${intervals.mkString(", ")}\n  " +
                    s"Predicate: ${Pretty(ctx, pred)}\n " + s"Post: ${Pretty(ctx, newCond)}")
                MatrixFilterRows(MatrixFilterIntervals(child, intervals, keep = true), newCond)
              }

            case _ => None
          }
        ).getOrElse(ir)
      }
    )
  }

  def extractPartitionFilters(ctx: ExecuteContext, cond: IR, ref: Ref, key: IndexedSeq[String]): Option[(IR, IndexedSeq[Interval])] = {
    if (key.isEmpty) None
    else {
      val extract = new ExtractIntervalFilters(ctx, ref.typ.asInstanceOf[TStruct].typeAfterSelectNames(key))
      val trueSet = extract.analyze(cond, ref.name)
      if (trueSet == extract.KeySetLattice.top)
        None
      else {
        val rw = extract.Rewrites(mutable.Set.empty, mutable.Set.empty)
        extract.analyze(cond, ref.name, Some(rw), trueSet)
        Some((extract.rewrite(cond, rw), trueSet))
      }
    }
  }

  def liftPosIntervalsToLocus(pos: IndexedSeq[Interval], rg: ReferenceGenome, ctx: ExecuteContext): IndexedSeq[Interval] = {
    val ord = PartitionBoundOrdering(ctx, TTuple(TInt32))
    val nonNull = rg.contigs.indices.flatMap { cont =>
      pos.flatMap { i =>
        i.intersect(ord, Interval(Row(1), Row(rg.contigLength(cont)), true, false))
          .map { interval =>
            Interval(
              Row(Locus(rg.contigs(cont), interval.start.asInstanceOf[Row].getAs[Int](0))),
              Row(Locus(rg.contigs(cont), interval.right.point.asInstanceOf[Row].getAs[Int](0))),
              interval.includesStart,
              interval.includesEnd)
          }
      }
    }
    if (pos.nonEmpty && pos.last.contains(ord, Row(null)))
      nonNull :+ Interval(Row(null), Row(), true, true)
    else
      nonNull
  }
}


// A set of key values, represented by an ordered sequence of disjoint intervals
// Supports lattice ops, plus complements.
class KeySetLattice(ctx: ExecuteContext, keyType: TStruct) extends AbstractLattice {
  type Value = IndexedSeq[Interval]
  type KeySet = Value

  object KeySet {
    def apply(intervals: Interval*): IndexedSeq[Interval] = {
      apply(intervals.toFastIndexedSeq)
    }

    def apply(intervals: IndexedSeq[Interval]): IndexedSeq[Interval] = {
      assert(intervals.isEmpty || KeySet.intervalIsReduced(intervals.last))
      intervals
    }

    val empty: IndexedSeq[Interval] = FastIndexedSeq()

    def reduce(intervals: IndexedSeq[Interval]): IndexedSeq[Interval] = intervals match {
      case Seq() => empty
      case init :+ last =>
        val reducedLast = if (intervalIsReduced(last))
          last
        else
          Interval(last.left, IntervalEndpoint(Row(), 1))
        (init :+ reducedLast).toFastIndexedSeq
    }

    private def intervalIsReduced(interval: Interval): Boolean = {
      interval.right != IntervalEndpoint(Row(null), 1)
    }
  }

  def keyOrd: ExtendedOrdering = PartitionBoundOrdering(ctx, keyType)
  val iord: IntervalEndpointOrdering = keyOrd.intervalEndpointOrdering

  def specializes(l: Value, r: Value): Boolean = {
    if (l == bottom) true
    else if (r == bottom) true
    else l.forall { i =>
      r.containsOrdered[Interval, Interval](i, (h: Interval, n: Interval) => iord.lt(h.right, n.right), (n: Interval, h: Interval) => iord.lt(n.left, h.left))
    }
  }

  def top: Value = KeySet(Interval(Row(), Row(), true, true))

  def bottom: Value = KeySet.empty

  def combine(l: Value, r: Value): Value = {
    if (l == bottom) r
    else if (r == bottom) l
    else KeySet(Interval.union(l ++ r, iord))
  }

  def combineMulti(vs: Value*): Value = {
    KeySet(Interval.union(vs.flatten.toFastIndexedSeq, iord))
  }

  def meet(l: Value, r: Value): Value = {
    if (l == top) r
    else if (r == top) l
    else KeySet(Interval.intersection(l, r, iord))
  }

  def complement(v: Value): Value = {
    if (v.isEmpty) return top

    val builder = mutable.ArrayBuilder.make[Interval]()
    var i = 0
    if (v.head.left != IntervalEndpoint(Row(), -1)) {
      builder += Interval(IntervalEndpoint(Row(), -1), v.head.left)
    }
    while (i + 1 < v.length) {
      builder += Interval(v(i).right, v(i + 1).left)
      i += 1
    }
    if (v.last.right != IntervalEndpoint(Row(), 1)) {
      builder += Interval(v.last.right, IntervalEndpoint(Row(), 1))
    }

    KeySet(builder.result())
  }
}

class ExtractIntervalFilters(ctx: ExecuteContext, keyType: TStruct) {
  import ExtractIntervalFilters._

  object KeySetLattice extends is.hail.expr.ir.KeySetLattice(ctx, keyType)
  import KeySetLattice.KeySet

  // The lattice used in the ExtractIntervalFilters analysis
  // A value can be
  // * a constant, e.g. I32(5),
  // * a key field, e.g. GetField(Ref("row"), "key_field")
  // * the contig or position of a locus key field (which must be the first key
  //   field, e.g. Apply("contig", ..., GetField(Ref("row"), "locus"), ...)
  // * a struct, which consists of a lattice value for each field
  // * a boolean, which consists of an overapproximating KeySet for each
  //   possible runtime value --- true, false, or missing ---. For example
  //   if the boolean is true in a row with key `k`, then `k` must be in the
  //   "true" KeySet, but the converse needn't hold. In particular, when we
  //   know nothing about the boolean (e.g. it's computed from a non-key field)
  //   all three KeySets are the set of all keys.
  // These categories are not disjoint. In particular, a value can be both a
  // (key field or constant) and a (struct or boolean). We represent these
  // overlaps with subclasses.
  object Lattice extends JoinLattice {
    abstract class Value

    object ConstantValue {
      def apply(v: Any, t: Type): ConstantValue = t match {
        case TBoolean => ConstantBool(v.asInstanceOf[Boolean], KeySetLattice.top)
        case t: TStruct => ConstantStruct(v.asInstanceOf[Row], t)
        case _ => ConcreteConstant(v)
      }

      def unapply(v: Value): Option[Any] = v match {
        case v: ConstantValue => Some(v.value)
        case _ => None
      }
    }

    trait ConstantValue extends Value {
      def value: Any
    }

    object KeyField {
      def apply(idx: Int): KeyField = keyType.types(idx) match {
        case TBoolean => KeyFieldBool(idx, KeySetLattice.top)
        case _: TStruct => KeyFieldStruct(idx)
        case _ => ConcreteKeyField(idx)
      }


      def unapply(v: Value): Option[Int] = v match {
        case v: KeyField => Some(v.idx)
        case _ => None
      }
    }

    trait KeyField extends Value {
      def idx: Int
    }

    private case class ConcreteConstant(value: Any) extends ConstantValue

    private case class ConcreteKeyField(idx: Int) extends KeyField

    case class Contig(rg: String) extends Value

    case class Position(rg: String) extends Value

    object StructValue {
      def apply(fields: Iterable[(String, Value)]): StructValue = {
        val filtered = fields.filter { case (_, v) => v != top }
        if (filtered.isEmpty)
          top
        else if (filtered.exists(_._2 == bottom))
          bottom
        else
          ConcreteStruct(filtered.toMap)
      }
    }

    trait StructValue extends Value {
      def apply(field: String): Value
      def values: Iterable[Value]
      def isKeyPrefix: Boolean
    }

    private case class ConcreteStruct(fields: Map[String, Value]) extends StructValue {
      def apply(field: String): Value = fields.getOrElse(field, top)

      def values: Iterable[Value] = fields.values

      def isKeyPrefix: Boolean = fields.values.view.zipWithIndex.forall {
        case (f: KeyField, i2) => f.idx == i2
        case _ => false
      }
    }

    object BoolValue {
      private def all: KeySet = KeySetLattice.top
      private def none: KeySet = KeySetLattice.bottom

      def allTrue(keySet: KeySet = KeySetLattice.top): BoolValue = ConcreteBool(keySet, none, none)
      def allFalse(keySet: KeySet = KeySetLattice.top): BoolValue = ConcreteBool(none, keySet, none)
      def allNA(keySet: KeySet = KeySetLattice.top): BoolValue = ConcreteBool(none, none, keySet)
      def top(keySet: KeySet = KeySetLattice.top): BoolValue = ConcreteBool(keySet, keySet, keySet)

      def apply(trueBound: KeySet, falseBound: KeySet, naBound: KeySet): BoolValue = {
        if (trueBound == all && falseBound == all && naBound == all)
          Lattice.top
        else if (trueBound == none && falseBound == none && naBound == none)
          Lattice.bottom
        else
          ConcreteBool(trueBound, falseBound, naBound)
      }

      def or(l: BoolValue, r: BoolValue): BoolValue = (l, r) match {
        case (ConstantBool(l, lKeySet), ConstantBool(r, rKeySet)) =>
          assert(lKeySet eq rKeySet)
          ConstantBool(l || r, lKeySet)
        case _ => ConcreteBool(
          KeySetLattice.combine(l.trueBound, r.trueBound),
          KeySetLattice.meet(l.falseBound, r.falseBound),
          KeySetLattice.combineMulti(
            KeySetLattice.meet(l.naBound, r.falseBound),
            KeySetLattice.meet(l.naBound, r.naBound),
            KeySetLattice.meet(l.falseBound, r.naBound)))
      }

      def and(l: BoolValue, r: BoolValue): BoolValue = (l, r) match {
        case (ConstantBool(l, lKeySet), ConstantBool(r, rKeySet)) =>
          assert(lKeySet eq rKeySet)
          ConstantBool(l && r, lKeySet)
        case _ => ConcreteBool(
          KeySetLattice.meet(l.trueBound, r.trueBound),
          KeySetLattice.combine(l.falseBound, r.falseBound),
          KeySetLattice.combineMulti(
            KeySetLattice.meet(l.naBound, r.trueBound),
            KeySetLattice.meet(l.naBound, r.naBound),
            KeySetLattice.meet(l.trueBound, r.naBound)))
      }

      def not(x: BoolValue): BoolValue = x match {
        case ConstantBool(x, keySet) => ConstantBool(!x, keySet)
        case _ => ConcreteBool(x.falseBound, x.trueBound, x.naBound)
      }

      // WIP: push these methods onto BoolValue class, make sure they preserve keySet
      def isNA(x: BoolValue): BoolValue = x match {
        case ConstantBool(x, keySet) => ConstantBool(false, keySet)
        case _ => ConcreteBool(x.naBound, KeySetLattice.combine(x.trueBound, x.falseBound), KeySetLattice.bottom)
      }

      def fromComparison(v: Any, op: ComparisonOp[_], wrapped: Boolean = true): BoolValue = {
        (op: @unchecked) match {
          case _: EQ => BoolValue( // value == key
                                   KeySet(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
                                   KeySet(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
                                   KeySet(Interval(endpoint(null, -1), posInf)))
          case _: NEQ => BoolValue( // value != key
                                    KeySet(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
                                    KeySet(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
                                    KeySet(Interval(endpoint(null, -1), posInf)))
          case _: GT => BoolValue( // value > key
                                   KeySet(Interval(negInf, endpoint(v, -1, wrapped))),
                                   KeySet(Interval(endpoint(v, -1, wrapped), endpoint(null, -1))),
                                   KeySet(Interval(endpoint(null, -1), posInf)))
          case _: GTEQ => BoolValue( // value >= key
                                     KeySet(Interval(negInf, endpoint(v, 1, wrapped))),
                                     KeySet(Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
                                     KeySet(Interval(endpoint(null, -1), posInf)))
          case _: LT => BoolValue( // value < key
                                   KeySet(Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
                                   KeySet(Interval(negInf, endpoint(v, 1, wrapped))),
                                   KeySet(Interval(endpoint(null, -1), posInf)))
          case _: LTEQ => BoolValue( // value <= key
                                     KeySet(Interval(endpoint(v, -1, wrapped), endpoint(null, -1))),
                                     KeySet(Interval(negInf, endpoint(v, -1, wrapped))),
                                     KeySet(Interval(endpoint(null, -1), posInf)))
          case _: EQWithNA => // value == key
            if (v == null)
              BoolValue(
                KeySet(Interval(endpoint(v, -1, wrapped), posInf)),
                KeySet(Interval(negInf, endpoint(v, -1, wrapped))),
                KeySetLattice.bottom)
            else
              BoolValue(
                KeySet(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
                KeySet(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), posInf)),
                KeySetLattice.bottom)
          case _: NEQWithNA => // value != key
            if (v == null)
              BoolValue(
                KeySet(Interval(negInf, endpoint(v, -1, wrapped))),
                KeySet(Interval(endpoint(v, -1, wrapped), posInf)),
                KeySetLattice.bottom)
            else
              BoolValue(
                KeySet(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), posInf)),
                KeySet(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
                KeySetLattice.bottom)
        }
      }

      def fromComparisonKeyPrefix(v: Row, op: ComparisonOp[_]): BoolValue = {
        (op: @unchecked) match {
          case _: EQ => BoolValue( // value == key
            KeySet(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
            KeySet(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            KeySetLattice.bottom)
          case _: NEQ => BoolValue( // value != key
            KeySet(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            KeySet(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
            KeySetLattice.bottom)
          case _: GT => BoolValue( // value > key
            KeySet(Interval(negInf, endpoint(v, -1, false))),
            KeySet(Interval(endpoint(v, -1, false), posInf)),
            KeySetLattice.bottom)
          case _: GTEQ => BoolValue( // value >= key
            KeySet(Interval(negInf, endpoint(v, 1, false))),
            KeySet(Interval(endpoint(v, 1, false), posInf)),
            KeySetLattice.bottom)
          case _: LT => BoolValue( // value < key
            KeySet(Interval(endpoint(v, 1, false), posInf)),
            KeySet(Interval(negInf, endpoint(v, 1, false))),
            KeySetLattice.bottom)
          case _: LTEQ => BoolValue( // value <= key
            KeySet(Interval(endpoint(v, -1, false), posInf)),
            KeySet(Interval(negInf, endpoint(v, 1, false))),
            KeySetLattice.bottom)
          case _: EQWithNA => BoolValue( // value == key
            KeySet(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
            KeySet(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            KeySetLattice.bottom)
          case _: NEQWithNA => BoolValue( // value != key
            KeySet(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            KeySet(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
            KeySetLattice.bottom)
        }
      }
    }

    trait BoolValue extends Value {
      def trueBound: KeySet
      def falseBound: KeySet
      def naBound: KeySet

      def restrict(keySet: KeySet): BoolValue

      def isNA(): BoolValue = ConcreteBool(naBound, KeySetLattice.combine(trueBound, falseBound), KeySetLattice.bottom)
    }

    private case class ConcreteBool(
      trueBound: KeySet,
      falseBound: KeySet,
      naBound: KeySet
    ) extends BoolValue {
      override def restrict(keySet: KeySet): BoolValue = {
        ConcreteBool(KeySetLattice.meet(trueBound, keySet), KeySetLattice.meet(falseBound, keySet), KeySetLattice.meet(naBound, keySet))
      }
    }

    private case class ConstantStruct(value: Row, t: TStruct) extends StructValue with ConstantValue {
      def apply(field: String): Value = this(t.field(field))
      def values: Iterable[Value] = t.fields.map(apply)
      private def apply(field: Field): ConstantValue = ConstantValue(value(field.index), field.typ)
      def isKeyPrefix: Boolean = false
    }

    private case class ConstantBool(value: Boolean, keySet: KeySet) extends BoolValue with ConstantValue {
      override def trueBound: KeySet =
        if (value) keySet else KeySetLattice.bottom

      override def falseBound: KeySet =
        if (value) KeySetLattice.bottom else keySet

      override def naBound: KeySet = KeySetLattice.bottom

      override def restrict(keySet: KeySet): BoolValue = ConstantBool(value, keySet)

      override def isNA(): BoolValue = ConstantBool(false, keySet)
    }

    private case class KeyFieldBool(idx: Int, keySet: KeySet) extends BoolValue with KeyField {
      override def trueBound: KeySet = if (idx == 0)
        KeySet(Interval(true, true, includesStart = true, includesEnd = true))
      else
        keySet

      override def falseBound: KeySet = if (idx == 0)
        KeySet(Interval(false, false, includesStart = true, includesEnd = true))
      else
        keySet

      override def naBound: KeySet = if (idx == 0)
        KeySet(Interval(null, null, includesStart = true, includesEnd = true))
      else
        keySet

      override def restrict(keySet: KeySet): BoolValue = KeyFieldBool(idx, keySet)
    }

    private case class KeyFieldStruct(idx: Int) extends StructValue with KeyField {
      def apply(field: String): Value = Top
      def values: Iterable[Value] = Iterable.empty
      def isKeyPrefix: Boolean = false
    }

    private case object Top extends StructValue with BoolValue {
      def apply(field: String): Value = Top
      def values: Iterable[Value] = Iterable.empty
      def isKeyPrefix: Boolean = false
      override def trueBound: KeySet = KeySetLattice.top
      override def falseBound: KeySet = KeySetLattice.top
      override def naBound: KeySet = KeySetLattice.top
      override def restrict(keySet: KeySet): BoolValue = BoolValue(keySet, keySet, keySet)
      override def isNA(): BoolValue = Top
    }

    private case object Bottom extends StructValue with BoolValue {
      def apply(field: String): Value = ???
      def values: Iterable[Value] = ???
      def isKeyPrefix: Boolean = ???
      override def trueBound: KeySet = KeySetLattice.bottom
      override def falseBound: KeySet = KeySetLattice.bottom
      override def naBound: KeySet = KeySetLattice.bottom
      override def restrict(keySet: KeySet): BoolValue = Bottom
      override def isNA(): BoolValue = Bottom
    }

    def top: StructValue with BoolValue = Top
    def bottom: StructValue with BoolValue = Bottom

    def combine(l: Value, r: Value): Value = (l, r) match {
      case (Bottom, x) => x
      case (x, Bottom) => x
      case (l: ConstantValue, r: ConstantValue) if l.value == r.value => l
      case (l: KeyField, r: KeyField) if l.idx == r.idx => l
      case (l: Contig, r: Contig) =>
        assert(l.rg == r.rg)
        l
      case (l: Position, r: Position) =>
        assert(l.rg == r.rg)
        l
      case (ConcreteStruct(l), ConcreteStruct(r)) =>
        StructValue(l.keySet.intersect(r.keySet).view.map {
          f => f -> combine(l(f), r(f))
        }.toMap)
      case (l: BoolValue, r: BoolValue) => BoolValue(
        KeySetLattice.combine(l.trueBound, r.trueBound),
        KeySetLattice.combine(l.falseBound, r.falseBound),
        KeySetLattice.combine(l.naBound, r.naBound))
      case _ => Top
    }

    def meet(l: Value, r: Value): Value = (l, r) match {
      case (Top, x) => x
      case (x, Top) => x
      case (l: ConstantValue, r: ConstantValue) if l.value == r.value => l
      case (l: KeyField, r: KeyField) if l.idx == r.idx => l
      case (l: Contig, r: Contig) =>
        assert(l.rg == r.rg)
        l
      case (l: Position, r: Position) =>
        assert(l.rg == r.rg)
        l
      case (l: ConcreteStruct, r: ConcreteStruct) =>
        StructValue(l.fields.keySet.union(r.fields.keySet).view.map {
          f => f -> meet(l(f), r(f))
        }.toMap)
      case (l: BoolValue, r: BoolValue) => BoolValue(
        KeySetLattice.meet(l.trueBound, r.trueBound),
        KeySetLattice.meet(l.falseBound, r.falseBound),
        KeySetLattice.meet(l.naBound, r.naBound))
      case _ => Top
    }

    def compare(l: Value, r: Value, op: ComparisonOp[_], keySet: KeySet): BoolValue = {
      if (opIsSupported(op)) (l, r) match {
        case (ConstantValue(l), r) => compareWithConstant(l, r, op, keySet)
        case (l, ConstantValue(r)) =>
          compareWithConstant(r, l, ComparisonOp.swap(op.asInstanceOf[ComparisonOp[Boolean]]), keySet)
        case _ => BoolValue.top(keySet)
      } else {
        BoolValue.top(keySet)
      }
    }

    private def compareWithConstant(l: Any, r: Value, op: ComparisonOp[_], keySet: KeySet): BoolValue = {
      if (op.strict && l == null) return BoolValue.allNA(keySet)
      r match {
        case r: KeyField if r.idx == 0 =>
          // simple key comparison
          BoolValue.fromComparison(l, op).restrict(keySet)
        case Contig(rgStr) =>
          // locus contig comparison
          assert(op.isInstanceOf[EQ])
          val b = getIntervalFromContig(l.asInstanceOf[String], ctx.getReference(rgStr)) match {
            case Some(i) =>
              BoolValue(
                KeySet(i),
                KeySet(Interval(negInf, i.left), Interval(i.right, endpoint(null, -1))),
                KeySet(Interval(endpoint(null, -1), posInf)))
            case None =>
              BoolValue(
                KeySetLattice.bottom,
                KeySet(Interval(negInf, endpoint(null, -1))),
                KeySet(Interval(endpoint(null, -1), posInf)))
          }
          b.restrict(keySet)
        case Position(rgStr) =>
          // locus position comparison
          val posBoolValue = BoolValue.fromComparison(l, op)
          val rg = ctx.getReference(rgStr)
          val b = BoolValue(
            KeySet(liftPosIntervalsToLocus(posBoolValue.trueBound, rg, ctx)),
            KeySet(liftPosIntervalsToLocus(posBoolValue.falseBound, rg, ctx)),
            KeySet(liftPosIntervalsToLocus(posBoolValue.naBound, rg, ctx)))
          b.restrict(keySet)
        case s: StructValue if s.isKeyPrefix =>
          BoolValue.fromComparisonKeyPrefix(l.asInstanceOf[Row], op).restrict(keySet)
        case _ => BoolValue.top(keySet)
      }
    }

    private def opIsSupported(op: ComparisonOp[_]): Boolean = op match {
      case _: EQ | _: NEQ | _: LTEQ | _: LT | _: GTEQ | _: GT | _: EQWithNA | _: NEQWithNA => true
      case _ => false
    }
  }

  import Lattice.{ Value => AbstractValue, ConstantValue, KeyField, StructValue, BoolValue, Contig, Position }

  case class AbstractEnv(keySet: KeySet, env: Env[AbstractValue]) {
    def lookupOption(name: String): Option[AbstractValue] =
      env.lookupOption(name)

    def bind(bindings: (String, AbstractValue)*): AbstractEnv =
      copy(env = env.bind(bindings: _*))

    def restrict(k: KeySet): AbstractEnv =
      copy(keySet = KeySetLattice.meet(keySet, k))
  }

  object AbstractEnvLattice extends JoinLattice {
    type Value = AbstractEnv
    val top: Value = AbstractEnv(KeySetLattice.top, Env.empty)

    def combine(l: Value, r: Value): Value = {
      val keys = l.env.m.keySet.intersect(r.env.m.keySet)
      val env = Env.fromSeq(keys.map { k =>
        k -> Lattice.combine(l.env(k), r.env(k))
      })
      AbstractEnv(KeySetLattice.combine(l.keySet, r.keySet), env)
    }
  }

  def firstKeyOrd: ExtendedOrdering = keyType.types.head.ordering(ctx.stateManager)
  def keyOrd: ExtendedOrdering = PartitionBoundOrdering(ctx, keyType)
  val iord: IntervalEndpointOrdering = keyOrd.intervalEndpointOrdering

  private def intervalsFromLiteral(lit: Any, ordering: Ordering[Any], wrapped: Boolean): IndexedSeq[Interval] =
    (lit: @unchecked) match {
      case x: Map[_, _] => intervalsFromCollection(x.keys, ordering, wrapped)
      case x: Traversable[_] => intervalsFromCollection(x, ordering, wrapped)
    }

  private def intervalsFromCollection(lit: Traversable[Any], ordering: Ordering[Any], wrapped: Boolean): IndexedSeq[Interval] =
    KeySet.reduce(
      lit.toArray.distinct.filter(x => wrapped || x != null).sorted(ordering)
        .map(elt => Interval(endpoint(elt, -1, wrapped), endpoint(elt, 1, wrapped)))
        .toFastIndexedSeq)

  private def intervalsFromLiteralContigs(contigs: Any, rg: ReferenceGenome): IndexedSeq[Interval] = {
    KeySet((contigs: @unchecked) match {
             case x: Map[_, _] => x.keys.asInstanceOf[Iterable[String]].toFastIndexedSeq
               .sortBy(rg.contigsIndex.get(_))(TInt32.ordering(null).toOrdering.asInstanceOf[Ordering[Integer]])
               .flatMap(getIntervalFromContig(_, rg))
             case x: Traversable[_] => x.asInstanceOf[Traversable[String]].toArray.toFastIndexedSeq
               .sortBy(rg.contigsIndex.get(_))(TInt32.ordering(null).toOrdering.asInstanceOf[Ordering[Integer]])
               .flatMap(getIntervalFromContig(_, rg))
           })
  }

  private def getIntervalFromContig(c: String, rg: ReferenceGenome): Option[Interval] = {
    if (rg.contigsSet.contains(c))
      Some(Interval(Row(Locus(c, 1)), Row(Locus(c, rg.contigLength(c))), true, false))
    else if (c == null) {
      warn(s"Filtered with null contig")
      Some(Interval(Row(null), Row(), true, true))
    } else {
      warn(
        s"Filtered with contig '$c', but '$c' is not a valid contig in reference genome ${rg.name}")
      None
    }
  }

  def endpoint(value: Any, sign: Int, wrapped: Boolean = true): IntervalEndpoint =
    IntervalEndpoint(if (wrapped) Row(value) else value, sign)

  private def literalSizeOkay(lit: Any): Boolean = lit.asInstanceOf[Iterable[_]].size <=
    MAX_LITERAL_SIZE

  private def wrapInRow(intervals: IndexedSeq[Interval]): IndexedSeq[Interval] = intervals
    .map { interval =>
      Interval(
        IntervalEndpoint(Row(interval.left.point), interval.left.sign),
        IntervalEndpoint(Row(interval.right.point), interval.right.sign))
    }

  private def intervalFromComparison(v: Any, op: ComparisonOp[_]): Interval = {
    (op: @unchecked) match {
      case _: EQ => Interval(endpoint(v, -1), endpoint(v, 1))
      case GT(_, _) => Interval(negInf, endpoint(v, -1)) // value > key
      case GTEQ(_, _) => Interval(negInf, endpoint(v, 1)) // value >= key
      case LT(_, _) => Interval(endpoint(v, 1), posInf) // value < key
      case LTEQ(_, _) => Interval(endpoint(v, -1), posInf) // value <= key
    }
  }

  private def posInf: IntervalEndpoint = IntervalEndpoint(Row(), 1)

  private def negInf: IntervalEndpoint = IntervalEndpoint(Row(), -1)

  def analyze(x: IR, rowName: String, rw: Option[Rewrites] = None, constraint: KeySet = KeySetLattice.top): KeySet = {
    val env = Env.empty[AbstractValue].bind(
      rowName,
      StructValue(
        Map(keyType.fieldNames.zipWithIndex.map(t => t._1 -> KeyField(t._2)): _*)))
    val bool = _analyze(x, AbstractEnv(constraint, env), rw).asInstanceOf[BoolValue]
    bool.trueBound
  }

  def rewrite(x: IR, rw: Rewrites): IR = {
    if (rw.replaceWithTrue.contains(RefEquality(x))) True()
    else if (rw.replaceWithFalse.contains(RefEquality(x))) False()
    else x.mapChildren {
      case child: IR => rewrite(child, rw)
      case child => child
    }
  }

  private def computeKeyOrConst(x: IR, children: IndexedSeq[AbstractValue]): AbstractValue = x match {
    case False() => ConstantValue(false, TBoolean)
    case True() => ConstantValue(true, TBoolean)
    case I32(v) => ConstantValue(v, x.typ)
    case I64(v) => ConstantValue(v, x.typ)
    case F32(v) => ConstantValue(v, x.typ)
    case F64(v) => ConstantValue(v, x.typ)
    case Str(v) => ConstantValue(v, x.typ)
    case NA(_) => ConstantValue(null, x.typ)
    case Literal(_, value) => ConstantValue(value, x.typ)
    case ApplySpecial("lor", _, _, _, _) => children match {
      case Seq(ConstantValue(l: Boolean), ConstantValue(r: Boolean)) => ConstantValue(l || r, TBoolean)
      case _ => Lattice.top
    }
    case ApplySpecial("land", _, _, _, _) => children match {
      case Seq(ConstantValue(l: Boolean), ConstantValue(r: Boolean)) => ConstantValue(l && r, TBoolean)
      case _ => Lattice.top
    }
    case Apply("contig", _, Seq(k), _, _) => children match {
      case Seq(KeyField(0)) => Contig(k.typ.asInstanceOf[TLocus].rg)
      case _ => Lattice.top
    }
    case Apply("position", _, Seq(k), _, _) => children match {
      case Seq(KeyField(0)) => Position(k.typ.asInstanceOf[TLocus].rg)
      case _ => Lattice.top
    }
    case _ => Lattice.top
  }

  private def computeBoolean(x: IR, children: IndexedSeq[AbstractValue], keySet: KeySet): BoolValue = (x, children) match {
    case (False(), _) => BoolValue.allFalse(keySet)
    case (True(), _) => BoolValue.allTrue(keySet)
    case (IsNA(_), Seq(KeyField(0))) => BoolValue(
      KeySet(Interval(endpoint(null, -1), posInf)),
      KeySet(Interval(negInf, endpoint(null, -1))),
      KeySetLattice.bottom)
      .restrict(keySet)
    case (IsNA(_), Seq(b: BoolValue)) => BoolValue.isNA(b)
    // collection contains
    case (ApplyIR("contains", _, _, _), Seq(ConstantValue(collectionVal), queryVal)) if literalSizeOkay(collectionVal) =>
      if (collectionVal == null) {
        BoolValue.allNA(keySet)
      } else queryVal match {
        case Contig(rgStr) =>
          val rg = ctx.stateManager.referenceGenomes(rgStr)
          val intervals = intervalsFromLiteralContigs(collectionVal, rg)
          BoolValue(intervals, KeySetLattice.complement(intervals), KeySetLattice.bottom).restrict(keySet)
        case KeyField(0) =>
          val intervals = intervalsFromLiteral(collectionVal, firstKeyOrd.toOrdering, true)
          BoolValue(intervals, KeySetLattice.complement(intervals), KeySetLattice.bottom).restrict(keySet)
        case struct: StructValue if struct.isKeyPrefix =>
          val intervals = intervalsFromLiteral(collectionVal, keyOrd.toOrdering, false)
          BoolValue(intervals, KeySetLattice.complement(intervals), KeySetLattice.bottom).restrict(keySet)
        case _ => BoolValue.top(keySet)
      }
    // interval contains
    case (ApplySpecial("contains", _, _, _, _), Seq(ConstantValue(intervalVal), queryVal)) =>
      (intervalVal: @unchecked) match {
        case null => BoolValue.allNA(keySet)
        case i: Interval => queryVal match {
          case KeyField(0) =>
            val l = IntervalEndpoint(Row(i.left.point), i.left.sign)
            val r = IntervalEndpoint(Row(i.right.point), i.right.sign)
            BoolValue(
              KeySet(Interval(l, r)),
              KeySet(Interval(negInf, l), Interval(r, endpoint(null, -1))),
              KeySet(Interval(endpoint(null, -1), posInf)))
              .restrict(keySet)
          case struct: StructValue if struct.isKeyPrefix =>
            BoolValue(
              KeySet(i),
              KeySet(Interval(negInf, i.left), Interval(i.right, posInf)),
              KeySet.empty)
              .restrict(keySet)
          case _ => BoolValue.top(keySet)
        }
      }
    case (ApplyComparisonOp(op, _, _), Seq(l, r)) =>
      Lattice.compare(l, r, op, keySet)
    case (ApplySpecial("lor", _, _, _, _), Seq(l: BoolValue, r: BoolValue)) =>
      BoolValue.or(l, r)
    case (ApplySpecial("land", _, _, _, _), Seq(l: BoolValue, r: BoolValue)) =>
      BoolValue.and(l, r)
    case (ApplyUnaryPrimOp(Bang, _), Seq(x: BoolValue)) =>
      BoolValue.not(x)
    case _ => BoolValue.top(keySet)
  }

  case class Rewrites(
    replaceWithTrue: mutable.Set[RefEquality[IR]],
    replaceWithFalse: mutable.Set[RefEquality[IR]])

  private def _analyze(x: IR, env: AbstractEnv, rewrites: Option[Rewrites]): AbstractValue = {
    def recur(x: IR, env: AbstractEnv = env): AbstractValue =
      _analyze(x, env, rewrites)

    var res: Lattice.Value = if (env.keySet == KeySetLattice.bottom)
      Lattice.bottom
    else x match {
      case Let(name, value, body) => recur(body, env.bind(name -> recur(value)))
      case Ref(name, _) => env.lookupOption(name).getOrElse(Lattice.top)
      case GetField(o, name) => recur(o).asInstanceOf[StructValue](name)
      case MakeStruct(fields) => StructValue(fields.view.map { case (name, field) =>
        name -> recur(field)
      })
      case SelectFields(old, fields) =>
        val oldVal = recur(old)
        StructValue(fields.view.map(name => name -> oldVal.asInstanceOf[StructValue](name)))
      case If(cond, cnsq, altr) =>
        val c = recur(cond).asInstanceOf[BoolValue]
        val res = Lattice.combine(
          recur(cnsq, env.restrict(c.trueBound)),
          recur(altr, env.restrict(c.falseBound)))
        if (x.typ == TBoolean)
          Lattice.combine(res, BoolValue(KeySetLattice.bottom, KeySetLattice.bottom, c.naBound))
        else
          res
      case ToStream(a, _) => recur(a)
      case StreamFold(a, zero, accumName, valueName, body) => recur(a) match {
          case ConstantValue(array) => array.asInstanceOf[Iterable[Any]]
              .foldLeft(recur(zero)) { (accum, value) =>
                recur(body, env.bind(accumName -> accum, valueName -> ConstantValue(value, TIterable.elementType(a.typ))))
              }
          case _ => Lattice.top
        }
      case x@Coalesce(values) =>
        val aVals = values.map(recur(_))
        if (x.typ == TBoolean) {
          val bVals = aVals.asInstanceOf[Seq[BoolValue]]
          val trueBound = bVals.foldRight(KeySetLattice.bottom) { (x, acc) =>
            KeySetLattice.combine(x.trueBound, KeySetLattice.meet(x.naBound, acc))
          }
          val falseBound = bVals.foldRight(KeySetLattice.bottom) { (x, acc) =>
            KeySetLattice.combine(x.falseBound, KeySetLattice.meet(x.naBound, acc))
          }
          val naBound = bVals.foldRight(KeySetLattice.top) { (x, acc) =>
            KeySetLattice.meet(x.naBound, acc)
          }
          BoolValue(trueBound, falseBound, naBound)
        } else {
          aVals.reduce(Lattice.combine)
        }
      case _ =>
        null
    }

    res = if (res == null) {
      val children = x.children.map(child => recur(child.asInstanceOf[IR])).toFastIndexedSeq
      val keyOrConstVal = computeKeyOrConst(x, children)
      if (x.typ == TBoolean) {
        if (keyOrConstVal == Lattice.top)
          computeBoolean(x, children, env.keySet)
        else
          keyOrConstVal.asInstanceOf[BoolValue].restrict(env.keySet)
      } else {
        keyOrConstVal
      }
    } else if (x.typ == TBoolean) {
      res.asInstanceOf[BoolValue].restrict(env.keySet)
    } else {
      res
    }

    res match {
      case res: BoolValue =>
        assert(KeySetLattice.specializes(res.trueBound, env.keySet), s"\n  trueBound = ${res.trueBound}\n  env = ${env.keySet}")
        assert(KeySetLattice.specializes(res.falseBound, env.keySet), s"\n  falseBound = ${res.falseBound}\n  env = ${env.keySet}")
        assert(KeySetLattice.specializes(res.naBound, env.keySet), s"\n  naBound = ${res.naBound}\n  env = ${env.keySet}")
      case _ =>
    }

    rewrites.foreach { rw =>
      if (x.typ == TBoolean) {
        val bool = res.asInstanceOf[BoolValue]
        if (KeySetLattice.meet(KeySetLattice.combine(bool.falseBound, bool.naBound), env.keySet) == KeySetLattice.bottom)
          rw.replaceWithTrue += RefEquality(x)
        else if (KeySetLattice.meet(KeySetLattice.combine(bool.trueBound, bool.naBound), env.keySet) == KeySetLattice.bottom) {
          rw.replaceWithFalse += RefEquality(x)
        }
      }
    }
    res
  }

}
