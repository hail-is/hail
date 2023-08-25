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
  type Value
  def top: Value
  def combine(l: Value, r: Value): Value
}

trait AbstractLattice extends JoinLattice {
  def bottom: Value
}

class LatticeFromJoin[T <: JoinLattice](val joinLattice: T) extends AbstractLattice {
  type Value = Option[joinLattice.Value]
  def top: Value = Some(joinLattice.top)
  def bottom: Value = None

  def combine(l: Value, r: Value): Value = (l, r) match {
    case (None, x) => x
    case (x, None) => x
    case (Some(l), Some(r)) => Some(joinLattice.combine(l, r))
  }

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
      if (trueSet == extract.KeySet.top)
        None
      else {
        val rw = extract.Rewrites(mutable.Set.empty, mutable.Set.empty)
        extract.analyze(cond, ref.name, Some(rw), trueSet)
        Some((extract.rewrite(cond, rw), trueSet.intervals))
      }
    }
  }

  private def intervalsFromLiteral(lit: Any, ordering: Ordering[Any], wrapped: Boolean): IntervalsSet =
    (lit: @unchecked) match {
      case x: Map[_, _] => intervalsFromCollection(x.keys, ordering, wrapped)
      case x: Traversable[_] => intervalsFromCollection(x, ordering, wrapped)
    }

  private def intervalsFromCollection(lit: Traversable[Any], ordering: Ordering[Any], wrapped: Boolean): IntervalsSet =
    IntervalsSet.reduce(
      lit.toArray.distinct.filter(x => wrapped || x != null).sorted(ordering)
        .map(elt => Interval(endpoint(elt, -1, wrapped), endpoint(elt, 1, wrapped)))
        .toFastIndexedSeq)

  private def intervalsFromLiteralContigs(contigs: Any, rg: ReferenceGenome): IntervalsSet = {
    IntervalsSet((contigs: @unchecked) match {
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

  def liftPosIntervalsToLocus(pos: IndexedSeq[Interval], rg: ReferenceGenome, ctx: ExecuteContext): IndexedSeq[Interval] = {
    val ord = PartitionBoundOrdering(ctx, TTuple(TInt32))
    val nonNull = rg.contigs.indices.flatMap { cont =>
      pos.flatMap { i =>
        i.intersect(ord, Interval(endpoint(1, -1), endpoint(rg.contigLength(cont), -1)))
          .map { interval =>
            Interval(
              endpoint(
                Locus(rg.contigs(cont), interval.left.point.asInstanceOf[Row].getAs[Int](0)),
                interval.left.sign),
              endpoint(
                Locus(rg.contigs(cont), interval.right.point.asInstanceOf[Row].getAs[Int](0)),
                interval.right.sign))
          }
      }
    }
    if (pos.nonEmpty && pos.last.contains(ord, Row(null)))
      nonNull :+ Interval(Row(null), Row(), true, true)
    else
      nonNull
  }

  private def posInf: IntervalEndpoint = IntervalEndpoint(Row(), 1)

  private def negInf: IntervalEndpoint = IntervalEndpoint(Row(), -1)


}

object IntervalsSet {
  def apply(intervals: Interval*): IntervalsSet = {
    apply(intervals.toFastIndexedSeq)
  }

  def apply(intervals: IndexedSeq[Interval]): IntervalsSet = {
    assert(intervals.isEmpty || IntervalsSet.intervalIsReduced(intervals.last))
    new IntervalsSet(intervals)
  }

  val empty: IntervalsSet = new IntervalsSet(FastIndexedSeq())

  def reduce(intervals: IndexedSeq[Interval]): IntervalsSet = intervals match {
    case Seq() => empty
    case init :+ last =>
      val reducedLast = if (intervalIsReduced(last))
        last
      else
        Interval(last.left, IntervalEndpoint(Row(), 1))
      new IntervalsSet((init :+ reducedLast).toFastIndexedSeq)
  }

  private def intervalIsReduced(interval: Interval): Boolean = {
    interval.right != IntervalEndpoint(Row(null), 1)
  }
}

class IntervalsSet private (val intervals: IndexedSeq[Interval]) extends AnyVal {
  def flatMap(f: Interval => GenTraversableOnce[Interval]): IntervalsSet =
    new IntervalsSet(intervals.flatMap(f))
}

class ExtractIntervalFilters(ctx: ExecuteContext, keyType: TStruct) {
  import ExtractIntervalFilters._

  object KeySet extends AbstractLattice {
    // FIXME: make this an AnyVal wrapper.
    // Make constructor simplify intervals involving null
    //   IntervalEndpoint(Row(a, b, null, null), -1) == IntervalEndpoint(Row(a, b), -1)
    type Value = IntervalsSet

    def top: Value = IntervalsSet(Interval(Row(), Row(), true, true))
    def bottom: Value = IntervalsSet.empty
    def combine(l: Value, r: Value): Value = {
      if (l == KeySet.bottom) r
      else if (r == KeySet.bottom) l
      else IntervalsSet(Interval.union(l.intervals ++ r.intervals, iord))
    }

    def combineMulti(vs: Value*): Value = {
      IntervalsSet(Interval.union(vs.flatMap(_.intervals).toFastIndexedSeq, iord))
    }

    def meet(l: Value, r: Value): Value = {
      if (l == KeySet.top) r
      else if (r == KeySet.top) l
      else IntervalsSet(Interval.intersection(l.intervals, r.intervals, iord))
    }

    def complement(v: Value): Value = {
      if (v.intervals.isEmpty) return top

      val builder = mutable.ArrayBuilder.make[Interval]()
      var i = 0
      if (v.intervals.head.left != IntervalEndpoint(Row(), -1)) {
        builder += Interval(IntervalEndpoint(Row(), -1), v.intervals.head.left)
      }
      while (i + 1 < v.intervals.length) {
        builder += Interval(v.intervals(i).right, v.intervals(i+1).left)
        i += 1
      }
      if (v.intervals.last.right != IntervalEndpoint(Row(), 1)) {
        builder += Interval(v.intervals.last.right, IntervalEndpoint(Row(), 1))
      }

      IntervalsSet(builder.result())
    }
  }

  object Lattice extends JoinLattice {
    abstract class Value

    object ConstantValue {
      def apply(v: Any, t: Type): ConstantValue = t match {
        case TBoolean => ConstantBool(v.asInstanceOf[Boolean])
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
        case TBoolean => KeyFieldBool(idx)
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
      private def all: KeySet.Value = KeySet.top
      private def none: KeySet.Value = KeySet.bottom

      val allTrue: BoolValue = ConcreteBool(all, none, none)
      val allFalse: BoolValue = ConcreteBool(none, all, none)
      val allNA: BoolValue = ConcreteBool(none, none, all)
      val top: BoolValue = Lattice.top

      def apply(trueBound: KeySet.Value, falseBound: KeySet.Value, naBound: KeySet.Value): BoolValue = {
        if (trueBound == all && falseBound == all && naBound == all)
          Lattice.top
        else if (trueBound == none && falseBound == none && naBound == none)
          Lattice.bottom
        else
          ConcreteBool(trueBound, falseBound, naBound)
      }

      def or(l: BoolValue, r: BoolValue): BoolValue = (l, r) match {
        case (ConstantBool(l), ConstantBool(r)) => ConstantBool(l || r)
        case _ => ConcreteBool(
          KeySet.combine(l.trueBound, r.trueBound),
          KeySet.meet(l.falseBound, r.falseBound),
          KeySet.combineMulti(
            KeySet.meet(l.naBound, r.falseBound),
            KeySet.meet(l.naBound, r.naBound),
            KeySet.meet(l.falseBound, r.naBound)))
      }

      def and(l: BoolValue, r: BoolValue): BoolValue = (l, r) match {
        case (ConstantBool(l), ConstantBool(r)) => ConstantBool(l && r)
        case _ => ConcreteBool(
          KeySet.meet(l.trueBound, r.trueBound),
          KeySet.combine(l.falseBound, r.falseBound),
          KeySet.combineMulti(
            KeySet.meet(l.naBound, r.trueBound),
            KeySet.meet(l.naBound, r.naBound),
            KeySet.meet(l.trueBound, r.naBound)))
      }

      def not(x: BoolValue): BoolValue = x match {
        case ConstantBool(x) => ConstantBool(!x)
        case _ => ConcreteBool(x.falseBound, x.trueBound, x.naBound)
      }

      def isNA(x: BoolValue): BoolValue = x match {
        case ConstantBool(x) => ConstantBool(false)
        case _ => ConcreteBool(x.naBound, KeySet.combine(x.trueBound, x.falseBound), KeySet.bottom)
      }

      def fromComparison(v: Any, op: ComparisonOp[_], wrapped: Boolean = true): BoolValue = {
        (op: @unchecked) match {
          case _: EQ => BoolValue( // value == key
            IntervalsSet(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
            IntervalsSet(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
            IntervalsSet(Interval(endpoint(null, -1), posInf)))
          case _: NEQ => BoolValue( // value != key
            IntervalsSet(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
            IntervalsSet(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
            IntervalsSet(Interval(endpoint(null, -1), posInf)))
          case _: GT => BoolValue( // value > key
            IntervalsSet(Interval(negInf, endpoint(v, -1, wrapped))),
            IntervalsSet(Interval(endpoint(v, -1, wrapped), endpoint(null,  -1))),
            IntervalsSet(Interval(endpoint(null, -1), posInf)))
          case _: GTEQ => BoolValue( // value >= key
            IntervalsSet(Interval(negInf, endpoint(v, 1, wrapped))),
            IntervalsSet(Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
            IntervalsSet(Interval(endpoint(null, -1), posInf)))
          case _: LT => BoolValue( // value < key
            IntervalsSet(Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
            IntervalsSet(Interval(negInf, endpoint(v, 1, wrapped))),
            IntervalsSet(Interval(endpoint(null, -1), posInf)))
          case _: LTEQ => BoolValue( // value <= key
            IntervalsSet(Interval(endpoint(v, -1, wrapped), endpoint(null, -1))),
            IntervalsSet(Interval(negInf, endpoint(v, -1, wrapped))),
            IntervalsSet(Interval(endpoint(null, -1), posInf)))
          case _: EQWithNA => // value == key
            if (v == null)
              BoolValue(
                IntervalsSet(Interval(endpoint(v, -1, wrapped), posInf)),
                IntervalsSet(Interval(negInf, endpoint(v, -1, wrapped))),
                KeySet.bottom)
            else
              BoolValue(
                IntervalsSet(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
                IntervalsSet(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), posInf)),
                KeySet.bottom)
          case _: NEQWithNA => // value != key
            if (v == null)
              BoolValue(
                IntervalsSet(Interval(negInf, endpoint(v, -1, wrapped))),
                IntervalsSet(Interval(endpoint(v, -1, wrapped), posInf)),
                KeySet.bottom)
            else
              BoolValue(
                IntervalsSet(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), posInf)),
                IntervalsSet(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
                KeySet.bottom)
        }
      }

      def fromComparisonKeyPrefix(v: Row, op: ComparisonOp[_]): BoolValue = {
        (op: @unchecked) match {
          case _: EQ => BoolValue( // value == key
            IntervalsSet(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
            IntervalsSet(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            KeySet.bottom)
          case _: NEQ => BoolValue( // value != key
            IntervalsSet(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            IntervalsSet(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
            KeySet.bottom)
          case _: GT => BoolValue( // value > key
            IntervalsSet(Interval(negInf, endpoint(v, -1, false))),
            IntervalsSet(Interval(endpoint(v, -1, false), posInf)),
            KeySet.bottom)
          case _: GTEQ => BoolValue( // value >= key
            IntervalsSet(Interval(negInf, endpoint(v, 1, false))),
            IntervalsSet(Interval(endpoint(v, 1, false), posInf)),
            KeySet.bottom)
          case _: LT => BoolValue( // value < key
            IntervalsSet(Interval(endpoint(v, 1, false), posInf)),
            IntervalsSet(Interval(negInf, endpoint(v, 1, false))),
            KeySet.bottom)
          case _: LTEQ => BoolValue( // value <= key
            IntervalsSet(Interval(endpoint(v, -1, false), posInf)),
            IntervalsSet(Interval(negInf, endpoint(v, 1, false))),
            KeySet.bottom)
          case _: EQWithNA => BoolValue( // value == key
            IntervalsSet(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
            IntervalsSet(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            KeySet.bottom)
          case _: NEQWithNA => BoolValue( // value != key
            IntervalsSet(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            IntervalsSet(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
            KeySet.bottom)
        }
      }
    }

    trait BoolValue extends Value {
      def trueBound: KeySet.Value
      def falseBound: KeySet.Value
      def naBound: KeySet.Value
    }

    private case class ConcreteBool(
      trueBound: KeySet.Value,
      falseBound: KeySet.Value,
      naBound: KeySet.Value
    ) extends BoolValue

    private case class ConstantStruct(value: Row, t: TStruct) extends StructValue with ConstantValue {
      def apply(field: String): Value = this(t.field(field))
      def values: Iterable[Value] = t.fields.map(apply)
      private def apply(field: Field): ConstantValue = ConstantValue(value(field.index), field.typ)
      def isKeyPrefix: Boolean = false
    }

    private case class ConstantBool(value: Boolean) extends BoolValue with ConstantValue {
      override def trueBound: KeySet.Value =
        if (value) KeySet.top else KeySet.bottom

      override def falseBound: KeySet.Value =
        if (value) KeySet.bottom else KeySet.top

      override def naBound: KeySet.Value = KeySet.bottom
    }

    private case class KeyFieldBool(idx: Int) extends BoolValue with KeyField {
      override def trueBound: KeySet.Value = if (idx == 0)
        IntervalsSet(Interval(true, true, includesStart = true, includesEnd = true))
      else
        KeySet.top

      override def falseBound: KeySet.Value = if (idx == 0)
        IntervalsSet(Interval(false, false, includesStart = true, includesEnd = true))
      else
        KeySet.top

      override def naBound: KeySet.Value = if (idx == 0)
        IntervalsSet(Interval(null, null, includesStart = true, includesEnd = true))
      else
        KeySet.top
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
      override def trueBound: IntervalsSet = KeySet.top
      override def falseBound: IntervalsSet = KeySet.top
      override def naBound: IntervalsSet = KeySet.top
    }

    private case object Bottom extends StructValue with BoolValue {
      def apply(field: String): Value = ???
      def values: Iterable[Value] = ???
      def isKeyPrefix: Boolean = ???
      override def trueBound: IntervalsSet = KeySet.bottom
      override def falseBound: IntervalsSet = KeySet.bottom
      override def naBound: IntervalsSet = KeySet.bottom
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
        KeySet.combine(l.trueBound, r.trueBound),
        KeySet.combine(l.falseBound, r.falseBound),
        KeySet.combine(l.naBound, r.naBound))
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
        KeySet.meet(l.trueBound, r.trueBound),
        KeySet.meet(l.falseBound, r.falseBound),
        KeySet.meet(l.naBound, r.naBound))
      case _ => Top
    }

    def compare(l: Value, r: Value, op: ComparisonOp[_]): BoolValue = {
      if (opIsSupported(op)) (l, r) match {
        case (ConstantValue(l), r) => compareWithConstant(l, r, op)
        case (l, ConstantValue(r)) =>
          compareWithConstant(r, l, ComparisonOp.swap(op.asInstanceOf[ComparisonOp[Boolean]]))
        case _ => top
      } else {
        top
      }
    }

    private def compareWithConstant(l: Any, r: Value, op: ComparisonOp[_]): BoolValue = {
      if (op.strict && l == null) return BoolValue.allNA
      r match {
        case r: KeyField if r.idx == 0 =>
          // simple key comparison
          BoolValue.fromComparison(l, op)
        case Contig(rgStr) =>
          // locus contig comparison
          assert(op.isInstanceOf[EQ])
          getIntervalFromContig(l.asInstanceOf[String], ctx.getReference(rgStr)) match {
            case Some(i) =>
              BoolValue(
                IntervalsSet(i),
                IntervalsSet(Interval(negInf, i.left), Interval(i.right, endpoint(null, -1))),
                IntervalsSet(Interval(endpoint(null, -1), posInf)))
            case None =>
              BoolValue(
                KeySet.bottom,
                IntervalsSet(Interval(negInf, endpoint(null, -1))),
                IntervalsSet(Interval(endpoint(null, -1), posInf)))
          }
        case Position(rgStr) =>
          // locus position comparison
          val posBoolValue = BoolValue.fromComparison(l, op)
          val rg = ctx.getReference(rgStr)
          BoolValue(
            IntervalsSet(liftPosIntervalsToLocus(posBoolValue.trueBound.intervals, rg, ctx)),
            IntervalsSet(liftPosIntervalsToLocus(posBoolValue.falseBound.intervals, rg, ctx)),
            IntervalsSet(liftPosIntervalsToLocus(posBoolValue.naBound.intervals, rg, ctx)))
        case s: StructValue if s.isKeyPrefix =>
          BoolValue.fromComparisonKeyPrefix(l.asInstanceOf[Row], op)
        case _ => top
      }
    }

    private def opIsSupported(op: ComparisonOp[_]): Boolean = op match {
      case _: EQ | _: NEQ | _: LTEQ | _: LT | _: GTEQ | _: GT | _: EQWithNA | _: NEQWithNA => true
      case _ => false
    }
  }

  import Lattice.{ Value => AbstractValue, ConstantValue, KeyField, StructValue, BoolValue, Contig, Position }

  case class AbstractEnv(keySet: KeySet.Value, env: Env[AbstractValue]) {
    def lookupOption(name: String): Option[AbstractValue] =
      env.lookupOption(name)

    def bind(bindings: (String, AbstractValue)*): AbstractEnv =
      copy(env = env.bind(bindings: _*))

    def restrict(k: KeySet.Value): AbstractEnv =
      copy(keySet = KeySet.meet(keySet, k))
  }

  object AbstractEnvLattice extends JoinLattice {
    type Value = AbstractEnv
    val top: Value = AbstractEnv(KeySet.top, Env.empty)

    def combine(l: Value, r: Value): Value = {
      val keys = l.env.m.keySet.intersect(r.env.m.keySet)
      val env = Env.fromSeq(keys.map { k =>
        k -> Lattice.combine(l.env(k), r.env(k))
      })
      AbstractEnv(KeySet.combine(l.keySet, r.keySet), env)
    }
  }

  def firstKeyOrd: ExtendedOrdering = keyType.types.head.ordering(ctx.stateManager)
  def keyOrd: ExtendedOrdering = PartitionBoundOrdering(ctx, keyType)
  val iord: IntervalEndpointOrdering = keyOrd.intervalEndpointOrdering

  def analyze(x: IR, rowName: String, rw: Option[Rewrites] = None, constraint: KeySet.Value = KeySet.top): IntervalsSet = {
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

  private def computeBoolean(x: IR, children: IndexedSeq[AbstractValue]): BoolValue = (x, children) match {
    case (False(), _) => BoolValue.allFalse
    case (True(), _) => BoolValue.allTrue
    case (IsNA(_), Seq(KeyField(0))) => BoolValue(
      IntervalsSet(Interval(endpoint(null, -1), posInf)),
      IntervalsSet(Interval(negInf, endpoint(null, -1))),
      KeySet.bottom)
    case (IsNA(_), Seq(b: BoolValue)) => BoolValue.isNA(b)
    // collection contains
    case (ApplyIR("contains", _, _, _), Seq(ConstantValue(collectionVal), queryVal)) if literalSizeOkay(collectionVal) =>
      if (collectionVal == null) {
        BoolValue.allNA
      } else queryVal match {
        case Contig(rgStr) =>
          val rg = ctx.stateManager.referenceGenomes(rgStr)
          val intervals = intervalsFromLiteralContigs(collectionVal, rg)
          BoolValue(intervals, KeySet.complement(intervals), KeySet.bottom)
        case KeyField(0) =>
          val intervals = intervalsFromLiteral(collectionVal, firstKeyOrd.toOrdering, true)
          BoolValue(intervals, KeySet.complement(intervals), KeySet.bottom)
        case struct: StructValue if struct.isKeyPrefix =>
          val intervals = intervalsFromLiteral(collectionVal, keyOrd.toOrdering, false)
          BoolValue(intervals, KeySet.complement(intervals), KeySet.bottom)
        case _ => BoolValue.top
      }
    // interval contains
    case (ApplySpecial("contains", _, _, _, _), Seq(ConstantValue(intervalVal), queryVal)) =>
      (intervalVal: @unchecked) match {
        case null => BoolValue.allNA
        case i: Interval => queryVal match {
          case KeyField(0) =>
            val l = IntervalEndpoint(Row(i.left.point), i.left.sign)
            val r = IntervalEndpoint(Row(i.right.point), i.right.sign)
            BoolValue(
              IntervalsSet(Interval(l, r)),
              IntervalsSet(Interval(negInf, l), Interval(r, endpoint(null, -1))),
              IntervalsSet(Interval(endpoint(null, -1), posInf)))
          case struct: StructValue if struct.isKeyPrefix =>
            BoolValue(
              IntervalsSet(i),
              IntervalsSet(Interval(negInf, i.left), Interval(i.right, posInf)),
              IntervalsSet.empty)
          case _ => BoolValue.top
        }
      }
    case (ApplyComparisonOp(op, _, _), Seq(l, r)) =>
      Lattice.compare(l, r, op)
    case (ApplySpecial("lor", _, _, _, _), Seq(l: BoolValue, r: BoolValue)) =>
      BoolValue.or(l, r)
    case (ApplySpecial("land", _, _, _, _), Seq(l: BoolValue, r: BoolValue)) =>
      BoolValue.and(l, r)
    case (ApplyUnaryPrimOp(Bang, _), Seq(x: BoolValue)) =>
      BoolValue.not(x)
    case _ => BoolValue.top
  }

  case class Rewrites(
    replaceWithTrue: mutable.Set[RefEquality[IR]],
    replaceWithFalse: mutable.Set[RefEquality[IR]])

  private def _analyze(x: IR, env: AbstractEnv, rewrites: Option[Rewrites]): AbstractValue = {
    def recur(x: IR, env: AbstractEnv = env): AbstractValue =
      _analyze(x, env, rewrites)

//    println(s"visiting:\n${Pretty(ctx, x)}")
//    println(s"env: ${env}")
    val res: Lattice.Value = if (env.keySet == KeySet.bottom)
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
        Lattice.combine(
          recur(cnsq, env.restrict(c.trueBound)),
          recur(altr, env.restrict(c.falseBound)))
      case ToStream(a, _) => recur(a)
      case StreamFold(a, zero, accumName, valueName, body) => recur(a) match {
          case ConstantValue(array) => array.asInstanceOf[Iterable[Any]]
              .foldLeft(recur(zero)) { (accum, value) =>
                recur(body, env.bind(accumName -> accum, valueName -> ConstantValue(value, a.typ.asInstanceOf[TStream].elementType)))
              }
          case _ => Lattice.top
        }
      case x@Coalesce(values) =>
        val aVals = values.map(recur(_))
        if (x.typ == TBoolean) {
          val bVals = aVals.asInstanceOf[Seq[BoolValue]]
          val trueBound = bVals.foldRight(KeySet.bottom) { (x, acc) =>
            KeySet.combine(x.trueBound, KeySet.meet(x.naBound, acc))
          }
          val falseBound = bVals.foldRight(KeySet.bottom) { (x, acc) =>
            KeySet.combine(x.falseBound, KeySet.meet(x.naBound, acc))
          }
          val naBound = bVals.foldRight(KeySet.top) { (x, acc) =>
            KeySet.meet(x.naBound, acc)
          }
          BoolValue(trueBound, falseBound, naBound)
        }
        aVals.reduce(Lattice.combine)
      case _ =>
        val children = x.children.map(child => recur(child.asInstanceOf[IR])).toFastIndexedSeq
        val keyOrConstVal = computeKeyOrConst(x, children)
        if (keyOrConstVal == Lattice.top && x.typ == TBoolean)
          computeBoolean(x, children)
        else
          keyOrConstVal
    }
//    println(s"finished visiting:\n${Pretty(ctx, x)}")
//    println(s"result: $res")

    rewrites.foreach { rw =>
      if (x.typ == TBoolean) {
        val bool = res.asInstanceOf[BoolValue]
        if (KeySet.meet(KeySet.combine(bool.falseBound, bool.naBound), env.keySet) == KeySet.bottom)
          rw.replaceWithTrue += RefEquality(x)
        else if (KeySet.meet(KeySet.combine(bool.trueBound, bool.naBound), env.keySet) == KeySet.bottom)
          rw.replaceWithFalse += RefEquality(x)
      }
    }
    res
  }

}
