package is.hail.expr.ir

import is.hail.annotations.IntervalEndpointOrdering
import is.hail.backend.ExecuteContext
import is.hail.rvd.PartitionBoundOrdering
import is.hail.types.virtual._
import is.hail.utils.{Interval, IntervalEndpoint, _}
import is.hail.variant.{Locus, ReferenceGenome}

import scala.Option.option2Iterable
import org.apache.spark.sql.Row

import scala.collection.mutable

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

  def extractPartitionFilters(ctx: ExecuteContext, cond: IR, ref: Ref, key: IndexedSeq[String])
      : Option[(IR, IndexedSeq[Interval])] = {
    if (key.isEmpty) None
    else {
      val extract = new ExtractIntervalFilters(ctx, ref.typ.asInstanceOf[TStruct])
      extract.analyze(cond, ref.name).map((cond, _))
    }
  }

  private def intervalsFromLiteral(lit: Any, wrapped: Boolean): IndexedSeq[Interval] =
    (lit: @unchecked) match {
      case x: Map[_, _] => intervalsFromCollection(x.keys, wrapped)
      case x: Traversable[_] => intervalsFromCollection(x, wrapped)
    }

  private def intervalsFromCollection(lit: Traversable[Any], wrapped: Boolean): IndexedSeq[Interval] =
    lit.map(elt => Interval(endpoint(elt, -1, wrapped), endpoint(elt, 1, wrapped))).toArray
      .toFastIndexedSeq

  private def intervalsFromLiteralContigs(contigs: Any, rg: ReferenceGenome): IndexedSeq[Interval] = {
    (contigs: @unchecked) match {
      case x: Map[_, _] => x.keys.flatMap(c => getIntervalFromContig(c.asInstanceOf[String], rg))
          .toArray.toFastIndexedSeq
      case x: Traversable[_] => x.flatMap(c => getIntervalFromContig(c.asInstanceOf[String], rg))
          .toArray.toFastIndexedSeq
    }
  }

  private def getIntervalFromContig(c: String, rg: ReferenceGenome): Option[Interval] = {
    if (rg.contigsSet.contains(c))
      Some(Interval(endpoint(Locus(c, 1), -1), endpoint(Locus(c, rg.contigLength(c)), -1)))
    else {
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


}

class ExtractIntervalFilters(ctx: ExecuteContext, keyType: TStruct) {
  import ExtractIntervalFilters._

  object KeySet extends AbstractLattice {
    type Value = IndexedSeq[Interval]

    def top: Value = FastIndexedSeq(Interval(Row(), Row(), true, true))
    def bottom: Value = FastIndexedSeq()
    def combine(l: Value, r: Value): Value = Interval.union(l ++ r, iord)
    def meet(l: Value, r: Value): Value = Interval.intersection(l, r, iord)

    def complement(v: Value): Value = {
      if (v.isEmpty) return top

      val builder = mutable.ArrayBuilder.make[Interval]()
      var i = 0
      if (v.head.left != IntervalEndpoint(Row(), -1)) {
        builder += Interval(IntervalEndpoint(Row(), -1), v.head.left)
      }
      while (i + 1 < v.length) {
        builder += Interval(v(i).right, v(i+1).left)
        i += 1
      }
      if (v.last.right != IntervalEndpoint(Row(), 1)) {
        builder += Interval(v.last.right, IntervalEndpoint(Row(), 1))
      }

      builder.result()
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
        else
          ConcreteBool(trueBound, falseBound, naBound)
      }

      def or(l: BoolValue, r: BoolValue): BoolValue = (l, r) match {
        case (ConstantBool(l), ConstantBool(r)) => ConstantBool(l || r)
        case _ => ConcreteBool(
          KeySet.combine(l.trueBound, r.trueBound),
          KeySet.meet(l.falseBound, r.falseBound),
          KeySet.combine(l.naBound, r.naBound))
      }

      def and(l: BoolValue, r: BoolValue): BoolValue = (l, r) match {
        case (ConstantBool(l), ConstantBool(r)) => ConstantBool(l && r)
        case _ => ConcreteBool(
          KeySet.meet(l.trueBound, r.trueBound),
          KeySet.combine(l.falseBound, r.falseBound),
          KeySet.combine(l.naBound, r.naBound))
      }

      def not(x: BoolValue): BoolValue = x match {
        case ConstantBool(x) => ConstantBool(!x)
        case _ => ConcreteBool(x.falseBound, x.trueBound, x.naBound)
      }

      def fromComparison(v: Any, op: ComparisonOp[_], wrapped: Boolean = true): BoolValue = {
        (op: @unchecked) match {
          case _: EQ => BoolValue( // value == key
            FastIndexedSeq(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
            FastIndexedSeq(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
            FastIndexedSeq(Interval(endpoint(null, -1), posInf)))
          case _: NEQ => BoolValue( // value != key
            FastIndexedSeq(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
            FastIndexedSeq(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
            FastIndexedSeq(Interval(endpoint(null, -1), posInf)))
          case _: GT => BoolValue( // value > key
            FastIndexedSeq(Interval(negInf, endpoint(v, -1, wrapped))),
            FastIndexedSeq(Interval(endpoint(v, -1, wrapped), endpoint(null,  -1))),
            FastIndexedSeq(Interval(endpoint(null, -1), posInf)))
          case _: GTEQ => BoolValue( // value >= key
            FastIndexedSeq(Interval(negInf, endpoint(v, 1, wrapped))),
            FastIndexedSeq(Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
            FastIndexedSeq(Interval(endpoint(null, -1), posInf)))
          case _: LT => BoolValue( // value < key
            FastIndexedSeq(Interval(endpoint(v, 1, wrapped), endpoint(null, -1))),
            FastIndexedSeq(Interval(negInf, endpoint(v, 1, wrapped))),
            FastIndexedSeq(Interval(endpoint(null, -1), posInf)))
          case _: LTEQ => BoolValue( // value <= key
            FastIndexedSeq(Interval(endpoint(v, -1, wrapped), endpoint(null, -1))),
            FastIndexedSeq(Interval(negInf, endpoint(v, 1, wrapped))),
            FastIndexedSeq(Interval(endpoint(null, -1), posInf)))
          case _: EQWithNA => BoolValue( // value == key
            FastIndexedSeq(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
            FastIndexedSeq(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), posInf)),
            KeySet.bottom)
          case _: NEQWithNA => BoolValue( // value != key
            FastIndexedSeq(Interval(negInf, endpoint(v, -1, wrapped)), Interval(endpoint(v, 1, wrapped), posInf)),
            FastIndexedSeq(Interval(endpoint(v, -1, wrapped), endpoint(v, 1, wrapped))),
            KeySet.bottom)
        }
      }

      def fromComparisonKeyPrefix(v: Row, op: ComparisonOp[_]): BoolValue = {
        (op: @unchecked) match {
          case _: EQ => BoolValue( // value == key
            FastIndexedSeq(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
            FastIndexedSeq(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            KeySet.bottom)
          case _: NEQ => BoolValue( // value != key
            FastIndexedSeq(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            FastIndexedSeq(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
            KeySet.bottom)
          case _: GT => BoolValue( // value > key
            FastIndexedSeq(Interval(negInf, endpoint(v, -1, false))),
            FastIndexedSeq(Interval(endpoint(v, -1, false), posInf)),
            KeySet.bottom)
          case _: GTEQ => BoolValue( // value >= key
            FastIndexedSeq(Interval(negInf, endpoint(v, 1, false))),
            FastIndexedSeq(Interval(endpoint(v, 1, false), posInf)),
            KeySet.bottom)
          case _: LT => BoolValue( // value < key
            FastIndexedSeq(Interval(endpoint(v, 1, false), posInf)),
            FastIndexedSeq(Interval(negInf, endpoint(v, 1, false))),
            KeySet.bottom)
          case _: LTEQ => BoolValue( // value <= key
            FastIndexedSeq(Interval(endpoint(v, -1, false), posInf)),
            FastIndexedSeq(Interval(negInf, endpoint(v, 1, false))),
            KeySet.bottom)
          case _: EQWithNA => BoolValue( // value == key
            FastIndexedSeq(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
            FastIndexedSeq(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            KeySet.bottom)
          case _: NEQWithNA => BoolValue( // value != key
            FastIndexedSeq(Interval(negInf, endpoint(v, -1, false)), Interval(endpoint(v, 1, false), posInf)),
            FastIndexedSeq(Interval(endpoint(v, -1, false), endpoint(v, 1, false))),
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
      override def trueBound: KeySet.Value = if (value)
        BoolValue.allTrue.trueBound
      else
        BoolValue.allFalse.trueBound

      override def falseBound: KeySet.Value = if (value)
        BoolValue.allTrue.falseBound
      else
        BoolValue.allFalse.falseBound

      override def naBound: KeySet.Value = if (value)
        BoolValue.allTrue.naBound
      else
        BoolValue.allFalse.naBound
    }

    private case class KeyFieldBool(idx: Int) extends BoolValue with KeyField {
      override def trueBound: KeySet.Value = if (idx == 0)
        FastIndexedSeq(Interval(true, true, includesStart = true, includesEnd = true))
      else
        BoolValue.top.trueBound

      override def falseBound: KeySet.Value = if (idx == 0)
        FastIndexedSeq(Interval(false, false, includesStart = true, includesEnd = true))
      else
        BoolValue.top.trueBound

      override def naBound: KeySet.Value = if (idx == 0)
        FastIndexedSeq(Interval(null, null, includesStart = true, includesEnd = true))
      else
        BoolValue.top.trueBound
    }

    private case class KeyFieldStruct(idx: Int) extends StructValue with KeyField {
      def apply(field: String): Value = Other
      def values: Iterable[Value] = Iterable.empty
      def isKeyPrefix: Boolean = false
    }

    case object Other extends StructValue with BoolValue {
      def apply(field: String): Value = Other
      def values: Iterable[Value] = Iterable.empty
      def isKeyPrefix: Boolean = false
      override def trueBound: IndexedSeq[Interval] = BoolValue.top.trueBound
      override def falseBound: IndexedSeq[Interval] = BoolValue.top.falseBound
      override def naBound: IndexedSeq[Interval] = BoolValue.top.naBound
    }

    def top: StructValue with BoolValue = Other

    def combine(l: Value, r: Value): Value = (l, r) match {
      case (l: ConstantValue, r: ConstantValue) if l.value == r.value => l
      case (l: KeyField, r: KeyField) if l.idx == r.idx => l
      case (l: Contig, r: Contig) =>
        assert(l.rg == r.rg)
        l
      case (l: Position, r: Position) =>
        assert(l.rg == r.rg)
        l
      case (ConcreteStruct(l), ConcreteStruct(r)) =>
        ConcreteStruct(l.keySet.intersect(r.keySet).view.map {
          f => f -> Lattice.combine(l(f), r(f))
        }.toMap)
      case (l: BoolValue, r: BoolValue) => ConcreteBool(
        KeySet.combine(l.trueBound, r.trueBound),
        KeySet.combine(l.falseBound, r.falseBound),
        KeySet.combine(l.naBound, r.naBound))
      case _ => Other
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
          val intervals =
            Array(getIntervalFromContig(l.asInstanceOf[String], ctx.getReference(rgStr))).flatten
          BoolValue(intervals, KeySet.complement(intervals), KeySet.bottom)
        case Position(rgStr) =>
          // locus position comparison
          val rg = ctx.getReference(rgStr)
          val ord = PartitionBoundOrdering(ctx, TTuple(TInt32))
          val posBoolValue = BoolValue.fromComparison(l, op)
          def liftPosInterval(pos: Interval): Iterable[Interval] = {
            rg.contigs.indices.flatMap { cont =>
              pos.intersect(ord, Interval(endpoint(1, -1), endpoint(rg.contigLength(cont), -1)))
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
          BoolValue(
            posBoolValue.trueBound.flatMap(liftPosInterval),
            posBoolValue.falseBound.flatMap(liftPosInterval),
            posBoolValue.naBound.flatMap(liftPosInterval))
        case s: StructValue if s.isKeyPrefix =>
          BoolValue.fromComparisonKeyPrefix(l.asInstanceOf[Row], op)
        case _ => top
      }
    }

    private def opIsSupported(op: ComparisonOp[_]): Boolean = op match {
      case _: EQ | _: LTEQ | _: LT | _: GTEQ | _: GT | _: EQWithNA | _: NEQWithNA => true
      case _ => false
    }
  }

  import Lattice.{ Value => AbstractValue, ConstantValue, KeyField, StructValue, BoolValue, Contig, Position }

  val iord: IntervalEndpointOrdering = PartitionBoundOrdering(ctx, keyType).intervalEndpointOrdering

  def analyze(x: IR, rowName: String): Option[IndexedSeq[Interval]] = {
    val env = Env.empty[AbstractValue].bind(
      rowName,
      StructValue(
        Map(keyType.fieldNames.zipWithIndex.map(t => t._1 -> KeyField(t._2)): _*)))
    val bool = _analyze(x, env).asInstanceOf[BoolValue]
    if (bool.trueBound.length == 1 && bool.trueBound(0) == Interval(Row(), Row(), true, true)) None
    else Some(bool.trueBound)
  }

  private def computeKeyOrConst(x: IR, children: IndexedSeq[AbstractValue]): AbstractValue = x match {
    case False() => ConstantValue(false, TBoolean)
    case True() => ConstantValue(true, TBoolean)
    case I32(v) => ConstantValue(v, x.typ)
    case I64(v) => ConstantValue(v, x.typ)
    case F32(v) => ConstantValue(v, x.typ)
    case F64(v) => ConstantValue(v, x.typ)
    case Str(v) => ConstantValue(v, x.typ)
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
          val intervals = intervalsFromLiteral(collectionVal, true)
          BoolValue(intervals, KeySet.complement(intervals), KeySet.bottom)
        case struct: StructValue if struct.isKeyPrefix =>
          val intervals = intervalsFromLiteral(collectionVal, false)
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
              Array(Interval(l, r)),
              Array(Interval(negInf, l), Interval(r, endpoint(null, -1))),
              Array(Interval(endpoint(null, -1), endpoint(null, 1))))
          case struct: StructValue if struct.isKeyPrefix =>
            val intervals = Array(i)
            BoolValue(intervals, KeySet.complement(intervals), KeySet.bottom)
          case _ => BoolValue.top
        }
      }
    case (ApplyComparisonOp(op, _, _), Seq(l, r)) =>
      Lattice.compare(l, r, op)
    case (ApplySpecial("lor", _, _, _, _), Seq(l: BoolValue, r: BoolValue)) =>
      BoolValue.or(l, r)
    case (ApplySpecial("land", _, _, _, _), Seq(l: BoolValue, r: BoolValue)) =>
      BoolValue.and(l, r)
    case (ApplyUnaryPrimOp(Bang(), _), Seq(x: BoolValue)) =>
      BoolValue.not(x)
    case _ => BoolValue.top
  }

  private def _analyze(x: IR, env: Env[AbstractValue]): AbstractValue = {
    def recur(x: IR, env: Env[AbstractValue] = env): AbstractValue = _analyze(x, env)

    println(s"visiting:\n${Pretty(ctx, x)}")
    println(s"env: ${env}")
    val res: Lattice.Value = x match {
      case Let(name, value, body) => recur(body, env.bind(name -> recur(value)))
      case Ref(name, _) => env.lookupOption(name).getOrElse(Lattice.top)
      case GetField(o, name) => recur(o).asInstanceOf[StructValue](name)
      case MakeStruct(fields) => StructValue(fields.view.map { case (name, field) =>
        name -> recur(field)
      })
      case SelectFields(old, fields) =>
        val oldVal = recur(old)
        StructValue(fields.view.map(name => name -> oldVal.asInstanceOf[StructValue](name)))
      /* TODO: when we support negation, if result is Boolean, handle like (cond & cnsq) | (~cond &
       * altr) */
      case x@If(cond, cnsq, altr) =>
        if (x.typ == TBoolean) {
          val c = recur(cond).asInstanceOf[BoolValue]
          BoolValue.or(
            BoolValue.and(c, recur(cnsq).asInstanceOf[BoolValue]),
            BoolValue.and(BoolValue.not(c), recur(altr).asInstanceOf[BoolValue]))
        } else {
          Lattice.combine(recur(cnsq), recur(altr))
        }
      case ToStream(a, _) => recur(a)
      case StreamFold(a, zero, accumName, valueName, body) => recur(a) match {
          case ConstantValue(array) => array.asInstanceOf[Iterable[Any]]
              .foldLeft(recur(zero)) { (accum, value) =>
                recur(body, env.bind(accumName -> accum, valueName -> ConstantValue(value, a.typ.asInstanceOf[TContainer].elementType)))
              }
          case _ => Lattice.top
        }
      case Coalesce(values) =>
        val first = recur(values.head)
        values.tail.foldLeft[AbstractValue](first)((acc, value) => Lattice.combine(acc, recur(value)))
      case _ =>
        val children = x.children.map(child => recur(child.asInstanceOf[IR])).toFastIndexedSeq
        val keyOrConstVal = computeKeyOrConst(x, children)
        if (keyOrConstVal == Lattice.top)
          computeBoolean(x, children)
        else
          keyOrConstVal
    }
    println(s"finished visiting:\n${Pretty(ctx, x)}")
    println(s"result: $res")
    res
  }

}
