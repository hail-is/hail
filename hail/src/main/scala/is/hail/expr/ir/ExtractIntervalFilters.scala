package is.hail.expr.ir

import is.hail.annotations.IntervalEndpointOrdering
import is.hail.backend.ExecuteContext
import is.hail.rvd.PartitionBoundOrdering
import is.hail.types.virtual._
import is.hail.utils.{Interval, IntervalEndpoint, _}
import is.hail.variant.{Locus, ReferenceGenome}

import scala.Option.option2Iterable
import org.apache.spark.sql.Row

trait JoinLattice {
  type Value
  def top: Value
  def combine(l: Value, r: Value): Value
}

object KeyFieldOrConstantJoinLattice extends JoinLattice {
  abstract class Value
  case class Constant(value: Any) extends Value
  case class KeyField(idx: Int) extends Value
  case class Contig(rg: String) extends Value
  case class Position(rg: String) extends Value
  case object Other extends Value

  def top: Value = Other

  def combine(l: Value, r: Value): Value = (l, r) match {
    case (Constant(l), Constant(r)) if l == r => Constant(l)
    case (KeyField(l), KeyField(r)) if l == r => KeyField(l)
    case (Contig(l), Contig(r)) =>
      assert(l == r)
      Contig(l)
    case (Position(l), Position(r)) =>
      assert(l == r)
      Position(l)
    case _ => Other
  }
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

  object Lattice extends JoinLattice {
    abstract class Value

    object ConstantValue {
      def apply(v: Any, t: Type): ConstantValue = t match {
        case t: TStruct => ConstantStruct(v.asInstanceOf[Row], t)
        case TBoolean => ConstantBool(v.asInstanceOf[Boolean])
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
      def apply(idx: Int): KeyField = ConcreteKeyField(idx)
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
      def apply(fields: Map[String, Value]): StructValue = ConcreteStruct(fields)
      def apply(fields: Iterable[(String, Value)]): StructValue = ConcreteStruct(fields.toMap)
    }
    trait StructValue extends Value {
      def apply(field: String): Value
      def values: Iterable[Value]
      def isKeyPrefix: Boolean

//      def combine(other: StructValue): StructValue
    }

    private case class ConcreteStruct(fields: Map[String, Value]) extends StructValue {
      def apply(field: String): Value = fields.getOrElse(field, top)

      def values: Iterable[Value] = fields.values

      def isKeyPrefix: Boolean = fields.values.view.zipWithIndex.forall {
        case (f: KeyField, i2) => f.idx == i2
        case _ => false
      }

//      def combine(other: StructValue): StructValue = other match {
//        case ConcreteStruct(rFields) =>
//          ConcreteStruct(fields.keySet.intersect(rFields.keySet).view.map {
//            f => f -> Lattice.combine(fields(f), rFields(f))
//          }.toMap)
//        case _ => Other
//      }
    }

    object BoolValue {
      val allTrue: BoolValue = ConcreteBool(FastIndexedSeq(Interval(Row(), Row(), true, true)))

      val allFalse: BoolValue = ConcreteBool(FastIndexedSeq())
      val top: BoolValue = ConcreteBool(FastIndexedSeq(Interval(Row(), Row(), true, true)))

      def apply(trueBound: IndexedSeq[Interval]): BoolValue = ConcreteBool(trueBound)

      def or(l: BoolValue, r: BoolValue): BoolValue = (l, r) match {
        case (ConstantBool(l), ConstantBool(r)) => ConstantBool(l || r)
        case _ => ConcreteBool(Interval.union(l.trueBound ++ r.trueBound, iord))
      }

      def and(l: BoolValue, r: BoolValue): BoolValue = (l, r) match {
        case (ConstantBool(l), ConstantBool(r)) => ConstantBool(l && r)
        case _ => ConcreteBool(Interval.intersection(l.trueBound, r.trueBound, iord))
      }
    }

    trait BoolValue extends Value {
      def trueBound: IndexedSeq[Interval]
    }

    private case class ConcreteBool(trueBound: IndexedSeq[Interval]) extends BoolValue

    private case class ConstantStruct(value: Row, t: TStruct) extends StructValue with ConstantValue {
      def apply(field: String): Value = this(t.field(field))
      def values: Iterable[Value] = t.fields.map(apply)
      private def apply(field: Field): ConstantValue = ConstantValue(value(field.index), field.typ)
      def isKeyPrefix: Boolean = false
    }

    private case class ConstantBool(value: Boolean) extends BoolValue with ConstantValue {
      override def trueBound: IndexedSeq[Interval] = if (value)
        BoolValue.allTrue.trueBound
      else
        BoolValue.allFalse.trueBound
    }

    private case class KeyFieldBool(idx: Int) extends BoolValue with KeyField {
      override def trueBound: IndexedSeq[Interval] = if (idx == 0)
        FastIndexedSeq(Interval(true, true, includesStart = true, includesEnd = true))
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
//      def combine(other: StructValue): StructValue = Other
      override def trueBound: IndexedSeq[Interval] = BoolValue.top.trueBound
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
      case (ConcreteBool(l), ConcreteBool(r)) => ConcreteBool(Interval.union(l ++ r, iord))
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
      r match {
        case r: KeyField if r.idx == 0 =>
          // simple key comparison
          BoolValue(Array(intervalFromComparison(l, op)))
        case Contig(rgStr) =>
          // locus contig comparison
          val intervals =
            Array(getIntervalFromContig(l.asInstanceOf[String], ctx.getReference(rgStr))).flatten
          BoolValue(intervals)
        case Position(rgStr) =>
          // locus position comparison
          val rg = ctx.getReference(rgStr)
          val ord = PartitionBoundOrdering(ctx, TTuple(TInt32))
          val intervals = rg.contigs.indices.flatMap { i =>
            intervalFromComparison(l, op)
              .intersect(ord, Interval(endpoint(1, -1), endpoint(rg.contigLength(i), -1)))
              .map { interval =>
                Interval(
                  endpoint(
                    Locus(rg.contigs(i), interval.left.point.asInstanceOf[Row].getAs[Int](0)),
                    interval.left.sign),
                  endpoint(
                    Locus(rg.contigs(i), interval.right.point.asInstanceOf[Row].getAs[Int](0)),
                    interval.right.sign)
                  )
              }
          }.toArray

          BoolValue(intervals)
        case s: StructValue if s.isKeyPrefix =>
          assert(op.isInstanceOf[EQ])
          BoolValue(Array(Interval(endpoint(l, -1, wrapped = false), endpoint(l, 1, wrapped = false))))
        case _ => BoolValue.top
      }
    }

    private def opIsSupported(op: ComparisonOp[_]): Boolean = op match {
      case _: EQ | _: LTEQ | _: LT | _: GTEQ | _: GT => true
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
      queryVal match {
        case Contig(rgStr) =>
          val rg = ctx.stateManager.referenceGenomes(rgStr)
          BoolValue(intervalsFromLiteralContigs(collectionVal, rg))
        case KeyField(0) =>
          BoolValue(intervalsFromLiteral(collectionVal, true))
        case struct: StructValue if struct.isKeyPrefix =>
          BoolValue(intervalsFromLiteral(collectionVal, false))
        case _ => BoolValue.top
      }
    // interval contains
    case (ApplySpecial("contains", _, _, _, _), Seq(ConstantValue(intervalVal), queryVal)) =>
      (intervalVal: @unchecked) match {
        case null => BoolValue.allFalse
        case i: Interval => queryVal match {
          case KeyField(0) => BoolValue(wrapInRow(Array(i)))
          case struct: StructValue if struct.isKeyPrefix => BoolValue(Array(i))
          case _ => BoolValue.top
        }
      }
    case (ApplyComparisonOp(op, _, _), Seq(l, r)) =>
      Lattice.compare(l, r, op)
    case (ApplySpecial("lor", _, _, _, _), Seq(l: BoolValue, r: BoolValue)) =>
      BoolValue.or(l, r)
    case (ApplySpecial("land", _, _, _, _), Seq(l: BoolValue, r: BoolValue)) =>
      BoolValue.and(l, r)
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
      case If(cond, cnsq, altr) => Lattice.combine(recur(cnsq), recur(altr))
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
