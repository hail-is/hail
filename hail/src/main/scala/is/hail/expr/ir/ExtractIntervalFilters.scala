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

  private def opIsSupported(op: ComparisonOp[_]): Boolean = op match {
    case _: EQ | _: LTEQ | _: LT | _: GTEQ | _: GT => true
    case _ => false
  }

}

class ExtractIntervalFilters(ctx: ExecuteContext, keyType: TStruct) {
  import ExtractIntervalFilters._
  import KeyFieldOrConstantJoinLattice.{ Constant, Contig, KeyField, Position }

  object StructJoinLattice extends JoinLattice {
    object Value {
      def apply(fields: Traversable[(String, Lattice.Value)]): Value =
        ConcreteValue(fields.filter { case (_, value) => value != Lattice.top }.toMap)
    }
    abstract class Value {
      def apply(field: String): Lattice.Value
      def isKeyPrefix: Boolean
    }

    private case class ConcreteValue(fields: Map[String, Lattice.Value]) extends Value {
      def apply(field: String): Lattice.Value = fields.getOrElse(field, Lattice.top)
      def values: Iterable[Lattice.Value] = fields.values

      def isKeyPrefix: Boolean = fields.values.zipWithIndex.forall {
        case (KeyOrConstant(KeyField(i1)), i2) => i1 == i2
        case _ => false
      }
    }

    private case class ConstantValue(fields: Row, typ: TStruct) extends Value {
      def apply(field: String): Lattice.Value = Lattice(Constant(fields(typ.field(field).index)))
      def isKeyPrefix: Boolean = false
    }

    def top: Value = ConcreteValue(Map.empty)

    def combine(l: Value, r: Value): Value = (l, r) match {
      case (ConcreteValue(l), ConcreteValue(r)) =>
        ConcreteValue(l.keySet.intersect(r.keySet).view.map {
          f => f -> Lattice.combine(l(f), r(f))
        }.toMap)
      case (ConstantValue(l, t), ConstantValue(r, _)) if l == r => ConstantValue(l, t)
      case _ => top
    }
  }

  object KeyPredicateJoinLattice extends JoinLattice {
    case class Value(trueBound: IndexedSeq[Interval])

    def top: Value = Value(FastIndexedSeq(Interval(Row(), Row(), true, true)))
    def allTrue: Value = Value(FastIndexedSeq(Interval(Row(), Row(), true, true)))
    def allFalse: Value = Value(FastIndexedSeq())

    def combine(l: Value, r: Value): Value = Value(Interval.union(l.trueBound ++ r.trueBound, iord))
    def or(l: Value, r: Value): Value = Value(Interval.union(l.trueBound ++ r.trueBound, iord))
    def and(l: Value, r: Value): Value = Value(Interval.intersection(l.trueBound, r.trueBound, iord))
  }

  object Lattice extends AbstractLattice {
    val keyOrConstantLattice = new LatticeFromJoin(KeyFieldOrConstantJoinLattice)
    val keyPredicateLattice = new LatticeFromJoin(KeyPredicateJoinLattice)
    val structLattice = new LatticeFromJoin(StructJoinLattice)

    type KeyOrConstValue = Lattice.keyOrConstantLattice.joinLattice.Value
    type StructValue = Lattice.structLattice.joinLattice.Value
    type BoolValue = Lattice.keyPredicateLattice.joinLattice.Value

    case class Value(
        keyOrConstLatticeValue: keyOrConstantLattice.Value,
        boolLatticeValue: keyPredicateLattice.Value,
        structLatticeValue: structLattice.Value) {
      def keyOrConstValue: KeyOrConstValue = keyOrConstLatticeValue.get

      def boolValue: BoolValue = boolLatticeValue.get

      def structValue: StructValue = structLatticeValue.get
    }

    def apply(x: keyOrConstantLattice.joinLattice.Value): Value =
      Value(Some(x), keyPredicateLattice.bottom, structLattice.bottom)

    def apply(x: keyPredicateLattice.joinLattice.Value, const: KeyOrConstValue = keyOrConstantLattice.joinLattice.top): Value =
      Value(Some(const), Some(x), structLattice.bottom)

    def apply(x: structLattice.joinLattice.Value): Value =
      Value(keyOrConstantLattice.top, keyPredicateLattice.bottom, Some(x))

    def top: Value = Value(keyOrConstantLattice.top, keyPredicateLattice.top, structLattice.top)

    def bottom: Value =
      Value(keyOrConstantLattice.bottom, keyPredicateLattice.bottom, structLattice.bottom)

    def combine(l: Value, r: Value): Value = Value(
      keyOrConstantLattice.combine(l.keyOrConstLatticeValue, r.keyOrConstLatticeValue),
      keyPredicateLattice.combine(l.boolLatticeValue, r.boolLatticeValue),
      structLattice.combine(l.structLatticeValue, r.structLatticeValue)
    )
  }

  import Lattice.{ BoolValue, KeyOrConstValue, StructValue, Value => AbstractValue }

  val iord: IntervalEndpointOrdering = PartitionBoundOrdering(ctx, keyType).intervalEndpointOrdering

  object KeyOrConstant {
    def unapply(x: AbstractValue): Option[KeyOrConstValue] = x.keyOrConstLatticeValue
  }

  object StructValue {
    def top: StructValue = StructJoinLattice.top
    def apply(fields: Traversable[(String, Lattice.Value)]): StructValue = StructJoinLattice.Value(fields)

    def unapply(x: AbstractValue): Option[StructValue] = x.structLatticeValue
  }

  object BoolValue {
//    def top: AbstractValue = (
//      Lattice.keyOrConstantLattice.bottom,
//      Lattice.keyPredicateLattice.top,
//      Lattice.structLattice.bottom)
    def top: BoolValue = KeyPredicateJoinLattice.top
    def allTrue: BoolValue = KeyPredicateJoinLattice.allTrue
    def allFalse: BoolValue = KeyPredicateJoinLattice.allFalse

    def apply(intervals: IndexedSeq[Interval]): BoolValue = KeyPredicateJoinLattice
      .Value(intervals)

    def unapply(x: AbstractValue): Option[BoolValue] = x.boolLatticeValue
  }

  def compareWithConstant(l: Any, r: AbstractValue, op: ComparisonOp[_]): BoolValue = {
    r match {
      case KeyOrConstant(KeyField(x)) =>
        // simple key comparison
        BoolValue(Array(intervalFromComparison(l, op)))
      case KeyOrConstant(Contig(rgStr)) =>
        // locus contig comparison
        val intervals =
          Array(getIntervalFromContig(l.asInstanceOf[String], ctx.getReference(rgStr))).flatten
        BoolValue(intervals)
      case KeyOrConstant(Position(rgStr)) =>
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
      case StructValue(s) if s.isKeyPrefix =>
        assert(op.isInstanceOf[EQ])
        BoolValue(Array(Interval(endpoint(l, -1, wrapped = false), endpoint(l, 1, wrapped = false))))
      case _ => BoolValue.top
    }
  }

  def analyze(x: IR, rowName: String): Option[IndexedSeq[Interval]] = {
    val env = Env.empty[AbstractValue].bind(
      rowName,
      Lattice(StructValue(
        Map(keyType.fieldNames.zipWithIndex.map(t => t._1 -> Lattice(KeyField(t._2))): _*))))
    val BoolValue(bool) = _analyze(x, env)
    if (bool.trueBound.length == 1 && bool.trueBound(0) == Interval(Row(), Row(), true, true)) None
    else Some(bool.trueBound)
  }

  private def computeKeyOrConst(x: IR, children: IndexedSeq[KeyOrConstValue]): KeyOrConstValue = x match {
    case False() => Constant(false)
    case True() => Constant(true)
    case I32(v) => Constant(v)
    case I64(v) => Constant(v)
    case F32(v) => Constant(v)
    case F64(v) => Constant(v)
    case Str(v) => Constant(v)
    case Literal(_, value) => Constant(value)
    case ApplySpecial("lor", _, _, _, _) => children match {
      case Seq(Constant(l: Boolean), Constant(r: Boolean)) => Constant(l || r)
      case _ => Lattice.keyOrConstantLattice.joinLattice.top
    }
    case ApplySpecial("land", _, _, _, _) => children match {
      case Seq(Constant(l: Boolean), Constant(r: Boolean)) => Constant(l && r)
      case _ => Lattice.keyOrConstantLattice.joinLattice.top
    }
    case Apply("contig", _, Seq(k), _, _) => children match {
      case Seq(KeyField(0)) => Contig(k.typ.asInstanceOf[TLocus].rg)
      case _ => Lattice.keyOrConstantLattice.joinLattice.top
    }
    case Apply("position", _, Seq(k), _, _) => children match {
      case Seq(KeyField(0)) => Position(k.typ.asInstanceOf[TLocus].rg)
      case _ => Lattice.keyOrConstantLattice.joinLattice.top
    }
    case _ => Lattice.keyOrConstantLattice.joinLattice.top
  }

  private def computeBoolean(x: IR, children: IndexedSeq[AbstractValue]): BoolValue = (x, children) match {
    case (False(), _) => BoolValue.allFalse
    case (True(), _) => BoolValue.allTrue
    // collection contains
    case (ApplyIR("contains", _, _, _), Seq(KeyOrConstant(Constant(collectionVal)), queryVal)) if literalSizeOkay(collectionVal) =>
      queryVal match {
        case KeyOrConstant(Contig(rgStr)) =>
          val rg = ctx.stateManager.referenceGenomes(rgStr)
          BoolValue(intervalsFromLiteralContigs(collectionVal, rg))
        case KeyOrConstant(KeyField(0)) =>
          BoolValue(intervalsFromLiteral(collectionVal, true))
        case StructValue(struct) if struct.isKeyPrefix =>
          BoolValue(intervalsFromLiteral(collectionVal, false))
        case _ => BoolValue.top
      }
    // interval contains
    case (ApplySpecial("contains", _, _, _, _), Seq(KeyOrConstant(Constant(intervalVal)), queryVal)) =>
      (intervalVal: @unchecked) match {
        case null => BoolValue.allFalse
        case i: Interval => queryVal match {
          case KeyOrConstant(KeyField(0)) => BoolValue(wrapInRow(Array(i)))
          case StructValue(struct) if struct.isKeyPrefix => BoolValue(Array(i))
          case _ => BoolValue.top
        }
      }
    case (ApplyComparisonOp(op, _, _), Seq(l, r)) if opIsSupported(op) => (l, r) match {
      case (KeyOrConstant(Constant(l)), r) => compareWithConstant(l, r, op)
      case (l, KeyOrConstant(Constant(r))) =>
        compareWithConstant(r, l, ComparisonOp.swap(op.asInstanceOf[ComparisonOp[Boolean]]))
      case _ => BoolValue.top
    }
    case (ApplySpecial("lor", _, _, _, _), Seq(l, r)) =>
      KeyPredicateJoinLattice.or(l.boolValue, r.boolValue)
    case (ApplySpecial("land", _, _, _, _), Seq(l, r)) =>
      KeyPredicateJoinLattice.and(l.boolValue, r.boolValue)
    case _ => BoolValue.top
  }

  private def _analyze(x: IR, env: Env[AbstractValue]): AbstractValue = {
    def recur(x: IR, env: Env[AbstractValue] = env): AbstractValue = _analyze(x, env)

    println(s"visiting:\n${Pretty(ctx, x)}")
    println(s"env: ${env}")
    val res: Lattice.Value = x match {
      case Let(name, value, body) => recur(body, env.bind(name -> recur(value)))
      case Ref(name, _) => env.lookupOption(name).getOrElse(Lattice.top)
      case GetField(o, name) => recur(o) match { case StructValue(s) => s(name) }
      case MakeStruct(fields) => Lattice(StructValue(fields.map { case (name, field) =>
        name -> recur(field)
      }))
      case SelectFields(old, fields) =>
        val oldVal = recur(old)
        Lattice(StructValue(fields.map(name => name -> oldVal.structValue(name))))
      /* TODO: when we support negation, if result is Boolean, handle like (cond & cnsq) | (~cond &
       * altr) */
      case If(cond, cnsq, altr) => Lattice.combine(recur(cnsq), recur(altr))
      case ToStream(a, _) => recur(a)
      case StreamFold(a, zero, accumName, valueName, body) => recur(a) match {
          case KeyOrConstant(Constant(array)) => array.asInstanceOf[Iterable[Any]]
              .foldLeft(recur(zero)) { (accum, value) =>
                recur(body, env.bind(accumName -> accum, valueName -> Lattice(Constant(value))))
              }
          case _ => Lattice.top
        }
      case Coalesce(values) =>
        values.foldLeft[AbstractValue](Lattice.bottom)((acc, value) => Lattice.combine(acc, recur(value)))
      case _ =>
        val children = x.children.map(child => recur(child.asInstanceOf[IR])).toFastIndexedSeq
        val boolVal = if (x.typ == TBoolean) {
          Some(computeBoolean(x, children))
        } else None
        val keyOrConstVal = computeKeyOrConst(x, children.map(_.keyOrConstValue))
        Lattice.Value(Some(keyOrConstVal), boolVal, None)
    }
    println(s"finished visiting:\n${Pretty(ctx, x)}")
    println(s"result: $res")
    res
  }

}
