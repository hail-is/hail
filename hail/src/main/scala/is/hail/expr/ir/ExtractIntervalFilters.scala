package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.annotations.IntervalEndpointOrdering
import is.hail.rvd.{PartitionBoundOrdering, RVDPartitioner}
import is.hail.types.virtual._
import is.hail.utils.{FastSeq, Interval, IntervalEndpoint, _}
import is.hail.variant.{Locus, ReferenceGenome}
import org.apache.spark.sql.Row

import scala.collection.immutable.SortedMap
import scala.Option.option2Iterable

object ExtractIntervalFilters {

  val MAX_LITERAL_SIZE = 4096

  sealed abstract class AbstractValue {
    def asBool: BoolValue = this.asInstanceOf[BoolValue]

    def asStruct: StructValue = this.asInstanceOf[StructValue]

    def isFirstKey: Boolean = this match {
      case KeyFieldValue(i) => i == 0
      case _ => false
    }

    def merge(other: AbstractValue, iord: IntervalEndpointOrdering): AbstractValue = (this, other) match {
      case (StructValue(l), StructValue(r)) =>
        val fields = l.keySet.intersect(r.keySet).view.map { f =>
          f -> l(f).merge(r(f), iord)
        }.toMap
        StructValue(fields)
      case (l: BoolValue, r: BoolValue) =>
        l.union(r, iord)
      case (ConstantValue(l), ConstantValue(r)) =>
        if (l == r) ConstantValue(l) else OtherValue
      case (KeyFieldValue(l), KeyFieldValue(r)) =>
        if (l == r) KeyFieldValue(l) else OtherValue
      case (ContigValue(l), ContigValue(r)) =>
        if (l == r) ContigValue(l) else OtherValue
      case (PositionValue(l), PositionValue(r)) =>
        if (l == r) PositionValue(l) else OtherValue
      case _ => OtherValue
    }
  }

  final case class StructValue(keyFields: Map[String, AbstractValue]) extends AbstractValue {
    def apply(name: String): AbstractValue = keyFields.getOrElse(name, OtherValue)

    def isKeyPrefix: Boolean = keyFields.values.view.zipWithIndex.forall {
      case (KeyFieldValue(i1), i2) => i1 == i2
      case _ => false
    }
  }

  object StructValue {
    val top: StructValue = StructValue(Map.empty)
  }

  // approximates runtime bool p by intervals, such that if p is true, then key is in intervals
  // equivalently p iff (p and key in intervals)
  final case class BoolValue(intervals: Array[Interval]) extends AbstractValue {
    override def toString: String = s"BoolValue(${intervals.toFastIndexedSeq})"
    def union(other: BoolValue, iord: IntervalEndpointOrdering): BoolValue =
      BoolValue(Interval.union(intervals ++ other.intervals, iord))
    def intersection(other: BoolValue, iord: IntervalEndpointOrdering): BoolValue = {
      log.info(s"intersecting list of ${intervals.length} intervals with list of ${other.intervals.length} intervals")
      val intersection = Interval.intersection(intervals, other.intervals, iord)
      log.info(s"intersect generated ${intersection.length} intersected intervals")
      BoolValue(intersection)
    }
  }

  object BoolValue {
    // interval containing all keys
    val top: BoolValue = BoolValue(Array(Interval(Row(), Row(), true, true)))
    val bottom: BoolValue = BoolValue(Array())
  }
  final case class ConstantValue(value: Any) extends AbstractValue

  final case class KeyFieldValue(idx: Int) extends AbstractValue

  final case class ContigValue(rg: String) extends AbstractValue

  final case class PositionValue(rg: String) extends AbstractValue

  final case object OtherValue extends AbstractValue

  private def intervalsFromLiteral(lit: Any, wrapped: Boolean): Array[Interval] = {
    (lit: @unchecked) match {
      case x: Map[_, _] => intervalsFromCollection(x.keys, wrapped)
      case x: Traversable[_] => intervalsFromCollection(x, wrapped)
    }
  }
  private def intervalsFromCollection(lit: Traversable[Any], wrapped: Boolean): Array[Interval] = {
    lit.map { elt =>
      Interval(endpoint(elt, -1, wrapped), endpoint(elt, 1, wrapped))
    }.toArray
  }
  private def intervalsFromLiteralContigs(contigs: Any, rg: ReferenceGenome): Array[Interval] = {
    (contigs: @unchecked) match {
      case x: Map[_, _] => x.keys.flatMap(c => getIntervalFromContig(c.asInstanceOf[String], rg)).toArray
      case x: Traversable[_] => x.flatMap(c => getIntervalFromContig(c.asInstanceOf[String], rg)).toArray
    }
  }

  private def literalSizeOkay(lit: Any): Boolean = lit.asInstanceOf[Iterable[_]].size <= MAX_LITERAL_SIZE

  private def wrapInRow(intervals: Array[Interval]): Array[Interval] = {
    intervals.map { interval =>
      Interval(IntervalEndpoint(Row(interval.left.point), interval.left.sign),
        IntervalEndpoint(Row(interval.right.point), interval.right.sign))
    }
  }

  def endpoint(value: Any, sign: Int, wrapped: Boolean = true): IntervalEndpoint = {
    IntervalEndpoint(if (wrapped) Row(value) else value, sign)
  }

  private def posInf: IntervalEndpoint = IntervalEndpoint(Row(), 1)
  private def negInf: IntervalEndpoint = IntervalEndpoint(Row(), -1)

  private def getIntervalFromContig(c: String, rg: ReferenceGenome): Option[Interval] = {
    if (rg.contigsSet.contains(c)) {
      Some(Interval(
        endpoint(Locus(c, 1), -1),
        endpoint(Locus(c, rg.contigLength(c)), -1)))
    } else {
      warn(s"Filtered with contig '${c}', but '${c}' is not a valid contig in reference genome ${rg.name}")
      None
    }
  }

  private def intervalFromComparison(v: Any, op: ComparisonOp[_]): Interval = {
    (op: @unchecked) match {
      case _: EQ =>
        Interval(endpoint(v, -1), endpoint(v, 1))
      case GT(_, _) =>
        Interval(negInf, endpoint(v, -1)) // value > key
      case GTEQ(_, _) =>
        Interval(negInf, endpoint(v, 1)) // value >= key
      case LT(_, _) =>
        Interval(endpoint(v, 1), posInf) // value < key
      case LTEQ(_, _) =>
        Interval(endpoint(v, -1), posInf) // value <= key
    }
  }

  private def opIsSupported(op: ComparisonOp[_]): Boolean = {
    op match {
      case _: EQ | _: LTEQ | _: LT | _: GTEQ | _: GT => true
      case _ => false
    }
  }

  def extractPartitionFilters(ctx: ExecuteContext, cond: IR, ref: Ref, key: IndexedSeq[String]): Option[(IR, Array[Interval])] = {
    if (key.isEmpty)
      None
    else {
      val extract = new ExtractIntervalFilters(ctx, ref.typ.asInstanceOf[TStruct])
      extract.analyze(cond, ref.name).map((cond, _))
    }
  }

  def apply(ctx: ExecuteContext, ir0: BaseIR): BaseIR = {
    MapIR.mapBaseIR(ir0, (ir: BaseIR) => {
      (ir match {
        case TableFilter(child, pred) =>
          extractPartitionFilters(ctx, pred, Ref("row", child.typ.rowType), child.typ.key)
            .map { case (newCond, intervals) =>
              log.info(s"generated TableFilterIntervals node with ${ intervals.length } intervals:\n  " +
                s"Intervals: ${ intervals.mkString(", ") }\n  " +
                s"Predicate: ${ Pretty(ctx, pred) }\n " +
                s"Post: ${ Pretty(ctx, newCond) }")
              TableFilter(
                TableFilterIntervals(child, intervals, keep = true),
                newCond)
            }
        case MatrixFilterRows(child, pred) =>
          extractPartitionFilters(ctx, pred, Ref("va", child.typ.rowType), child.typ.rowKey)
            .map { case (newCond, intervals) =>
              log.info(s"generated MatrixFilterIntervals node with ${ intervals.length } intervals:\n  " +
                s"Intervals: ${ intervals.mkString(", ") }\n  " +
                s"Predicate: ${ Pretty(ctx, pred) }\n " +
                s"Post: ${ Pretty(ctx, newCond) }")
              MatrixFilterRows(
                MatrixFilterIntervals(child, intervals, keep = true),
                newCond)
            }

        case _ => None
      }).getOrElse(ir)
    })
  }
}

class ExtractIntervalFilters(ctx: ExecuteContext, keyType: TStruct) {
  import ExtractIntervalFilters._

  val iord: IntervalEndpointOrdering = PartitionBoundOrdering(ctx, keyType).intervalEndpointOrdering

  def compareWithConstant(l: Any, r: AbstractValue, op: ComparisonOp[_]): BoolValue = {
    r match {
      case KeyFieldValue(0) =>
        // simple key comparison
        BoolValue(Array(intervalFromComparison(l, op)))
      case struct: StructValue if struct.isKeyPrefix =>
        assert(op.isInstanceOf[EQ])
        BoolValue(Array(Interval(endpoint(l, -1, wrapped = false), endpoint(l, 1, wrapped = false))))
      case ContigValue(rgStr) =>
        // locus contig comparison
        val intervals = Array(getIntervalFromContig(l.asInstanceOf[String], ctx.getReference(rgStr))).flatten
        BoolValue(intervals)
      case PositionValue(rgStr) =>
        // locus position comparison
        val rg = ctx.getReference(rgStr)
        val ord = PartitionBoundOrdering(ctx, TTuple(TInt32))
        val intervals = rg.contigs.indices
          .flatMap { i =>
            intervalFromComparison(l, op)
              .intersect(
                ord,
                Interval(endpoint(1, -1), endpoint(rg.contigLength(i), -1)))
              .map { interval =>
                Interval(
                  endpoint(Locus(rg.contigs(i), interval.left.point.asInstanceOf[Row].getAs[Int](0)), interval.left.sign),
                  endpoint(Locus(rg.contigs(i), interval.right.point.asInstanceOf[Row].getAs[Int](0)), interval.right.sign))
              }
          }.toArray

        BoolValue(intervals)
      case _ => BoolValue.top
    }
  }

  def analyze(x: IR, rowName: String): Option[Array[Interval]] = {
    val env = Env.empty[AbstractValue].bind(
      rowName,
      StructValue(Map(keyType.fieldNames.zipWithIndex.map(t => t._1 -> KeyFieldValue(t._2)): _*)))
    val BoolValue(intervals) = _analyze(x, env)
    if (intervals.length == 1 && intervals(0) == Interval(Row(), Row(), true, true))
      None
    else {
      Some(intervals)
    }
  }

  private def _analyze(x: IR, env: Env[AbstractValue]): AbstractValue = {
    def recur(x: IR, env: Env[AbstractValue] = env): AbstractValue = _analyze(x, env)

//    println(s"visiting:\n${Pretty(ctx, x)}")
//    println(s"env: ${env}")
    val res = x match {
      case Let(name, value, body) => recur(body, env.bind(name -> recur(value)))
      case Ref(name, _) => env.lookupOption(name).getOrElse(OtherValue)
      case False() => BoolValue.bottom
      case True() => BoolValue.top
      case I32(v) => ConstantValue(v)
      case I64(v) => ConstantValue(v)
      case F32(v) => ConstantValue(v)
      case F64(v) => ConstantValue(v)
      case Str(v) => ConstantValue(v)
      case Literal(_, value) => ConstantValue(value)
      case GetField(o, name) => recur(o).asStruct(name)
      // TODO: when we support negation, if result is Boolean, handle like (cond & cnsq) | (~cond & altr)
      case If(cond, cnsq, altr) => recur(cnsq).merge(recur(altr), iord)
      case ToStream(a, _) => recur(a)
      case Apply("contig", _, Seq(k), _, _) =>
        if (recur(k).isFirstKey) ContigValue(k.typ.asInstanceOf[TLocus].rg) else OtherValue
      case Apply("position", _, Seq(k), _, _) =>
        if (recur(k).isFirstKey) PositionValue(k.typ.asInstanceOf[TLocus].rg) else OtherValue
      case ApplySpecial("lor", _, Seq(l, r), t, _) =>
        recur(l).asBool.union(recur(r).asBool, iord)
      case ApplySpecial("land", _, Seq(l, r), t, _) =>
        recur(l).asBool.intersection(recur(r).asBool, iord)
      case StreamFold(a, zero, accumName, valueName, body) =>
        recur(a) match {
          case ConstantValue(array) =>
            array.asInstanceOf[Iterable[Any]].foldLeft(recur(zero)) { (accum, value) =>
              recur(body, env.bind(accumName -> accum, valueName -> ConstantValue(value)))
            }
          case _ => OtherValue
        }
      case Coalesce(bools) =>
        // if Coalesce is true, then one of the bools must be true, so can conservatively treat like an Or
        val intervals = Interval.union(Array.concat(bools.view.map(recur(_).asBool.intervals): _*), iord)
        BoolValue(intervals)
      // collection contains
      case ApplyIR("contains", _, Seq(collection, query), _) =>
        recur(collection) match {
          case ConstantValue(collectionVal) =>
            val queryVal = recur(query)
            if (literalSizeOkay(collectionVal)) {
              queryVal match {
                case ContigValue(rgStr) =>
                  val rg = ctx.stateManager.referenceGenomes(rgStr)
                  BoolValue(intervalsFromLiteralContigs(collectionVal, rg))
                case KeyFieldValue(0) =>
                  BoolValue(intervalsFromLiteral(collectionVal, true))
                case struct: StructValue if struct.isKeyPrefix =>
                  BoolValue(intervalsFromLiteral(collectionVal, false))
              }
            } else {
              BoolValue.top
            }
          case _ => OtherValue
        }
      // interval contains
      case ApplySpecial("contains", _, Seq(interval, query), _, _) =>
        recur(interval) match {
          case ConstantValue(intervalVal) =>
            val queryVal = recur(query)
            (intervalVal: @unchecked) match {
              case null => BoolValue.bottom
              case i: Interval =>
                queryVal match {
                  case KeyFieldValue(0) =>
                    BoolValue(wrapInRow(Array(i)))
                  case struct: StructValue if struct.isKeyPrefix =>
                    BoolValue(Array(i))
                  case _ =>
                    BoolValue.top
                }
            }
          case _ => OtherValue
        }
      case ApplyComparisonOp(op, l, r) if opIsSupported(op) =>
        (recur(l), recur(r)) match {
          case (ConstantValue(l), r) => compareWithConstant(l, r, op)
          case (l, ConstantValue(r)) => compareWithConstant(r, l, ComparisonOp.swap(op.asInstanceOf[ComparisonOp[Boolean]]))
          case _ => BoolValue.top
        }
      case x if x.typ == TBoolean => BoolValue.top
      case x if x.typ.isInstanceOf[TStruct] => StructValue.top
      case _ => OtherValue
    }
//    println(s"result: $res")
    res
  }
}