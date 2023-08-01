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

  abstract class AbstractValue {
    def asBool: BoolValue = this.asInstanceOf[BoolValue]
    def isFirstKey: Boolean = this match {
      case KeyFieldValue(i) => i == 0
      case _ => false
    }
  }
  final case class StructValue(keyFields: SortedMap[String, AbstractValue]) extends AbstractValue {
    def isKeyPrefix: Boolean = keyFields.values.view.zipWithIndex.forall {
      case (KeyFieldValue(i1), i2) => i1 == i2
      case _ => false
    }
  }
  // approximates runtime bool p by intervals, such that if p is true, then key is in intervals
  // equivalently p iff (p and key in intervals)
  final case class BoolValue(intervals: Array[Interval]) extends AbstractValue
  object BoolValue {
    // interval containing all keys
    val top: BoolValue = BoolValue(Array(Interval(Row(), Row(), true, true)))
  }
  final case class KeyFieldValue(idx: Int) extends AbstractValue
  final case class ContigValue(rg: String) extends AbstractValue
  final case class PositionValue(rg: String) extends AbstractValue
  final case object OtherValue extends AbstractValue

  def intervalsFromLiteral(lit: Literal, wrapped: Boolean): Array[Interval] = {
    (lit.value: @unchecked) match {
      case x: Map[_, _] => intervalsFromLiteral(x.keys, wrapped)
      case x: Traversable[_] => intervalsFromLiteral(x, wrapped)
    }
  }
  def intervalsFromLiteral(lit: Traversable[Any], wrapped: Boolean): Array[Interval] = {
    lit.map { elt =>
      Interval(endpoint(elt, -1, wrapped), endpoint(elt, 1, wrapped))
    }.toArray
  }
  def intervalsFromLiteralContigs(contigs: Literal, rg: ReferenceGenome): Array[Interval] = {
    (contigs.value: @unchecked) match {
      case x: Map[_, _] => x.keys.flatMap(c => getIntervalFromContig(c.asInstanceOf[String], rg)).toArray
      case x: Traversable[_] => x.flatMap(c => getIntervalFromContig(c.asInstanceOf[String], rg)).toArray
    }
  }

  def analyze(x: IR, env: Env[AbstractValue], ctx: ExecuteContext, iord: IntervalEndpointOrdering): AbstractValue = {
    def recur(x: IR, env: Env[AbstractValue] = env): AbstractValue = analyze(x, env, ctx, iord)

    x match {
      case False() =>
        BoolValue(Array())
      case True() =>
        // the interval containing all keys
        BoolValue(Array(Interval(Row(), Row(), true, true)))
      case Apply("contig", _, Seq(k), _, _) =>
        if (recur(k).isFirstKey) ContigValue(k.typ.asInstanceOf[TLocus].rg) else OtherValue
      case Apply("position", _, Seq(k), _, _) =>
        if (recur(k).isFirstKey) PositionValue(k.typ.asInstanceOf[TLocus].rg) else OtherValue
      case ApplySpecial("lor", _, Seq(l, r), t, _) =>
        val ll = recur(l).asBool.intervals
        val rr = recur(r).asBool.intervals
        val union = Interval.union(ll ++ rr, iord)
        BoolValue(union)
      case ApplySpecial("land", _, Seq(l, r), t, _) =>
        val ll = recur(l).asBool.intervals
        val rr = recur(r).asBool.intervals
        log.info(s"intersecting list of ${ll.length} intervals with list of ${rr.length} intervals")
        val intersection = Interval.intersection(ll, rr, iord)
        log.info(s"intersect generated ${intersection.length} intersected intervals")
        BoolValue(intersection)
      case StreamFold(
        ToStream(lit: Literal, _), False(), acc, value,
          ApplySpecial(
            "lor", _,
            Seq(
              Ref(acc2, _),
              ApplySpecial("contains", _, Seq(Ref(value2, _), k), _, _)
            ),
            _, _)) =>
        assert(lit.typ.asInstanceOf[TContainer].elementType.isInstanceOf[TInterval])
        if (acc == acc2 && value == value2 && recur(k).isFirstKey) {
          val intervals = Interval.union(
            constValue(lit).asInstanceOf[Iterable[_]]
              .filter(_ != null)
              .map { v =>
                val i = v.asInstanceOf[Interval]
                Interval(
                  IntervalEndpoint(Row(i.left.point), i.left.sign),
                  IntervalEndpoint(Row(i.right.point), i.right.sign))
              }.toArray,
            iord)
          BoolValue(intervals)
        } else BoolValue(Array.empty)
      case Coalesce(bools) =>
        // if Coalesce is true, then one of the bools must be true, so can conservatively treat like an Or
        val intervals = Interval.union(Array.concat(bools.view.map(recur(_).asBool.intervals): _*), iord)
        BoolValue(intervals)
      // collection contains
      case ApplyIR("contains", _, Seq(lit: Literal, query), _) =>
        val queryVal = recur(query)
        if (literalSizeOkay(lit)) {
          queryVal match {
            case ContigValue(rgStr) =>
              val rg = ctx.stateManager.referenceGenomes(rgStr)
              BoolValue(intervalsFromLiteralContigs(lit, rg))
            case KeyFieldValue(0) =>
              BoolValue(intervalsFromLiteral(lit, true))
            case struct: StructValue if struct.isKeyPrefix =>
              BoolValue(intervalsFromLiteral(lit, false))
          }
        } else {
          BoolValue.top
        }
      // interval contains
      case ApplySpecial("contains", _, Seq(lit: Literal, query), _, _) =>
        val queryVal = recur(query)
        (lit.value: @unchecked) match {
          case null => BoolValue(Array())
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
      case ApplyComparisonOp(op, l, r) if opIsSupported(op) =>
        if (!IsConstant(l) && !IsConstant(r)) {
          BoolValue.top
        } else {
          val (const, k, flipped) = if (IsConstant(l)) (l, r, false) else (r, l, true)
          recur(k) match {
            case KeyFieldValue(0) =>
              // simple key comparison
              BoolValue(Array(openInterval(constValue(const), const.typ, op, ctx, flipped)))
            case struct: StructValue if struct.isKeyPrefix =>
              assert(op.isInstanceOf[EQ])
              val c = constValue(const)
              BoolValue(Array(Interval(endpoint(c, -1, wrapped = false), endpoint(c, 1, wrapped = false))))
            case ContigValue(rgStr) =>
              // locus contig comparison
              val intervals = (constValue(const): @unchecked) match {
                case s: String =>
                  Array(getIntervalFromContig(s, ctx.getReference(rgStr))).flatten
              }
              BoolValue(intervals)
            case PositionValue(rgStr) =>
              // locus position comparison
              val pos = constValue(const).asInstanceOf[Int]
              val rg = ctx.getReference(rgStr)
              val ord = TTuple(TInt32).ordering(ctx.stateManager)
              val intervals = rg.contigs.indices
                .flatMap { i =>
                  openInterval(pos, TInt32, op, ctx, flipped)
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
      case Let(name, value, body) =>
        recur(body, env.bind(name -> recur(value)))
      case x if x.typ == TBoolean =>
        BoolValue.top
      case x if x.typ.isInstanceOf[TStruct] =>
        StructValue(SortedMap.empty)
      case _ =>
        OtherValue
    }
  }

  object ExtractionState {
    def apply(
      ctx: ExecuteContext,
      rowRef: Ref,
      keyFields: IndexedSeq[String]
    ): ExtractionState = {
      new ExtractionState(ctx, Env.empty[IR], rowRef, keyFields)
    }
  }

  class ExtractionState(
    val ctx: ExecuteContext,
    env: Env[IR],
    val rowRef: Ref,
    val keyFields: IndexedSeq[String]
  ) {
    val rowType: TStruct = rowRef.typ.asInstanceOf[TStruct]
    val rowKeyType: TStruct = rowType.select(keyFields)._1
    val firstKeyType: Type = rowKeyType.types.head
    val iOrd: IntervalEndpointOrdering = PartitionBoundOrdering(ctx, rowKeyType).intervalEndpointOrdering

    def getReferenceGenome(rg: String): ReferenceGenome = ctx.stateManager.referenceGenomes(rg)

    def bind(name: String, v: IR): ExtractionState =
      new ExtractionState(ctx, env.bind(name, v), rowRef, keyFields)

    private def isRowRef(ir: IR): Boolean = ir match {
      case ref: Ref =>
        ref == rowRef || env.lookupOption(ref.name).exists(isRowRef)
      case _ => false
    }

    private def isKeyField(ir: IR, keyFieldName: String): Boolean = ir match {
      case GetField(struct, name) => fieldIsKeyField(struct, name, keyFieldName)
      case Ref(name, _) => env.lookupOption(name).exists(isKeyField(_, keyFieldName))
      case _ => false
    }

    private def fieldIsKeyField(struct: IR, fieldName: String, keyFieldName: String): Boolean = struct match {
      case MakeStruct(fields) => fields.exists { case (n, f) =>
        n == fieldName && isKeyField(f, keyFieldName)
      }
      case SelectFields(o, fields) => fields.exists { f =>
        f == fieldName && fieldIsKeyField(o, fieldName, keyFieldName)
      }
      case _ => isRowRef(struct) && fieldName == keyFieldName
    }

    def isFirstKey(ir: IR): Boolean = isKeyField(ir, rowKeyType.fieldNames.head)

    def isKeyStructPrefix(ir: IR): Boolean = ir match {
      case MakeStruct(fields) => fields.view.zipWithIndex.forall { case ((_, fd), idx) =>
        idx < rowKeyType.size && isKeyField(fd, rowKeyType.fieldNames(idx))
      }
      case SelectFields(old, fields) => fields.view.zipWithIndex.forall { case (fd, idx) =>
        idx < rowKeyType.size && fieldIsKeyField(old, fd, rowKeyType.fieldNames(idx))
      }
      case _ => false
    }
  }

  def literalSizeOkay(lit: Literal): Boolean = lit.value.asInstanceOf[Iterable[_]].size <= MAX_LITERAL_SIZE

  def wrapInRow(intervals: Array[Interval]): Array[Interval] = {
    intervals.map { interval =>
      Interval(IntervalEndpoint(Row(interval.left.point), interval.left.sign),
        IntervalEndpoint(Row(interval.right.point), interval.right.sign))
    }
  }

  def minimumValueByType(t: Type, ctx: ExecuteContext): Any = {
    t match {
      case TInt32 => Int.MinValue
      case TInt64 => Long.MinValue
      case TFloat32 => Float.NegativeInfinity
      case TFloat64 => Double.PositiveInfinity
      case TBoolean => false
      case t: TLocus =>
        val rg = ctx.getReference(t.rg)
        Locus(rg.contigs.head, 1)
      case tbs: TBaseStruct => Row.fromSeq(tbs.types.map(minimumValueByType(_, ctx)))
    }
  }

  def maximumValueByType(t: Type, ctx: ExecuteContext): Any = {
    t match {
      case TInt32 => Int.MaxValue
      case TInt64 => Long.MaxValue
      case TFloat32 => Float.PositiveInfinity
      case TFloat64 => Double.PositiveInfinity
      case TBoolean => false
      case t: TLocus =>
        val rg = ctx.getReference(t.rg)
        val contig = rg.contigs.last
        Locus(contig, rg.contigLength(contig) - 1)
      case tbs: TBaseStruct => Row.fromSeq(tbs.types.map(maximumValueByType(_, ctx)))
    }
  }

  def constValue(x: IR): Any = (x: @unchecked) match {
    case I32(v) => v
    case I64(v) => v
    case F32(v) => v
    case F64(v) => v
    case Str(v) => v
    case Literal(_, v) => v
  }

  def endpoint(value: Any, inclusivity: Int, wrapped: Boolean = true): IntervalEndpoint = {
    IntervalEndpoint(if (wrapped) Row(value) else value, inclusivity)
  }

  def getIntervalFromContig(c: String, rg: ReferenceGenome): Option[Interval] = {
    if (rg.contigsSet.contains(c)) {
      Some(Interval(
        endpoint(Locus(c, 1), -1),
        endpoint(Locus(c, rg.contigLength(c)), -1)))
    } else {
      warn(s"Filtered with contig '${c}', but '${c}' is not a valid contig in reference genome ${rg.name}")
      None
    }
  }

  def openInterval(v: Any, typ: Type, op: ComparisonOp[_], ctx: ExecuteContext, flipped: Boolean = false): Interval = {
    (op: @unchecked) match {
      case _: EQ =>
        Interval(endpoint(v, -1), endpoint(v, 1))
      case GT(_, _) =>
        if (flipped)
          Interval(endpoint(v, 1), endpoint(maximumValueByType(typ, ctx), 1)) // key > value
        else
          Interval(endpoint(minimumValueByType(typ, ctx), -1), endpoint(v, -1)) // value > key
      case GTEQ(_, _) =>
        if (flipped)
          Interval(endpoint(v, -1), endpoint(maximumValueByType(typ, ctx), 1)) // key >= value
        else
          Interval(endpoint(minimumValueByType(typ, ctx), -1), endpoint(v, 1)) // value >= key
      case LT(_, _) =>
        if (flipped)
          Interval(endpoint(minimumValueByType(typ, ctx), -1), endpoint(v, -1)) // key < value
        else
          Interval(endpoint(v, 1), endpoint(maximumValueByType(typ, ctx), 1)) // value < key
      case LTEQ(_, _) =>
        if (flipped)
          Interval(endpoint(minimumValueByType(typ, ctx), -1), endpoint(v, 1)) // key <= value
        else
          Interval(endpoint(v, -1), endpoint(maximumValueByType(typ, ctx), 1)) // value <= key
    }
  }

  def canGenerateOpenInterval(t: Type): Boolean = t match {
    case _: TNumeric => true
    case TBoolean => true
    case _: TLocus => true
    case ts: TBaseStruct => ts.fields.forall(f => canGenerateOpenInterval(f.typ))
    case _ => false
  }

  def opIsSupported(op: ComparisonOp[_]): Boolean = {
    op match {
      case EQ(_, _) => true
      case _: LTEQ | _: LT | _: GTEQ | _: GT => canGenerateOpenInterval(op.t1)
      case _ => false
    }
  }

  def extractAndRewrite(cond1: IR, es: ExtractionState): Option[(IR, Array[Interval])] = {
    val env = Env.empty[AbstractValue].bind(
      es.rowRef.name,
      StructValue(SortedMap(es.keyFields.zipWithIndex.map(t => t._1 -> KeyFieldValue(t._2)): _*)))
    println(env)
    val BoolValue(intervals) = analyze(cond1, env, es.ctx, es.iOrd)
    if (intervals.length == 1 && intervals(0) == Interval(Row(), Row(), true, true))
      None
    else
      Some((cond1, intervals))
  }

  def extractPartitionFilters(ctx: ExecuteContext, cond: IR, ref: Ref, key: IndexedSeq[String]): Option[(IR, Array[Interval])] = {
    println("in extract")
    if (key.isEmpty)
      None
    else
      extractAndRewrite(cond, ExtractionState(ctx, ref, key))
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
