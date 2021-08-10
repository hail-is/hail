package is.hail.expr.ir

import is.hail.annotations.IntervalEndpointOrdering
import is.hail.types.virtual._
import is.hail.utils.{FastSeq, Interval, IntervalEndpoint, _}
import is.hail.variant.{Locus, ReferenceGenome}
import org.apache.spark.sql.Row

object ExtractIntervalFilters {

  val MAX_LITERAL_SIZE = 4096

  case class ExtractionState(rowRef: Ref, keyFields: IndexedSeq[String]) {
    val rowType: TStruct = rowRef.typ.asInstanceOf[TStruct]
    val rowKeyType: TStruct = rowType.select(keyFields)._1
    val firstKeyType: Type = rowKeyType.types.head
    val iOrd: IntervalEndpointOrdering = rowKeyType.ordering.intervalEndpointOrdering

    def isFirstKey(ir: IR): Boolean = ir == GetField(rowRef, rowKeyType.fieldNames.head)

    def isKeyStructPrefix(ir: IR): Boolean = ir match {
      case MakeStruct(fields) => fields
        .iterator
        .map(_._2)
        .zipWithIndex
        .forall { case (fd, idx) => idx < rowKeyType.size && fd == GetField(rowRef, rowKeyType.fieldNames(idx)) }
      case SelectFields(`rowRef`, fields) => keyFields.startsWith(fields)
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

  def minimumValueByType(t: Type): Any = {
    t match {
      case TInt32 => Int.MinValue
      case TInt64 => Long.MinValue
      case TFloat32 => Float.NegativeInfinity
      case TFloat64 => Double.PositiveInfinity
      case TBoolean => false
      case t: TLocus =>
        val rg = t.rg.asInstanceOf[ReferenceGenome]
        Locus(rg.contigs.head, 1)
      case tbs: TBaseStruct => Row.fromSeq(tbs.types.map(minimumValueByType))
    }
  }

  def maximumValueByType(t: Type): Any = {
    t match {
      case TInt32 => Int.MaxValue
      case TInt64 => Long.MaxValue
      case TFloat32 => Float.PositiveInfinity
      case TFloat64 => Double.PositiveInfinity
      case TBoolean => false
      case t: TLocus =>
        val rg = t.rg.asInstanceOf[ReferenceGenome]
        val contig = rg.contigs.last
        Locus(contig, rg.contigLength(contig) - 1)
      case tbs: TBaseStruct => Row.fromSeq(tbs.types.map(maximumValueByType))
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

  def getIntervalFromContig(c: String, rg: ReferenceGenome): Interval = {
    Interval(
      endpoint(Locus(c, 1), -1),
      endpoint(Locus(c, rg.contigLength(c)), -1))
  }

  def openInterval(v: Any, typ: Type, op: ComparisonOp[_], flipped: Boolean = false): Interval = {
    (op: @unchecked) match {
      case _: EQ =>
        Interval(endpoint(v, -1), endpoint(v, 1))
      case GT(_, _) =>
        if (flipped)
          Interval(endpoint(v, 1), endpoint(maximumValueByType(typ), 1)) // key > value
        else
          Interval(endpoint(minimumValueByType(typ), -1), endpoint(v, -1)) // value > key
      case GTEQ(_, _) =>
        if (flipped)
          Interval(endpoint(v, -1), endpoint(maximumValueByType(typ), 1)) // key >= value
        else
          Interval(endpoint(minimumValueByType(typ), -1), endpoint(v, 1)) // value >= key
      case LT(_, _) =>
        if (flipped)
          Interval(endpoint(minimumValueByType(typ), -1), endpoint(v, -1)) // key < value
        else
          Interval(endpoint(v, 1), endpoint(maximumValueByType(typ), 1)) // value < key
      case LTEQ(_, _) =>
        if (flipped)
          Interval(endpoint(minimumValueByType(typ), -1), endpoint(v, 1)) // key <= value
        else
          Interval(endpoint(v, -1), endpoint(maximumValueByType(typ), 1)) // value <= key
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
    cond1 match {
      case ApplySpecial("lor", _, Seq(l, r), t) =>
        extractAndRewrite(l, es)
          .liftedZip(extractAndRewrite(r, es))
          .flatMap {
            case ((True(), i1), (True(), i2)) =>
              Some((True(), Interval.union(i1 ++ i2, es.iOrd)))
            case _ => None
          }
      case ApplySpecial("land", _, Seq(l, r), t) =>
        val ll = extractAndRewrite(l, es)
        val rr = extractAndRewrite(r, es)
        (ll, rr) match {
          case (Some((ir1, i1)), Some((ir2, i2))) =>
            log.info(s"intersecting list of ${ i1.length } intervals with list of ${ i2.length } intervals")
            val intersection = Interval.intersection(i1, i2, es.iOrd)
            log.info(s"intersect generated ${ intersection.length } intersected intervals")
            Some((invoke("land", t, ir1, ir2), intersection))
          case (Some((ir1, i1)), None) =>
            Some((invoke("land", t, ir1, r), i1))
          case (None, Some((ir2, i2))) =>
            Some((invoke("land", t, l, ir2), i2))
          case (None, None) =>
            None
        }
      case StreamFold(ToStream(lit: Literal, _), False(), acc, value, body) =>
        body match {
          case ApplySpecial("lor", _, Seq(Ref(`acc`, _), ApplySpecial("contains", _, Seq(Ref(`value`, _), k), _)), _) if es.isFirstKey(k) =>
            assert(lit.typ.asInstanceOf[TContainer].elementType.isInstanceOf[TInterval])
            Some((True(),
              Interval.union(constValue(lit).asInstanceOf[Iterable[_]]
                .filter(_ != null)
                .map { v =>
                  val i = v.asInstanceOf[Interval]
                  Interval(
                    IntervalEndpoint(Row(i.left.point), i.left.sign),
                    IntervalEndpoint(Row(i.right.point), i.right.sign))
                }.toArray,
                es.iOrd)))
          case _ => None
        }
      case Coalesce(Seq(x, False())) => extractAndRewrite(x, es)
        .map { case (ir, intervals) => (Coalesce(FastSeq(ir, False())), intervals) }
      case ApplyIR("contains", _, Seq(lit: Literal, Apply("contig", _, Seq(k), _))) if es.isFirstKey(k) =>
        val rg = k.typ.asInstanceOf[TLocus].rg.asInstanceOf[ReferenceGenome]

        val intervals = (lit.value: @unchecked) match {
          case x: IndexedSeq[_] => x.map(elt => getIntervalFromContig(elt.asInstanceOf[String], rg)).toArray
          case x: Set[_] => x.map(elt => getIntervalFromContig(elt.asInstanceOf[String], rg)).toArray
          case x: Map[_, _] => x.keys.map(elt => getIntervalFromContig(elt.asInstanceOf[String], rg)).toArray
        }
        Some((True(), intervals))
      case ApplyIR("contains", _, Seq(lit: Literal, k)) if literalSizeOkay(lit) =>
        val wrap = if (es.isFirstKey(k)) Some(true) else if (es.isKeyStructPrefix(k)) Some(false) else None
        wrap.map { wrapStruct =>
          val intervals = (lit.value: @unchecked) match {
            case x: IndexedSeq[_] => x.map(elt => Interval(
              endpoint(elt, -1, wrapped = wrapStruct),
              endpoint(elt, 1, wrapped = wrapStruct))).toArray
            case x: Set[_] => x.map(elt => Interval(
              endpoint(elt, -1, wrapped = wrapStruct),
              endpoint(elt, 1, wrapped = wrapStruct))).toArray
            case x: Map[_, _] => x.keys.map(elt => Interval(
              endpoint(elt, -1, wrapped = wrapStruct),
              endpoint(elt, 1, wrapped = wrapStruct))).toArray
          }
          (True(), intervals)
        }
      case ApplySpecial("contains", _, Seq(lit: Literal, k), _) =>
        k match {
          case x if es.isFirstKey(x) =>
            val intervals = (lit.value: @unchecked) match {
              case null => Array[Interval]()
              case i: Interval => Array(i)
            }
            Some((True(), wrapInRow(intervals)))
          case x if es.isKeyStructPrefix(x) =>
            val intervals = (lit.value: @unchecked) match {
              case null => Array[Interval]()
              case i: Interval => Array(i)
            }
            Some((True(), intervals))
          case _ => None
        }
      case ApplyComparisonOp(op, l, r) if opIsSupported(op) =>
        val comparisonData = if (IsConstant(l))
          Some((l, r, false))
        else if (IsConstant(r))
          Some((r, l, true))
        else
          None
        comparisonData.flatMap { case (const, k, flipped) =>
          k match {
            case x if es.isFirstKey(x) =>
              // simple key comparison
              Some((True(), Array(openInterval(constValue(const), const.typ, op, flipped))))
            case x if es.isKeyStructPrefix(x) =>
              assert(op.isInstanceOf[EQ])
              val c = constValue(const)
              Some((True(), Array(Interval(endpoint(c, -1), endpoint(c, 1)))))
            case Apply("contig", _, Seq(x), _) if es.isFirstKey(x) =>
              // locus contig comparison
              val intervals = (constValue(const): @unchecked) match {
                case s: String => Array(getIntervalFromContig(s, es.firstKeyType.asInstanceOf[TLocus].rg.asInstanceOf[ReferenceGenome]))
              }
              Some((True(), intervals))
            case Apply("position", _, Seq(x), _) if es.isFirstKey(x) =>
              // locus position comparison
              val pos = constValue(const).asInstanceOf[Int]
              val rg = es.firstKeyType.asInstanceOf[TLocus].rg.asInstanceOf[ReferenceGenome]
              val ord = TTuple(TInt32).ordering
              val intervals = rg.contigs.indices
                .flatMap { i =>
                  openInterval(pos, TInt32, op, flipped).intersect(ord,
                    Interval(endpoint(1, -1), endpoint(rg.contigLength(i), -1)))
                    .map { interval =>
                      Interval(
                        endpoint(Locus(rg.contigs(i), interval.left.point.asInstanceOf[Row].getAs[Int](0)), interval.left.sign),
                        endpoint(Locus(rg.contigs(i), interval.right.point.asInstanceOf[Row].getAs[Int](0)), interval.right.sign))
                    }
                }.toArray

              Some((True(), intervals))
            case _ => None
          }
        }
      case Let(name, value, body) if name != es.rowRef.name =>
        // TODO: thread key identity through values, since this will break when CSE arrives
        // TODO: thread predicates in `value` through `body` as a ref
        extractAndRewrite(body, es)
          .map { case (ir, intervals) => (Let(name, value, ir), intervals) }
      case _ => None
    }
  }

  def extractPartitionFilters(cond: IR, ref: Ref, key: IndexedSeq[String]): Option[(IR, Array[Interval])] = {
    if (key.isEmpty)
      None
    else
      extractAndRewrite(cond, ExtractionState(ref, key))
  }

  def apply(ir0: BaseIR): BaseIR = {
    MapIR.mapBaseIR(ir0, (ir: BaseIR) => {
      (ir match {
        case TableFilter(child, pred) =>
          extractPartitionFilters(pred, Ref("row", child.typ.rowType), child.typ.key)
            .map { case (newCond, intervals) =>
              log.info(s"generated TableFilterIntervals node with ${ intervals.length } intervals:\n  " +
                s"Intervals: ${ intervals.mkString(", ") }\n  " +
                s"Predicate: ${ Pretty(pred) }\n " +
                s"Post: ${ Pretty(newCond) }")
              TableFilter(
                TableFilterIntervals(child, intervals, keep = true),
                newCond)
            }
        case MatrixFilterRows(child, pred) =>
          extractPartitionFilters(pred, Ref("va", child.typ.rowType), child.typ.rowKey)
            .map { case (newCond, intervals) =>
              log.info(s"generated MatrixFilterIntervals node with ${ intervals.length } intervals:\n  " +
                s"Intervals: ${ intervals.mkString(", ") }\n  " +
                s"Predicate: ${ Pretty(pred) }\n " +
                s"Post: ${ Pretty(newCond) }")
              MatrixFilterRows(
                MatrixFilterIntervals(child, intervals, keep = true),
                newCond)
            }

        case _ => None
      }).getOrElse(ir)
    })
  }
}