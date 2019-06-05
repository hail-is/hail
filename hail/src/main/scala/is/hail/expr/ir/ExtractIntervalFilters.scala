package is.hail.expr.ir

import is.hail.expr.types.virtual._
import is.hail.utils.{FastSeq, Interval, IntervalEndpoint, _}
import is.hail.variant.{Locus, ReferenceGenome}
import org.apache.spark.sql.Row

object ExtractIntervalFilters {
  def wrapInRow(intervals: Array[Interval]): Array[Interval] = {
    intervals.map { interval =>
      Interval(IntervalEndpoint(Row(interval.left.point), interval.left.sign),
        IntervalEndpoint(Row(interval.right.point), interval.right.sign))
    }
  }

  def minimumValueByType(t: Type): IntervalEndpoint = {
    t match {
      case _: TInt32 => endpoint(Int.MinValue, -1)
      case _: TInt64 => endpoint(Long.MinValue, -1)
      case _: TFloat32 => endpoint(Float.NegativeInfinity, -1)
      case _: TFloat64 => endpoint(Double.PositiveInfinity, -1)
    }
  }

  def maximumValueByType(t: Type): IntervalEndpoint = {
    t match {
      case _: TInt32 => endpoint(Int.MaxValue, 1)
      case _: TInt64 => endpoint(Long.MaxValue, 1)
      case _: TFloat32 => endpoint(Float.PositiveInfinity, 1)
      case _: TFloat64 => endpoint(Double.PositiveInfinity, 1)
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

  def endpoint(value: Any, inclusivity: Int): IntervalEndpoint = {
    IntervalEndpoint(value, inclusivity)
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
          Interval(endpoint(v, 1), maximumValueByType(typ)) // key > value
        else
          Interval(minimumValueByType(typ), endpoint(v, -1)) // value > key
      case GTEQ(_, _) =>
        if (flipped)
          Interval(endpoint(v, -1), maximumValueByType(typ)) // key >= value
        else
          Interval(minimumValueByType(typ), endpoint(v, 1)) // value >= key
      case LT(_, _) =>
        if (flipped)
          Interval(minimumValueByType(typ), endpoint(v, -1)) // key < value
        else
          Interval(endpoint(v, 1), maximumValueByType(typ)) // value < key
      case LTEQ(_, _) =>
        if (flipped)
          Interval(minimumValueByType(typ), endpoint(v, 1)) // key <= value
        else
          Interval(endpoint(v, -1), maximumValueByType(typ)) // value <= key
    }
  }

  def opIsSupported(op: ComparisonOp[_]): Boolean = {
    op match {
      case _: Compare => false
      case _: NEQ => false
      case _: NEQWithNA => false
      case _: EQWithNA => false
      case _ => true
    }
  }

  def extractAndRewrite(cond1: IR, ref: Ref, k: IR): Option[(IR, Array[Interval])] = {
    cond1 match {
      case ApplySpecial("||", Seq(l, r)) =>
        extractAndRewrite(l, ref, k)
          .liftedZip(extractAndRewrite(r, ref, k))
          .map { case ((_, i1), (_, i2)) =>
            (True(), Interval.union(i1 ++ i2, k.typ.ordering.intervalEndpointOrdering))
          }
      case ApplySpecial("&&", Seq(l, r)) =>
        val ll = extractAndRewrite(l, ref, k)
        val rr = extractAndRewrite(r, ref, k)
        (ll, rr) match {
          case (Some((ir1, i1)), Some((ir2, i2))) =>
            log.info(s"intersecting list of ${ i1.length } intervals with list of ${ i2.length } intervals")
            val intersection = Interval.intersection(i1, i2, k.typ.ordering.intervalEndpointOrdering)
            log.info(s"intersect generated ${ intersection.length } intersected intervals")
            Some((invoke("&&", ir1, ir2), intersection))
          case (Some((ir1, i1)), None) =>
            Some((invoke("&&", ir1, r), i1))
          case (None, Some((ir2, i2))) =>
            Some((invoke("&&", l, ir2), i2))
          case (None, None) =>
            None
        }
      case ArrayFold(Literal(t, lit), False(), acc, value, body) =>
        body match {
          case ApplySpecial("||", Seq(Ref(`acc`, _), ApplySpecial("contains", Seq(Ref(`value`, _), `k`)))) =>
            assert(t.asInstanceOf[TContainer].elementType.isInstanceOf[TInterval])
            Some((True(),
              Interval.union(lit.asInstanceOf[Iterable[_]]
                .filter(_ != null)
                .map(_.asInstanceOf[Interval])
                .toArray,
                k.typ.ordering.intervalEndpointOrdering)))
          case _ => None
        }
      case Coalesce(Seq(x, False())) => extractAndRewrite(x, ref, k)
        .map { case (ir, intervals) => (Coalesce(FastSeq(ir, False())), intervals) }
      case ApplyIR("contains", Seq(lit: Literal, `k`)) =>
        val intervals = (lit.value: @unchecked) match {
          case x: IndexedSeq[_] => x.map(elt => Interval(endpoint(elt, -1), endpoint(elt, 1))).toArray
          case x: Set[_] => x.map(elt => Interval(endpoint(elt, -1), endpoint(elt, 1))).toArray
          case x: Map[_, _] => x.keys.map(elt => Interval(endpoint(elt, -1), endpoint(elt, 1))).toArray
        }
        Some((True(), intervals))
      case ApplySpecial("contains", Seq(lit: Literal, `k`)) =>
        val intervals = (lit.value: @unchecked) match {
          case null => Array[Interval]()
          case i: Interval => Array(i)
        }
        Some((True(), intervals))
      case ApplyIR("contains", Seq(lit: Literal, Apply("contig", Seq(`k`)))) =>
        val rg = k.typ.asInstanceOf[TLocus].rg.asInstanceOf[ReferenceGenome]

        val intervals = (lit.value: @unchecked) match {
          case x: IndexedSeq[_] => x.map(elt => getIntervalFromContig(elt.asInstanceOf[String], rg)).toArray
          case x: Set[_] => x.map(elt => getIntervalFromContig(elt.asInstanceOf[String], rg)).toArray
          case x: Map[_, _] => x.keys.map(elt => getIntervalFromContig(elt.asInstanceOf[String], rg)).toArray
        }
        Some((True(), intervals))
      case ApplyComparisonOp(op, l, r) if opIsSupported(op) =>
        if (IsConstant(l) && r == k || l == k && IsConstant(r)) {
          // simple key comparison
          // TODO: need to look for casts, since patterns like [ `k` > 1.5 ] will not match if `k` is an integer
          val (v, isFlipped) = if (IsConstant(l)) (l, false) else (r, true)
          Some((True(), Array(openInterval(constValue(v), v.typ, op, isFlipped))))
        } else if ((IsConstant(l) && r == Apply("contig", FastSeq(k)) || l == Apply("contig", FastSeq(k)) && IsConstant(r)) && op.isInstanceOf[EQ]) {
          // locus contig comparison
          val v = if (IsConstant(l)) l else r
          val intervals = (constValue(v): @unchecked) match {
            case s: String => Array(getIntervalFromContig(s, k.typ.asInstanceOf[TLocus].rg.asInstanceOf[ReferenceGenome]))
          }
          Some((True(), intervals))
        } else if (IsConstant(l) && r == Apply("position", FastSeq(k)) || l == Apply("position", FastSeq(k)) && IsConstant(r)) {
          // locus position comparison
          val (v, isFlipped) = if (IsConstant(l)) (l, false) else (r, true)
          val pos = constValue(v).asInstanceOf[Int]
          val rg = k.typ.asInstanceOf[TLocus].rg.asInstanceOf[ReferenceGenome]
          val ord = TInt32().ordering
          val intervals = rg.contigs.indices
            .flatMap { i =>
              openInterval(pos, TInt32(), op, isFlipped).intersect(ord,
                Interval(endpoint(1, -1), endpoint(rg.contigLength(i), -1)))
                .map { interval =>
                  Interval(endpoint(Locus(rg.contigs(i), interval.left.point.asInstanceOf[Int]), interval.left.sign),
                    endpoint(Locus(rg.contigs(i), interval.right.point.asInstanceOf[Int]), interval.right.sign))
                }
            }.toArray

          Some((True(), intervals))
        } else None
      case Let(name, value, body) if name != ref.name =>
        // TODO: thread key identity through values, since this will break when CSE arrives
        // TODO: thread predicates in `value` through `body` as a ref
        extractAndRewrite(body, ref, k)
          .map { case (ir, intervals) => (Let(name, value, ir), intervals) }
      case _ => None
    }
  }

  def extractPartitionFilters(cond: IR, ref: Ref, key: IndexedSeq[String]): Option[(IR, Array[Interval])] = {
    if (key.isEmpty)
      None
    else
      extractAndRewrite(cond, ref, GetField(ref, key.head))
  }

  def apply(ir0: BaseIR): BaseIR = {

    RewriteBottomUp(ir0, {
      case TableFilter(child, pred) =>
        extractPartitionFilters(pred, Ref("row", child.typ.rowType), child.typ.key)
          .map { case (newCond, intervals) =>
            log.info(s"generated TableFilterIntervals node with ${ intervals.length } intervals:\n  " +
              s"Intervals: ${ intervals.mkString(", ") }\n  " +
              s"Predicate: ${ Pretty(pred) }")
            TableFilter(
              TableFilterIntervals(child, wrapInRow(intervals), keep = true),
              newCond)
          }
      case MatrixFilterRows(child, pred) =>
        extractPartitionFilters(pred, Ref("va", child.typ.rowType), child.typ.rowKey)
          .map { case (newCond, intervals) =>
            log.info(s"generated MatrixFilterIntervals node with ${ intervals.length } intervals:\n  " +
              s"Intervals: ${ intervals.mkString(", ") }\n  " +
              s"Predicate: ${ Pretty(pred) }")
            MatrixFilterRows(
              MatrixFilterIntervals(child, wrapInRow(intervals), keep = true),
              newCond)
          }

      case _ => None
    })
  }
}
