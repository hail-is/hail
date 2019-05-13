package is.hail.expr.ir

import is.hail.expr.types.virtual._
import is.hail.methods.{MatrixFilterIntervals, TableFilterIntervals}
import is.hail.utils.{FastSeq, Interval, IntervalEndpoint, _}
import is.hail.variant.{Locus, ReferenceGenome}
import org.apache.spark.sql.Row

sealed trait KeyFilterPredicate

case class KeyComparison(comp: ApplyComparisonOp) extends KeyFilterPredicate

case class LiteralContains(comp: IR) extends KeyFilterPredicate

case class IntervalContains(comp: IR) extends KeyFilterPredicate

case class LocusContigComparison(comp: ApplyComparisonOp) extends KeyFilterPredicate

case class LocusPositionComparison(comp: ApplyComparisonOp) extends KeyFilterPredicate

case class LocusContigContains(comp: IR) extends KeyFilterPredicate

case class Disjunction(l: KeyFilterPredicate, r: KeyFilterPredicate) extends KeyFilterPredicate

case class Conjunction(l: KeyFilterPredicate, r: KeyFilterPredicate) extends KeyFilterPredicate

case object Unknown extends KeyFilterPredicate

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

  private val noData: (Set[IR], Array[Interval]) = (Set(), Array())

  def processPredicates(p: KeyFilterPredicate, kType: Type): (Set[IR], Array[Interval]) = {
    p match {
      case KeyComparison(comp) =>
        val (v, isFlipped) = if (IsConstant(comp.l)) (comp.l, false) else (comp.r, true)
        Set[IR](comp) -> Array(openInterval(constValue(v), v.typ, comp.op, isFlipped))

      case LiteralContains(comp: IR) =>
        val ApplyIR(_, Seq(Literal(_, lit), _)) = comp
        val intervals = (lit: @unchecked) match {
          case x: IndexedSeq[_] => x.map(elt => Interval(endpoint(elt, -1), endpoint(elt, 1))).toArray
          case x: Set[_] => x.map(elt => Interval(endpoint(elt, -1), endpoint(elt, 1))).toArray
          case x: Map[_, _] => x.keys.map(elt => Interval(endpoint(elt, -1), endpoint(elt, 1))).toArray
        }
        Set[IR](comp) -> intervals

      case IntervalContains(comp: IR) =>
        val ApplySpecial(_, Seq(Literal(_, lit), _)) = comp
        val intervals = lit match {
          case null => Array[Interval]()
          case i: Interval => Array(i)
        }
        Set[IR](comp) -> intervals

      case LocusContigComparison(comp) =>
        val (v, Apply(_, Seq(locus))) = if (IsConstant(comp.l)) (comp.l, comp.r) else (comp.r, comp.l)
        val interval = (constValue(v): @unchecked) match {
          case s: String => getIntervalFromContig(s, locus.typ.asInstanceOf[TLocus].rg.asInstanceOf[ReferenceGenome])
        }
        Set[IR](comp) -> Array(interval)

      case LocusPositionComparison(comp) =>
        val (v, isFlipped) = if (IsConstant(comp.l)) (comp.l, false) else (comp.r, true)
        val pos = constValue(v).asInstanceOf[Int]
        val rg = kType.asInstanceOf[TLocus].rg.asInstanceOf[ReferenceGenome]
        val ord = TInt32().ordering
        val intervals = rg.contigs.indices
          .flatMap { i =>
            openInterval(pos, TInt32(), comp.op, isFlipped).intersect(ord,
              Interval(endpoint(1, -1), endpoint(rg.contigLength(i), -1)))
              .map { interval =>
                Interval(endpoint(Locus(rg.contigs(i), interval.left.point.asInstanceOf[Int]), interval.left.sign),
                  endpoint(Locus(rg.contigs(i), interval.right.point.asInstanceOf[Int]), interval.right.sign))
              }
          }.toArray

        Set[IR](comp) -> intervals


      case LocusContigContains(comp) =>
        val ApplyIR(_, Seq(Literal(_, lit), Apply("contig", Seq(locus)))) = comp

        val rg = locus.typ.asInstanceOf[TLocus].rg.asInstanceOf[ReferenceGenome]

        val intervals = (lit: @unchecked) match {
          case x: IndexedSeq[_] => x.map(elt => getIntervalFromContig(elt.asInstanceOf[String], rg)).toArray
          case x: Set[_] => x.map(elt => getIntervalFromContig(elt.asInstanceOf[String], rg)).toArray
          case x: Map[_, _] => x.keys.map(elt => getIntervalFromContig(elt.asInstanceOf[String], rg)).toArray
        }

        Set(comp) -> intervals

      case Disjunction(x1, x2) =>
        val (s1, i1) = processPredicates(x1, kType)
        val (s2, i2) = processPredicates(x2, kType)
        (s1.union(s2), Interval.union(i1 ++ i2, kType.ordering.intervalEndpointOrdering))

      case Conjunction(x1, x2) =>
        val (s1, i1) = processPredicates(x1, kType)
        val (s2, i2) = processPredicates(x2, kType)
        log.info(s"intersecting list of ${ i1.length } intervals with list of ${ i2.length } intervals")
        val intersection = Interval.intersection(i1, i2, kType.ordering.intervalEndpointOrdering)
        log.info(s"intersect generated ${ intersection.length } intersected intervals")
        (s1.union(s2), intersection)

      case Unknown => noData // should only be found in a conjunction
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

  def extract(cond1: IR, ref: Ref, k: IR): KeyFilterPredicate = {
    cond1 match {
      case ApplySpecial("||", Seq(l, r)) =>
        val ll = extract(l, ref, k)
        val rr = extract(r, ref, k)
        if (ll == Unknown || rr == Unknown)
          Unknown
        else
          Disjunction(ll, rr)
      case ApplySpecial("&&", Seq(l, r)) =>
        Conjunction(extract(l, ref, k), extract(r, ref, k))
      case Coalesce(Seq(x, False())) => extract(x, ref, k)
      case x@ApplyIR("contains", Seq(_: Literal, `k`)) => LiteralContains(x) // don't match string contains
      case x@ApplySpecial("contains", Seq(_: Literal, `k`)) => IntervalContains(x)
      case x@ApplyIR("contains", Seq(_: Literal, Apply("contig", Seq(`k`)))) => LocusContigContains(x)
      case x@ApplyComparisonOp(op, l, r) if opIsSupported(op) =>
        if (IsConstant(l) && r == k || l == k && IsConstant(r))
          // TODO: need to look for casts, since patterns like [ `k` > 1.5 ] will not match if `k` is an integer
          KeyComparison(x)
        else if ((IsConstant(l) && r == Apply("contig", FastSeq(k))
          || l == Apply("contig", FastSeq(k)) && IsConstant(r)) && op.isInstanceOf[EQ])
          LocusContigComparison(x)
        else if (IsConstant(l) && r == Apply("position", FastSeq(k))
          || l == Apply("position", FastSeq(k)) && IsConstant(r))
          LocusPositionComparison(x)
        else
          Unknown
      case Let(name, _, body) if name != ref.name =>
        // TODO: thread key identity through values, since this will break when CSE arrives
        // TODO: thread predicates in `value` through `body` as a ref
        extract(body, ref, k)
      case _ => Unknown
    }
  }

  def extractPartitionFilters(cond: IR, ref: Ref, key: IndexedSeq[String]): Option[(Array[Interval], IR)] = {
    if (key.isEmpty)
      return None

    val k1 = GetField(ref, key.head)

    val (nodes, intervals) = processPredicates(extract(cond, ref, k1), k1.typ)
    if (nodes.nonEmpty) {
      val refSet = nodes.map(RefEquality(_))

      def rewrite(ir: IR): IR = if (refSet.contains(RefEquality(ir))) True() else MapIR(rewrite)(ir)

      Some(intervals -> rewrite(cond))
    } else {
      assert(intervals.isEmpty)
      None
    }
  }

  def apply(ir0: BaseIR): BaseIR = {

    RewriteBottomUp(ir0, {
      case TableFilter(child, pred) =>
        extractPartitionFilters(pred, Ref("row", child.typ.rowType), child.typ.key)
          .map { case (intervals, newCond) =>
            log.info(s"generated TableFilterIntervals node with ${ intervals.length } intervals:\n  " +
              s"Intervals: ${ intervals.mkString(", ") }\n  " +
              s"Predicate: ${ Pretty(pred) }")
            TableFilter(
              TableToTableApply(
                child,
                TableFilterIntervals(child.typ.keyType, wrapInRow(intervals), keep = true)),
              newCond)
          }
      case MatrixFilterRows(child, pred) =>
        extractPartitionFilters(pred, Ref("va", child.typ.rvRowType), child.typ.rowKey)
          .map { case (intervals, newCond) =>
            log.info(s"generated MatrixFilterIntervals node with ${ intervals.length } intervals:\n  " +
              s"Intervals: ${ intervals.mkString(", ") }\n  " +
              s"Predicate: ${ Pretty(pred) }")
            MatrixFilterRows(
              MatrixToMatrixApply(
                child,
                MatrixFilterIntervals(child.typ.rowKeyStruct, wrapInRow(intervals), keep = true)),
              newCond)
          }

      case _ => None
    })
  }
}
