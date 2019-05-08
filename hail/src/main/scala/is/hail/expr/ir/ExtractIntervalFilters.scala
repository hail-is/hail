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

case class Disjunction(xs: Array[KeyFilterPredicate]) extends KeyFilterPredicate

case class Conjunction(xs: Array[KeyFilterPredicate]) extends KeyFilterPredicate

case object Unknown extends KeyFilterPredicate

object ExtractIntervalFilters {
  def wrapInRow(intervals: Array[Interval]): Array[Interval] = {
    intervals.map { interval =>
      Interval(IntervalEndpoint(Row(interval.left.point), interval.left.sign),
        IntervalEndpoint(Row(interval.right.point), interval.right.sign))
    }
  }

  def simplifyPredicates(p: KeyFilterPredicate): KeyFilterPredicate = {
    p match {
      case Disjunction(xs) =>
        if (xs.contains(Unknown))
          Unknown
        else {
          val (ors, other) = xs.map(simplifyPredicates).partition(_.isInstanceOf[Disjunction])
          Disjunction(ors.flatMap(o => o.asInstanceOf[Disjunction].xs) ++ other)
        }
      case Conjunction(xs) =>
        if (xs.forall(_ == Unknown))
          Unknown
        else {
          val (ands, other) = xs.map(simplifyPredicates).partition(_.isInstanceOf[Conjunction])
          val elts = ands.flatMap(o => o.asInstanceOf[Conjunction].xs) ++ other.filter(_ != Unknown)
          if (elts.length == 1)
            elts.head
          else
            Conjunction(elts)
        }
      case _ => p
    }
  }

  def minimumValueByType(t: Type): IntervalEndpoint = {
    t match {
      case _: TInt32 => endpoint(Int.MinValue, -1)
      case _: TInt64 => endpoint(Long.MinValue, -1)
      case _: TFloat32 => endpoint(Float.MinValue, -1)
      case _: TFloat64 => endpoint(Double.MinValue, -1)
    }
  }

  def maximumValueByType(t: Type): IntervalEndpoint = {
    t match {
      case _: TInt32 => endpoint(Int.MaxValue, 1)
      case _: TInt64 => endpoint(Long.MaxValue, 1)
      case _: TFloat32 => endpoint(Float.MaxValue, 1)
      case _: TFloat64 => endpoint(Double.MaxValue, 1)
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
      endpoint(Locus(c, rg.contigLength(c)), 1))
  }

  def openInterval(v: Any, typ: Type, op: ComparisonOp[_], flipped: Boolean = false): Interval = {
    (op: @unchecked) match {
      case _: EQ =>
        Interval(endpoint(v, -1), endpoint(v, 1))
      case GT(_, _) =>
        if (flipped)
        // key > value
          Interval(endpoint(v, 1), maximumValueByType(typ))
        else
        // value > key
          Interval(minimumValueByType(typ), endpoint(v, -1))
      case GTEQ(_, _) =>
        if (flipped)
        // key >= value
          Interval(endpoint(v, -1), maximumValueByType(typ))
        else
        // value >= key
          Interval(minimumValueByType(typ), endpoint(v, 1))
      case LT(_, _) =>
        if (flipped)
        // key < value
          Interval(minimumValueByType(typ), endpoint(v, -1))
        else
        // value < key
          Interval(endpoint(v, 1), maximumValueByType(typ))
      case LTEQ(_, _) =>
        if (flipped)
        // key <= value
          Interval(minimumValueByType(typ), endpoint(v, 1))
        else
        // value <= key
          Interval(endpoint(v, -1), maximumValueByType(typ))
    }
  }

  def transitiveComparisonNodes(f: KeyFilterPredicate): Set[IR] = f match {
    case KeyComparison(x) => Set(x)
    case LiteralContains(x) => Set(x)
    case LocusContigContains(x) => Set(x)
    case LocusPositionComparison(x) => Set(x)
    case LocusContigComparison(x) => Set(x)
    case Disjunction(xs) => xs.map(transitiveComparisonNodes).fold(Set())(_.union(_))
    case Conjunction(xs) => xs.map(transitiveComparisonNodes).fold(Set())(_.union(_))
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
          case i: Interval => Array(Interval(endpoint(i.left.point, i.left.sign), endpoint(i.right.point, i.right.sign)))
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
        val intOrd = TInt32().ordering.intervalEndpointOrdering
        val intervals = rg.contigs.indices
          .flatMap { i =>
            Interval.intersection(Array(openInterval(pos, TInt32(), comp.op, isFlipped)),
              Array(Interval(endpoint(0, -1), endpoint(rg.contigLength(i), -1))),
              intOrd)
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

      case Disjunction(xs) =>
        val (nodes, intervals) = xs.map(processPredicates(_, kType)).unzip
        nodes.fold(Set())(_.union(_)) -> intervals.flatten

      case Conjunction(xs) =>
        xs.map(processPredicates(_, kType)).reduce[(Set[IR], Array[Interval])] {
          case ((s1, i1), (s2, i2)) =>
            log.info(s"intersecting list of ${ i1.length } intervals with list of ${ i2.length } intervals")
            val intersection = Interval.intersection(i1, i2, kType.ordering.intervalEndpointOrdering)
            log.info(s"intersect generated ${ intersection.length } intersected intervals")
            (s1.union(s2), intersection)
        }

      case Unknown => noData
    }
  }

  def extractPartitionFilters(cond: IR, ref: Ref, key: IndexedSeq[String]): Option[(Array[Interval], IR)] = {
    if (key.isEmpty)
      return None

    val k1 = GetField(ref, key.head)

    def recur[T](cond1: IR): KeyFilterPredicate = {
      cond1 match {
        case ApplySpecial("||", Seq(l, r)) =>
          Disjunction(Array(recur(l), recur(r)))
        case ApplySpecial("&&", Seq(l, r)) =>
          Conjunction(Array(recur(l), recur(r)))
        case Coalesce(Seq(x, False())) => recur(x)
        case x@ApplyIR("contains", Seq(_: Literal, `k1`)) => LiteralContains(x) // don't match string contains
        case x@ApplySpecial("contains", Seq(_: Literal, `k1`)) => IntervalContains(x)
        case x@ApplyIR("contains", Seq(_: Literal, Apply("contig", Seq(`k1`)))) => LocusContigContains(x)
        case x@ApplyComparisonOp(op, l, r) if !op.isInstanceOf[Compare] && !op.isInstanceOf[EQWithNA] =>
          if ((IsConstant(l) && r == k1 || l == k1 && IsConstant(r)) && !op.t1.isInstanceOf[TString])
            KeyComparison(x)
          else if ((IsConstant(l) && r == Apply("contig", FastSeq(k1))
            || l == Apply("contig", FastSeq(k1)) && IsConstant(r)) && op.isInstanceOf[EQ])
            LocusContigComparison(x)
          else if (IsConstant(l) && r == Apply("position", FastSeq(k1))
            || l == Apply("position", FastSeq(k1)) && IsConstant(r))
            LocusPositionComparison(x)
          else
            Unknown
        case Let(name, _, body) if name != ref.name =>
          // TODO: thread key identity through values, since this will break when CSE arrives
          // TODO: thread predicates in `value` through `body` as a ref
          recur(body)
        case _ => Unknown
      }
    }

    val predicates = recur(cond)
    val (nodes, intervals) = processPredicates(simplifyPredicates(predicates), k1.typ)
    if (nodes.nonEmpty) {
      val refSet = nodes.map(RefEquality(_))

      def rewrite(ir: IR): IR = if (refSet.contains(RefEquality(ir))) True() else MapIR(rewrite)(ir)

      Some(intervals -> rewrite(cond))
    } else None
  }

  def apply(ir0: BaseIR): BaseIR = {

    RewriteBottomUp(ir0, {
      case TableFilter(child, pred) =>
        extractPartitionFilters(pred, Ref("row", child.typ.rowType), child.typ.key)
          .map { case (intervals, newCond) =>
            log.info(s"generated TableFilterIntervals node:\n  " +
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
            log.info(s"generated MatrixFilterIntervals node:\n  " +
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
