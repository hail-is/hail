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

case class LocusContigContains(comp: IR) extends KeyFilterPredicate

case class LocusPositionComparison(comp: IR) extends KeyFilterPredicate

case class Disjunction(xs: Array[KeyFilterPredicate]) extends KeyFilterPredicate

case class Conjunction(xs: Array[KeyFilterPredicate]) extends KeyFilterPredicate

case object Unknown extends KeyFilterPredicate

object ExtractIntervalFilters {

  def getIntervalIntersection(intervals: Iterable[Interval], t: Type): Option[Interval] = {
    if (intervals.isEmpty)
      return Some(Interval(minimumValueByType(t), maximumValueByType(t)))
    val tOrd = t.ordering
    val iOrd = tOrd.intervalEndpointOrdering
    val ord = iOrd.toOrdering
    val minValue = intervals.map(_.left).min(ord)
    val maxValue = intervals.map(_.right).max(ord)
    tOrd.compare(minValue.point, maxValue.point) match {
      case x if x < 0 => Some(Interval(minValue, maxValue))
      case x if x == 0 => if (minValue.sign < 0 && maxValue.sign > 0) Some(Interval(minValue, maxValue)) else None
      case _ => None
    }
  }

  def intersectIntervalLists(i1: Array[Interval], i2: Array[Interval], t: Type): Array[Interval] = {
    // FIXME: this is quadratic
    log.info(s"intersecting list of ${ i1.length } intervals with list of ${ i2.length } intervals")
    val r = i1.flatMap(i => i2.foldLeft(Option(i)) { case (comb, ii) => comb.flatMap(c => getIntervalIntersection(Array(c, ii), t)) })
    log.info(s"intersect generated ${ r.length } intersected intervals")
    r
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
    IntervalEndpoint(Row(value), inclusivity)
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

      case LocusPositionComparison(_) => Set[IR]() -> Array() // don't generate intervals per contig due to number of GRCh38 contigs?


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

      case x@Conjunction(xs) =>
        // search for contig and position combination filters
        val (extractedLocusFilters, other) = xs.partition {
          case _: LocusPositionComparison | _: LocusContigComparison => true
          case _ => false
        }

        val (contigComparisons, positionComparisons) = extractedLocusFilters.partition(_.isInstanceOf[LocusContigComparison])

        val base: (Set[IR], Array[Interval]) = if (contigComparisons.isEmpty && positionComparisons.nonEmpty)
          noData // could generate intervals per contig, but potentially bad with GRCh38?
        else {
          val cc = contigComparisons.map {
            case LocusContigComparison(ApplyComparisonOp(_, l, r)) =>
              if (IsConstant(l))
                constValue(l)
              else
                constValue(r)
          }.toSet

          if (cc.size > 2)
          // rewrite all pieces of the conjunction, since we have proven it will return no rows
            transitiveComparisonNodes(x) -> Array()
          else if (cc.size == 1) {
            val pc = positionComparisons.map {
              case LocusPositionComparison(ApplyComparisonOp(op, l, r)) =>
                if (IsConstant(l))
                  (op, constValue(l), true)
                else
                  (op, constValue(r), false)
            }.map { case (op, v, isFlipped) => openInterval(v, TInt32(), op, isFlipped) }
            getIntervalIntersection(pc, TTuple(TInt32())) match {
              case Some(i) =>
                val start = i.start.asInstanceOf[Row].getAs[Int](0)
                val end = i.end.asInstanceOf[Row].getAs[Int](0)
                val ccHead = contigComparisons.head.asInstanceOf[LocusContigComparison].comp
                val Apply(_, Seq(k)) = if (IsConstant(ccHead.l)) ccHead.r else ccHead.l
                val rg = k.typ.asInstanceOf[TLocus].rg.asInstanceOf[ReferenceGenome]
                val contig = cc.head.asInstanceOf[String]
                val contigLength = rg.contigLength(contig)

                val intervals = if (end < 1 || start > contigLength)
                  Array[Interval]()
                else {
                  val start2 = if (start < 1)
                    endpoint(Locus(contig, 1), -1)
                  else
                    endpoint(Locus(contig, start), i.left.sign)
                  val end2 = if (end > contigLength)
                    endpoint(Locus(contig, contigLength), 1)
                  else
                    endpoint(Locus(contig, end), i.right.sign)
                  Array(Interval(start2, end2))
                }
                (contigComparisons.map(_.asInstanceOf[LocusContigComparison].comp).toSet[IR]
                  .union(positionComparisons.map(_.asInstanceOf[LocusPositionComparison].comp).toSet[IR]),
                  intervals)
              case None => transitiveComparisonNodes(x) -> Array()
            }
          } else {
            noData
          }
        }
        other.foldLeft(base) { case ((s, i), f) =>
          val (s2, i2) = processPredicates(f, kType)
          (s2.union(s), intersectIntervalLists(i, i2, kType))
        }
      case _ => noData
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
                TableFilterIntervals(child.typ.keyType, intervals, keep = true)),
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
                MatrixFilterIntervals(child.typ.rowKeyStruct, intervals, keep = true)),
              newCond)
          }

      case _ => None
    })
  }
}
