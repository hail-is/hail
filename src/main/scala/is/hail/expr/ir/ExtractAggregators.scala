package is.hail.expr.ir

import is.hail.annotations.aggregators._
import is.hail.expr.{RegionValueAggregator, TAggregable}
import is.hail.utils._

import scala.language.existentials

object ExtractAggregators {

  private case class IRAgg(in: In, agg: RegionValueAggregator) { }

  def apply(ir: IR, tAggIn: TAggregable): (IR, RegionValueAggregator) = {
    val (ir2, aggs) = extract(ir, tAggIn)
    val combined = new ZippedRegionValueAggregator(aggs map (_.agg))
    aggs.foreach(_.in.typ = combined.typ)
    (ir2, combined)
  }

  private def extract(ir: IR, tAggIn: TAggregable): (IR, Array[IRAgg]) = {
    val ab = new ArrayBuilder[IRAgg]()
    val ir2 = extract(ir, ab, tAggIn)
    (ir2, ab.result())
  }

  private def extract(ir: IR, ab: ArrayBuilder[IRAgg], tAggIn: TAggregable): IR = {
    def extract(ir: IR): IR = this.extract(ir, ab, tAggIn)
    ir match {
      case Ref(name, typ) =>
        assert(typ.isRealizable)
        ir
      case AggIn(_) =>
        throw new RuntimeException(s"AggIn must be used inside an AggSum, but found: $ir")
      case AggMap(a, name, body, typ) =>
        throw new RuntimeException(s"AggMap must be used inside an AggSum, but found: $ir")
      case AggSum(a, typ) =>
        val tChildAgg = a.typ.asInstanceOf[TAggregable]
        val in = In(0, null)
        ab += IRAgg(in,
          TransformedRegionValueAggregator(tAggIn,
            aggin => tChildAgg.getElement(lower(a, aggin, tAggIn)),
            RegionValueSumAggregator(tChildAgg.elementType)))
        GetField(in, (ab.length - 1).toString(), typ)
      case _ => Recur(extract)(ir)
    }
  }

  private def lower(ir: IR, aggIn: IR, tAggIn: TAggregable): IR = {
    def lower(ir: IR): IR = this.lower(ir, aggIn, tAggIn)
    ir match {
      case AggIn(typ) =>
        assert(tAggIn == typ)
        aggIn
      case AggMap(a, name, body, typ) =>
        val tA = a.typ.asInstanceOf[TAggregable]
        val tLoweredA = tA.elementAndScopeStruct
        val la = lower(a)
        assert(la.typ == tLoweredA, s"type after lowering, ${la.typ}, should be the carrier struct of ${a.typ}; $ir")
        tA.inScope(la, e => Let(name, e, lower(body), typ.elementType))
      case AggSum(_, _) =>
        throw new RuntimeException(s"No nested aggregations allowed: $ir")
      case In(i, typ) =>
        throw new RuntimeException(s"No inputs may be referenced inside an aggregator: $ir")
      case InMissingness(i) =>
        throw new RuntimeException(s"No inputs may be referenced inside an aggregator: $ir")
      case _ => Recur(lower)(ir)
    }
  }
}
