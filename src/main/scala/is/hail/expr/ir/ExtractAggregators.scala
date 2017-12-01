package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.expr.{Type, RegionValueAggregator, TAggregable, TBoolean, TInt32, TInt64, TFloat64, TFloat32}
import is.hail.utils._

import scala.language.existentials

object ExtractAggregators {

  private case class IRAgg(in: In, typ: Type, agg: RegionValueAggregator) { }

  def apply(ir: IR, tAggIn: TAggregable): (IR, RegionValueAggregator) = {
    val (ir2, aggs) = extract(ir, tAggIn)
    val combined = new ZippedRegionValueAggregator(aggs.map(_.typ), aggs.map(_.agg))
    // mutate the type of the input IR node now that we know what the combined
    // struct's type is
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
      case AggIn(_) | AggMap(_, _, _, _) =>
        throw new RuntimeException(s"Aggregable manipulations must appear inside lexical scope of an Aggregation: $ir")
      case AggSum(a, typ) =>
        val tChildAgg = a.typ.asInstanceOf[TAggregable]
        val in = In(0, null)

        val transform = tChildAgg.getElement(lower(a, In(0, tAggIn.elementAndScopeStruct), tAggIn))
        assert(transform.typ == tChildAgg.elementType)
        val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, RegionValueAggregator, Unit]
        val (dov, mv, v) = Compile.toCode(transform, fb)
        val aggregator = fb.getArg[RegionValueAggregator](4)
        val (callSeq, agg) = transform.typ match {
          case _: TBoolean =>
            (mv.mux(
              Code.checkcast[RegionValueSumBooleanAggregator](aggregator).invoke[Boolean, Boolean, Unit](
                "seqOp", coerce[Boolean](defaultValue(transform.typ)), true),
              Code.checkcast[RegionValueSumBooleanAggregator](aggregator).invoke[Boolean, Boolean, Unit](
                "seqOp", coerce[Boolean](v), false)),
              new RegionValueSumBooleanAggregator())
          case _: TInt32 =>
            (mv.mux(
              Code.checkcast[RegionValueSumIntAggregator](aggregator).invoke[Int, Boolean, Unit](
                "seqOp", coerce[Int](defaultValue(transform.typ)), true),
              Code.checkcast[RegionValueSumIntAggregator](aggregator).invoke[Int, Boolean, Unit](
                "seqOp", coerce[Int](v), false)),
              new RegionValueSumIntAggregator())
          case _: TInt64 =>
            (mv.mux(
              Code.checkcast[RegionValueSumLongAggregator](aggregator).invoke[Long, Boolean, Unit](
                "seqOp", coerce[Long](defaultValue(transform.typ)), true),
              Code.checkcast[RegionValueSumLongAggregator](aggregator).invoke[Long, Boolean, Unit](
                "seqOp", coerce[Long](v), false)),
              new RegionValueSumLongAggregator())
          case _: TFloat32 =>
            (mv.mux(
              Code.checkcast[RegionValueSumFloatAggregator](aggregator).invoke[Float, Boolean, Unit](
                "seqOp", coerce[Float](defaultValue(transform.typ)), true),
              Code.checkcast[RegionValueSumFloatAggregator](aggregator).invoke[Float, Boolean, Unit](
                "seqOp", coerce[Float](v), false)),
              new RegionValueSumFloatAggregator())
          case _: TFloat64 =>
            (mv.mux(
              Code.checkcast[RegionValueSumDoubleAggregator](aggregator).invoke[Double, Boolean, Unit](
                "seqOp", coerce[Double](defaultValue(transform.typ)), true),
              Code.checkcast[RegionValueSumDoubleAggregator](aggregator).invoke[Double, Boolean, Unit](
                "seqOp", coerce[Double](v), false)),
              new RegionValueSumDoubleAggregator())
          case _ => throw new IllegalArgumentException(s"Cannot sum over values of type ${transform.typ}")
        }
        fb.emit(Code(dov, callSeq))

        ab += IRAgg(in,
          transform.typ, // sum always produces same value as input
          new TransformedRegionValueAggregator(fb.result(), agg))

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
      case In(_, _) | InMissingness(_) =>
        throw new RuntimeException(s"No inputs may be referenced inside an aggregator: $ir")
      case _ => Recur(lower)(ir)
    }
  }
}
