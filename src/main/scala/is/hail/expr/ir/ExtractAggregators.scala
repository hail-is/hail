package is.hail.expr.ir

import is.hail.methods._
import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr
import is.hail.expr.{Type, Aggregator, TAggregable, TArray, TBoolean, TContainer, TFloat32, TFloat64, TInt32, TInt64, TStruct, RegionValueAggregator}
import is.hail.utils._
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

import scala.language.existentials

object ExtractAggregators {

  private case class IRAgg(in: In, agg: RegionValueAggregator) { }

  private object TransformedRegionValueAggregator {
    def apply(aggT: TAggregable,
      makeTransform: IR => IR,
      next: RegionValueAggregator): TransformedRegionValueAggregator = {
      val transform = makeTransform(In(0, aggT.carrierStruct))
      // without a struct we cannot return the missingness of `transform`
      assert(transform.typ != null, s"found null type: $transform")
      println(s"transform: $transform")
      val outT = TStruct(true, "it" -> transform.typ)
      val itIndex = outT.fieldIdx("it")
      val out = MakeStruct(Array(("it", transform.typ, transform)))
      val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Long]
      Compile(out, fb)
      new TransformedRegionValueAggregator(aggT, fb.result(), outT, itIndex, next)
    }
  }

  private class TransformedRegionValueAggregator(
    aggT: TAggregable,
    getTransformer: () => AsmFunction3[MemoryBuffer, Long, Boolean, Long],
    outT: TStruct,
    itIndex: Int,
    val next: RegionValueAggregator) extends RegionValueAggregator {
    val typ = next.typ

    def seqOp(region: MemoryBuffer, off: Long, missing: Boolean) {
      val outOff = getTransformer()(region, off, missing)
      println(s"seqOp ${outT.loadField(region, outOff, itIndex)} ${!outT.isFieldDefined(region, outOff, itIndex)}")
      next.seqOp(region, outT.loadField(region, outOff, itIndex), !outT.isFieldDefined(region, outOff, itIndex))
    }

    def combOp(agg2: RegionValueAggregator) {
      next.combOp(agg2.asInstanceOf[TransformedRegionValueAggregator].next)
    }

    def result(region: MemoryBuffer): Long = {
      next.result(region)
    }

    def copy(): TransformedRegionValueAggregator =
      new TransformedRegionValueAggregator(aggT, getTransformer, outT, itIndex, next)
  }

  class ZippedRegionValueAggregator(val aggs: Array[RegionValueAggregator]) {
    private val fields = aggs.map(_.typ).zipWithIndex.map { case (t, i) => (i.toString -> t) }
    val typ: TStruct = TStruct(true, fields:_*)

    def seqOp(region: MemoryBuffer, off: Long, missing: Boolean) {
      aggs.foreach(_.seqOp(region, off, missing))
    }

    def combOp(agg2: RegionValueAggregator) {
      (aggs zip agg2.asInstanceOf[ZippedRegionValueAggregator].aggs)
        .foreach { case (l,r) => l.combOp(r) }
    }

    def result(region: MemoryBuffer): Long = {
      val rvb = new RegionValueBuilder(region)
      rvb.start(typ)
      rvb.startStruct()
      aggs.foreach(agg => rvb.addRegionValue(agg.typ, region, agg.result(region)))
      rvb.endStruct()
      rvb.end()
    }

    def copy(): ZippedRegionValueAggregator =
      new ZippedRegionValueAggregator(aggs.map(_.copy))
  }

  def apply(ir: IR, aggT: TAggregable): (IR, ZippedRegionValueAggregator) = {
    val (ir2, aggs) = extract(ir, aggT)
    val zrva = new ZippedRegionValueAggregator(aggs map (_.agg))
    aggs.foreach(_.in.typ = zrva.typ)
    (ir2, zrva)
  }

  private def extract(ir: IR, aggT: TAggregable): (IR, Array[IRAgg]) = {
    val ab = new ArrayBuilder[IRAgg]()
    val ir2 = extract(ir, ab, aggT)
    (ir2, ab.result())
  }

  private def extract(ir: IR, ab: ArrayBuilder[IRAgg], aggT: TAggregable): IR = {
    def extract(ir: IR): IR = this.extract(ir, ab, aggT)
    ir match {
      case Ref(name, typ) =>
        assert(typ.isRealizable)
        ir
      case AggIn(_) =>
        throw new RuntimeException(s"AggMap must be used inside an AggSum, but found: $ir")
      case AggMap(a, name, body, typ) =>
        throw new RuntimeException(s"AggMap must be used inside an AggSum, but found: $ir")
      case AggSum(a, typ) =>
        val tAgg = a.typ.asInstanceOf[TAggregable]
        val in = In(0, null)
        ab += IRAgg(in,
          TransformedRegionValueAggregator(aggT,
            aggin => tAgg.getElement(lower(a, aggin)),
            RegionValueSumAggregator(tAgg.elementType)))
        GetField(in, (ab.length - 1).toString(), typ)
      case _ => Recur(extract)(ir)
    }
  }

  private def lower(ir: IR, aggIn: IR): IR = {
    def lower(ir: IR): IR = this.lower(ir, aggIn)
    ir match {
      case AggIn(typ) =>
        aggIn
      case AggMap(a, name, body, typ) =>
        val tAgg = a.typ.asInstanceOf[TAggregable]
        val tA = tAgg.carrierStruct
        val la = lower(a)
        assert(la.typ == tA, s"should have same type ${la.typ} $tA, $la")
        tAgg.inContext(la, e => Let(name, e, lower(body), typ.elementType))
      case AggSum(_, _) =>
        throw new RuntimeException(s"Found aggregator inside an aggregator: $ir")
      case In(i, typ) =>
        throw new RuntimeException(s"Referenced input inside an aggregator, that's a no-no: $ir")
      case InMissingness(i) =>
        throw new RuntimeException(s"Referenced input inside an aggregator, that's a no-no: $ir")
      case _ => Recur(lower)(ir)
    }
  }
}
