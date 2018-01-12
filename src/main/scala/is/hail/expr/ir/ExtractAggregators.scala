package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

import scala.language.{existentials, postfixOps}

object ExtractAggregators {

  private case class IRAgg(in: In, applyAggOp: ApplyAggOp) { }

  def apply(ir: IR, tAggIn: TAggregable, nSpecialArguments: Int): (IR, TStruct, Array[(IR, RegionValueAggregator)]) = {
    val (ir2, aggs) = extract(ir, tAggIn)
    val rvas = aggs.map(_.applyAggOp).map(x => (x: IR, newAggregator(x)))

    val fields = aggs.map(_.applyAggOp.typ).zipWithIndex.map { case (t, i) => i.toString -> t }
    val resultStruct = TStruct(fields: _*)
    // mutate the type of the input IR node now that we know what the combined
    // struct's type is
    aggs.foreach(_.in.typ = resultStruct)

    (ir2, resultStruct, rvas)
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
      case _: AggIn | _: AggMap | _: AggFilter | _: AggFlatMap =>
        throw new RuntimeException(s"Aggregable manipulations must appear inside the lexical scope of an Aggregation: $ir")
      case x: ApplyAggOp =>
        val in = In(0, null)

        ab += IRAgg(in, x)

        GetField(in, (ab.length - 1).toString, x.typ)
      case _ => Recur(extract)(ir)
    }
  }

  private def newAggregator(ir: ApplyAggOp): RegionValueAggregator = ir match {
    case x@ApplyAggNullaryOp(a, op, typ) =>
      AggOp.getNullary(op, x.inputType).aggregator
    case x@ApplyAggUnaryOp(a, op, arg1, typ) =>
      val constfb = FunctionBuilder.functionBuilder[Region, RegionValueAggregator]
      val (doarg1, marg1, varg1) = Emit.toCode(arg1, constfb, 1)
      constfb.emit(Code(
        doarg1,
        AggOp.getUnary(op, arg1.typ, x.inputType).stagedNew(varg1, coerce[Boolean](marg1))))
      constfb.result()()(Region())
    case x@ApplyAggTernaryOp(a, op, arg1, arg2, arg3, typ) =>
      val constfb = FunctionBuilder.functionBuilder[Region, RegionValueAggregator]
      val (doarg1, marg1, varg1) = Emit.toCode(arg1, constfb, 1)
      val (doarg2, marg2, varg2) = Emit.toCode(arg2, constfb, 1)
      val (doarg3, marg3, varg3) = Emit.toCode(arg3, constfb, 1)
      constfb.emit(Code(
        doarg1,
        doarg2,
        doarg3,
        AggOp.getTernary(op, arg1.typ, arg2.typ, arg3.typ, x.inputType)
          .stagedNew(varg1, marg1, varg2, marg2, varg3, marg3)))
      constfb.result()()(Region())
  }
}
