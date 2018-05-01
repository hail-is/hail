package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

import scala.language.{existentials, postfixOps}

object ExtractAggregators {

  private case class IRAgg(ref: Ref, applyAggOp: ApplyAggOp) {}

  def apply(ir: IR, tAggIn: TAggregable): (IR, TStruct, IR, Array[RegionValueAggregator]) = {
    val (ir2, aggs) = extract(ir, tAggIn)
    val aggir = Begin(
      aggs.map(_.applyAggOp)
        .zipWithIndex
        .map { case (x, i) =>
          val agg = AggOp.get(x.op, x.inputType, x.args.map(_.typ))
          SeqOp(x.a, I32(i), agg)
        })

    val rvas = aggs.map(_.applyAggOp)
      .map { x =>
        newAggregator(x)
      }

    val fields = aggs.map(_.applyAggOp.typ).zipWithIndex.map { case (t, i) => i.toString -> t }
    val resultStruct = TStruct(fields: _*)
    // mutate the type of the input IR node now that we know what the combined
    // struct's type is
    aggs.foreach(_.ref.typ = resultStruct)

    (ir2, resultStruct, aggir, rvas)
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
        val ref = Ref("AGGR", null)
        ab += IRAgg(ref, x)

        GetField(ref, (ab.length - 1).toString)
      case _ => Recur(extract)(ir)
    }
  }

  private def newAggregator(ir: ApplyAggOp): RegionValueAggregator = ir match {
    case x@ApplyAggOp(a, op, args) =>
      val constfb = EmitFunctionBuilder[Region, RegionValueAggregator]
      val codeArgs = args.map(Emit.toCode(_, constfb, 1))
      constfb.emit(Code(
        Code(codeArgs.map(_.setup): _*),
        AggOp.get(op, x.inputType, args.map(_.typ))
          .stagedNew(codeArgs.map(_.v).toArray, codeArgs.map(_.m).toArray)))
      constfb.result()()(Region())
  }
}
