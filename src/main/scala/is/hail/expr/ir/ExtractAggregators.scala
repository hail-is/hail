package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

import scala.language.{existentials, postfixOps}

object ExtractAggregators {

  private case class IRAgg(ref: Ref, applyAggOp: ApplyAggOp) {}

  def apply(ir: IR): (IR, TStruct, IR, IR, Array[RegionValueAggregator]) = {
    def rewriteSeqOps(x: IR, i: Int): IR = {
      def rewrite(x: IR): IR = rewriteSeqOps(x, i)
      x match {
        case SeqOp(a, _, aggSig) =>
          SeqOp(a, I32(i), aggSig)
        case _ => Recur(rewrite)(x)
      }
    }

    val (ir2, aggs) = extract(ir.unwrap)

    val (initOps, seqOps) = aggs.map(_.applyAggOp)
      .zipWithIndex
      .map { case (x, i) =>
        (x.initOpArgs.map(args => InitOp(I32(i), args, x.aggSig)), rewriteSeqOps(x.a, i))
      }.unzip

    val seqOpIR = Begin(seqOps)
    val initOpIR = Begin(initOps.flatten[InitOp])

    val rvas = aggs.map(_.applyAggOp)
      .map { x =>
        newAggregator(x)
      }

    val fields = aggs.map(_.applyAggOp.typ).zipWithIndex.map { case (t, i) => i.toString -> t }
    val resultStruct = TStruct(fields: _*)
    // mutate the type of the input IR node now that we know what the combined
    // struct's type is
    aggs.foreach(_.ref.typ = resultStruct)

    (ir2, resultStruct, initOpIR, seqOpIR, rvas)
  }

  private def extract(ir: IR): (IR, Array[IRAgg]) = {
    val ab = new ArrayBuilder[IRAgg]()
    val ir2 = extract(ir, ab)
    (ir2, ab.result())
  }

  private def extract(ir: IR, ab: ArrayBuilder[IRAgg]): IR = {
    def extract(ir: IR): IR = this.extract(ir, ab)

    ir match {
      case Ref(name, typ) =>
        assert(typ.isRealizable)
        ir
      case x: ApplyAggOp =>
        val ref = Ref("AGGR", null)
        ab += IRAgg(ref, x)

        GetField(ref, (ab.length - 1).toString)
      case _ => Recur(extract)(ir)
    }
  }

  private def newAggregator(ir: ApplyAggOp): RegionValueAggregator = ir match {
    case x@ApplyAggOp(a, constructorArgs, initOpArgs, aggSig) =>
      val constfb = EmitFunctionBuilder[Region, RegionValueAggregator]
      val codeConstructorArgs = constructorArgs.map(Emit.toCode(_, constfb, 1))
      constfb.emit(Code(
        Code(codeConstructorArgs.map(_.setup): _*),
        AggOp.get(aggSig)
          .stagedNew(codeConstructorArgs.map(_.v).toArray, codeConstructorArgs.map(_.m).toArray)))
      Region.scoped(constfb.result()()(_))
  }
}
