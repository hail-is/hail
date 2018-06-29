package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

import scala.language.{existentials, postfixOps}

object ExtractAggregators {

  private case class IRAgg(ref: Ref, applyAggOp: ApplyAggOp) {}

  def apply(ir: IR, resultName: String = "AGGR"): (IR, TStruct, IR, IR, Array[RegionValueAggregator]) = {
    def rewriteSeqOps(x: IR, i: Int): IR = {
      def rewrite(x: IR): IR = rewriteSeqOps(x, i)
      x match {
        case SeqOp(_, args, aggSig) =>
          SeqOp(I32(i), args, aggSig)
        case _ => Recur(rewrite)(x)
      }
    }

    val (ir2, aggs) = extract(ir, resultName)

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

  private def extract(ir: IR, resultName: String): (IR, Array[IRAgg]) = {
    val ab = new ArrayBuilder[IRAgg]()
    val ir2 = extract(ir, ab, resultName)
    (ir2, ab.result())
  }

  private def extract(ir: IR, ab: ArrayBuilder[IRAgg], resultName: String): IR = {
    def extract(ir: IR): IR = this.extract(ir, ab, resultName)

    ir match {
      case Ref(name, typ) =>
        assert(typ.isRealizable)
        ir
      case x: ApplyAggOp =>
        val ref = Ref(resultName, null)
        ab += IRAgg(ref, x)

        GetField(ref, (ab.length - 1).toString)
      case _ => Recur(extract)(ir)
    }
  }

  private def newAggregator(ir: ApplyAggOp): RegionValueAggregator = ir match {
    case x@ApplyAggOp(a, constructorArgs, _, aggSig) =>
      val fb = EmitFunctionBuilder[Region, RegionValueAggregator]
      var codeConstructorArgs = constructorArgs.map(Emit.toCode(_, fb, 1))

      aggSig match {
        case AggSignature(Collect() | Take() | CollectAsSet(), _, _, Seq(t@(_: TBoolean | _: TInt32 | _: TInt64 | _: TFloat32 | _: TFloat64 | _: TCall))) =>
        case AggSignature(Collect() | Take() | CollectAsSet(), _, _, Seq(t)) =>
          codeConstructorArgs ++= FastIndexedSeq(EmitTriplet(Code._empty, const(false), fb.getType(t)))
        case AggSignature(Counter(), _, _, Seq(t@(_: TBoolean))) =>
        case AggSignature(Counter(), _, _, Seq(t)) =>
          codeConstructorArgs = FastIndexedSeq(EmitTriplet(Code._empty, const(false), fb.getType(t)))
        case AggSignature(TakeBy(), _, _, Seq(aggType, keyType)) =>
          codeConstructorArgs ++= FastIndexedSeq(EmitTriplet(Code._empty, const(false), fb.getType(aggType)),
            EmitTriplet(Code._empty, const(false), fb.getType(keyType)))
        case _ =>
      }

      fb.emit(Code(
        Code(codeConstructorArgs.map(_.setup): _*),
        AggOp.get(aggSig)
          .stagedNew(codeConstructorArgs.map(_.v).toArray, codeConstructorArgs.map(_.m).toArray)))

      Region.scoped(fb.result()()(_))
  }
}
