package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

import scala.language.{existentials, postfixOps}

case class ExtractedAggregators(postAggIR: IR, resultType: TTuple, init: IR, perElt: IR, rvAggs: Array[RegionValueAggregator])

object ExtractAggregators {

  private case class IRAgg(i: Int, rvAgg: RegionValueAggregator, rt: Type)
  private case class AggOps(initOp: Option[IR], seqOp: IR)

  def apply(ir: IR, resultName: String = "AGGR"): ExtractedAggregators = {
    val ab = new ArrayBuilder[IRAgg]()
    val ab2 = new ArrayBuilder[AggOps]()
    val ref = Ref(resultName, null)
    val postAgg = extract(ir, ab, ab2, ref)
    val aggs = ab.result()
    val rt = TTuple(aggs.map(_.rt): _*)
    ref.typ = rt
    val ops = ab2.result()
    ExtractedAggregators(
      postAgg,
      rt,
      Begin(ops.flatMap(_.initOp)),
      Begin(ops.map(_.seqOp)),
      aggs.map(_.rvAgg))
  }

  private def extract(ir: IR, ab: ArrayBuilder[IRAgg], ab2: ArrayBuilder[AggOps], result: IR): IR = {
    def extract(node: IR): IR = this.extract(node, ab, ab2, result)

    ir match {
      case Ref(name, typ) =>
        assert(typ.isRealizable)
        ir
      case x: ApplyAggOp =>
        val i = ab.length
        ab += IRAgg(i, newAggregator(x), x.typ)
        ab2 += AggOps(
          x.initOpArgs.map(InitOp(i, _, x.aggSig)),
          SeqOp(i, x.seqOpArgs, x.aggSig))
        GetTupleElement(result, i)
      case AggFilter(cond, aggIR) =>
        val newBuilder = new ArrayBuilder[AggOps]()
        val transformed = this.extract(aggIR, ab, newBuilder, result)
        val (initOp, seqOp) = newBuilder.result().map { case AggOps(x, y) => (x, y) }.unzip
        val io = if (initOp.flatten.isEmpty) None else Some(Begin(initOp.flatten.toFastIndexedSeq))
        ab2 += AggOps(io,
          If(cond, Begin(seqOp), Begin(FastIndexedSeq())))
        transformed
      case AggExplode(array, name, aggBody) =>
        val newBuilder = new ArrayBuilder[AggOps]()
        val transformed = this.extract(aggBody, ab, newBuilder, result)
        val (initOp, seqOp) = newBuilder.result().map { case AggOps(x, y) => (x, y) }.unzip
        val io = if (initOp.flatten.isEmpty) None else Some(Begin(initOp.flatten.toFastIndexedSeq))
        ab2 += AggOps(
          io,
          ArrayFor(array, name, Begin(seqOp)))
        transformed
      case AggGroupBy(key, aggIR) =>

        val newRVAggBuilder = new ArrayBuilder[IRAgg]()
        val newBuilder = new ArrayBuilder[AggOps]()
        val newRef = Ref(genUID(), null)
        val transformed = this.extract(aggIR, newRVAggBuilder, newBuilder, GetField(newRef, "value"))

        val nestedAggs = newRVAggBuilder.result()
        val agg = KeyedRegionValueAggregator(nestedAggs.map(_.rvAgg), key.typ)
        val aggSig = AggSignature(Group(), Seq(), Some(Seq(TVoid)), Seq(key.typ, TVoid))
        val rt = TDict(key.typ, TTuple(nestedAggs.map(_.rt): _*))
        newRef.typ = -rt.elementType

        val (initOp, seqOp) = newBuilder.result().map { case AggOps(x, y) => (x, y) }.unzip
        val i = ab.length
        ab += IRAgg(i, agg, rt)
        ab2 += AggOps(
          Some(InitOp(i, FastIndexedSeq(Begin(initOp.flatten.toFastIndexedSeq)), aggSig)),
          SeqOp(I32(i), FastIndexedSeq(key, Begin(seqOp)), aggSig))

        ToDict(ArrayMap(ToArray(GetTupleElement(result, i)), newRef.name, MakeTuple(FastSeq(GetField(newRef, "key"), transformed))))
      case _ => MapIR(extract)(ir)
    }
  }

  private def newAggregator(ir: ApplyAggOp): RegionValueAggregator = ir match {
    case x@ApplyAggOp(_, constructorArgs, _, aggSig) =>
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
        case AggSignature(InfoScore(), _, _, Seq(t)) =>
          codeConstructorArgs = FastIndexedSeq(EmitTriplet(Code._empty, const(false), fb.getType(t)))
        case AggSignature(LinearRegression(), _, _, Seq(_, xType)) =>
          codeConstructorArgs ++= FastIndexedSeq(EmitTriplet(Code._empty, const(false), fb.getType(xType)))
        case AggSignature(Sum(), _, _, Seq(t@(_: TInt64 | _: TFloat64))) =>
        case AggSignature(Sum(), _, _, Seq(t)) =>
          codeConstructorArgs = FastIndexedSeq(EmitTriplet(Code._empty, const(false), fb.getType(t)))
        case _ =>
      }

        fb.emit(Code(
          Code(codeConstructorArgs.map(_.setup): _*),
          AggOp.get(aggSig).asInstanceOf[CodeAggregator[_]]
            .stagedNew(codeConstructorArgs.map(_.v).toArray, codeConstructorArgs.map(_.m).toArray)))
        Region.scoped(fb.resultWithIndex()(0)(_))
  }
}
