package is.hail.expr.ir

import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types.physical.PTuple
import is.hail.expr.types.virtual._
import is.hail.utils._

import scala.language.{existentials, postfixOps}
import scala.reflect.ClassTag

case class StagedExtractedAggregators(postAggIR: IR, resultType: PTuple, init: IR, perElt: IR, rvAggs: Code[Array[RegionValueAggregator]])

object StagedExtractAggregators {

  private case class IRAgg(i: Int, rvAgg: Code[RegionValueAggregator], rt: Type)

  private case class AggOps(initOp: Option[IR], seqOp: IR)

  def newArray[T](fb: EmitFunctionBuilder[_], a: Array[Code[T]])(implicit tct: ClassTag[T], tti: TypeInfo[T]): Code[Array[T]] = {
    val sa = fb.newField[Array[T]]("a")
    Code(
      sa := Code.newArray[T](const(a.length)),
      Code(a.zipWithIndex.map { case (ai, i) => sa.load().update(const(i), ai) }: _*),
      sa.load())
  }

  def apply(fb: EmitFunctionBuilder[_], ir: IR, resultName: String = "AGGR"): StagedExtractedAggregators = {
    val ab = new ArrayBuilder[IRAgg]()
    val ab2 = new ArrayBuilder[AggOps]()
    val ab3 = new ArrayBuilder[AggLet]()
    val ref = Ref(resultName, null)
    val postAgg = extract(fb, ir, ab, ab2, ab3, ref)
    val aggs = ab.result()
    val aggLets = ab3.result()
    assert(aggLets.areDistinct())
    val rt = TTuple(aggs.map(_.rt): _*)
    ref._typ = rt
    val ops = ab2.result()
    StagedExtractedAggregators(
      postAgg,
      rt.physicalType,
      Begin(ops.flatMap(_.initOp)),
      aggLets.foldRight[IR](Begin(ops.map(_.seqOp))) { case (al, comb) => Let(al.name, al.value, comb)},
      newArray(fb, aggs.map(_.rvAgg)))
  }

  private def extract(fb: EmitFunctionBuilder[_], ir: IR, ab: ArrayBuilder[IRAgg], ab2: ArrayBuilder[AggOps], ab3: ArrayBuilder[AggLet], result: IR): IR = {
    def extract(node: IR): IR = this.extract(fb, node, ab, ab2, ab3, result)

    ir match {
      case Ref(name, typ) =>
        assert(typ.isRealizable)
        ir
      case x@AggLet(name, value, body) =>
        ab3 += x
        extract(body)
      case x: ApplyAggOp =>
        val i = ab.length
        ab += IRAgg(i, newAggregator(fb, x), x.typ)
        ab2 += AggOps(
          x.initOpArgs.map(InitOp(i, _, x.aggSig)),
          SeqOp(i, x.seqOpArgs, x.aggSig))
        GetTupleElement(result, i)
      case AggFilter(cond, aggIR) =>
        val newBuilder = new ArrayBuilder[AggOps]()
        val transformed = this.extract(fb, aggIR, ab, newBuilder, ab3, result)
        val (initOp, seqOp) = newBuilder.result().map { case AggOps(x, y) => (x, y) }.unzip
        val io = if (initOp.flatten.isEmpty) None else Some(Begin(initOp.flatten.toFastIndexedSeq))
        ab2 += AggOps(io,
          If(cond, Begin(seqOp), Begin(FastIndexedSeq())))
        transformed
      case AggExplode(array, name, aggBody) =>
        val newBuilder = new ArrayBuilder[AggOps]()
        val transformed = this.extract(fb, aggBody, ab, newBuilder, ab3, result)
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
        val transformed = this.extract(fb, aggIR, newRVAggBuilder, newBuilder, ab3, GetField(newRef, "value"))

        val nestedAggs = newRVAggBuilder.result()
        val agg = Code.newInstance[KeyedRegionValueAggregator, Array[RegionValueAggregator], Type](newArray(fb, nestedAggs.map(_.rvAgg)), fb.getType(key.typ))
        val aggSig = AggSignature(Group(), Seq(), Some(Seq(TVoid)), Seq(key.typ, TVoid))
        val rt = TDict(key.typ, TTuple(nestedAggs.map(_.rt): _*))
        newRef._typ = -rt.elementType

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

  private def newAggregator(fb: EmitFunctionBuilder[_], ir: ApplyAggOp): Code[RegionValueAggregator] = ir match {
    case x@ApplyAggOp(constructorArgs, _, _, aggSig) =>
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

      Code(
        Code(codeConstructorArgs.map(_.setup): _*),
        AggOp.get(aggSig).asInstanceOf[CodeAggregator[_]]
          .stagedNew(codeConstructorArgs.map(_.v).toArray, codeConstructorArgs.map(_.m).toArray))
        .asInstanceOf[Code[RegionValueAggregator]]
  }
}
