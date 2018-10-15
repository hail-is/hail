package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

import scala.language.{existentials, postfixOps}

case class ExtractedAggregators(postAggIR: IR, resultType: TTuple, init: IR, perElt: IR, rvAggs: Array[RegionValueAggregator])

object ExtractAggregators2 {

  private case class IRAgg(i: Int, rvAgg: RegionValueAggregator, seqOp: IR, init: Option[IR], typ: Type)

  def apply(ir: IR, resultName: String = "AGGR"): ExtractedAggregators = {
    val ab = new ArrayBuilder[IRAgg]()
    val postAgg = extract(ir, ab, Ref(resultName, null))
    val aggs = ab.result()
    ExtractedAggregators(
      postAgg,
      TTuple(aggs.map(_.typ): _*),
      Begin(aggs.flatMap(_.init)),
      Begin(aggs.map(_.seqOp)),
      aggs.map(_.rvAgg))
  }

  private def fromAggIR(i: Int, aggOp: ApplyAggOp): IRAgg = {
    IRAgg(i, newAggregator(aggOp), SeqOp(I32(i), aggOp.seqOpArgs, aggOp.aggSig), aggOp.initOpArgs.map(InitOp(I32(i), _, aggOp.aggSig)), aggOp.typ)
  }
  private def extract(ir: IR, ab: ArrayBuilder[IRAgg], result: Ref, transform: (Int, ApplyAggOp) => IRAgg = fromAggIR): IR = {
    def extractWithTransform(node: IR, transform: (Int, ApplyAggOp) => IRAgg): IR = this.extract(node, ab, result, transform)

    def extract(node: IR): IR = extractWithTransform(node, transform)

    ir match {
      case Ref(name, typ) =>
        assert(typ.isRealizable)
        ir
      case x: ApplyAggOp =>
        val i = ab.length - 1
        ab += transform(IRAgg(i, newAggregator(x), ))
        GetTupleElement(result, ab.length - 1)
      case AggFilter(cond, aggIR) =>
        val filtered = { node: IR =>
          If(cond,
            transform(node),
            Begin(FastIndexedSeq()))
        }
        extractWithTransform(aggIR, filtered)
      case AggExplode(array, name, aggBody) =>
        val exploded = { node: IR =>
          ArrayFor(
            array,
            name,
            transform(node))
        }
        extractWithTransform(aggBody, exploded)
      case AggGroupBy(key, aggIR) =>

      case _ => MapIR(extract)(ir)
    }
  }

  private def newAggregator(ir: ApplyAggOp): RegionValueAggregator = ir match {
    case x@ApplyAggOp(_, constructorArgs, _, aggSig) =>

      def getAggregator(op: AggOp, aggSig: AggSignature): RegionValueAggregator = op match {
        case Keyed(subop) =>
          val newAggSig = aggSig.copy(op = subop, seqOpArgs = aggSig.seqOpArgs.drop(1))
          KeyedRegionValueAggregator(getAggregator(subop, newAggSig), aggSig.seqOpArgs.head)
        case _ =>
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

      getAggregator(x.op, aggSig)
  }
}

object ExtractAggregators {

  private case class IRAgg(ref: Ref, applyAggOp: ApplyAggOp) {}

  def apply(ir: IR, resultName: String = "AGGR"): (IR, TStruct, IR, IR, Array[RegionValueAggregator]) = {
    def rewriteSeqOps(x: IR, i: Int): IR = {
      def rewrite(x: IR): IR = rewriteSeqOps(x, i)
      x match {
        case SeqOp(_, args, aggSig) =>
          SeqOp(I32(i), args, aggSig)
        case _ => MapIR(rewrite)(x)
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
      case _ => MapIR(extract)(ir)
    }
  }

  private def newAggregator(ir: ApplyAggOp): RegionValueAggregator = ir match {
    case x@ApplyAggOp(_, constructorArgs, _, aggSig) =>

      def getAggregator(op: AggOp, aggSig: AggSignature): RegionValueAggregator = op match {
          case Keyed(subop) =>
            val newAggSig = aggSig.copy(op = subop, seqOpArgs = aggSig.seqOpArgs.drop(1))
            KeyedRegionValueAggregator(getAggregator(subop, newAggSig), aggSig.seqOpArgs.head)
          case _ =>
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

      getAggregator(x.op, aggSig)
  }
}
