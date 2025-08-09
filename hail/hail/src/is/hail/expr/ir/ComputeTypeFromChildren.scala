package is.hail.expr.ir

import is.hail.annotations.Annotation
import is.hail.expr.Nat
import is.hail.expr.ir.agg.{AggStateSig, PhysicalAggSig}
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.{
  BlockMatrixToValueFunction, MatrixToValueFunction, TableToValueFunction,
}
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec}
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._

object ComputeTypeFromChildren {
  @inline def I32(x: Int): TInt32.type = TInt32
  @inline def I64(x: Long) = TInt64
  @inline def F32(x: Float) = TFloat32
  @inline def F64(x: Double) = TFloat64
  @inline def Str(x: String) = TString
  @inline def UUID4(id: String) = TString
  @inline def Literal(t: Type, value: Annotation) = t

  @inline def EncodedLiteral(codec: AbstractTypedCodecSpec, value: WrappedByteArrays) =
    codec.encodedVirtualType

  @inline def True() = TBoolean
  @inline def False() = TBoolean
  @inline def Void() = TVoid

  @inline def Cast(v: IR, typ: Type) =
    if (!Casts.valid(v.typ, typ))
      throw new RuntimeException(s"invalid cast:\n  " +
        s"child type: ${v.typ.parsableString()}\n  " +
        s"cast type:  ${typ.parsableString()}")

  @inline def CastRename(v: IR, typ: Type) =
    if (!v.typ.isIsomorphicTo(typ))
      throw new RuntimeException(s"invalid cast:\n  " +
        s"child type: ${v.typ.parsableString()}\n  " +
        s"cast type:  ${typ.parsableString()}")

  @inline def NA(typ: Type) = {}
  @inline def IsNA(value: IR) = TBoolean

  @inline def Coalesce(values: IndexedSeq[IR]) = {
    assert(values.tail.forall(_.typ == values.head.typ))
    values.head.typ
  }

  @inline def Consume(value: IR) = TInt64

  @inline def Ref(name: Name, typ: Type) = {}

  @inline def RelationalRef(name: Name, typ: Type) = {}
  @inline def RelationalLet(name: Name, value: IR, body: IR) = body.typ

  @inline def In(i: Int, paramType: EmitParamType) = {
    assert(paramType != null)
    paramType.virtualType match {
      case stream: TStream => assert(stream.elementType.isRealizable)
      case _ =>
    }
    paramType.virtualType
  }

  @inline def MakeArray(args: IndexedSeq[IR], typ: TArray) = {
    assert(typ.elementType.isRealizable, typ.elementType)
    args.map(_.typ).zipWithIndex.foreach { case (x, i) =>
      assert(
        x == typ.elementType,
        s"at position $i type mismatch: ${typ.parsableString()} ${x.parsableString()}",
      )
    }
    typ
  }

  @inline def MakeStream(
    args: IndexedSeq[IR],
    typ: TStream,
    requiresMemoryManagementPerElement: Boolean,
  ) = {
    assert(typ.elementType.isRealizable, typ.elementType)
    args.map(_.typ).zipWithIndex.foreach { case (x, i) =>
      assert(
        x == typ.elementType,
        s"at position $i type mismatch: ${typ.elementType.parsableString()} ${x.parsableString()}",
      )
    }
    typ
  }

  @inline def MakeNDArray(data: IR, shape: IR, rowMajor: IR, errorID: Int) = {
    val shapeTyp = tcoerce[TTuple](shape.typ)
    assert(data.typ.isInstanceOf[TArray] || data.typ.isInstanceOf[TStream])
    assert(shapeTyp.types.forall(t => t == TInt64))
    assert(rowMajor.typ == TBoolean)
    TNDArray(tcoerce[TIterable](data.typ).elementType, Nat(shapeTyp.size))
  }

  @inline def StreamBufferedAggregate(
    streamChild: IR,
    initAggs: IR,
    newKey: IR,
    seqOps: IR,
    name: Name,
    aggSignature: IndexedSeq[PhysicalAggSig],
    bufferSize: Int,
  ) = {
    assert(streamChild.typ.isInstanceOf[TStream])
    assert(initAggs.typ == TVoid)
    assert(seqOps.typ == TVoid)
    val tupleFieldTypes = TTuple(aggSignature.map(_ => TBinary): _*)
    TStream(tcoerce[TStruct](newKey.typ).insertFields(IndexedSeq(("agg", tupleFieldTypes))))
  }

  @inline def ArrayLen(a: IR) = {
    assert(a.typ.isInstanceOf[TArray])
    TInt32
  }

  @inline def StreamIota(
    start: IR,
    step: IR,
    requiresMemoryManagementPerElement: Boolean = false,
  ) = {
    assert(start.typ == TInt32)
    assert(step.typ == TInt32)
    TStream(TInt32)
  }

  @inline def StreamRange(
    start: IR,
    stop: IR,
    step: IR,
    requiresMemoryManagementPerElement: Boolean = false,
    errorID: Int = ErrorIDs.NO_ERROR,
  ) = {
    assert(start.typ == TInt32)
    assert(stop.typ == TInt32)
    assert(step.typ == TInt32)
    TStream(TInt32)
  }

  @inline def SeqSample(
    totalRange: IR,
    numToSample: IR,
    rngState: IR,
    requiresMemoryManagementPerElement: Boolean = false,
  ) = {
    assert(totalRange.typ == TInt32)
    assert(numToSample.typ == TInt32)
    assert(rngState.typ == TRNGState)
    TStream(TInt32)
  }

  @inline def ArrayZeros(length: IR) = {
    assert(length.typ == TInt32)
    TArray(TInt32)
  }

  @inline def LowerBoundOnOrderedCollection(orderedCollection: IR, elem: IR, onKey: Boolean) = {
    val elt = tcoerce[TContainer](orderedCollection.typ).elementType
    assert(elem.typ == (if (onKey) elt match {
                          case t: TBaseStruct => t.types(0)
                          case t: TInterval => t.pointType
                        }
                        else elt))
    TInt32
  }

  @inline def StreamFor(a: IR, valueName: Name, body: IR) = {
    assert(a.typ.isInstanceOf[TStream])
    assert(body.typ == TVoid)
    TVoid
  }

  @inline def InitOp(i: Int, args: IndexedSeq[IR], aggSig: PhysicalAggSig) = {
    assert(
      args.map(_.typ) == aggSig.initOpTypes,
      s"${args.map(_.typ)} !=  ${aggSig.initOpTypes}",
    )
    TVoid
  }

  @inline def SeqOp(i: Int, args: IndexedSeq[IR], aggSig: PhysicalAggSig) = {
    assert(args.map(_.typ) == aggSig.seqOpTypes)
    TVoid
  }

  @inline def CombOp(i1: Int, i2: Int, aggSig: PhysicalAggSig) = TVoid
  @inline def ResultOp(idx: Int, aggSig: PhysicalAggSig) = aggSig.resultType
  @inline def AggStateValue(i: Int, aggSig: AggStateSig) = TBinary

  @inline def CombOpValue(i: Int, value: IR, aggSig: PhysicalAggSig) = {
    assert(value.typ == TBinary)
    TVoid
  }

  @inline def InitFromSerializedValue(i: Int, value: IR, aggSig: AggStateSig) = {
    assert(value.typ == TBinary)
    TVoid
  }

  @inline def SerializeAggs(
    startIdx: Int,
    serializedIdx: Int,
    spec: BufferSpec,
    aggSigs: IndexedSeq[AggStateSig],
  ) = TVoid

  @inline def DeserializeAggs(
    startIdx: Int,
    serializedIdx: Int,
    spec: BufferSpec,
    aggSigs: IndexedSeq[AggStateSig],
  ) = TVoid

  @inline def Die(message: IR, typ: Type, errorID: Int) =
    assert(message.typ == TString)

  @inline def Trap(child: IR) = TTuple(TTuple(TString, TInt32), child.typ)

  @inline def ConsoleLog(message: IR, result: IR) = {
    assert(message.typ == TString)
    result.typ
  }

  @inline def If(cond: IR, cnsq: IR, altr: IR) = {
    assert(cond.typ == TBoolean)
    assert(cnsq.typ == altr.typ)
    cnsq.typ match {
      case tstream: TStream => assert(tstream.elementType.isRealizable)
      case _ =>
    }
    cnsq.typ
  }

  @inline def Switch(x: IR, default: IR, cases: IndexedSeq[IR]) = {
    assert(x.typ == TInt32)
    assert(cases.forall(_.typ == default.typ))
    default.typ
  }

  @inline def Block(bindings: IndexedSeq[Binding], body: IR) = body.typ

  @inline def TailLoop(name: Name, params: IndexedSeq[(Name, IR)], typ: Type, body: IR) = {}

  @inline def Recur(name: Name, args: IndexedSeq[IR], typ: Type) = {}

  @inline def ApplyBinaryPrimOp(op: BinaryOp, l: IR, r: IR) =
    BinaryOp.getReturnType(op, l.typ, r.typ)

  @inline def ApplyUnaryPrimOp(op: UnaryOp, x: IR) =
    UnaryOp.getReturnType(op, x.typ)

  @inline def ApplyComparisonOp(op: ComparisonOp[_], l: IR, r: IR) = {
    assert(op.t1 == l.typ)
    assert(op.t2 == r.typ)
    ComparisonOp.checkCompatible(op.t1, op.t2)
    op match {
      case _: Compare => TInt32
      case _ => TBoolean
    }
  }

  @inline def ApplyIR(
    function: String,
    typeArgs: Seq[Type],
    args: IndexedSeq[IR],
    typ: Type,
    errorID: Int = ErrorIDs.NO_ERROR,
  ) = {}

  @inline def Apply(
    function: String,
    typeArgs: Seq[Type],
    args: IndexedSeq[IR],
    typ: Type,
    errorID: Int = ErrorIDs.NO_ERROR,
  ) = {}

  @inline def ApplySeeded(
    function: String,
    _args: IndexedSeq[IR],
    rngState: IR,
    staticUID: Long,
    typ: Type,
  ) = {}

  @inline def ApplySpecial(
    function: String,
    typeArgs: Seq[Type],
    args: IndexedSeq[IR],
    typ: Type,
    errorID: Int = ErrorIDs.NO_ERROR,
  ) = {}

  @inline def ArrayRef(a: IR, i: IR, errorID: Int) = {
    assert(i.typ == TInt32)
    tcoerce[TArray](a.typ).elementType
  }

  @inline def ArraySlice(a: IR, start: IR, stop: Option[IR], step: IR, errorID: Int) = {
    assert(start.typ == TInt32)
    stop.foreach(ir => assert(ir.typ == TInt32))
    assert(step.typ == TInt32)
    tcoerce[TArray](a.typ)
  }

  @inline def ArraySort(a: IR, left: Name, right: Name, lessThan: IR) = {
    assert(lessThan.typ == TBoolean)
    val et = tcoerce[TStream](a.typ).elementType
    TArray(et)
  }

  @inline def ArrayMaximalIndependentSet(edges: IR, tieBreaker: Option[(Name, Name, IR)]) = {
    val edgeType = tcoerce[TBaseStruct](tcoerce[TArray](edges.typ).elementType)
    val Array(leftType, rightType) = edgeType.types
    assert(leftType == rightType)
    tieBreaker.foreach { case (_, _, tb) => assert(tb.typ == TFloat64) }
    TArray(leftType)
  }

  @inline def ToSet(a: IR) = {
    val et = tcoerce[TStream](a.typ).elementType
    TSet(et)
  }

  @inline def ToDict(a: IR) = {
    val elt = tcoerce[TBaseStruct](tcoerce[TStream](a.typ).elementType)
    assert(elt.size == 2)
    TDict(elt.types(0), elt.types(1))
  }

  @inline def ToArray(a: IR) = {
    val elt = tcoerce[TStream](a.typ).elementType
    TArray(elt)
  }

  @inline def CastToArray(a: IR) = {
    val elt = tcoerce[TContainer](a.typ).elementType
    TArray(elt)
  }

  @inline def ToStream(a: IR, requiresMemoryManagementPerElement: Boolean = false) = {
    val elt = tcoerce[TContainer](a.typ).elementType
    TStream(elt)
  }

  @inline def RNGStateLiteral() = TRNGState

  @inline def RNGSplit(state: IR, dynBitstring: IR) = {
    assert(state.typ == TRNGState)
    def isValid: Type => Boolean = {
      case tuple: TTuple => tuple.types.forall(_ == TInt64)
      case t => t == TInt64
    }
    assert(isValid(dynBitstring.typ))
    TRNGState
  }

  @inline def StreamLen(a: IR) = {
    assert(a.typ.isInstanceOf[TStream])
    TInt32
  }

  @inline def GroupByKey(collection: IR) = {
    val elt = tcoerce[TBaseStruct](tcoerce[TStream](collection.typ).elementType)
    TDict(elt.types(0), TArray(elt.types(1)))
  }

  @inline def StreamTake(a: IR, num: IR) = {
    assert(num.typ == TInt32)
    tcoerce[TStream](a.typ)
  }

  @inline def StreamDrop(a: IR, num: IR) = {
    assert(num.typ == TInt32)
    tcoerce[TStream](a.typ)
  }

  @inline def StreamGrouped(a: IR, groupSize: IR) = {
    assert(a.typ.isInstanceOf[TStream])
    assert(groupSize.typ == TInt32)
    TStream(a.typ)
  }

  @inline def StreamGroupByKey(a: IR, key: IndexedSeq[String], missingEqual: Boolean) = {
    val structType = tcoerce[TStruct](tcoerce[TStream](a.typ).elementType)
    assert(key.forall(structType.hasField))
    TStream(a.typ)
  }

  @inline def StreamMap(a: IR, name: Name, body: IR) = {
    assert(a.typ.isInstanceOf[TStream])
    TStream(body.typ)
  }

  @inline def StreamZip(
    as: IndexedSeq[IR],
    names: IndexedSeq[Name],
    body: IR,
    behavior: ArrayZipBehavior.ArrayZipBehavior,
    errorID: Int = ErrorIDs.NO_ERROR,
  ) = {
    assert(as.length == names.length)
    assert(as.forall(_.typ.isInstanceOf[TStream]))
    TStream(body.typ)
  }

  @inline def StreamZipJoin(
    as: IndexedSeq[IR],
    key: IndexedSeq[String],
    curKey: Name,
    curVals: Name,
    joinF: IR,
  ) = {
    val streamType = tcoerce[TStream](as.head.typ)
    assert(as.forall(_.typ == streamType))
    val eltType = tcoerce[TStruct](streamType.elementType)
    assert(key.forall(eltType.hasField))
    TStream(joinF.typ)
  }

  @inline def StreamZipJoinProducers(
    contexts: IR,
    ctxName: Name,
    makeProducer: IR,
    key: IndexedSeq[String],
    curKey: Name,
    curVals: Name,
    joinF: IR,
  ) = {
    assert(contexts.typ.isInstanceOf[TArray])
    val streamType = tcoerce[TStream](makeProducer.typ)
    val eltType = tcoerce[TStruct](streamType.elementType)
    assert(key.forall(eltType.hasField))
    TStream(joinF.typ)
  }

  @inline def StreamMultiMerge(as: IndexedSeq[IR], key: IndexedSeq[String]) = {
    val streamType = tcoerce[TStream](as.head.typ)
    assert(as.forall(_.typ == streamType))
    val eltType = tcoerce[TStruct](streamType.elementType)
    assert(key.forall(eltType.hasField))
    TStream(streamType.elementType)
  }

  @inline def StreamFilter(a: IR, name: Name, cond: IR) = {
    val streamType = tcoerce[TStream](a.typ)
    assert(streamType.elementType.isRealizable)
    assert(cond.typ == TBoolean, cond.typ)
    streamType
  }

  @inline def StreamTakeWhile(a: IR, elementName: Name, cond: IR) = {
    val streamType = tcoerce[TStream](a.typ)
    assert(streamType.elementType.isRealizable)
    assert(cond.typ == TBoolean)
    streamType
  }

  @inline def StreamDropWhile(a: IR, elementName: Name, cond: IR) = {
    val streamType = tcoerce[TStream](a.typ)
    assert(streamType.elementType.isRealizable)
    assert(cond.typ == TBoolean)
    streamType
  }

  @inline def StreamFlatMap(a: IR, name: Name, cond: IR) = {
    assert(a.typ.isInstanceOf[TStream])
    TStream(tcoerce[TStream](cond.typ).elementType)
  }

  @inline def StreamFold(a: IR, zero: IR, accumName: Name, valueName: Name, body: IR) = {
    assert(tcoerce[TStream](a.typ).elementType.isRealizable)
    assert(body.typ == zero.typ)
    zero.typ
  }

  @inline def StreamFold2(
    a: IR,
    accum: IndexedSeq[(Name, IR)],
    valueName: Name,
    seq: IndexedSeq[IR],
    result: IR,
  ) = {
    assert(a.typ.isInstanceOf[TStream])
    assert(accum.zip(seq).forall { case ((_, z), s) => s.typ == z.typ })
    result.typ
  }

  @inline def StreamDistribute(
    child: IR,
    pivots: IR,
    path: IR,
    comparisonOp: ComparisonOp[_],
    spec: AbstractTypedCodecSpec,
  ) = {
    assert(path.typ == TString)
    assert(child.typ.isInstanceOf[TStream])
    val keyType = tcoerce[TStruct](tcoerce[TArray](pivots.typ).elementType)
    TArray(TStruct(
      ("interval", TInterval(keyType)),
      ("fileName", TString),
      ("numElements", TInt32),
      ("numBytes", TInt64),
    ))
  }

  @inline def StreamWhiten(
    stream: IR,
    newChunk: String,
    prevWindow: String,
    vecSize: Int,
    windowSize: Int,
    chunkSize: Int,
    blockSize: Int,
    normalizeAfterWhiten: Boolean,
  ) = {
    val streamTyp = tcoerce[TStream](stream.typ)
    val structTyp = tcoerce[TStruct](streamTyp.elementType)
    val matTyp = TNDArray(TFloat64, Nat(2))
    assert(structTyp.field(newChunk).typ == matTyp)
    assert(structTyp.field(prevWindow).typ == matTyp)
    assert(windowSize % chunkSize == 0)
    streamTyp
  }

  @inline def StreamScan(a: IR, zero: IR, accumName: Name, valueName: Name, body: IR) = {
    assert(a.typ.isInstanceOf[TStream])
    assert(body.typ == zero.typ)
    assert(zero.typ.isRealizable)
    TStream(zero.typ)
  }

  @inline def StreamAgg(a: IR, name: Name, query: IR) = {
    assert(a.typ.isInstanceOf[TStream])
    query.typ
  }

  @inline def StreamAggScan(a: IR, name: Name, query: IR) = {
    assert(a.typ.isInstanceOf[TStream])
    TStream(query.typ)
  }

  @inline def StreamLocalLDPrune(
    child: IR,
    r2Threshold: IR,
    windowSize: IR,
    maxQueueSize: IR,
    nSamples: IR,
  ) = {
    val eltType = tcoerce[TStruct](tcoerce[TStream](child.typ).elementType)
    assert(r2Threshold.typ == TFloat64)
    assert(windowSize.typ == TInt32)
    assert(maxQueueSize.typ == TInt32)
    assert(nSamples.typ == TInt32)
    val allelesType = eltType.fieldType("alleles")
    assert(tcoerce[TArray](allelesType).elementType == TString)
    assert(tcoerce[TArray](eltType.fieldType("genotypes")).elementType == TCall)
    TStream(TStruct(
      "locus" -> tcoerce[TLocus](eltType.fieldType("locus")),
      "alleles" -> allelesType,
      "mean" -> TFloat64,
      "centered_length_rec" -> TFloat64,
    ))
  }

  @inline def RunAgg(body: IR, result: IR, signature: IndexedSeq[AggStateSig]) = {
    assert(body.typ == TVoid)
    result.typ
  }

  @inline def RunAggScan(
    array: IR,
    name: Name,
    init: IR,
    seqs: IR,
    result: IR,
    signature: IndexedSeq[AggStateSig],
  ) = {
    assert(array.typ.isInstanceOf[TStream])
    assert(init.typ == TVoid)
    assert(seqs.typ == TVoid)
    TStream(result.typ)
  }

  @inline def StreamLeftIntervalJoin(
    left: IR,
    right: IR,
    lKeyFieldName: String,
    rIntervalFieldName: String,
    lname: Name,
    rname: Name,
    body: IR,
  ) = {
    val lEltTy = tcoerce[TStruct](tcoerce[TStream](left.typ).elementType)
    val rPointTy = tcoerce[TStruct](tcoerce[TStream](right.typ).elementType)
      .fieldType(rIntervalFieldName)
      .asInstanceOf[TInterval]
      .pointType
    assert(lEltTy.fieldType(lKeyFieldName) == rPointTy)
    assert(body.typ.isInstanceOf[TStruct])
    TStream(body.typ)
  }

  @inline def StreamJoinRightDistinct(
    left: IR,
    right: IR,
    lKey: IndexedSeq[String],
    rKey: IndexedSeq[String],
    l: Name,
    r: Name,
    joinF: IR,
    joinType: String,
  ) = {
    val lEltTyp = tcoerce[TStruct](tcoerce[TStream](left.typ).elementType)
    val rEltTyp = tcoerce[TStruct](tcoerce[TStream](right.typ).elementType)
    assert(lKey.forall(lEltTyp.hasField))
    assert(rKey.forall(rEltTyp.hasField))
    if (defs.StreamJoinRightDistinct.isIntervalJoin(lEltTyp, rEltTyp, lKey, rKey)) {
      val lKeyTyp = lEltTyp.fieldType(lKey(0))
      val rKeyTyp = rEltTyp.fieldType(rKey(0)).asInstanceOf[TInterval]
      assert(lKeyTyp == rKeyTyp.pointType)
      assert((joinType == "left") || (joinType == "inner"))
    } else {
      assert((lKey, rKey).zipped.forall { case (lk, rk) =>
        lEltTyp.fieldType(lk) == rEltTyp.fieldType(rk)
      })
    }
    TStream(joinF.typ)
  }

  @inline def NDArrayShape(nd: IR) = tcoerce[TNDArray](nd.typ).shapeType

  @inline def NDArrayReshape(nd: IR, shape: IR, errorID: Int = ErrorIDs.NO_ERROR) = {
    val shapeTyp = tcoerce[TTuple](shape.typ)
    assert(shapeTyp.types.forall(t => t == TInt64))
    TNDArray(tcoerce[TNDArray](nd.typ).elementType, Nat(shapeTyp.size))
  }

  @inline def NDArrayConcat(nds: IR, axis: Int) = {
    val ndType = tcoerce[TNDArray](tcoerce[TArray](nds.typ).elementType)
    assert(axis < ndType.nDims)
    ndType
  }

  @inline def NDArrayMap(nd: IR, valueName: Name, body: IR) =
    TNDArray(body.typ, tcoerce[TNDArray](nd.typ).nDimsBase)

  @inline def NDArrayMap2(
    l: IR,
    r: IR,
    lName: Name,
    rName: Name,
    body: IR,
    errorID: Int = ErrorIDs.NO_ERROR,
  ) = {
    val lTyp = tcoerce[TNDArray](l.typ)
    val rTyp = tcoerce[TNDArray](r.typ)
    assert(lTyp.nDims == rTyp.nDims)
    TNDArray(body.typ, lTyp.nDimsBase)
  }

  @inline def NDArrayReindex(nd: IR, indexExpr: IndexedSeq[Int]) = {
    val ndTyp = tcoerce[TNDArray](nd.typ)
    val nInputDims = ndTyp.nDims
    val nOutputDims = indexExpr.length
    assert(nInputDims <= nOutputDims)
    assert(indexExpr.forall(i => i < nOutputDims))
    assert((0 until nOutputDims).forall(i => indexExpr.contains(i)))
    TNDArray(ndTyp.elementType, Nat(indexExpr.length))
  }

  @inline def NDArrayAgg(nd: IR, axes: IndexedSeq[Int]) = {
    val childType = tcoerce[TNDArray](nd.typ)
    val nInputDims = childType.nDims
    assert(axes.length <= nInputDims)
    assert(axes.forall(i => i < nInputDims))
    assert(axes.distinct.length == axes.length)
    TNDArray(childType.elementType, Nat(nInputDims - axes.length))
  }

  @inline def NDArrayRef(nd: IR, idxs: IndexedSeq[IR], errorID: Int = ErrorIDs.NO_ERROR) = {
    val childType = tcoerce[TNDArray](nd.typ)
    assert(childType.nDims == idxs.length)
    assert(idxs.forall(_.typ == TInt64))
    childType.elementType
  }

  @inline def NDArraySlice(nd: IR, slices: IR) = {
    val childTyp = tcoerce[TNDArray](nd.typ)
    val slicesTyp = tcoerce[TTuple](slices.typ)
    assert(slicesTyp.size == childTyp.nDims)
    assert(slicesTyp.types.forall(t => (t == TTuple(TInt64, TInt64, TInt64)) || (t == TInt64)))
    val tuplesOnly = slicesTyp.types.collect {
      case x: TTuple => x
    }
    val remainingDims = Nat(tuplesOnly.length)
    TNDArray(childTyp.elementType, remainingDims)
  }

  @inline def NDArrayFilter(nd: IR, keep: IndexedSeq[IR]) = {
    val ndtyp = tcoerce[TNDArray](nd.typ)
    assert(ndtyp.nDims == keep.length)
    assert(keep.forall(f => tcoerce[TArray](f.typ).elementType == TInt64))
    ndtyp
  }

  @inline def NDArrayMatMul(l: IR, r: IR, errorID: Int = ErrorIDs.NO_ERROR) = {
    val lTyp = tcoerce[TNDArray](l.typ)
    val rTyp = tcoerce[TNDArray](r.typ)
    assert(lTyp.elementType == rTyp.elementType, "element type did not match")
    assert(lTyp.nDims > 0)
    assert(rTyp.nDims > 0)
    assert(lTyp.nDims == 1 || rTyp.nDims == 1 || lTyp.nDims == rTyp.nDims)
    TNDArray(lTyp.elementType, Nat(TNDArray.matMulNDims(lTyp.nDims, rTyp.nDims)))
  }

  @inline def NDArrayQR(nd: IR, mode: String, errorID: Int = ErrorIDs.NO_ERROR) = {
    val ndType = nd.typ.asInstanceOf[TNDArray]
    assert(ndType.elementType == TFloat64)
    assert(ndType.nDims == 2)
    if (Array("complete", "reduced").contains(mode)) {
      TTuple(TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)))
    } else if (mode == "raw") {
      TTuple(TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(1)))
    } else if (mode == "r") {
      TNDArray(TFloat64, Nat(2))
    } else {
      throw new NotImplementedError(s"Cannot infer type for mode $mode")
    }
  }

  @inline def NDArraySVD(
    nd: IR,
    fullMatrices: Boolean,
    computeUV: Boolean,
    errorID: Int = ErrorIDs.NO_ERROR,
  ) = {
    val ndType = nd.typ.asInstanceOf[TNDArray]
    assert(ndType.elementType == TFloat64)
    assert(ndType.nDims == 2)
    if (computeUV) {
      TTuple(TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(1)), TNDArray(TFloat64, Nat(2)))
    } else {
      TNDArray(TFloat64, Nat(1))
    }
  }

  @inline def NDArrayEigh(nd: IR, eigvalsOnly: Boolean, errorID: Int = ErrorIDs.NO_ERROR) = {
    val ndType = nd.typ.asInstanceOf[TNDArray]
    assert(ndType.elementType == TFloat64)
    assert(ndType.nDims == 2)
    if (eigvalsOnly) {
      TNDArray(TFloat64, Nat(1))
    } else {
      TTuple(TNDArray(TFloat64, Nat(1)), TNDArray(TFloat64, Nat(2)))
    }
  }

  @inline def NDArrayInv(nd: IR, errorID: Int = ErrorIDs.NO_ERROR) = {
    val ndType = nd.typ.asInstanceOf[TNDArray]
    assert(ndType.elementType == TFloat64)
    assert(ndType.nDims == 2)
    TNDArray(TFloat64, Nat(2))
  }

  @inline def NDArrayWrite(nd: IR, path: IR) = {
    assert(nd.typ.isInstanceOf[TNDArray])
    assert(path.typ == TString)
    TVoid
  }

  @inline def AggFilter(cond: IR, aggIR: IR, isScan: Boolean) = {
    assert(cond.typ == TBoolean)
    aggIR.typ
  }

  @inline def AggExplode(array: IR, name: Name, aggBody: IR, isScan: Boolean) = {
    assert(array.typ.isInstanceOf[TStream])
    aggBody.typ
  }

  @inline def AggGroupBy(key: IR, aggIR: IR, isScan: Boolean) = TDict(key.typ, aggIR.typ)

  @inline def AggArrayPerElement(
    a: IR,
    elementName: Name,
    indexName: Name,
    aggBody: IR,
    knownLength: Option[IR],
    isScan: Boolean,
  ) = {
    assert(a.typ.isInstanceOf[TArray])
    assert(knownLength.forall(_.typ == TInt32))
    TArray(aggBody.typ)
  }

  @inline def ApplyAggOp(
    initOpArgs: IndexedSeq[IR],
    seqOpArgs: IndexedSeq[IR],
    aggSig: AggSignature,
  ) = {
    assert(initOpArgs.map(_.typ).zip(aggSig.initOpArgs).forall { case (l, r) => l == r })
    assert(seqOpArgs.map(_.typ).zip(aggSig.seqOpArgs).forall { case (l, r) => l == r })
    aggSig.returnType
  }

  @inline def ApplyScanOp(
    initOpArgs: IndexedSeq[IR],
    seqOpArgs: IndexedSeq[IR],
    aggSig: AggSignature,
  ) = {
    assert(initOpArgs.map(_.typ).zip(aggSig.initOpArgs).forall { case (l, r) => l == r })
    assert(seqOpArgs.map(_.typ).zip(aggSig.seqOpArgs).forall { case (l, r) => l == r })
    aggSig.returnType
  }

  @inline def AggFold(
    zero: IR,
    seqOp: IR,
    combOp: IR,
    accumName: Name,
    otherAccumName: Name,
    isScan: Boolean,
  ) = {
    assert(zero.typ == seqOp.typ)
    assert(zero.typ == combOp.typ)
    zero.typ
  }

  @inline def MakeStruct(fields: IndexedSeq[(String, IR)]) =
    TStruct(fields.map { case (name, a) =>
      (name, a.typ)
    }: _*)

  @inline def SelectFields(old: IR, fields: IndexedSeq[String]) = {
    val tbs = tcoerce[TStruct](old.typ)
    val oldfields = tbs.fieldNames.toSet
    assert(fields.forall(id => oldfields.contains(id)))
    tbs.select(fields.toFastSeq)._1
  }

  @inline def InsertFields(
    old: IR,
    fields: IndexedSeq[(String, IR)],
    fieldOrder: Option[IndexedSeq[String]] = None,
  ) = {
    val tbs = tcoerce[TStruct](old.typ)
    val s = tbs.insertFields(fields.map(f => (f._1, f._2.typ)))
    fieldOrder.map { fds =>
      assert(fds.areDistinct())
      assert(fds.size == s.size)
      TStruct(fds.map(f => f -> s.fieldType(f)): _*)
    }.getOrElse(s)
  }

  @inline def GetField(o: IR, name: String) = {
    val t = tcoerce[TStruct](o.typ)
    if (t.index(name).isEmpty)
      throw new RuntimeException(s"$name not in $t")
    t.field(name).typ
  }

  @inline def MakeTuple(fields: IndexedSeq[(Int, IR)]) = {
    val indices = fields.map(_._1)
    assert(indices.areDistinct())
    assert(indices.isSorted)
    TTuple(fields.map { case (i, value) => TupleField(i, value.typ) }.toFastSeq)
  }

  @inline def GetTupleElement(o: IR, idx: Int) = {
    val t = tcoerce[TTuple](o.typ)
    val fd = t.fields(t.fieldIndex(idx)).typ
    fd
  }

  @inline def TableCount(child: TableIR) = TInt64
  @inline def MatrixCount(child: MatrixIR) = TTuple(TInt64, TInt32)
  @inline def TableAggregate(child: TableIR, query: IR) = query.typ
  @inline def MatrixAggregate(child: MatrixIR, query: IR) = query.typ
  @inline def TableWrite(child: TableIR, writer: TableWriter) = TVoid

  @inline def TableMultiWrite(
    children: IndexedSeq[TableIR],
    writer: WrappedMatrixNativeMultiWriter,
  ) = {
    val t = children.head.typ
    assert(children.forall(_.typ == t))
    TVoid
  }

  @inline def MatrixWrite(child: MatrixIR, writer: MatrixWriter) = TVoid

  @inline def MatrixMultiWrite(children: IndexedSeq[MatrixIR], writer: MatrixNativeMultiWriter) = {
    val t = children.head.typ
    assert(
      !t.rowType.hasField(MatrixReader.rowUIDFieldName) &&
        !t.colType.hasField(MatrixReader.colUIDFieldName),
      t,
    )
    assert(children.forall(_.typ == t))
    TVoid
  }

  @inline def BlockMatrixCollect(child: BlockMatrixIR) = TNDArray(TFloat64, Nat(2))
  @inline def BlockMatrixWrite(child: BlockMatrixIR, writer: BlockMatrixWriter) = writer.loweredTyp

  @inline def BlockMatrixMultiWrite(
    blockMatrices: IndexedSeq[BlockMatrixIR],
    writer: BlockMatrixMultiWriter,
  ) = TVoid

  @inline def TableGetGlobals(child: TableIR) = child.typ.globalType

  @inline def TableCollect(child: TableIR) = {
    assert(child.typ.key.isEmpty)
    TStruct("rows" -> TArray(child.typ.rowType), "global" -> child.typ.globalType)
  }

  @inline def TableToValueApply(child: TableIR, function: TableToValueFunction) =
    function.typ(child.typ)

  @inline def MatrixToValueApply(child: MatrixIR, function: MatrixToValueFunction) =
    function.typ(child.typ)

  @inline def BlockMatrixToValueApply(child: BlockMatrixIR, function: BlockMatrixToValueFunction) =
    function.typ(child.typ)

  @inline def CollectDistributedArray(
    contexts: IR,
    globals: IR,
    cname: Name,
    gname: Name,
    body: IR,
    dynamicID: IR,
    staticID: String,
    tsd: Option[TableStageDependency] = None,
  ) = {
    assert(contexts.typ.isInstanceOf[TStream])
    assert(dynamicID.typ == TString)
    TArray(body.typ)
  }

  @inline def ReadPartition(context: IR, rowType: TStruct, reader: PartitionReader) = {
    assert(rowType.isRealizable)
    assert(context.typ == reader.contextType)
    assert(PruneDeadFields.isSupertype(rowType, reader.fullRowType))
    TStream(rowType)
  }

  @inline def WritePartition(value: IR, writeCtx: IR, writer: PartitionWriter) = {
    assert(value.typ.isInstanceOf[TStream])
    assert(writeCtx.typ == writer.ctxType)
    writer.returnType
  }

  @inline def WriteMetadata(writeAnnotations: IR, writer: MetadataWriter) = {
    assert(writeAnnotations.typ == writer.annotationType)
    TVoid
  }

  @inline def ReadValue(path: IR, reader: ValueReader, typ: Type) = {
    assert(path.typ == TString)
    reader match {
      case reader: ETypeValueReader =>
        assert(reader.spec.encodedType.decodedPType(typ).virtualType == typ)
      case _ => // do nothing, we can't in general typecheck an arbitrary value reader
    }
  }

  @inline def WriteValue(
    value: IR,
    path: IR,
    writer: ValueWriter,
    stagingFile: Option[IR] = None,
  ) = {
    assert(path.typ == TString)
    assert(stagingFile.forall(_.typ == TString))
    TString
  }

  @inline def LiftMeOut(child: IR) = child.typ
}
