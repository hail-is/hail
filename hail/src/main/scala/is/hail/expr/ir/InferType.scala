package is.hail.expr.ir

import is.hail.expr.Nat
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._

object InferType {
  def apply(ir: IR): Type = {
    ir match {
      case I32(_) => TInt32
      case I64(_) => TInt64
      case F32(_) => TFloat32
      case F64(_) => TFloat64
      case Str(_) => TString
      case UUID4(_) => TString
      case Literal(t, _) => t
      case EncodedLiteral(codec, _) => codec.encodedVirtualType
      case True() | False() => TBoolean
      case Void() => TVoid
      case Cast(_, t) => t
      case CastRename(_, t) => t
      case NA(t) => t
      case IsNA(_) => TBoolean
      case Coalesce(values) => values.head.typ
      case Consume(_) => TInt64
      case Ref(_, t) => t
      case RelationalRef(_, t) => t
      case RelationalLet(_, _, body) => body.typ
      case In(_, t) => t.virtualType
      case MakeArray(_, t) => t
      case MakeStream(_, t, _) => t
      case MakeNDArray(data, shape, _, _) =>
        TNDArray(tcoerce[TIterable](data.typ).elementType, Nat(shape.typ.asInstanceOf[TTuple].size))
      case StreamBufferedAggregate(_, _, newKey, _, _, aggSignatures, _) =>
        val tupleFieldTypes = TTuple(aggSignatures.map(_ => TBinary):_*)
        TStream(newKey.typ.asInstanceOf[TStruct].insertFields(IndexedSeq(("agg", tupleFieldTypes))))
      case _: ArrayLen => TInt32
      case _: StreamIota => TStream(TInt32)
      case _: StreamRange => TStream(TInt32)
      case _: SeqSample => TStream(TInt32)
      case _: ArrayZeros => TArray(TInt32)
      case _: LowerBoundOnOrderedCollection => TInt32
      case _: StreamFor => TVoid
      case _: InitOp => TVoid
      case _: SeqOp => TVoid
      case _: CombOp => TVoid
      case ResultOp(_, aggSig) =>
        aggSig.resultType
      case AggStateValue(i, sig) => TBinary
      case _: CombOpValue => TVoid
      case _: InitFromSerializedValue => TVoid
      case _: SerializeAggs => TVoid
      case _: DeserializeAggs => TVoid
      case _: Begin => TVoid
      case Die(_, t, _) => t
      case Trap(child) => TTuple(TTuple(TString, TInt32), child.typ)
      case ConsoleLog(message, result) => result.typ
      case If(cond, cnsq, altr) =>
        assert(cond.typ == TBoolean)
        assert(cnsq.typ == altr.typ)
        cnsq.typ
      case Switch(_, default, _) =>
        default.typ
      case Let(name, value, body) =>
        body.typ
      case AggLet(name, value, body, _) =>
        body.typ
      case TailLoop(_, _, resultType, _) =>
        resultType
      case Recur(_, _, typ) =>
        typ
      case ApplyBinaryPrimOp(op, l, r) =>
        BinaryOp.getReturnType(op, l.typ, r.typ)
      case ApplyUnaryPrimOp(op, v) =>
        UnaryOp.getReturnType(op, v.typ)
      case ApplyComparisonOp(op, l, r) =>
        assert(l.typ == r.typ)
        op match {
          case _: Compare => TInt32
          case _ => TBoolean
        }
      case a: ApplyIR => a.returnType
      case a: AbstractApplyNode[_] =>
        val typeArgs = a.typeArgs
        val argTypes = a.args.map(_.typ)
        assert(a.implementation.unify(typeArgs, argTypes, a.returnType))
        a.returnType
      case ArrayRef(a, i, _) =>
        assert(i.typ == TInt32)
        tcoerce[TArray](a.typ).elementType
      case ArraySlice(a, start, stop, step, _) =>
        assert(start.typ == TInt32)
        stop.foreach(ir => assert(ir.typ == TInt32))
        assert(step.typ == TInt32)
        tcoerce[TArray](a.typ)
      case ArraySort(a, _, _, lessThan) =>
        assert(lessThan.typ == TBoolean)
        val et = tcoerce[TStream](a.typ).elementType
        TArray(et)
      case ArrayMaximalIndependentSet(edges, _) =>
        val et = tcoerce[TArray](edges.typ).elementType.asInstanceOf[TBaseStruct].types.head
        TArray(et)
      case ToSet(a) =>
        val et = tcoerce[TStream](a.typ).elementType
        TSet(et)
      case ToDict(a) =>
        val elt = tcoerce[TBaseStruct](tcoerce[TStream](a.typ).elementType)
        TDict(elt.types(0), elt.types(1))
      case ta@ToArray(a) =>
        val elt = tcoerce[TStream](a.typ).elementType
        TArray(elt)
      case CastToArray(a) =>
        val elt = tcoerce[TContainer](a.typ).elementType
        TArray(elt)
      case ToStream(a, _) =>
        val elt = tcoerce[TIterable](a.typ).elementType
        TStream(elt)
      case RNGStateLiteral() =>
        TRNGState
      case RNGSplit(_, _) =>
        TRNGState
      case StreamLen(a) => TInt32
      case GroupByKey(collection) =>
        val elt = tcoerce[TBaseStruct](tcoerce[TStream](collection.typ).elementType)
        TDict(elt.types(0), TArray(elt.types(1)))
      case StreamTake(a, _) =>
        a.typ
      case StreamDrop(a, _) =>
        a.typ
      case StreamGrouped(a, _) =>
        TStream(a.typ)
      case StreamGroupByKey(a, _, _) =>
        TStream(a.typ)
      case StreamMap(a, name, body) =>
        TStream(body.typ)
      case StreamZip(as, _, body, _, _) =>
        TStream(body.typ)
      case StreamZipJoin(_, _, _, _, joinF) =>
        TStream(joinF.typ)
      case StreamZipJoinProducers(_, _, _, _, _, _, joinF) =>
        TStream(joinF.typ)
      case StreamMultiMerge(as, _) =>
        TStream(tcoerce[TStream](as.head.typ).elementType)
      case StreamFilter(a, name, cond) =>
        a.typ
      case StreamTakeWhile(a, name, cond) =>
        a.typ
      case StreamDropWhile(a, name, cond) =>
        a.typ
      case StreamFlatMap(a, name, body) =>
        TStream(tcoerce[TStream](body.typ).elementType)
      case StreamFold(a, zero, accumName, valueName, body) =>
        assert(body.typ == zero.typ)
        zero.typ
      case StreamFold2(_, _, _, _, result) => result.typ
      case StreamDistribute(child, pivots, pathPrefix, _, _) =>
        val keyType = pivots.typ.asInstanceOf[TContainer].elementType
        TArray(TStruct(("interval", TInterval(keyType)), ("fileName", TString), ("numElements", TInt32), ("numBytes", TInt64)))
      case StreamWhiten(stream, _, _, _, _, _, _, _) =>
        stream.typ
      case StreamScan(a, zero, accumName, valueName, body) =>
        assert(body.typ == zero.typ)
        TStream(zero.typ)
      case StreamAgg(_, _, query) =>
        query.typ
      case StreamAggScan(_, _, query) =>
        TStream(query.typ)
      case StreamLocalLDPrune(streamChild, _, _, _, _) =>
        val childType = tcoerce[TStruct](tcoerce[TStream](streamChild.typ).elementType)
        TStream(TStruct(
          "locus" -> childType.fieldType("locus"),
          "alleles" -> childType.fieldType("alleles"),
          "mean" -> TFloat64,
          "centered_length_rec" -> TFloat64))
      case RunAgg(body, result, _) =>
        result.typ
      case RunAggScan(_, _, _, _, result, _) =>
        TStream(result.typ)
      case StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType) =>
        TStream(join.typ)
      case NDArrayShape(nd) =>
        val ndType = nd.typ.asInstanceOf[TNDArray]
        ndType.shapeType
      case NDArrayReshape(nd, shape, _) =>
        TNDArray(tcoerce[TNDArray](nd.typ).elementType, Nat(shape.typ.asInstanceOf[TTuple].size))
      case NDArrayConcat(nds, _) =>
        tcoerce[TArray](nds.typ).elementType
      case NDArrayMap(nd, _, body) =>
        TNDArray(body.typ, tcoerce[TNDArray](nd.typ).nDimsBase)
      case NDArrayMap2(l, _, _, _, body, _) =>
        TNDArray(body.typ, tcoerce[TNDArray](l.typ).nDimsBase)
      case NDArrayReindex(nd, indexExpr) =>
        TNDArray(tcoerce[TNDArray](nd.typ).elementType, Nat(indexExpr.length))
      case NDArrayAgg(nd, axes) =>
        val childType = tcoerce[TNDArray](nd.typ)
        TNDArray(childType.elementType, Nat(childType.nDims - axes.length))
      case NDArrayRef(nd, idxs, _) =>
        assert(idxs.forall(_.typ == TInt64))
        tcoerce[TNDArray](nd.typ).elementType
      case NDArraySlice(nd, slices) =>
        val childTyp = tcoerce[TNDArray](nd.typ)
        val slicesTyp = tcoerce[TTuple](slices.typ)
        val tuplesOnly = slicesTyp.types.collect { case x: TTuple => x}
        val remainingDims = Nat(tuplesOnly.length)
        TNDArray(childTyp.elementType, remainingDims)
      case NDArrayFilter(nd, _) =>
        nd.typ
      case NDArrayMatMul(l, r, _) =>
        val lTyp = tcoerce[TNDArray](l.typ)
        val rTyp = tcoerce[TNDArray](r.typ)
        TNDArray(lTyp.elementType, Nat(TNDArray.matMulNDims(lTyp.nDims, rTyp.nDims)))
      case NDArrayQR(nd, mode, _) =>
        if (Array("complete", "reduced").contains(mode)) {
          TTuple(TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)))
        } else if (mode == "raw") {
          TTuple(TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(1)))
        } else if (mode == "r") {
          TNDArray(TFloat64, Nat(2))
        } else {
          throw new NotImplementedError(s"Cannot infer type for mode $mode")
        }
      case NDArraySVD(nd, _, compute_uv, _) =>
        if (compute_uv) {
          TTuple(TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(1)), TNDArray(TFloat64, Nat(2)))
        } else {
          TNDArray(TFloat64, Nat(1))
        }
      case NDArrayEigh(nd, eigvalsOnly, _) =>
        if (eigvalsOnly) {
          TNDArray(TFloat64, Nat(1))
        } else {
          TTuple(TNDArray(TFloat64, Nat(1)), TNDArray(TFloat64, Nat(2)))
        }
      case NDArrayInv(_, _) =>
        TNDArray(TFloat64, Nat(2))
      case NDArrayWrite(_, _) => TVoid
      case AggFilter(_, aggIR, _) =>
        aggIR.typ
      case AggExplode(array, name, aggBody, _) =>
        aggBody.typ
      case AggGroupBy(key, aggIR, _) =>
        TDict(key.typ, aggIR.typ)
      case AggArrayPerElement(a, _, _, aggBody, _, _) => TArray(aggBody.typ)
      case ApplyAggOp(_, _, aggSig) =>
        aggSig.returnType
      case ApplyScanOp(_, _, aggSig) =>
        aggSig.returnType
      case AggFold(zero, _, _, _, _, _) =>
        zero.typ
      case MakeStruct(fields) =>
        TStruct(fields.map { case (name, a) =>
          (name, a.typ)
        }: _*)
      case SelectFields(old, fields) =>
        val tbs = tcoerce[TStruct](old.typ)
        tbs.select(fields.toFastSeq)._1
      case InsertFields(old, fields, fieldOrder) =>
        val tbs = tcoerce[TStruct](old.typ)
        val s = tbs.insertFields(fields.map(f => (f._1, f._2.typ)))
        fieldOrder.map { fds =>
          assert(fds.length == s.size, s"${fds} != ${s.types.toIndexedSeq}")
          TStruct(fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      case GetField(o, name) =>
        val t = tcoerce[TStruct](o.typ)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        t.field(name).typ
      case MakeTuple(values) =>
        TTuple(values.map { case (i, value) => TupleField(i, value.typ) }.toFastSeq)
      case GetTupleElement(o, idx) =>
        val t = tcoerce[TTuple](o.typ)
        val fd = t.fields(t.fieldIndex(idx)).typ
        fd
      case TableCount(_) => TInt64
      case MatrixCount(_) => TTuple(TInt64, TInt32)
      case TableAggregate(child, query) =>
        query.typ
      case MatrixAggregate(child, query) =>
        query.typ
      case _: TableWrite => TVoid
      case _: TableMultiWrite => TVoid
      case _: MatrixWrite => TVoid
      case _: MatrixMultiWrite => TVoid
      case _: BlockMatrixCollect => TNDArray(TFloat64, Nat(2))
      case BlockMatrixWrite(_, writer) => writer.loweredTyp
      case _: BlockMatrixMultiWrite => TVoid
      case TableGetGlobals(child) => child.typ.globalType
      case TableCollect(child) => TStruct("rows" -> TArray(child.typ.rowType), "global" -> child.typ.globalType)
      case TableToValueApply(child, function) => function.typ(child.typ)
      case MatrixToValueApply(child, function) => function.typ(child.typ)
      case BlockMatrixToValueApply(child, function) => function.typ(child.typ)
      case CollectDistributedArray(_, _, _, _, body, _, _, _) => TArray(body.typ)
      case ReadPartition(_, rowType, _) => TStream(rowType)
      case WritePartition(value, writeCtx, writer) => writer.returnType
      case _: WriteMetadata => TVoid
      case ReadValue(_, _, typ) => typ
      case _: WriteValue => TString
      case LiftMeOut(child) => child.typ
    }
  }
}
