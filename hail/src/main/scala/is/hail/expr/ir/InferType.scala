package is.hail.expr.ir

import is.hail.expr.Nat
import is.hail.types.virtual._
import is.hail.utils._

// FIXME: strip all requiredness logic when possible
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
      case MakeStream(_, t) => t
      case MakeNDArray(data, shape, _) =>
        TNDArray(coerce[TArray](data.typ).elementType, Nat(shape.typ.asInstanceOf[TTuple].size))
      case _: ArrayLen => TInt32
      case _: StreamRange => TStream(TInt32)
      case _: ArrayZeros => TArray(TInt32)
      case _: LowerBoundOnOrderedCollection => TInt32
      case _: StreamFor => TVoid
      case _: InitOp => TVoid
      case _: SeqOp => TVoid
      case _: CombOp => TVoid
      case ResultOp(_, aggSigs) =>
        TTuple(aggSigs.map(_.resultType): _*)
      case AggStateValue(i, sig) => TBinary
      case _: CombOpValue => TVoid
      case _: InitFromSerializedValue => TVoid
      case _: SerializeAggs => TVoid
      case _: DeserializeAggs => TVoid
      case _: Begin => TVoid
      case Die(_, t) => t
      case If(cond, cnsq, altr) =>
        assert(cond.typ == TBoolean)
        assert(cnsq.typ == altr.typ)
        cnsq.typ
      case Let(name, value, body) =>
        body.typ
      case AggLet(name, value, body, _) =>
        body.typ
      case TailLoop(_, _, body) =>
        body.typ
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
      case a: ApplyIR => a.explicitNode.typ
      case a: AbstractApplyNode[_] =>
        val typeArgs = a.typeArgs
        val argTypes = a.args.map(_.typ)
        assert(a.implementation.unify(typeArgs, argTypes, a.returnType))
        a.returnType
      case ArrayRef(a, i, s) =>
        assert(i.typ == TInt32)
        coerce[TArray](a.typ).elementType
      case ArraySort(a, _, _, lessThan) =>
        assert(lessThan.typ == TBoolean)
        val et = coerce[TStream](a.typ).elementType
        TArray(et)
      case ToSet(a) =>
        val et = coerce[TStream](a.typ).elementType
        TSet(et)
      case ToDict(a) =>
        val elt = coerce[TBaseStruct](coerce[TStream](a.typ).elementType)
        TDict(elt.types(0), elt.types(1))
      case ToArray(a) =>
        val elt = coerce[TStream](a.typ).elementType
        TArray(elt)
      case CastToArray(a) =>
        val elt = coerce[TContainer](a.typ).elementType
        TArray(elt)
      case ToStream(a) =>
        val elt = coerce[TIterable](a.typ).elementType
        TStream(elt)
      case StreamLen(a) => TInt32
      case GroupByKey(collection) =>
        val elt = coerce[TBaseStruct](coerce[TStream](collection.typ).elementType)
        TDict(elt.types(0), TArray(elt.types(1)))
      case StreamTake(a, _) =>
        a.typ
      case StreamDrop(a, _) =>
        a.typ
      case StreamGrouped(a, _) =>
        TStream(a.typ)
      case StreamGroupByKey(a, _) =>
        TStream(a.typ)
      case StreamMap(a, name, body) =>
        TStream(body.typ)
      case StreamMerge(l, r, key) =>
        assert(l.typ == r.typ)
        l.typ
      case StreamZip(as, _, body, _) =>
        TStream(body.typ)
      case StreamZipJoin(_, _, _, _, joinF) =>
        TStream(joinF.typ)
      case StreamMultiMerge(as, _) =>
        TStream(coerce[TStream](as.head.typ).elementType)
      case StreamFilter(a, name, cond) =>
        a.typ
      case StreamFlatMap(a, name, body) =>
        TStream(coerce[TStream](body.typ).elementType)
      case StreamFold(a, zero, accumName, valueName, body) =>
        assert(body.typ == zero.typ)
        zero.typ
      case StreamFold2(_, _, _, _, result) => result.typ
      case StreamScan(a, zero, accumName, valueName, body) =>
        assert(body.typ == zero.typ)
        TStream(zero.typ)
      case StreamAgg(_, _, query) =>
        query.typ
      case StreamAggScan(_, _, query) =>
        TStream(query.typ)
      case RunAgg(body, result, _) =>
        result.typ
      case RunAggScan(_, _, _, _, result, _) =>
        TStream(result.typ)
      case StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType) =>
        TStream(join.typ)
      case NDArrayShape(nd) =>
        val ndType = nd.typ.asInstanceOf[TNDArray]
        ndType.shapeType
      case NDArrayReshape(nd, shape) =>
        TNDArray(coerce[TNDArray](nd.typ).elementType, Nat(shape.typ.asInstanceOf[TTuple].size))
      case NDArrayConcat(nds, _) =>
        coerce[TArray](nds.typ).elementType
      case NDArrayMap(nd, _, body) =>
        TNDArray(body.typ, coerce[TNDArray](nd.typ).nDimsBase)
      case NDArrayMap2(l, _, _, _, body) =>
        TNDArray(body.typ, coerce[TNDArray](l.typ).nDimsBase)
      case NDArrayReindex(nd, indexExpr) =>
        TNDArray(coerce[TNDArray](nd.typ).elementType, Nat(indexExpr.length))
      case NDArrayAgg(nd, axes) =>
        val childType = coerce[TNDArray](nd.typ)
        TNDArray(childType.elementType, Nat(childType.nDims - axes.length))
      case NDArrayRef(nd, idxs) =>
        assert(idxs.forall(_.typ == TInt64))
        coerce[TNDArray](nd.typ).elementType
      case NDArraySlice(nd, slices) =>
        val childTyp = coerce[TNDArray](nd.typ)
        val slicesTyp = coerce[TTuple](slices.typ)
        val tuplesOnly = slicesTyp.types.collect { case x: TTuple => x}
        val remainingDims = Nat(tuplesOnly.length)
        TNDArray(childTyp.elementType, remainingDims)
      case NDArrayFilter(nd, _) =>
        nd.typ
      case NDArrayMatMul(l, r) =>
        val lTyp = coerce[TNDArray](l.typ)
        val rTyp = coerce[TNDArray](r.typ)
        TNDArray(lTyp.elementType, Nat(TNDArray.matMulNDims(lTyp.nDims, rTyp.nDims)))
      case NDArrayQR(nd, mode) =>
        if (Array("complete", "reduced").contains(mode)) {
          TTuple(TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)))
        } else if (mode == "raw") {
          TTuple(TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(1)))
        } else if (mode == "r") {
          TNDArray(TFloat64, Nat(2))
        } else {
          throw new NotImplementedError(s"Cannot infer type for mode $mode")
        }
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
      case MakeStruct(fields) =>
        TStruct(fields.map { case (name, a) =>
          (name, a.typ)
        }: _*)
      case SelectFields(old, fields) =>
        val tbs = coerce[TStruct](old.typ)
        tbs.select(fields.toFastIndexedSeq)._1
      case InsertFields(old, fields, fieldOrder) =>
        val tbs = coerce[TStruct](old.typ)
        val s = tbs.insertFields(fields.map(f => (f._1, f._2.typ)))
        fieldOrder.map { fds =>
          assert(fds.length == s.size)
          TStruct(fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      case GetField(o, name) =>
        val t = coerce[TStruct](o.typ)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        t.field(name).typ
      case MakeTuple(values) =>
        TTuple(values.map { case (i, value) => TupleField(i, value.typ) }.toFastIndexedSeq)
      case GetTupleElement(o, idx) =>
        val t = coerce[TTuple](o.typ)
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
      case _: BlockMatrixWrite => TVoid
      case _: BlockMatrixMultiWrite => TVoid
      case _: UnpersistBlockMatrix => TVoid
      case TableGetGlobals(child) => child.typ.globalType
      case TableCollect(child) => TStruct("rows" -> TArray(child.typ.rowType), "global" -> child.typ.globalType)
      case TableToValueApply(child, function) => function.typ(child.typ)
      case MatrixToValueApply(child, function) => function.typ(child.typ)
      case BlockMatrixToValueApply(child, function) => function.typ(child.typ)
      case CollectDistributedArray(_, _, _, _, body) => TArray(body.typ)
      case ReadPartition(_, rowType, _) => TStream(rowType)
      case WritePartition(value, writeCtx, writer) => writer.returnType
      case _: WriteMetadata => TVoid
      case ReadValue(_, _, typ) => typ
      case WriteValue(value, pathPrefix, spec) => TString
      case LiftMeOut(child) => child.typ
      case ShuffleStart(_, _, _, _) =>
        TBinary
      case ShuffleWrite(id, partitionId, rows) =>
        TBinary
      case ShuffleWritingFinished(id, successfulPartitionIds) =>
        TVoid
      case ShuffleGetPartitionBounds(_, _, _, rowType, _) =>
        TArray(rowType)
      case ShuffleRead(id, keyRange, rowType, rowEType) =>
        TStream(rowType)
      case ShuffleDelete(id) =>
        TVoid
    }
  }
}
