package is.hail.expr.ir

import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils._

object InferPType {
  private[this] def propagateStreamable(t: PStreamable, elt: PType): PType =
    t match {
      case _: PStream => PStream(elt, t.required)
      case _: PArray => PArray(elt, t.required)
    }

  def apply(ir: IR): PType = {
    ir match {
      case I32(_) => PInt32()
      case I64(_) => PInt64()
      case F32(_) => PFloat32()
      case F64(_) => PFloat64()
      case Str(_) => PString()
      case Literal(t, _) => PType.canonical(t)
      case True() | False() => PBoolean()
      case Void() => PVoid
      case Cast(_, t) => PType.canonical(t)
      case NA(t) => PType.canonical(t)
      case IsNA(_) => PBoolean()
      case Ref(_, t) => PType.canonical(t) // FIXME fill in with supplied physical type
      case In(_, t) => PType.canonical(t) // FIXME fill in with supplied physical type
      case MakeArray(_, t) => PType.canonical(t)
      case MakeStream(_, t) => PType.canonical(t)
      case MakeNDArray(nDim, data, _, _) => PNDArray(coerce[PArray](data.typ.physicalType).elementType, nDim)
      case _: ArrayLen => PInt32()
      case _: ArrayRange => PArray(PInt32())
      case _: StreamRange => PStream(PInt32())
      case _: LowerBoundOnOrderedCollection => PInt32()
      case _: ArrayFor => PVoid
      case _: InitOp => PVoid
      case _: SeqOp => PVoid
      case _: Begin => PVoid
      case Die(_, t) => PType.canonical(t)
      case If(cond, cnsq, altr) =>
        assert(cond.typ.isOfType(TBoolean()))
        assert(cnsq.typ.isOfType(altr.typ))
        if (cnsq.pType != altr.pType)
          PType.canonical(cnsq.typ.deepOptional())
        else
          cnsq.pType
      case Let(name, value, body) =>
        body.pType
      case AggLet(name, value, body, _) =>
        body.pType
      case ApplyBinaryPrimOp(op, l, r) =>
        PType.canonical(BinaryOp.getReturnType(op, l.typ, r.typ)).setRequired(l.pType.required && r.pType.required)
      case ApplyUnaryPrimOp(op, v) =>
        PType.canonical(UnaryOp.getReturnType(op, v.typ)).setRequired(v.pType.required)
      case ApplyComparisonOp(op, l, r) =>
        assert(l.typ isOfType r.typ, s"${l.typ.parsableString()} vs ${r.typ.parsableString()}")
        op match {
          case _: Compare => PInt32(l.pType.required && r.pType.required)
          case _ => PBoolean(l.pType.required && r.pType.required)
        }
      case a: ApplyIR => a.explicitNode.pType
      case a: AbstractApplyNode[_] => PType.canonical(a.typ)
      case _: Uniroot => PFloat64()
      case ArrayRef(a, i) =>
        coerce[PIterable](a.pType).elementType.setRequired(a.pType.required && i.pType.required)
      case ArraySort(a, _, _, _) =>
        val et = coerce[PIterable](a.pType).elementType
        PArray(et, a.pType.required)
      case ToSet(a) =>
        val elt = coerce[PIterable](a.pType).elementType
        PSet(elt, a.pType.required)
      case ToDict(a) =>
        val elt = coerce[PBaseStruct](coerce[PIterable](a.pType).elementType)
        PDict(elt.types(0), elt.types(1), a.pType.required)
      case ToArray(a) =>
        val elt = coerce[PIterable](a.pType).elementType
        PArray(elt, a.pType.required)
      case ToStream(a) =>
        val elt = coerce[PIterable](a.pType).elementType
        PStream(elt, a.pType.required)
      case GroupByKey(collection) =>
        val elt = coerce[PBaseStruct](coerce[PIterable](collection.pType).elementType)
        // FIXME requiredness
        PDict(elt.types(0), PArray(elt.types(1), required = false), collection.pType.required)
      case ArrayMap(a, name, body) =>
        // FIXME: requiredness artifact of former IR bug, remove when possible
        propagateStreamable(coerce[PStreamable](a.pType), body.pType.setRequired(false))
      case ArrayFilter(a, name, cond) =>
        a.pType
      case ArrayFlatMap(a, name, body) =>
        propagateStreamable(coerce[PStreamable](a.pType), coerce[PIterable](body.pType).elementType)
      case ArrayFold(a, zero, accumName, valueName, body) =>
        zero.pType
      case ArrayScan(a, zero, accumName, valueName, body) =>
        propagateStreamable(coerce[PStreamable](a.pType), zero.pType)
      case ArrayAgg(_, _, query) =>
        query.pType
      case ArrayLeftJoinDistinct(left, right, l, r, compare, join) =>
        propagateStreamable(coerce[PStreamable](left.pType), join.pType)
        PArray(join.pType)
      case NDArrayMap(nd, _, body) =>
        PNDArray(body.pType, coerce[TNDArray](nd.typ).nDims, nd.typ.required)
      case NDArrayMap2(l, _, _, _, body) =>
        PNDArray(body.pType, coerce[TNDArray](l.typ).nDims, l.typ.required)
      case NDArrayRef(nd, idxs) =>
        coerce[PNDArray](nd.pType).elementType.setRequired(nd.pType.required && 
          idxs.pType.required &&
          coerce[PStreamable](idxs.pType).elementType.required)
      case AggFilter(_, aggIR, _) =>
        aggIR.pType
      case AggExplode(array, name, aggBody, _) =>
        aggBody.pType
      case AggGroupBy(key, aggIR, _) =>
        PDict(PType.canonical(key.pType), aggIR.pType)
      case AggArrayPerElement(a, name, aggBody, _) => PArray(aggBody.pType)
      case ApplyAggOp(_, _, _, aggSig) =>
        PType.canonical(AggOp.getType(aggSig))
      case ApplyScanOp(_, _, _, aggSig) =>
        PType.canonical(AggOp.getType(aggSig))
      case MakeStruct(fields) =>
        // FIXME requiredness
        PStruct(required = false,
          fields.map { case (name, a) =>
          (name, a.pType)
        }: _*)
      case SelectFields(old, fields) =>
        val tbs = coerce[PStruct](old.pType)
        tbs.selectFields(fields)
      case InsertFields(old, fields, fieldOrder) =>
        val tbs = coerce[TStruct](old.typ)
        val s = tbs.insertFields(fields.map(f => (f._1, f._2.typ)))
        fieldOrder.map(fds => TStruct(fds.map(f => f -> s.fieldType(f)): _*)).getOrElse(s).physicalType
      case GetField(o, name) =>
        val t = coerce[PStruct](o.pType)
        val fd = t.field(name).typ
        fd.setRequired(t.required && fd.required)
      case MakeTuple(values) =>
        // FIXME requiredness
        PTuple(values.map(_.pType).toFastIndexedSeq, required = false)
      case GetTupleElement(o, idx) =>
        val t = coerce[PTuple](o.pType)
        val fd = t.types(idx)
        fd.setRequired(t.required && fd.required)
      case TableCount(_) => PInt64()
      case TableAggregate(child, query) =>
        query.pType
      case MatrixAggregate(child, query) =>
        query.pType
      case _: TableWrite => PVoid
      case _: MatrixWrite => PVoid
      case _: MatrixMultiWrite => PVoid
      case _: BlockMatrixWrite => PVoid
      case TableGetGlobals(child) => PType.canonical(child.typ.globalType)
      case TableCollect(child) => PStruct("rows" -> PArray(PType.canonical(child.typ.rowType)), "global" -> PType.canonical(child.typ.globalType))
      case TableToValueApply(child, function) => PType.canonical(function.typ(child.typ))
      case MatrixToValueApply(child, function) => PType.canonical(function.typ(child.typ))
      case BlockMatrixToValueApply(child, function) => PType.canonical(function.typ(child.typ))
      case CollectDistributedArray(_, _, _, _, body) => PArray(body.pType)
      case ReadPartition(_, _, _, rowType) => PType.canonical(TStream(rowType))
    }
  }
}
