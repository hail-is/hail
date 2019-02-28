package is.hail.expr.ir

import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils._

object InferPType {
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
      case MakeNDArray(data, _, _) => PNDArray(data.pType.asInstanceOf[PArray].elementType)
      case _: ArrayLen => PInt32()
      case _: ArrayRange => PArray(PInt32())
      case _: LowerBoundOnOrderedCollection => PInt32()
      case _: ArrayFor => PVoid
      case _: InitOp => PVoid
      case _: SeqOp => PVoid
      case _: Begin => PVoid
      case _: StringLength => PInt32()
      case _: StringSlice => PString()
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
      case AggLet(name, value, body) =>
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
        coerce[PArray](a.pType).elementType.setRequired(a.pType.required && i.pType.required)
      case ArraySort(a, _, _, _) =>
        val et = coerce[PArray](a.pType).elementType
        PArray(et, a.pType.required)
      case ToSet(a) =>
        val et = coerce[PArray](a.pType).elementType
        PSet(et, a.pType.required)
      case ToDict(a) =>
        val elt = coerce[PBaseStruct](coerce[PArray](a.pType).elementType)
        PDict(elt.types(0), elt.types(1), a.pType.required)
      case ToArray(a) =>
        val et = coerce[PContainer](a.pType).elementType
        PArray(et, a.pType.required)
      case GroupByKey(collection) =>
        val elt = coerce[PBaseStruct](coerce[PArray](collection.pType).elementType)
        // FIXME requiredness
        PDict(elt.types(0), PArray(elt.types(1), required = false), collection.pType.required)
      case ArrayMap(a, name, body) =>
        // FIXME: requiredness artifact of former IR bug, remove when possible
        PArray(body.pType.setRequired(false), a.typ.required)
      case ArrayFilter(a, name, cond) =>
        PArray(coerce[PArray](a.pType).elementType, a.pType.required)
      case ArrayFlatMap(a, name, body) =>
        PArray(coerce[PContainer](body.pType).elementType, a.pType.required)
      case ArrayFold(a, zero, accumName, valueName, body) =>
        zero.pType
      case ArrayScan(a, zero, accumName, valueName, body) =>
        PArray(zero.pType)
      case ArrayAgg(_, _, query) =>
        query.pType
      case ArrayLeftJoinDistinct(left, right, l, r, compare, join) =>
        PArray(join.pType)
      case NDArrayRef(nd, idxs) =>
        coerce[PNDArray](nd.pType).elementType.setRequired(nd.pType.required && 
          idxs.pType.required &&
          coerce[PArray](idxs.pType).elementType.required)
      case AggFilter(_, aggIR) =>
        aggIR.pType
      case AggExplode(array, name, aggBody) =>
        aggBody.pType
      case AggGroupBy(key, aggIR) =>
        PDict(PType.canonical(key.pType), aggIR.pType)
      case AggArrayPerElement(a, name, aggBody) => PArray(aggBody.pType)
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
      case _: TableExport => PVoid
      case TableGetGlobals(child) => PType.canonical(child.typ.globalType)
      case TableCollect(child) => PStruct("rows" -> PArray(PType.canonical(child.typ.rowType)), "global" -> PType.canonical(child.typ.globalType))
      case TableToValueApply(child, function) => PType.canonical(function.typ(child.typ))
      case MatrixToValueApply(child, function) => PType.canonical(function.typ(child.typ))
      case BlockMatrixToValueApply(child, function) => PType.canonical(function.typ(child.typ))
      case CollectDistributedArray(_, _, _, _, body) => PArray(body.pType)
    }
  }
}
