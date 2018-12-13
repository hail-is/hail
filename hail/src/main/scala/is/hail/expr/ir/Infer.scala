package is.hail.expr.ir

import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils._

object Infer {
  def apply(ir: InferIR): PType = {
    ir match {
      case If(cond, cnsq, altr) =>
        assert(cond.typ.isOfType(TBoolean()))
        assert(cnsq.typ.isOfType(altr.typ))
        if (cnsq.pType != altr.pType)
          PType.canonical(cnsq.typ.deepOptional())
        else
          cnsq.pType
      case Let(name, value, body) =>
        body.pType
      case ApplyBinaryPrimOp(op, l, r) =>
        PType.canonical(BinaryOp.getReturnType(op, l.typ, r.typ)).setRequired(l.pType.required && r.pType.required)
      case ApplyUnaryPrimOp(op, v) =>
        PType.canonical(UnaryOp.getReturnType(op, v.typ)).setRequired(v.pType.required)
      case ApplyComparisonOp(op, l, r) =>
        assert(l.typ isOfType r.typ, s"${l.typ.parsableString()} vs ${r.typ.parsableString()}")
        PBoolean(l.pType.required && r.pType.required)
      case ArrayRef(a, i) =>
        assert(i.typ.isOfType(TInt32()))
        coerce[PArray](a.pType).elementType.setRequired(a.pType.required && i.pType.required)
      case ArraySort(a, ascending, _) =>
        assert(ascending.typ.isOfType(TBoolean()))
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
        assert(body.pType == zero.pType)
        zero.pType
      case ArrayScan(a, zero, accumName, valueName, body) =>
        assert(body.pType == zero.pType)
        PArray(zero.pType)
      case AggFilter(_, aggIR) =>
        aggIR.pType
      case AggExplode(array, name, aggBody) =>
        aggBody.pType
      case AggGroupBy(key, aggIR) =>
        PDict(PType.canonical(key.pType), aggIR.pType)
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
      case InsertFields(old, fields) =>
        val tbs = coerce[PStruct](old.pType)
        tbs.insertFields(fields.map(f => (f._1, f._2.pType)))
      case GetField(o, name) =>
        val t = coerce[PStruct](o.pType)
        assert(t.index(name).nonEmpty, s"$name not in $t")
        val fd = t.field(name).typ
        fd.setRequired(t.required && fd.required)
      case MakeTuple(values) =>
        // FIXME requiredness
        PTuple(values.map(_.pType).toFastIndexedSeq, required = false)
      case GetTupleElement(o, idx) =>
        val t = coerce[PTuple](o.pType)
        assert(idx >= 0 && idx < t.size)
        val fd = t.types(idx)
        fd.setRequired(t.required && fd.required)
      case TableAggregate(child, query) =>
        query.pType
      case MatrixAggregate(child, query) =>
        query.pType
      case TableGetGlobals(child) => PType.canonical(child.typ.globalType)
      case TableCollect(child) => PStruct("rows" -> PArray(PType.canonical(child.typ.rowType)), "global" -> PType.canonical(child.typ.globalType))
    }
  }
}
