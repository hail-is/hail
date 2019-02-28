package is.hail.expr.ir

import is.hail.expr.types.virtual._
import is.hail.utils._

// FIXME: strip all requiredness logic when possible
object InferType {
  def apply(ir: IR): Type = {
    ir match {
      case I32(_) => TInt32()
      case I64(_) => TInt64()
      case F32(_) => TFloat32()
      case F64(_) => TFloat64()
      case Str(_) => TString()
      case Literal(t, _) => t
      case True() | False() => TBoolean()
      case Void() => TVoid
      case Cast(_, t) => t
      case NA(t) => t
      case IsNA(_) => TBoolean()
      case Ref(_, t) => t
      case In(_, t) => t
      case MakeArray(_, t) => t
      case MakeNDArray(data, _, _) => TNDArray(data.typ.asInstanceOf[TArray].elementType)
      case _: ArrayLen => TInt32()
      case _: ArrayRange => TArray(TInt32())
      case _: LowerBoundOnOrderedCollection => TInt32()
      case _: ArrayFor => TVoid
      case _: InitOp => TVoid
      case _: SeqOp => TVoid
      case _: Begin => TVoid
      case _: StringLength => TInt32()
      case _: StringSlice => TString()
      case Die(_, t) => t
      case If(cond, cnsq, altr) =>
        assert(cond.typ.isOfType(TBoolean()))
        assert(cnsq.typ.isOfType(altr.typ))
        if (cnsq.typ != altr.typ)
          cnsq.typ.deepOptional()
        else
          cnsq.typ
      case Let(name, value, body) =>
        body.typ
      case AggLet(name, value, body) =>
        body.typ
      case ApplyBinaryPrimOp(op, l, r) =>
        BinaryOp.getReturnType(op, l.typ, r.typ).setRequired(l.typ.required && r.typ.required)
      case ApplyUnaryPrimOp(op, v) =>
        UnaryOp.getReturnType(op, v.typ).setRequired(v.typ.required)
      case ApplyComparisonOp(op, l, r) =>
        assert(l.typ isOfType r.typ)
        op match {
          case _: Compare => TInt32(l.pType.required && r.pType.required)
          case _ => TBoolean(l.pType.required && r.pType.required)
        }
      case a: ApplyIR => a.explicitNode.typ
      case a: AbstractApplyNode[_] =>
        val argTypes = a.args.map(_.typ)
        a.implementation.unify(argTypes)
        a.implementation.returnType.subst()
      case _: Uniroot => TFloat64()
      case ArrayRef(a, i) =>
        assert(i.typ.isOfType(TInt32()))
        coerce[TArray](a.typ).elementType.setRequired(a.typ.required && i.typ.required)
      case ArraySort(a, _, _, compare) =>
        assert(compare.typ.isOfType(TBoolean()))
        val et = coerce[TArray](a.typ).elementType
        TArray(et, a.typ.required)
      case ToSet(a) =>
        val et = coerce[TArray](a.typ).elementType
        TSet(et, a.typ.required)
      case ToDict(a) =>
        val elt = coerce[TBaseStruct](coerce[TArray](a.typ).elementType)
        TDict(elt.types(0), elt.types(1), a.typ.required)
      case ToArray(a) =>
        val et = coerce[TContainer](a.typ).elementType
        TArray(et, a.typ.required)
      case GroupByKey(collection) =>
        val elt = coerce[TBaseStruct](coerce[TArray](collection.typ).elementType)
        TDict(elt.types(0), TArray(elt.types(1)), collection.typ.required)
      case ArrayMap(a, name, body) =>
        TArray(body.typ.setRequired(false), a.typ.required)
      case ArrayFilter(a, name, cond) =>
        TArray(coerce[TArray](a.typ).elementType, a.typ.required)
      case ArrayFlatMap(a, name, body) =>
        TArray(coerce[TContainer](body.typ).elementType, a.typ.required)
      case ArrayFold(a, zero, accumName, valueName, body) =>
        assert(body.typ == zero.typ)
        zero.typ
      case ArrayScan(a, zero, accumName, valueName, body) =>
        assert(body.typ == zero.typ)
        TArray(zero.typ)
      case ArrayAgg(_, _, query) =>
        query.typ
      case ArrayLeftJoinDistinct(left, right, l, r, compare, join) =>
        TArray(join.typ)
      case NDArrayRef(nd, idxs) =>
        assert(idxs.typ.isOfType(TArray(TInt64())))
        coerce[TNDArray](nd.typ).elementType.setRequired(nd.typ.required && 
          idxs.typ.required && 
          coerce[TArray](idxs.typ).elementType.required)
      case AggFilter(_, aggIR) =>
        aggIR.typ
      case AggExplode(array, name, aggBody) =>
        aggBody.typ
      case AggGroupBy(key, aggIR) =>
        TDict(key.typ, aggIR.typ)
      case AggArrayPerElement(a, name, aggBody) => TArray(aggBody.typ)
      case ApplyAggOp(_, _, _, aggSig) =>
        AggOp.getType(aggSig)
      case ApplyScanOp(_, _, _, aggSig) =>
        AggOp.getType(aggSig)
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
        val fd = t.field(name).typ
        fd.setRequired(t.required && fd.required)
      case MakeTuple(values) =>
        TTuple(values.map(_.typ).toFastIndexedSeq)
      case GetTupleElement(o, idx) =>
        val t = coerce[TTuple](o.typ)
        assert(idx >= 0 && idx < t.size)
        val fd = t.types(idx)
        fd.setRequired(t.required && fd.required)
      case TableCount(_) => TInt64()
      case TableAggregate(child, query) =>
        query.typ
      case MatrixAggregate(child, query) =>
        query.typ
      case _: TableWrite => TVoid
      case _: MatrixWrite => TVoid
      case _: MatrixMultiWrite => TVoid
      case _: BlockMatrixWrite => TVoid
      case _: TableExport => TVoid
      case TableGetGlobals(child) => child.typ.globalType
      case TableCollect(child) => TStruct("rows" -> TArray(child.typ.rowType), "global" -> child.typ.globalType)
      case TableToValueApply(child, function) => function.typ(child.typ)
      case MatrixToValueApply(child, function) => function.typ(child.typ)
      case BlockMatrixToValueApply(child, function) => function.typ(child.typ)
      case CollectDistributedArray(_, _, _, _, body) => TArray(body.typ)
    }
  }
}
