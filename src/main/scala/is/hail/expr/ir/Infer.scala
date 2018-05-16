package is.hail.expr.ir

import is.hail.expr.types._

object Infer {
  def apply(ir: IR): Type = {
    ir match {
      case x@If(cond, cnsq, altr) =>
        assert(cond.typ.isOfType(TBoolean()))
        assert(cnsq.typ == altr.typ, s"${ cnsq.typ }, ${ altr.typ }, $cond")
        cnsq.typ

      case x@Let(name, value, body) =>
        body.typ
      case x@ApplyBinaryPrimOp(op, l, r) =>
        BinaryOp.getReturnType(op, l.typ, r.typ)
      case x@ApplyUnaryPrimOp(op, v) =>
        UnaryOp.getReturnType(op, v.typ)
      case x@ArrayRef(a, i) =>
        assert(i.typ.isOfType(TInt32()))
        -coerce[TArray](a.typ).elementType
      case x@ArraySort(a) =>
        a.typ
      case x@ToSet(a) =>
        TSet(coerce[TArray](a.typ).elementType)
      case x@ToDict(a) =>
        val elt = coerce[TBaseStruct](coerce[TArray](a.typ).elementType)
        TDict(elt.types(0), elt.types(1))
      case x@ToArray(a) =>
        TArray(coerce[TContainer](a.typ).elementType)
      case x@DictGet(dict, key) =>
        -coerce[TDict](dict.typ).valueType
      case x@ArrayMap(a, name, body) =>
        TArray(-body.typ)
      case x@ArrayFilter(a, name, cond) =>
        a.typ
      case x@ArrayFlatMap(a, name, body) =>
        TArray(coerce[TContainer](body.typ).elementType)
      case x@ArrayFold(a, zero, accumName, valueName, body) =>
        assert(body.typ == zero.typ)
        zero.typ
      case x@ApplyAggOp(a, constructorArgs, initOpArgs, aggSig) =>
        AggOp.getType(aggSig)
      case x@MakeStruct(fields) =>
        TStruct(fields.map { case (name, a) =>
          (name, a.typ)
        }: _*)
      case x@InsertFields(old, fields) =>
        fields.foldLeft(old.typ) { case (t, (name, a)) =>
          t match {
            case t2: TStruct =>
              t2.selfField(name) match {
                case Some(f2) => t2.updateKey(name, f2.index, a.typ)
                case None => t2.appendKey(name, a.typ)
              }
            case _ => TStruct(name -> a.typ)
          }
        }.asInstanceOf[TStruct]
      case x@GetField(o, name) =>
        val t = coerce[TStruct](o.typ)
        assert(t.index(name).nonEmpty, s"$name not in $t")
        -t.field(name).typ
      case x@MakeTuple(types) =>
        TTuple(types.map(_.typ): _*)
      case x@GetTupleElement(o, idx) =>
        val t = coerce[TTuple](o.typ)
        assert(idx >= 0 && idx < t.size)
        -t.types(idx)
      case x@TableAggregate(child, query) =>
        query.typ
    }
  }
}
