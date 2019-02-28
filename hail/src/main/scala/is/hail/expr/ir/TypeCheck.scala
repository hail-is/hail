package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.utils._

object TypeCheck {
  def apply(ir: IR, aggEnv: Option[Env[Type]] = None) {
    apply(ir, new Env[Type](), aggEnv)
  }

  def apply(ir: IR, env: Env[Type], aggEnv: Option[Env[Type]]): Unit = {
    try {
      _apply(ir, env, aggEnv)
    } catch {
      case e: Throwable => fatal(s"Error while typechecking IR:\n${ Pretty(ir) }", e)
    }
  }

  private def _apply(ir: IR, env: Env[Type], aggEnv: Option[Env[Type]]) {
    def check(ir: IR, env: Env[Type] = env, aggEnv: Option[Env[Type]] = aggEnv) {
      _apply(ir, env, aggEnv)
    }

    ir match {
      case I32(x) =>
      case I64(x) =>
      case F32(x) =>
      case F64(x) =>
      case True() =>
      case False() =>
      case Str(x) =>
      case Literal(_, _) =>
      case Void() =>

      case Cast(v, typ) =>
        check(v)
        assert(Casts.valid(v.typ, typ))

      case NA(t) =>
        assert(t != null)
        assert(!t.required)
      case IsNA(v) =>
        check(v)

      case x@If(cond, cnsq, altr) =>
        check(cond)
        check(cnsq)
        check(altr)
        assert(cond.typ.isOfType(TBoolean()))
        assert(cnsq.typ == altr.typ, s"Type mismatch:\n  cnsq: ${ cnsq.typ.parsableString() }\n  altr: ${ altr.typ.parsableString() }\n  $x")
        assert(x.typ == cnsq.typ)

      case x@Let(name, value, body) =>
        check(value)
        check(body, env = env.bind(name, value.typ))
        assert(x.typ == body.typ)
      case x@AggLet(name, value, body) =>
        check(value, env = aggEnv.get, aggEnv = None)
        check(body, env, aggEnv = aggEnv.map(_.bind(name, value.typ)))
        assert(x.typ == body.typ)
      case x@Ref(name, _) =>
        val expected = env.lookup(x)
        assert(x.typ == expected, s"type mismatch:\n  name: $name\n  actual: ${ x.typ.parsableString() }\n  expect: ${ expected.parsableString() }")
      case x@ApplyBinaryPrimOp(op, l, r) =>
        check(l)
        check(r)
        assert(x.typ == BinaryOp.getReturnType(op, l.typ, r.typ))
      case x@ApplyUnaryPrimOp(op, v) =>
        check(v)
        assert(x.typ == UnaryOp.getReturnType(op, v.typ))
      case x@ApplyComparisonOp(op, l, r) =>
        check(l)
        check(r)
        assert(-op.t1.fundamentalType == -l.typ.fundamentalType)
        assert(-op.t2.fundamentalType == -r.typ.fundamentalType)
        op match {
          case _: Compare => assert(x.typ.isInstanceOf[TInt32])
          case _ => assert(x.typ.isInstanceOf[TBoolean])
        }
      case x@MakeArray(args, typ) =>
        assert(typ != null)
        args.map(_.typ).zipWithIndex.foreach { case (x, i) => assert(x == typ.elementType,
          s"at position $i type mismatch: ${ typ.parsableString() } ${ x.parsableString() }")
        }
        args.foreach(check(_))
      case x@ArrayRef(a, i) =>
        check(a)
        check(i)
        assert(i.typ.isOfType(TInt32()))
        assert(x.typ == -coerce[TArray](a.typ).elementType)
      case ArrayLen(a) =>
        check(a)
        assert(a.typ.isInstanceOf[TArray])
      case x@ArrayRange(a, b, c) =>
        check(a)
        check(b)
        check(c)
        assert(a.typ.isOfType(TInt32()))
        assert(b.typ.isOfType(TInt32()))
        assert(c.typ.isOfType(TInt32()))
      case x@MakeNDArray(data, shape, row_major) =>
        check(data)
        check(shape)
        check(row_major)
        assert(data.typ.isInstanceOf[TArray])
        assert(coerce[TNDArray](x.typ).elementType == coerce[TArray](data.typ).elementType)
        assert(shape.typ.isOfType(TArray(TInt64())))
        assert(row_major.typ.isOfType(TBoolean()))
      case x@NDArrayRef(nd, idxs) =>
        check(nd)
        check(idxs)
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(idxs.typ.isOfType(TArray(TInt64())))
      case x@ArraySort(a, l, r, compare) =>
        check(a)
        val tarray = coerce[TArray](a.typ)
        check(compare, env = env.bind(l, -tarray.elementType).bind(r, -tarray.elementType))
        assert(compare.typ.isOfType(TBoolean()))
      case x@ToSet(a) =>
        check(a)
        assert(a.typ.isInstanceOf[TArray])
      case x@ToDict(a) =>
        check(a)
        assert(a.typ.isInstanceOf[TArray])
        assert(coerce[TBaseStruct](coerce[TArray](a.typ).elementType).size == 2)
      case x@ToArray(a) =>
        check(a)
        assert(a.typ.isInstanceOf[TContainer])
      case x@LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        check(orderedCollection)
        check(elem)
        val elt = -coerce[TContainer](orderedCollection.typ).elementType
        assert(-elem.typ == (if (onKey) -coerce[TStruct](elt).types(0) else elt))
      case x@GroupByKey(collection) =>
        check(collection)
        val telt = coerce[TBaseStruct](coerce[TArray](collection.typ).elementType)
        val td = coerce[TDict](x.typ)
        assert(td.keyType == telt.types(0))
        assert(td.valueType == TArray(telt.types(1)))
      case x@ArrayMap(a, name, body) =>
        check(a)
        val tarray = coerce[TArray](a.typ)
        check(body, env = env.bind(name, -tarray.elementType))
        assert(x.elementTyp == body.typ)
      case x@ArrayFilter(a, name, cond) =>
        check(a)
        val tarray = coerce[TArray](a.typ)
        check(cond, env = env.bind(name, -tarray.elementType))
        assert(cond.typ.isOfType(TBoolean()))
      case x@ArrayFlatMap(a, name, body) =>
        check(a)
        val tarray = coerce[TArray](a.typ)
        check(body, env = env.bind(name, -tarray.elementType))
        assert(body.typ.isInstanceOf[TArray])
      case x@ArrayFold(a, zero, accumName, valueName, body) =>
        check(a)
        val tarray = coerce[TArray](a.typ)
        check(zero)
        check(body, env = env.bind(accumName -> zero.typ, valueName -> -tarray.elementType))
        assert(body.typ == zero.typ)
        assert(x.typ == zero.typ)
      case x@ArrayScan(a, zero, accumName, valueName, body) =>
        check(a)
        val tarray = coerce[TArray](a.typ)
        check(zero)
        check(body, env = env.bind(accumName -> zero.typ, valueName -> -tarray.elementType))
        assert(body.typ == zero.typ)
        assert(x.typ == TArray(zero.typ))
      case x@ArrayLeftJoinDistinct(left, right, l, r, compare, join) =>
        check(left)
        check(right)
        val ltyp = coerce[TArray](left.typ)
        val rtyp = coerce[TArray](right.typ)
        check(compare, env = env.bind(l -> -ltyp.elementType, r -> -rtyp.elementType))
        check(join, env = env.bind(l -> -ltyp.elementType, r -> -rtyp.elementType))
        assert(compare.typ.isOfType(TInt32()))
        assert(x.typ == TArray(join.typ))
      case x@ArrayFor(a, valueName, body) =>
        check(a)
        val tarray = coerce[TArray](a.typ)
        check(body, env = env.bind(valueName -> -tarray.elementType))
        assert(body.typ == TVoid)
      case x@ArrayAgg(a, name, query) =>
        check(a)
        val tarray = coerce[TArray](a.typ)
        assert(aggEnv.isEmpty)
        check(query, env, Some(env.bind(name, tarray.elementType)))
      case x@AggFilter(cond, aggIR) =>
        check(cond, env = aggEnv.get)
        check(aggIR)
        assert(cond.typ isOfType TBoolean())
        assert(x.typ == aggIR.typ)
      case x@AggExplode(array, name, aggBody) =>
        check(array, env = aggEnv.get)
        assert(array.typ.isInstanceOf[TArray])
        check(aggBody, env = env, aggEnv = aggEnv.map(_.bind(name -> -coerce[TArray](array.typ).elementType)))
        assert(x.typ == aggBody.typ)
      case x@AggGroupBy(key, aggIR) =>
        check(key, env = aggEnv.get)
        check(aggIR)
        assert(x.typ == TDict(key.typ, aggIR.typ))
      case x@AggArrayPerElement(a, name, aggBody) =>
        check(a, env = aggEnv.get)
        check(aggBody, env = env, aggEnv = aggEnv.map(_.bind(name -> -coerce[TArray](a.typ).elementType)))
        assert(x.typ == TArray(aggBody.typ))
      case x@InitOp(i, args, aggSig) =>
        args.foreach(check(_))
        check(i)
        assert(Some(args.map(_.typ)) == aggSig.initOpArgs)
        assert(i.typ.isInstanceOf[TInt32])
      case x@SeqOp(i, args, aggSig) =>
        check(i)
        args.foreach(check(_))
        assert(args.map(_.typ) == aggSig.seqOpArgs)
        assert(i.typ.isInstanceOf[TInt32])
      case x@Begin(xs) =>
        xs.foreach { x =>
          check(x)
          assert(x.typ == TVoid)
        }
      case x@ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
        seqOpArgs.foreach(check(_, env = aggEnv.get))
        constructorArgs.foreach(check(_))
        initOpArgs.foreach(_.foreach(check(_)))
        assert(x.typ == AggOp.getType(aggSig))
      case x@ApplyScanOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
        constructorArgs.foreach(check(_))
        initOpArgs.foreach(_.foreach(check(_)))
        seqOpArgs.foreach(check(_, env = aggEnv.get))
        assert(x.typ == AggOp.getType(aggSig))
      case x@MakeStruct(fields) =>
        fields.foreach { case (name, a) => check(a) }
        assert(x.typ == TStruct(fields.map { case (name, a) =>
          (name, a.typ)
        }: _*))
      case x@SelectFields(old, fields) =>
        check(old)
        assert {
          val oldfields = coerce[TStruct](old.typ).fieldNames.toSet
          fields.forall { id => oldfields.contains(id) }
        }
      case x@InsertFields(old, fields, fieldOrder) =>
        fieldOrder.foreach { fds =>
          val newFieldSet = fields.map(_._1).toSet
          val oldFieldNames = old.typ.asInstanceOf[TStruct].fieldNames
          val oldFieldNameSet = oldFieldNames.toSet
          assert(fds.length == x.typ.size)
          assert(oldFieldNames
            .filter(f => !newFieldSet.contains(f))
            .sameElements(fds.filter(f => !newFieldSet.contains(f))))
          assert(fds.areDistinct())
          assert(fds.toSet.forall(f => newFieldSet.contains(f) || oldFieldNameSet.contains(f)))
        }
        check(old)
        fields.foreach { case (name, a) => check(a) }
      case x@GetField(o, name) =>
        check(o)
        val t = coerce[TStruct](o.typ)
        assert(t.index(name).nonEmpty, s"$name not in $t")
        assert(x.typ == -t.field(name).typ)
      case x@MakeTuple(types) =>
        types.foreach { a => check(a) }
        assert(x.typ == TTuple(types.map(_.typ): _*))
      case x@GetTupleElement(o, idx) =>
        check(o)
        val t = coerce[TTuple](o.typ)
        assert(idx >= 0 && idx < t.size)
        assert(x.typ == -t.types(idx))
      case StringSlice(s, start, end) =>
        check(s)
        check(start)
        check(end)
        assert(s.typ isOfType TString())
        assert(start.typ isOfType TInt32())
        assert(end.typ isOfType TInt32())
      case StringLength(s) =>
        check(s)
        assert(s.typ isOfType TString())
      case In(i, typ) =>
        assert(typ != null)
      case Die(msg, typ) =>
        check(msg)
        assert(msg.typ isOfType TString())
      case x@ApplyIR(fn, args) =>
        check(x.explicitNode)
      case x: AbstractApplyNode[_] =>
        x.args.foreach(check(_))
        assert(x.implementation.unify(x.args.map(_.typ)))
      case Uniroot(_, fn, min, max) =>
        assert(fn.typ.isInstanceOf[TFloat64])
        assert(min.typ.isInstanceOf[TFloat64])
        assert(max.typ.isInstanceOf[TFloat64])
      case MatrixWrite(_, _) =>
      case MatrixMultiWrite(_, _) => // do nothing
      case x@TableAggregate(child, query) =>
        check(query,
          env = child.typ.globalEnv,
          aggEnv = Some(child.typ.rowEnv))
        assert(x.typ == query.typ)
      case x@MatrixAggregate(child, query) =>
        check(query,
          env = child.typ.globalEnv,
          aggEnv = Some(child.typ.entryEnv))
        assert(x.typ == query.typ)
      case TableWrite(_, _, _, _, _) =>
      case TableExport(_, _, _, _, _, _) =>
      case TableCount(_) =>
      case TableGetGlobals(_) =>
      case TableCollect(_) =>
      case TableToValueApply(_, _) =>
      case MatrixToValueApply(_, _) =>
      case BlockMatrixToValueApply(_, _) =>
      case BlockMatrixWrite(_, _, _, _, _) =>
      case CollectDistributedArray(ctxs, globals, cname, gname, body) =>
        check(ctxs)
        assert(ctxs.typ.isInstanceOf[TArray])
        check(globals)
        check(body, env = env.bind(cname, coerce[TArray](ctxs.typ).elementType).bind(gname, globals.typ))
    }
  }
}
