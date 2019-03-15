package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.utils._

object TypeCheck {
  def apply(ir: BaseIR): Unit = apply(ir, Env.empty, None)


  def apply(ir: BaseIR, env: Env[Type], aggEnv: Option[Env[Type]]): Unit = {
    try {
      ir match {
        case ir: IR => _apply(ir, env, aggEnv)
        case _ =>
          assert(env.m.isEmpty)
          assert(aggEnv.isEmpty)
      }

    } catch {
      case e: Throwable => fatal(s"Error while typechecking IR:\n${ Pretty(ir) }", e)
    }
  }

  private def _apply(ir: IR, env: Env[Type], aggEnv: Option[Env[Type]]): Unit = {
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
      case Cast(v, typ) => assert(Casts.valid(v.typ, typ))
      case NA(t) =>
        assert(t != null)
        assert(!t.required)
      case IsNA(v) =>
      case x@If(cond, cnsq, altr) =>
        assert(cond.typ.isOfType(TBoolean()))
        assert(cnsq.typ == altr.typ, s"Type mismatch:\n  cnsq: ${ cnsq.typ.parsableString() }\n  altr: ${ altr.typ.parsableString() }\n  $x")
        assert(x.typ == cnsq.typ)

      case x@Let(_, _, body) =>
        assert(x.typ == body.typ)
      case x@AggLet(_, _, body) =>
        assert(x.typ == body.typ)
      case x@Ref(name, _) =>
        val expected = env.lookup(name)
        assert(x.typ == expected, s"type mismatch:\n  name: $name\n  actual: ${ x.typ.parsableString() }\n  expect: ${ expected.parsableString() }")
      case x@ApplyBinaryPrimOp(op, l, r) =>
        assert(x.typ == BinaryOp.getReturnType(op, l.typ, r.typ))
      case x@ApplyUnaryPrimOp(op, v) =>
        assert(x.typ == UnaryOp.getReturnType(op, v.typ))
      case x@ApplyComparisonOp(op, l, r) =>
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
      case x@ArrayRef(a, i) =>
        assert(i.typ.isOfType(TInt32()))
        assert(x.typ == -coerce[TArray](a.typ).elementType)
      case ArrayLen(a) =>
        assert(a.typ.isInstanceOf[TArray])
      case x@ArrayRange(a, b, c) =>
        assert(a.typ.isOfType(TInt32()))
        assert(b.typ.isOfType(TInt32()))
        assert(c.typ.isOfType(TInt32()))
      case x@MakeNDArray(data, shape, row_major) =>
        assert(data.typ.isInstanceOf[TArray])
        assert(coerce[TNDArray](x.typ).elementType == coerce[TArray](data.typ).elementType)
        assert(shape.typ.isOfType(TArray(TInt64())))
        assert(row_major.typ.isOfType(TBoolean()))
      case x@NDArrayRef(nd, idxs) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(idxs.typ.isOfType(TArray(TInt64())))
      case x@ArraySort(a, l, r, compare) =>
        val tarray = coerce[TArray](a.typ)
        assert(compare.typ.isOfType(TBoolean()))
      case x@ToSet(a) =>
        assert(a.typ.isInstanceOf[TArray])
      case x@ToDict(a) =>
        assert(a.typ.isInstanceOf[TArray])
        assert(coerce[TBaseStruct](coerce[TArray](a.typ).elementType).size == 2)
      case x@ToArray(a) =>
        assert(a.typ.isInstanceOf[TContainer])
      case x@LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val elt = -coerce[TContainer](orderedCollection.typ).elementType
        assert(-elem.typ == (if (onKey) -coerce[TStruct](elt).types(0) else elt))
      case x@GroupByKey(collection) =>
        val telt = coerce[TBaseStruct](coerce[TArray](collection.typ).elementType)
        val td = coerce[TDict](x.typ)
        assert(td.keyType == telt.types(0))
        assert(td.valueType == TArray(telt.types(1)))
      case x@ArrayMap(a, name, body) =>
        assert(x.elementTyp == body.typ)
      case x@ArrayFilter(a, name, cond) =>
        val tarray = coerce[TArray](a.typ)
        assert(cond.typ.isOfType(TBoolean()))
      case x@ArrayFlatMap(a, name, body) =>
        val tarray = coerce[TArray](a.typ)
        assert(body.typ.isInstanceOf[TArray])
      case x@ArrayFold(a, zero, accumName, valueName, body) =>
        val tarray = coerce[TArray](a.typ)
        assert(body.typ == zero.typ)
        assert(x.typ == zero.typ)
      case x@ArrayScan(a, zero, accumName, valueName, body) =>
        val tarray = coerce[TArray](a.typ)
        assert(body.typ == zero.typ)
        assert(x.typ == TArray(zero.typ))
      case x@ArrayLeftJoinDistinct(left, right, l, r, compare, join) =>
        val ltyp = coerce[TArray](left.typ)
        val rtyp = coerce[TArray](right.typ)
        assert(compare.typ.isOfType(TInt32()))
        assert(x.typ == TArray(join.typ))
      case x@ArrayFor(a, valueName, body) =>
        val tarray = coerce[TArray](a.typ)
        assert(body.typ == TVoid)
      case x@ArrayAgg(a, name, query) =>
        val tarray = coerce[TArray](a.typ)
        assert(aggEnv.isEmpty)
      case x@AggFilter(cond, aggIR) =>
        assert(cond.typ isOfType TBoolean())
        assert(x.typ == aggIR.typ)
      case x@AggExplode(array, name, aggBody) =>
        assert(array.typ.isInstanceOf[TArray])
        assert(x.typ == aggBody.typ)
      case x@AggGroupBy(key, aggIR) =>
        assert(x.typ == TDict(key.typ, aggIR.typ))
      case x@AggArrayPerElement(a, name, aggBody) =>
        assert(x.typ == TArray(aggBody.typ))
      case x@InitOp(i, args, aggSig) =>
        assert(Some(args.map(_.typ)) == aggSig.initOpArgs)
        assert(i.typ.isInstanceOf[TInt32])
      case x@SeqOp(i, args, aggSig) =>
        assert(args.map(_.typ) == aggSig.seqOpArgs)
        assert(i.typ.isInstanceOf[TInt32])
      case x@Begin(xs) =>
        xs.foreach { x =>
          assert(x.typ == TVoid)
        }
      case x@ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
        assert(x.typ == AggOp.getType(aggSig))
      case x@ApplyScanOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
        assert(x.typ == AggOp.getType(aggSig))
      case x@MakeStruct(fields) =>
        assert(x.typ == TStruct(fields.map { case (name, a) =>
          (name, a.typ)
        }: _*))
      case x@SelectFields(old, fields) =>
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
      case x@GetField(o, name) =>
        val t = coerce[TStruct](o.typ)
        assert(t.index(name).nonEmpty, s"$name not in $t")
        assert(x.typ == -t.field(name).typ)
      case x@MakeTuple(types) =>
        assert(x.typ == TTuple(types.map(_.typ): _*))
      case x@GetTupleElement(o, idx) =>
        val t = coerce[TTuple](o.typ)
        assert(idx >= 0 && idx < t.size)
        assert(x.typ == -t.types(idx))
      case StringSlice(s, start, end) =>
        assert(s.typ isOfType TString())
        assert(start.typ isOfType TInt32())
        assert(end.typ isOfType TInt32())
      case StringLength(s) =>
        assert(s.typ isOfType TString())
      case In(i, typ) =>
        assert(typ != null)
      case Die(msg, typ) =>
        assert(msg.typ isOfType TString())
      case x@ApplyIR(fn, args) =>
        check(x.explicitNode, env, aggEnv)
      case x: AbstractApplyNode[_] =>
        x.args.foreach(check(_))
        assert(x.implementation.unify(x.args.map(_.typ)))
      case Uniroot(name, fn, min, max) =>
        assert(fn.typ.isInstanceOf[TFloat64])
        assert(min.typ.isInstanceOf[TFloat64])
        assert(max.typ.isInstanceOf[TFloat64])
      case MatrixWrite(_, _) =>
      case MatrixMultiWrite(_, _) => // do nothing
      case x@TableAggregate(child, query) =>
        assert(x.typ == query.typ)
      case x@MatrixAggregate(child, query) =>
        assert(x.typ == query.typ)
      case TableWrite(_, _, _, _, _) =>
      case TableExport(_, _, _, _, _, _) =>
      case TableCount(_) =>
      case TableGetGlobals(_) =>
      case TableCollect(_) =>
      case TableToValueApply(_, _) =>
      case MatrixToValueApply(_, _) =>
      case BlockMatrixToValueApply(_, _) =>
      case BlockMatrixWrite(_, _) =>
      case CollectDistributedArray(ctxs, globals, cname, gname, body) =>
        assert(ctxs.typ.isInstanceOf[TArray])
      case x@ReadPartition(path, _, _, rowType) =>
        check(path.typ == TString())
        assert(x.typ == TArray(rowType))
    }

    ir.children
      .iterator
      .zipWithIndex
      .foreach {
        case (child: IR, i) =>
          val (e, ae) = ChildEnv(ir, i, env, aggEnv)
          check(child, e, ae)
        case (tir: TableIR, _) =>
        case (mir: MatrixIR, _) =>
        case (bmir: BlockMatrixIR, _) =>
      }
  }
}
