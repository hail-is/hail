package is.hail.expr.ir

import is.hail.expr.types.virtual._
import is.hail.utils._

object TypeCheck {
  def apply(ir: BaseIR): Unit = {
    try {
      ir match {
        case ir: IR => check(ir, BindingEnv.empty)
        case tir: TableIR => check(tir)
        case mir: MatrixIR => check(mir)
        case bmir: BlockMatrixIR => check(bmir)
      }
    } catch {
      case e: Throwable => fatal(s"Error while typechecking IR:\n${ Pretty(ir) }", e)
    }
  }

  def apply(ir: IR, env: BindingEnv[Type]): Unit = {
    try {
      check(ir, env)
    } catch {
      case e: Throwable => fatal(s"Error while typechecking IR:\n${ Pretty(ir) }", e)
    }
  }

  private def check(tir: TableIR): Unit = tir.children
    .iterator
    .zipWithIndex
    .foreach {
      case (child: IR, i) => check(child, NewBindings(tir, i))
      case (tir: TableIR, _) => check(tir)
      case (mir: MatrixIR, _) => check(mir)
      case (bmir: BlockMatrixIR, _) => check(bmir)
    }


  private def check(mir: MatrixIR): Unit = mir.children
    .iterator
    .zipWithIndex
    .foreach {
      case (child: IR, i) => check(child, NewBindings(mir, i))
      case (tir: TableIR, _) => check(tir)
      case (mir: MatrixIR, _) => check(mir)
      case (bmir: BlockMatrixIR, _) => check(bmir)
    }

  private def check(bmir: BlockMatrixIR): Unit = bmir.children
    .iterator
    .zipWithIndex
    .foreach {
      case (child: IR, i) => check(child, NewBindings(bmir, i))
      case (tir: TableIR, _) => check(tir)
      case (mir: MatrixIR, _) => check(mir)
      case (bmir: BlockMatrixIR, _) => check(bmir)
    }

  private def check(ir: IR, env: BindingEnv[Type]): Unit = {
    ir.children
      .iterator
      .zipWithIndex
      .foreach {
        case (child: IR, i) => check(child, ChildBindings(ir, i, env))
        case (tir: TableIR, _) => check(tir)
        case (mir: MatrixIR, _) => check(mir)
        case (bmir: BlockMatrixIR, _) => check(bmir)
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
      case CastRename(v, typ) =>
        if (!v.typ.canCastTo(typ))
          throw new RuntimeException(s"invalid cast:\n  " +
            s"child type: ${ v.typ.parsableString() }\n  " +
            s"cast type:  ${ typ.parsableString() }")
      case NA(t) =>
        assert(t != null)
        assert(!t.required)
      case IsNA(v) =>
      case Coalesce(values) =>
        val t1 = values.head.typ
        if (!values.tail.forall(_.typ == t1))
          throw new RuntimeException(s"Coalesce expects all children to have the same type:" +
            s"${ values.map(v => s"\n  ${ v.typ.parsableString() }").mkString }")
      case x@If(cond, cnsq, altr) =>
        assert(cond.typ.isOfType(TBoolean()))
        assert(cnsq.typ == altr.typ, s"Type mismatch:\n  cnsq: ${ cnsq.typ.parsableString() }\n  altr: ${ altr.typ.parsableString() }\n  $x")
        assert(x.typ == cnsq.typ)

      case x@Let(_, _, body) =>
        assert(x.typ == body.typ)
      case x@AggLet(_, _, body, _) =>
        assert(x.typ == body.typ)
      case x@Ref(name, _) =>
        val expected = env.eval.lookup(name)
        assert(x.typ == expected, s"type mismatch:\n  name: $name\n  actual: ${ x.typ.parsableString() }\n  expect: ${ expected.parsableString() }")
      case RelationalRef(_, _) =>
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
      case x@MakeStream(args, typ) =>
        assert(typ != null)
        args.map(_.typ).zipWithIndex.foreach { case (x, i) => assert(x == typ.elementType,
          s"at position $i type mismatch: ${ typ.parsableString() } ${ x.parsableString() }")
        }
      case x@ArrayRef(a, i) =>
        assert(i.typ.isOfType(TInt32()))
        assert(x.typ == -coerce[TStreamable](a.typ).elementType)
      case ArrayLen(a) =>
        assert(a.typ.isInstanceOf[TStreamable])
      case x@ArrayRange(a, b, c) =>
        assert(a.typ.isOfType(TInt32()))
        assert(b.typ.isOfType(TInt32()))
        assert(c.typ.isOfType(TInt32()))
      case x@StreamRange(a, b, c) =>
        assert(a.typ.isOfType(TInt32()))
        assert(b.typ.isOfType(TInt32()))
        assert(c.typ.isOfType(TInt32()))
      case x@MakeNDArray(data, shape, rowMajor) =>
        assert(data.typ.isInstanceOf[TArray])
        assert(shape.typ.asInstanceOf[TTuple].types.forall(t => t.isInstanceOf[TInt64]))
        assert(rowMajor.typ.isOfType(TBoolean()))
      case x@NDArrayShape(nd) =>
        assert(nd.typ.isInstanceOf[TNDArray])
      case x@NDArrayReshape(nd, shape) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(shape.typ.asInstanceOf[TTuple].types.forall(t => t.isInstanceOf[TInt64]))
      case x@NDArrayRef(nd, idxs) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(nd.typ.asInstanceOf[TNDArray].nDims == idxs.length)
        assert(idxs.forall(_.typ.isOfType(TInt64())))
      case x@NDArraySlice(nd, slices) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        val childTyp =nd.typ.asInstanceOf[TNDArray]
        val slicesTuple = slices.typ.asInstanceOf[TTuple]
        assert(slicesTuple.size == childTyp.nDims)
        assert(slicesTuple.types.forall { t =>
          t == TTuple(TInt64(), TInt64(), TInt64()) || t == TInt64()
        })
      case x@NDArrayMap(_, _, body) =>
        assert(x.elementTyp isOfType body.typ)
      case x@NDArrayMap2(l, r, _, _, body) =>
        val lTyp = coerce[TNDArray](l.typ)
        val rTyp = coerce[TNDArray](r.typ)
        assert(lTyp.nDims == rTyp.nDims)
        assert(x.elementTyp == body.typ)
      case x@NDArrayReindex(nd, indexExpr) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        val nInputDims = coerce[TNDArray](nd.typ).nDims
        val nOutputDims = indexExpr.length
        assert(nInputDims <= nOutputDims)
        assert(indexExpr.forall(i => i < nOutputDims))
        assert((0 until nOutputDims).forall(i => indexExpr.contains(i)))
      case x@NDArrayAgg(nd, axes) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        val nInputDims = coerce[TNDArray](nd.typ).nDims
        assert(axes.length <= nInputDims)
        assert(axes.forall(i => i < nInputDims))
        assert(axes.distinct.length == axes.length)
      case x@NDArrayWrite(nd, path) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(path.typ.isInstanceOf[TString])
      case x@NDArrayMatMul(l, r) =>
        assert(l.typ.isInstanceOf[TNDArray])
        assert(r.typ.isInstanceOf[TNDArray])
        val lType = l.typ.asInstanceOf[TNDArray]
        val rType = r.typ.asInstanceOf[TNDArray]
        assert(lType.elementType == rType.elementType)
        assert(lType.nDims > 0)
        assert(rType.nDims > 0)
        assert(lType.nDims == 1 || rType.nDims == 1 || lType.nDims == rType.nDims)
      case x@ArraySort(a, l, r, compare) =>
        assert(a.typ.isInstanceOf[TStreamable])
        assert(compare.typ.isOfType(TBoolean()))
      case x@ToSet(a) =>
        assert(a.typ.isInstanceOf[TIterable])
      case x@ToDict(a) =>
        assert(a.typ.isInstanceOf[TIterable])
        assert(coerce[TBaseStruct](coerce[TIterable](a.typ).elementType).size == 2)
      case x@ToArray(a) =>
        assert(a.typ.isInstanceOf[TIterable])
      case x@ToStream(a) =>
        assert(a.typ.isInstanceOf[TIterable])
      case x@LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val elt = -coerce[TIterable](orderedCollection.typ).elementType
        assert(-elem.typ == (if (onKey) -coerce[TStruct](elt).types(0) else elt))
      case x@GroupByKey(collection) =>
        val telt = coerce[TBaseStruct](coerce[TStreamable](collection.typ).elementType)
        val td = coerce[TDict](x.typ)
        assert(td.keyType == telt.types(0))
        assert(td.valueType == TArray(telt.types(1)))
      case x@ArrayMap(a, name, body) =>
        assert(a.typ.isInstanceOf[TStreamable])
        assert(x.elementTyp == body.typ)
      case x@ArrayFilter(a, name, cond) =>
        assert(a.typ.isInstanceOf[TStreamable])
        assert(cond.typ.isOfType(TBoolean()))
      case x@ArrayFlatMap(a, name, body) =>
        assert(a.typ.isInstanceOf[TStreamable])
        assert(body.typ.isInstanceOf[TArray])
      case x@ArrayFold(a, zero, accumName, valueName, body) =>
        assert(a.typ.isInstanceOf[TStreamable])
        assert(body.typ == zero.typ)
        assert(x.typ == zero.typ)
      case x@ArrayScan(a, zero, accumName, valueName, body) =>
        assert(a.typ.isInstanceOf[TStreamable])
        assert(body.typ == zero.typ)
        assert(coerce[TStreamable](x.typ).elementType == zero.typ)
      case x@ArrayLeftJoinDistinct(left, right, l, r, compare, join) =>
        val ltyp = coerce[TStreamable](left.typ)
        val rtyp = coerce[TStreamable](right.typ)
        assert(compare.typ.isOfType(TInt32()))
        assert(coerce[TStreamable](x.typ).elementType == join.typ)
      case x@ArrayFor(a, valueName, body) =>
        assert(a.typ.isInstanceOf[TStreamable])
        assert(body.typ == TVoid)
      case x@ArrayAgg(a, name, query) =>
        assert(a.typ.isInstanceOf[TStreamable])
        assert(env.agg.isEmpty)
      case x@ArrayAggScan(a, name, query) =>
        assert(a.typ.isInstanceOf[TStreamable])
        assert(env.scan.isEmpty)
      case x@AggFilter(cond, aggIR, _) =>
        assert(cond.typ isOfType TBoolean())
        assert(x.typ == aggIR.typ)
      case x@AggExplode(array, name, aggBody, _) =>
        assert(array.typ.isInstanceOf[TStreamable])
        assert(x.typ == aggBody.typ)
      case x@AggGroupBy(key, aggIR, _) =>
        assert(x.typ == TDict(key.typ, aggIR.typ))
      case x@AggArrayPerElement(a, _, _, aggBody, knownLength, _) =>
        assert(x.typ == TArray(aggBody.typ))
        assert(knownLength.forall(_.typ == TInt32()))
      case x@InitOp(i, args, aggSig) =>
        assert(Some(args.map(_.typ)) == aggSig.initOpArgs)
        assert(i.typ.isInstanceOf[TInt32])
      case x@SeqOp(i, args, aggSig) =>
        assert(args.map(_.typ) == aggSig.seqOpArgs)
        assert(i.typ.isInstanceOf[TInt32])
      case x@InitOp2(_, args, aggSig) =>
        assert(args.map(_.typ) == aggSig.initOpArgs)
      case x@SeqOp2(_, args, aggSig) =>
        assert(args.map(_.typ) == aggSig.seqOpArgs)
      case _: CombOp2 =>
      case _: ResultOp2 =>
      case _: SerializeAggs =>
      case _: DeserializeAggs =>
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
      case x@MakeTuple(fields) =>
        val indices = fields.map(_._1)
        assert(indices.areDistinct())
        assert(indices.isSorted)
        assert(x.typ == TTuple(fields.map { case (idx, f) => TupleField(idx, f.typ)}.toFastIndexedSeq))
      case x@GetTupleElement(o, idx) =>
        val t = coerce[TTuple](o.typ)
        val fd = t.fields(t.fieldIndex(idx))
        assert(x.typ == -fd.typ)
      case In(i, typ) =>
        assert(typ != null)
      case Die(msg, typ) =>
        assert(msg.typ isOfType TString())
      case x@ApplyIR(fn, args) =>
      case x: AbstractApplyNode[_] =>
        assert(x.implementation.unify(x.args.map(_.typ) :+ x.returnType))
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
      case RelationalLet(_, _, _) =>
      case TableWrite(_, _) =>
      case TableMultiWrite(_, _) =>
      case TableCount(_) =>
      case TableGetGlobals(_) =>
      case TableCollect(_) =>
      case TableToValueApply(_, _) =>
      case MatrixToValueApply(_, _) =>
      case BlockMatrixToValueApply(_, _) =>
      case BlockMatrixWrite(_, _) =>
      case BlockMatrixMultiWrite(_, _) =>
      case CollectDistributedArray(ctxs, globals, cname, gname, body) =>
        assert(ctxs.typ.isInstanceOf[TArray])
      case x@ReadPartition(path, _, rowType) =>
        assert(path.typ == TString())
        assert(x.typ == TStream(rowType))
    }
  }
}
