package is.hail.expr.ir

import is.hail.types.physical.PStream
import is.hail.types.virtual._
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

  private def check(tir: TableIR): Unit = {
    tir match {
      case TableMapRows(child, newRow) =>
        val newFieldSet = newRow.typ.asInstanceOf[TStruct].fieldNames.toSet
        assert(child.typ.key.forall(newFieldSet.contains))
      case _ =>
    }

    tir.children
      .iterator
      .zipWithIndex
      .foreach {
        case (child: IR, i) => check(child, NewBindings(tir, i))
        case (tir: TableIR, _) => check(tir)
        case (mir: MatrixIR, _) => check(mir)
        case (bmir: BlockMatrixIR, _) => check(bmir)
      }
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
      case UUID4(_) =>
      case Literal(_, _) =>
      case Void() =>
      case Cast(v, typ) => if (!Casts.valid(v.typ, typ))
        throw new RuntimeException(s"invalid cast:\n  " +
          s"child type: ${ v.typ.parsableString() }\n  " +
          s"cast type:  ${ typ.parsableString() }")
      case CastRename(v, typ) =>
        if (!v.typ.canCastTo(typ))
          throw new RuntimeException(s"invalid cast:\n  " +
            s"child type: ${ v.typ.parsableString() }\n  " +
            s"cast type:  ${ typ.parsableString() }")
      case NA(t) =>
        assert(t != null)
      case IsNA(v) =>
      case Coalesce(values) =>
        assert(values.tail.forall(_.typ == values.head.typ))
      case x@If(cond, cnsq, altr) =>
        assert(cond.typ == TBoolean)
        assert(x.typ == cnsq.typ && x.typ == altr.typ)
        x.typ match {
          case tstream: TStream => assert(tstream.elementType.isRealizable)
          case _ =>
        }
      case x@Let(_, _, body) =>
        assert(x.typ == body.typ)
      case x@AggLet(_, _, body, _) =>
        assert(x.typ == body.typ)
      case x@Ref(name, _) =>
        val expected = env.eval.lookup(name)
        assert(x.typ == expected, s"type mismatch:\n  name: $name\n  actual: ${ x.typ.parsableString() }\n  expect: ${ expected.parsableString() }")
      case RelationalRef(_, _) =>
      case x@TailLoop(name, _, body) =>
        assert(x.typ == body.typ)
        def recurInTail(node: IR, tailPosition: Boolean): Boolean = node match {
          case x: Recur =>
            x.name != name || tailPosition
          case _ =>
            node.children.zipWithIndex
              .forall {
                case (c: IR, i) => recurInTail(c, tailPosition && InTailPosition(node, i))
                case _ => true
              }
        }
        assert(recurInTail(body, tailPosition = true))
      case x@Recur(name, args, typ) =>
        val TTuple(IndexedSeq(TupleField(_, argTypes), TupleField(_, rt))) = env.eval.lookup(name)
        assert(argTypes.asInstanceOf[TTuple].types.zip(args).forall { case (t, ir) => t == ir.typ } )
        assert(typ == rt)
      case x@ApplyBinaryPrimOp(op, l, r) =>
        assert(x.typ == BinaryOp.getReturnType(op, l.typ, r.typ))
      case x@ApplyUnaryPrimOp(op, v) =>
        assert(x.typ == UnaryOp.getReturnType(op, v.typ))
      case x@ApplyComparisonOp(op, l, r) =>
        assert(op.t1.fundamentalType == l.typ.fundamentalType)
        assert(op.t2.fundamentalType == r.typ.fundamentalType)
        op match {
          case _: Compare => assert(x.typ == TInt32)
          case _ => assert(x.typ == TBoolean)
        }
      case x@MakeArray(args, typ) =>
        assert(typ != null)
        args.map(_.typ).zipWithIndex.foreach { case (x, i) => assert(x == typ.elementType,
          s"at position $i type mismatch: ${ typ.parsableString() } ${ x.parsableString() }")
        }
      case x@MakeStream(args, typ) =>
        assert(typ != null)
        assert(typ.elementType.isRealizable)

        args.map(_.typ).zipWithIndex.foreach { case (x, i) => assert(x == typ.elementType,
          s"at position $i type mismatch: ${ typ.elementType.parsableString() } ${ x.parsableString() }")
        }
      case x@ArrayRef(a, i, s) =>
        assert(i.typ == TInt32)
        assert(s.typ == TString)
        assert(x.typ == coerce[TArray](a.typ).elementType)
      case ArrayLen(a) =>
        assert(a.typ.isInstanceOf[TArray])
      case x@StreamRange(a, b, c) =>
        assert(a.typ == TInt32)
        assert(b.typ == TInt32)
        assert(c.typ == TInt32)
      case x@ArrayZeros(length) =>
        assert(length.typ == TInt32)
      case x@MakeNDArray(data, shape, rowMajor) =>
        assert(data.typ.isInstanceOf[TArray])
        assert(shape.typ.asInstanceOf[TTuple].types.forall(t => t == TInt64))
        assert(rowMajor.typ == TBoolean)
      case x@NDArrayShape(nd) =>
        assert(nd.typ.isInstanceOf[TNDArray])
      case x@NDArrayReshape(nd, shape) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(shape.typ.asInstanceOf[TTuple].types.forall(t => t == TInt64))
      case x@NDArrayConcat(nds, axis) =>
        assert(coerce[TArray](nds.typ).elementType.isInstanceOf[TNDArray])
        assert(axis < x.typ.nDims)
      case x@NDArrayRef(nd, idxs) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(nd.typ.asInstanceOf[TNDArray].nDims == idxs.length)
        assert(idxs.forall(_.typ == TInt64))
      case x@NDArraySlice(nd, slices) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        val childTyp =nd.typ.asInstanceOf[TNDArray]
        val slicesTuple = slices.typ.asInstanceOf[TTuple]
        assert(slicesTuple.size == childTyp.nDims)
        assert(slicesTuple.types.forall { t =>
          (t == TTuple(TInt64, TInt64, TInt64)) || (t == TInt64)
        })
      case NDArrayFilter(nd, filters) =>
        val ndtyp = coerce[TNDArray](nd.typ)
        assert(ndtyp.nDims == filters.length)
        assert(filters.forall(f => coerce[TArray](f.typ).elementType == TInt64))
      case x@NDArrayMap(_, _, body) =>
        assert(x.elementTyp == body.typ)
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
        assert(path.typ == TString)
      case x@NDArrayMatMul(l, r) =>
        assert(l.typ.isInstanceOf[TNDArray])
        assert(r.typ.isInstanceOf[TNDArray])
        val lType = l.typ.asInstanceOf[TNDArray]
        val rType = r.typ.asInstanceOf[TNDArray]
        assert(lType.elementType == rType.elementType, "element type did not match")
        assert(lType.nDims > 0)
        assert(rType.nDims > 0)
        assert(lType.nDims == 1 || rType.nDims == 1 || lType.nDims == rType.nDims)
      case x@NDArrayQR(nd, mode) =>
        val ndType = nd.typ.asInstanceOf[TNDArray]
        assert(ndType.elementType == TFloat64)
        assert(ndType.nDims == 2)
      case x@ArraySort(a, l, r, lessThan) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(lessThan.typ == TBoolean)
      case x@ToSet(a) =>
        assert(a.typ.isInstanceOf[TStream])
      case x@ToDict(a) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(coerce[TBaseStruct](coerce[TStream](a.typ).elementType).size == 2)
      case x@ToArray(a) =>
        assert(a.typ.isInstanceOf[TStream])
      case x@CastToArray(a) =>
        assert(a.typ.isInstanceOf[TContainer])
      case x@ToStream(a) =>
        assert(a.typ.isInstanceOf[TContainer])
      case x@LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val elt = coerce[TIterable](orderedCollection.typ).elementType
        assert(elem.typ == (if (onKey) elt match {
          case t: TBaseStruct => t.types(0)
          case t: TInterval => t.pointType
        } else elt))
      case x@GroupByKey(collection) =>
        val telt = coerce[TBaseStruct](coerce[TStream](collection.typ).elementType)
        val td = coerce[TDict](x.typ)
        assert(td.keyType == telt.types(0))
        assert(td.valueType == TArray(telt.types(1)))
      case StreamLen(a) =>
        assert(a.typ.isInstanceOf[TStream])
      case x@StreamTake(a, num) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.typ == a.typ)
        assert(num.typ == TInt32)
      case x@StreamDrop(a, num) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.typ == a.typ)
        assert(num.typ == TInt32)
      case x@StreamGrouped(a, size) =>
        val ts = coerce[TStream](x.typ)
        assert(a.typ.isInstanceOf[TStream])
        assert(ts.elementType == a.typ)
        assert(size.typ == TInt32)
      case x@StreamGroupByKey(a, key) =>
        val ts = coerce[TStream](x.typ)
        assert(ts.elementType == a.typ)
        val structType = coerce[TStruct](coerce[TStream](a.typ).elementType)
        assert(key.forall(structType.hasField))
      case x@StreamMap(a, name, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.elementTyp == body.typ)
      case x@StreamMerge(l, r, key) =>
        assert(l.typ == r.typ)
        assert(x.typ == l.typ)
        val structType = coerce[TStruct](coerce[TStream](l.typ).elementType)
        assert(key.forall(structType.hasField))
      case x@StreamZip(as, names, body, _) =>
        assert(as.length == names.length)
        assert(x.typ.elementType == body.typ)
        assert(as.forall(_.typ.isInstanceOf[TStream]))
      case x@StreamZipJoin(as, key, curKey, curVals, joinF) =>
        val streamType = coerce[TStream](as.head.typ)
        assert(as.forall(_.typ == streamType))
        val eltType = coerce[TStruct](streamType.elementType)
        assert(key.forall(eltType.hasField))
        assert(x.typ.elementType == joinF.typ)
      case x@StreamMultiMerge(as, key) =>
        val streamType = coerce[TStream](as.head.typ)
        assert(as.forall(_.typ == streamType))
        val eltType = coerce[TStruct](streamType.elementType)
        assert(x.typ.elementType == eltType)
        assert(key.forall(eltType.hasField))
      case x@StreamFilter(a, name, cond) =>
        assert(a.typ.asInstanceOf[TStream].elementType.isRealizable)
        assert(cond.typ == TBoolean)
        assert(x.typ == a.typ)
      case x@StreamFlatMap(a, name, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(body.typ.isInstanceOf[TStream])
      case x@StreamFold(a, zero, accumName, valueName, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(a.typ.asInstanceOf[TStream].elementType.isRealizable, Pretty(x))
        assert(body.typ == zero.typ)
        assert(x.typ == zero.typ)
      case x@StreamFold2(a, accum, valueName, seq, res) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.typ == res.typ)
        assert(accum.zip(seq).forall { case ((_, z), s) => s.typ == z.typ })
      case x@StreamScan(a, zero, accumName, valueName, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(body.typ == zero.typ)
        assert(coerce[TStream](x.typ).elementType == zero.typ)
        assert(zero.typ.isRealizable)
      case x@StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType) =>
        val lEltTyp = coerce[TStruct](coerce[TStream](left.typ).elementType)
        val rEltTyp = coerce[TStruct](coerce[TStream](right.typ).elementType)
        assert(coerce[TStream](x.typ).elementType == join.typ)
        assert(lKey.forall(lEltTyp.hasField))
        assert(rKey.forall(rEltTyp.hasField))
        assert((lKey, rKey).zipped.forall { case (lk, rk) =>
          lEltTyp.fieldType(lk) == rEltTyp.fieldType(rk)
        })
      case x@StreamFor(a, valueName, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(body.typ == TVoid)
      case x@StreamAgg(a, name, query) =>
        assert(a.typ.isInstanceOf[TStream])
      case x@StreamAggScan(a, name, query) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.typ.asInstanceOf[TStream].elementType == query.typ)
      case x@RunAgg(body, result, _) =>
        assert(x.typ == result.typ)
        assert(body.typ == TVoid)
      case x@RunAggScan(array, _, init, seqs, result, _) =>
        assert(array.typ.isInstanceOf[TStream])
        assert(init.typ == TVoid)
        assert(seqs.typ == TVoid)
        assert(x.typ.asInstanceOf[TStream].elementType == result.typ)
      case x@AggFilter(cond, aggIR, _) =>
        assert(cond.typ == TBoolean)
        assert(x.typ == aggIR.typ)
      case x@AggExplode(array, name, aggBody, _) =>
        assert(array.typ.isInstanceOf[TStream])
        assert(x.typ == aggBody.typ)
      case x@AggGroupBy(key, aggIR, _) =>
        assert(x.typ == TDict(key.typ, aggIR.typ))
      case x@AggArrayPerElement(a, _, _, aggBody, knownLength, _) =>
        assert(x.typ == TArray(aggBody.typ))
        assert(knownLength.forall(_.typ == TInt32))
      case x@InitOp(_, args, aggSig) =>
        assert(args.map(_.typ) == aggSig.initOpTypes)
      case x@SeqOp(_, args, aggSig) =>
        assert(args.map(_.typ) == aggSig.seqOpTypes)
      case _: CombOp =>
      case _: ResultOp =>
      case AggStateValue(i, sig) =>
      case CombOpValue(i, value, sig) => assert(value.typ == TBinary)
      case InitFromSerializedValue(i, value, sig) => assert(value.typ == TBinary)
      case _: SerializeAggs =>
      case _: DeserializeAggs =>
      case x@Begin(xs) =>
        xs.foreach { x =>
          assert(x.typ == TVoid)
        }
      case x@ApplyAggOp(initOpArgs, seqOpArgs, aggSig) =>
        assert(x.typ == aggSig.returnType)
        assert(initOpArgs.map(_.typ).zip(aggSig.initOpArgs).forall { case (l, r) => l == r })
        assert(seqOpArgs.map(_.typ).zip(aggSig.seqOpArgs).forall { case (l, r) => l == r })
      case x@ApplyScanOp(initOpArgs, seqOpArgs, aggSig) =>
        assert(x.typ == aggSig.returnType)
        assert(initOpArgs.map(_.typ).zip(aggSig.initOpArgs).forall { case (l, r) => l == r })
        assert(seqOpArgs.map(_.typ).zip(aggSig.seqOpArgs).forall { case (l, r) => l == r })
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
          assert(fds.areDistinct())
          assert(fds.toSet.forall(f => newFieldSet.contains(f) || oldFieldNameSet.contains(f)))
        }
      case x@GetField(o, name) =>
        val t = coerce[TStruct](o.typ)
        assert(t.index(name).nonEmpty, s"$name not in $t")
        assert(x.typ == t.field(name).typ)
      case x@MakeTuple(fields) =>
        val indices = fields.map(_._1)
        assert(indices.areDistinct())
        assert(indices.isSorted)
        assert(x.typ == TTuple(fields.map { case (idx, f) => TupleField(idx, f.typ)}.toFastIndexedSeq))
      case x@GetTupleElement(o, idx) =>
        val t = coerce[TTuple](o.typ)
        val fd = t.fields(t.fieldIndex(idx))
        assert(x.typ == fd.typ)
      case In(i, typ) =>
        assert(typ != null)
        typ match {
          case pstream: PStream => assert(pstream.elementType.isRealizable)
          case _ =>
        }
      case Die(msg, typ) =>
        assert(msg.typ == TString)
      case x@ApplyIR(fn, typeArgs, args) =>
      case x: AbstractApplyNode[_] =>
        assert(x.implementation.unify(x.typeArgs, x.args.map(_.typ), x.returnType))
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
      case MatrixCount(_) =>
      case TableGetGlobals(_) =>
      case TableCollect(child) =>
        assert(child.typ.key.isEmpty)
      case TableToValueApply(_, _) =>
      case MatrixToValueApply(_, _) =>
      case BlockMatrixToValueApply(_, _) =>
      case BlockMatrixCollect(_) =>
      case BlockMatrixWrite(_, _) =>
      case BlockMatrixMultiWrite(_, _) =>
      case UnpersistBlockMatrix(_) =>
      case CollectDistributedArray(ctxs, globals, cname, gname, body) =>
        assert(ctxs.typ.isInstanceOf[TStream])
      case x@ReadPartition(context, rowType, reader) =>
        assert(rowType.isRealizable)
        assert(context.typ == reader.contextType)
        assert(x.typ == TStream(rowType))
        assert(PruneDeadFields.isSupertype(rowType, reader.fullRowType))
      case x@WritePartition(value, writeCtx, writer) =>
        assert(value.typ.isInstanceOf[TStream])
        assert(writeCtx.typ == writer.ctxType)
        assert(x.typ == writer.returnType)
      case WriteMetadata(writeAnnotations, writer) =>
        assert(writeAnnotations.typ == writer.annotationType)
      case x@ReadValue(path, spec, requestedType) =>
        assert(path.typ == TString)
        assert(spec.encodedType.decodedPType(requestedType).virtualType == requestedType)
      case x@WriteValue(value, pathPrefix, spec) =>
        assert(pathPrefix.typ == TString)
      case LiftMeOut(_) =>
    }
  }
}
