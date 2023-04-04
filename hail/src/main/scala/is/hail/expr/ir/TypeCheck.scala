package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.streams.StreamUtils
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils.StackSafe._
import is.hail.utils._

import scala.reflect.ClassTag

object TypeCheck {
  def apply(ctx: ExecuteContext, ir: BaseIR): Unit = {
    try {
      check(ctx, ir, BindingEnv.empty).run()
    } catch {
      case e: Throwable => fatal(s"Error while typechecking IR:\n${ Pretty(ctx, ir) }", e)
    }
  }

  def apply(ctx: ExecuteContext, ir: IR, env: BindingEnv[Type]): Unit = {
    try {
      check(ctx, ir, env).run()
    } catch {
      case e: Throwable => fatal(s"Error while typechecking IR:\n${ Pretty(ctx, ir) }", e)
    }
  }

  def check(ctx: ExecuteContext, ir: BaseIR, env: BindingEnv[Type]): StackFrame[Unit] = {
    for {
      _ <- ir.children
        .iterator
        .zipWithIndex
        .foreachRecur { case (child, i) =>
          for {
            _ <- call(check(ctx, child, ChildBindings(ir, i, env)))
          } yield {
            if (child.typ == TVoid) {
              checkVoidTypedChild(ctx, ir, i, env)
            } else ()
          }
        }
    } yield checkSingleNode(ctx, ir, env)
  }

  private def checkVoidTypedChild(ctx: ExecuteContext, ir: BaseIR, i: Int, env: BindingEnv[Type]): Unit = ir match {
    case _: Let if i == 1 =>
    case _: StreamFor if i == 1 =>
    case _: RunAggScan if (i == 1 || i == 2) =>
    case _: StreamBufferedAggregate if (i == 1 || i == 3) =>
    case _: RunAgg if i == 0 =>
    case _: SeqOp => // let seqop checking below catch bad void arguments
    case _: InitOp => // let initop checking below catch bad void arguments
    case _: If if i != 0 =>
    case _: RelationalLet if i == 1 =>
    case _: Begin =>
    case _: WriteMetadata =>
    case _ =>
      throw new RuntimeException(s"unexpected void-typed IR at child $i of ${ ir.getClass.getSimpleName }" +
        s"\n  IR: ${ Pretty(ctx, ir) }")
  }

  private def checkSingleNode(ctx: ExecuteContext, ir: BaseIR, env: BindingEnv[Type]): Unit = {
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
      case EncodedLiteral(_, _) =>
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
        env.eval.lookupOption(name) match {
          case Some(expected) =>
            assert(x.typ == expected,
              s"type mismatch:\n  name: $name\n  actual: ${x.typ.parsableString()}\n  expect: ${expected.parsableString()}")
          case None =>
            throw new NoSuchElementException(s"Ref with name ${name} could not be resolved in env ${env}")
        }

      case RelationalRef(name, t) =>
        env.relational.lookupOption(name) match {
          case Some(t2) =>
            if (t != t2)
              throw new RuntimeException(s"RelationalRef type mismatch:\n  node=${t}\n   env=${t2}")
          case None =>
            throw new RuntimeException(s"RelationalRef not found in env: $name")
        }
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
        assert(op.t1 == l.typ)
        assert(op.t2 == r.typ)
        op match {
          case _: Compare => assert(x.typ == TInt32)
          case _ => assert(x.typ == TBoolean)
        }
      case x@MakeArray(args, typ) =>
        assert(typ != null)
        args.map(_.typ).zipWithIndex.foreach { case (x, i) => assert(x == typ.elementType,
          s"at position $i type mismatch: ${ typ.parsableString() } ${ x.parsableString() }")
        }
      case x@MakeStream(args, typ, _) =>
        assert(typ != null)
        assert(typ.elementType.isRealizable)

        args.map(_.typ).zipWithIndex.foreach { case (x, i) => assert(x == typ.elementType,
          s"at position $i type mismatch: ${ typ.elementType.parsableString() } ${ x.parsableString() }")
        }
      case x@ArrayRef(a, i, _) =>
        assert(i.typ == TInt32)
        assert(x.typ == tcoerce[TArray](a.typ).elementType)
      case x@ArraySlice(a, start, stop, step, _) =>
        assert(start.typ == TInt32)
        stop.foreach(ir => assert(ir.typ == TInt32))
        assert(step.typ == TInt32)
        assert(x.typ == tcoerce[TArray](a.typ))
      case ArrayLen(a) =>
        assert(a.typ.isInstanceOf[TArray])
      case ArrayMaximalIndependentSet(edges, tieBreaker) =>
        assert(edges.typ.isInstanceOf[TArray])
        val edgeType = tcoerce[TArray](edges.typ).elementType
        assert(edgeType.isInstanceOf[TBaseStruct])
        val Array(leftType, rightType) = edgeType.asInstanceOf[TBaseStruct].types
        assert(leftType == rightType)
        tieBreaker.foreach { case (_, _, tb) => assert(tb.typ == TFloat64) }
      case StreamIota(start, step, _) =>
        assert(start.typ == TInt32)
        assert(step.typ == TInt32)
      case x@StreamRange(a, b, c, _, _) =>
        assert(a.typ == TInt32)
        assert(b.typ == TInt32)
        assert(c.typ == TInt32)
      case SeqSample(totalRange, numToSample, rngState, _) =>
        assert(totalRange.typ == TInt32)
        assert(numToSample.typ == TInt32)
        assert(rngState.typ == TRNGState)
      case StreamDistribute(child, pivots, path, _, _) =>
        assert(path.typ == TString)
        assert(child.typ.isInstanceOf[TStream])
        assert(pivots.typ.isInstanceOf[TArray])
        assert(pivots.typ.asInstanceOf[TArray].elementType.isInstanceOf[TStruct])
      case x@ArrayZeros(length) =>
        assert(length.typ == TInt32)
      case x@MakeNDArray(data, shape, rowMajor, _) =>
        assert(data.typ.isInstanceOf[TArray] || data.typ.isInstanceOf[TStream])
        assert(shape.typ.asInstanceOf[TTuple].types.forall(t => t == TInt64))
        assert(rowMajor.typ == TBoolean)
      case x@NDArrayShape(nd) =>
        assert(nd.typ.isInstanceOf[TNDArray])
      case x@NDArrayReshape(nd, shape, _) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(shape.typ.asInstanceOf[TTuple].types.forall(t => t == TInt64))
      case x@NDArrayConcat(nds, axis) =>
        assert(tcoerce[TArray](nds.typ).elementType.isInstanceOf[TNDArray])
        assert(axis < x.typ.nDims)
      case x@NDArrayRef(nd, idxs, _) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(nd.typ.asInstanceOf[TNDArray].nDims == idxs.length)
        assert(idxs.forall(_.typ == TInt64))
      case x@NDArraySlice(nd, slices) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        val childTyp = nd.typ.asInstanceOf[TNDArray]
        val slicesTuple = slices.typ.asInstanceOf[TTuple]
        assert(slicesTuple.size == childTyp.nDims)
        assert(slicesTuple.types.forall { t =>
          (t == TTuple(TInt64, TInt64, TInt64)) || (t == TInt64)
        })
      case NDArrayFilter(nd, filters) =>
        val ndtyp = tcoerce[TNDArray](nd.typ)
        assert(ndtyp.nDims == filters.length)
        assert(filters.forall(f => tcoerce[TArray](f.typ).elementType == TInt64))
      case x@NDArrayMap(_, _, body) =>
        assert(x.elementTyp == body.typ)
      case x@NDArrayMap2(l, r, _, _, body, _) =>
        val lTyp = tcoerce[TNDArray](l.typ)
        val rTyp = tcoerce[TNDArray](r.typ)
        assert(lTyp.nDims == rTyp.nDims)
        assert(x.elementTyp == body.typ)
      case x@NDArrayReindex(nd, indexExpr) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        val nInputDims = tcoerce[TNDArray](nd.typ).nDims
        val nOutputDims = indexExpr.length
        assert(nInputDims <= nOutputDims)
        assert(indexExpr.forall(i => i < nOutputDims))
        assert((0 until nOutputDims).forall(i => indexExpr.contains(i)))
      case x@NDArrayAgg(nd, axes) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        val nInputDims = tcoerce[TNDArray](nd.typ).nDims
        assert(axes.length <= nInputDims)
        assert(axes.forall(i => i < nInputDims))
        assert(axes.distinct.length == axes.length)
      case x@NDArrayWrite(nd, path) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(path.typ == TString)
      case x@NDArrayMatMul(l, r, _) =>
        assert(l.typ.isInstanceOf[TNDArray])
        assert(r.typ.isInstanceOf[TNDArray])
        val lType = l.typ.asInstanceOf[TNDArray]
        val rType = r.typ.asInstanceOf[TNDArray]
        assert(lType.elementType == rType.elementType, "element type did not match")
        assert(lType.nDims > 0)
        assert(rType.nDims > 0)
        assert(lType.nDims == 1 || rType.nDims == 1 || lType.nDims == rType.nDims)
      case x@NDArrayQR(nd, mode, _) =>
        val ndType = nd.typ.asInstanceOf[TNDArray]
        assert(ndType.elementType == TFloat64)
        assert(ndType.nDims == 2)
      case x@NDArraySVD(nd, _, _, _) =>
        val ndType = nd.typ.asInstanceOf[TNDArray]
        assert(ndType.elementType == TFloat64)
        assert(ndType.nDims == 2)
      case x@NDArrayInv(nd, _) =>
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
        assert(tcoerce[TBaseStruct](tcoerce[TStream](a.typ).elementType).size == 2)
      case x@ToArray(a) =>
        assert(a.typ.isInstanceOf[TStream])
      case x@CastToArray(a) =>
        assert(a.typ.isInstanceOf[TContainer])
      case x@ToStream(a, _) =>
        assert(a.typ.isInstanceOf[TContainer])
      case x@LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val elt = tcoerce[TIterable](orderedCollection.typ).elementType
        assert(elem.typ == (if (onKey) elt match {
          case t: TBaseStruct => t.types(0)
          case t: TInterval => t.pointType
        } else elt))
      case x@GroupByKey(collection) =>
        val telt = tcoerce[TBaseStruct](tcoerce[TStream](collection.typ).elementType)
        val td = tcoerce[TDict](x.typ)
        assert(td.keyType == telt.types(0))
        assert(td.valueType == TArray(telt.types(1)))
      case x@RNGStateLiteral() =>
        assert(x.typ == TRNGState)
      case RNGSplit(state, dynBitstring) =>
        assert(state.typ == TRNGState)
        def isValid: Type => Boolean = {
          case tuple: TTuple => tuple.types.forall(_ == TInt64)
          case t => t == TInt64
        }
        assert(isValid(dynBitstring.typ))
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
        val ts = tcoerce[TStream](x.typ)
        assert(a.typ.isInstanceOf[TStream])
        assert(ts.elementType == a.typ)
        assert(size.typ == TInt32)
      case x@StreamGroupByKey(a, key, _) =>
        val ts = tcoerce[TStream](x.typ)
        assert(ts.elementType == a.typ)
        val structType = tcoerce[TStruct](tcoerce[TStream](a.typ).elementType)
        assert(key.forall(structType.hasField))
      case x@StreamMap(a, name, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.elementTyp == body.typ)
      case x@StreamZip(as, names, body, _, _) =>
        assert(as.length == names.length)
        assert(x.typ.elementType == body.typ)
        assert(as.forall(_.typ.isInstanceOf[TStream]))
      case x@StreamZipJoin(as, key, curKey, curVals, joinF) =>
        val streamType = tcoerce[TStream](as.head.typ)
        assert(as.forall(_.typ == streamType))
        val eltType = tcoerce[TStruct](streamType.elementType)
        assert(key.forall(eltType.hasField))
        assert(x.typ.elementType == joinF.typ)
      case x@StreamMultiMerge(as, key) =>
        val streamType = tcoerce[TStream](as.head.typ)
        assert(as.forall(_.typ == streamType))
        val eltType = tcoerce[TStruct](streamType.elementType)
        assert(x.typ.elementType == eltType)
        assert(key.forall(eltType.hasField))
      case x@StreamFilter(a, name, cond) =>
        assert(a.typ.asInstanceOf[TStream].elementType.isRealizable)
        assert(cond.typ == TBoolean)
        assert(x.typ == a.typ)
      case x@StreamTakeWhile(a, name, cond) =>
        assert(a.typ.asInstanceOf[TStream].elementType.isRealizable)
        assert(cond.typ == TBoolean)
        assert(x.typ == a.typ)
      case x@StreamDropWhile(a, name, cond) =>
        assert(a.typ.asInstanceOf[TStream].elementType.isRealizable)
        assert(cond.typ == TBoolean)
        assert(x.typ == a.typ)
      case x@StreamFlatMap(a, name, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(body.typ.isInstanceOf[TStream])
      case x@StreamFold(a, zero, accumName, valueName, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(a.typ.asInstanceOf[TStream].elementType.isRealizable, Pretty(ctx, x))
        assert(body.typ == zero.typ)
        assert(x.typ == zero.typ)
      case x@StreamFold2(a, accum, valueName, seq, res) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.typ == res.typ)
        assert(accum.zip(seq).forall { case ((_, z), s) => s.typ == z.typ })
      case x@StreamScan(a, zero, accumName, valueName, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(body.typ == zero.typ)
        assert(tcoerce[TStream](x.typ).elementType == zero.typ)
        assert(zero.typ.isRealizable)
      case x@StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType) =>
        val lEltTyp = tcoerce[TStruct](tcoerce[TStream](left.typ).elementType)
        val rEltTyp = tcoerce[TStruct](tcoerce[TStream](right.typ).elementType)
        assert(tcoerce[TStream](x.typ).elementType == join.typ)
        assert(lKey.forall(lEltTyp.hasField))
        assert(rKey.forall(rEltTyp.hasField))
        if (x.isIntervalJoin) {
          val lKeyTyp = lEltTyp.fieldType(lKey(0))
          val rKeyTyp = rEltTyp.fieldType(rKey(0)).asInstanceOf[TInterval]
          assert(lKeyTyp == rKeyTyp.pointType)
          assert((joinType == "left") || (joinType == "inner"))
        } else {
          assert((lKey, rKey).zipped.forall { case (lk, rk) =>
            lEltTyp.fieldType(lk) == rEltTyp.fieldType(rk)
          })
        }
      case x@StreamFor(a, valueName, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(body.typ == TVoid)
      case x@StreamAgg(a, name, query) =>
        assert(a.typ.isInstanceOf[TStream])
      case x@StreamAggScan(a, name, query) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.typ.asInstanceOf[TStream].elementType == query.typ)
      case x@StreamBufferedAggregate(streamChild, initAggs, newKey, seqOps, _, _, _) =>
        assert(streamChild.typ.isInstanceOf[TStream])
        assert(initAggs.typ == TVoid)
        assert(seqOps.typ == TVoid)
        assert(newKey.typ.isInstanceOf[TStruct])
        assert(x.typ.isInstanceOf[TStream])
      case x@StreamLocalLDPrune(streamChild, r2Threshold, windowSize, maxQueueSize, nSamples) =>
        assert(streamChild.typ.isInstanceOf[TStream])
        assert(r2Threshold.typ == TFloat64)
        assert(windowSize.typ == TInt32)
        assert(maxQueueSize.typ == TInt32)
        assert(nSamples.typ == TInt32)
        val eltType = streamChild.typ.asInstanceOf[TStream].elementType
        assert(eltType.isInstanceOf[TStruct])
        val structType = eltType.asInstanceOf[TStruct]
        assert(structType.fieldType("locus").isInstanceOf[TLocus])
        val allelesType = structType.fieldType("alleles")
        assert(allelesType.isInstanceOf[TArray])
        assert(allelesType.asInstanceOf[TArray].elementType == TString)
        val gtType = structType.fieldType("genotypes")
        assert(gtType.isInstanceOf[TArray])
        assert(gtType.asInstanceOf[TArray].elementType == TCall)
        assert(x.typ.isInstanceOf[TStream])
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
        assert(args.map(_.typ) == aggSig.initOpTypes, s"${args.map(_.typ)} !=  ${aggSig.initOpTypes}")
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
      case x@AggFold(zero, seqOp, combOp, elementName, accumName, _) =>
        assert(zero.typ == seqOp.typ)
        assert(zero.typ == combOp.typ)
      case x@MakeStruct(fields) =>
        assert(x.typ == TStruct(fields.map { case (name, a) =>
          (name, a.typ)
        }: _*))
      case x@SelectFields(old, fields) =>
        assert {
          val oldfields = tcoerce[TStruct](old.typ).fieldNames.toSet
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
        val t = tcoerce[TStruct](o.typ)
        assert(t.index(name).nonEmpty, s"$name not in $t")
        assert(x.typ == t.field(name).typ)
      case x@MakeTuple(fields) =>
        val indices = fields.map(_._1)
        assert(indices.areDistinct())
        assert(indices.isSorted)
        assert(x.typ == TTuple(fields.map { case (idx, f) => TupleField(idx, f.typ)}.toFastIndexedSeq))
      case x@GetTupleElement(o, idx) =>
        val t = tcoerce[TTuple](o.typ)
        val fd = t.fields(t.fieldIndex(idx))
        assert(x.typ == fd.typ)
      case In(i, typ) =>
        assert(typ != null)
        typ.virtualType match {
          case stream: TStream => assert(stream.elementType.isRealizable)
          case _ =>
        }
      case Die(msg, typ, _) =>
        assert(msg.typ == TString)
      case Trap(child) =>
      case ConsoleLog(msg, _) => assert(msg.typ == TString)
      case x@ApplyIR(fn, typeArgs, args, _) =>
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
      case BlockMatrixWrite(_, writer) => writer.loweredTyp
      case BlockMatrixMultiWrite(_, _) =>
      case ValueToBlockMatrix(child, _, _) =>
        assert(child.typ.isInstanceOf[TArray] || child.typ.isInstanceOf[TNDArray] ||  child.typ == TFloat64)
      case CollectDistributedArray(ctxs, globals, cname, gname, body, dynamicID, _, _) =>
        assert(ctxs.typ.isInstanceOf[TStream])
        assert(dynamicID.typ == TString)
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
      case WriteValue(_, path, _, stagingFile) =>
        assert(path.typ == TString)
        assert(stagingFile.forall(_.typ == TString))
      case LiftMeOut(_) =>
      case Consume(_) =>
      case TableMapRows(child, newRow) =>
        val newFieldSet = newRow.typ.asInstanceOf[TStruct].fieldNames.toSet
        assert(child.typ.key.forall(newFieldSet.contains))
      case TableMapPartitions(child, globalName, partitionStreamName, body, requestedKey, allowedOverlap) =>
        assert(StreamUtils.isIterationLinear(body, partitionStreamName), "must iterate over the partition exactly once")
        val newRowType = body.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
        child.typ.key.foreach { k => if (!newRowType.hasField(k)) throw new RuntimeException(s"prev key: ${child.typ.key}, new row: ${newRowType}")}

      case MatrixUnionCols(left, right, joinType) =>
        assert(left.typ.rowKeyStruct == right.typ.rowKeyStruct, s"${left.typ.rowKeyStruct} != ${right.typ.rowKeyStruct}")
        assert(left.typ.colType == right.typ.colType, s"${left.typ.colType} != ${right.typ.colType}")
        assert(left.typ.entryType == right.typ.entryType, s"${left.typ.entryType} != ${right.typ.entryType}")

      case _: TableIR =>
      case _: MatrixIR =>
      case _: BlockMatrixIR =>
    }
  }

  def coerce[A <: Type](argname: String, typ: Type)(implicit tag: ClassTag[A]): A =
    if (tag.runtimeClass.isInstance(typ)) typ.asInstanceOf[A]
    else throw new IllegalArgumentException(
      s"""'$argname': Type mismatch.
         |  Expected: ${tag.runtimeClass.getName}
         |    Actual: ${typ.getClass.getName}""".stripMargin
    )

}
