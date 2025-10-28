package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.Nat
import is.hail.expr.ir.defs._
import is.hail.expr.ir.streams.StreamUtils
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.StackSafe._

import scala.collection.compat._
import scala.reflect.ClassTag

object TypeCheck {
  def apply(ctx: ExecuteContext, ir: BaseIR): Unit =
    apply(ctx, ir, BindingEnv.empty)

  def apply(ctx: ExecuteContext, ir: BaseIR, env: BindingEnv[Type]): Unit =
    ctx.time {
      try
        check(ctx, ir, env).run()
      catch {
        case e: Throwable =>
          fatal(
            s"Error while typechecking IR:\n${Pretty(ctx, ir, preserveNames = true, allowUnboundRefs = true)}",
            e,
          )
      }
    }

  def check(ctx: ExecuteContext, ir: BaseIR, env: BindingEnv[Type]): StackFrame[Unit] = {
    for {
      _ <- ir.forEachChildWithEnvStackSafe(env) { (child, i, childEnv) =>
        for {
          _ <- call(check(ctx, child, childEnv))
        } yield
          if (child.typ == TVoid) {
            checkVoidTypedChild(ctx, ir, i, env)
          } else ()
      }
    } yield checkSingleNode(ctx, ir, env)
  }

  private def checkVoidTypedChild(ctx: ExecuteContext, ir: BaseIR, i: Int, env: BindingEnv[Type])
    : Unit = ir match {
    case l: Block if i == l.bindings.length || l.body.typ == TVoid =>
    case _: StreamFor if i == 1 =>
    case _: RunAggScan if (i == 1 || i == 2) =>
    case _: StreamBufferedAggregate if (i == 1 || i == 3) =>
    case _: RunAgg if i == 0 =>
    case _: SeqOp => // let seqop checking below catch bad void arguments
    case _: InitOp => // let initop checking below catch bad void arguments
    case _: If if i != 0 =>
    case _: RelationalLet if i == 1 =>
    case _: WriteMetadata =>
    case _ =>
      throw new RuntimeException(
        s"unexpected void-typed IR at child $i of ${ir.getClass.getSimpleName}" +
          s"\n  IR: ${Pretty(ctx, ir)}"
      )
  }

  private def checkSingleNode(ctx: ExecuteContext, ir: BaseIR, env: BindingEnv[Type]): Unit = {
    ir match {
      case I32(_) =>
      case I64(_) =>
      case F32(_) =>
      case F64(_) =>
      case True() =>
      case False() =>
      case Str(_) =>
      case UUID4(_) =>
      case Literal(_, _) =>
      case EncodedLiteral(_, _) =>
      case Void() =>
      case Cast(v, typ) => if (!Casts.valid(v.typ, typ))
          throw new RuntimeException(s"invalid cast:\n  " +
            s"child type: ${v.typ.parsableString()}\n  " +
            s"cast type:  ${typ.parsableString()}")
      case CastRename(v, typ) =>
        if (!v.typ.isIsomorphicTo(typ))
          throw new RuntimeException(s"invalid cast:\n  " +
            s"child type: ${v.typ.parsableString()}\n  " +
            s"cast type:  ${typ.parsableString()}")
      case NA(t) =>
        assert(t != null)
      case IsNA(_) =>
      case Coalesce(values) =>
        assert(values.tail.forall(_.typ == values.head.typ))
      case x @ If(cond, cnsq, altr) =>
        assert(cond.typ == TBoolean)
        assert(x.typ == cnsq.typ && x.typ == altr.typ)
        x.typ match {
          case tstream: TStream => assert(tstream.elementType.isRealizable)
          case _ =>
        }
      case Switch(x, default, cases) =>
        assert(x.typ == TInt32)
        assert(cases.forall(_.typ == default.typ))
      case x @ Block(_, body) =>
        assert(x.typ == body.typ)
      case x @ Ref(name, _) =>
        env.eval.lookupOption(name) match {
          case Some(expected) =>
            assert(
              x.typ == expected,
              s"type mismatch:\n  name: $name\n  actual: ${x.typ.parsableString()}\n  expect: ${expected.parsableString()}",
            )
          case None =>
            throw new NoSuchElementException(
              s"Ref with name $name could not be resolved in env $env"
            )
        }

      case RelationalRef(name, t) =>
        env.relational.lookupOption(name) match {
          case Some(t2) =>
            if (t != t2)
              throw new RuntimeException(s"RelationalRef type mismatch:\n  node=$t\n   env=$t2")
          case None =>
            throw new RuntimeException(s"RelationalRef not found in env: $name")
        }
      case x @ TailLoop(name, _, rt, body) =>
        assert(x.typ == rt)
        assert(body.typ == rt)
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
      case Recur(name, args, typ) =>
        val TTuple(IndexedSeq(TupleField(_, argTypes), TupleField(_, rt))) = env.eval.lookup(name)
        assert(argTypes.asInstanceOf[TTuple].types.zip(args).forall { case (t, ir) => t == ir.typ })
        assert(typ == rt)
      case x @ ApplyBinaryPrimOp(op, l, r) =>
        assert(x.typ == BinaryOp.getReturnType(op, l.typ, r.typ))
      case x @ ApplyUnaryPrimOp(op, v) =>
        assert(x.typ == UnaryOp.getReturnType(op, v.typ))
      case x @ ApplyComparisonOp(op, l, r) =>
        ComparisonOp.checkCompatible(l.typ, r.typ)
        op match {
          case Compare => assert(x.typ == TInt32)
          case _ => assert(x.typ == TBoolean)
        }
      case MakeArray(args, typ) =>
        assert(typ != null)
        args.map(_.typ).zipWithIndex.foreach { case (x, i) =>
          assert(
            x == typ.elementType && x.isRealizable,
            s"at position $i type mismatch: ${typ.parsableString()} ${x.parsableString()}",
          )
        }
      case MakeStream(args, typ, _) =>
        assert(typ != null)
        assert(typ.elementType.isRealizable, typ.elementType)

        args.map(_.typ).zipWithIndex.foreach { case (x, i) =>
          assert(
            x == typ.elementType,
            s"at position $i type mismatch: ${typ.elementType.parsableString()} ${x.parsableString()}",
          )
        }
      case x @ ArrayRef(a, i, _) =>
        assert(i.typ == TInt32)
        assert(x.typ == tcoerce[TArray](a.typ).elementType)
      case x @ ArraySlice(a, start, stop, step, _) =>
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
      case StreamRange(a, b, c, _, _) =>
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
      case StreamWhiten(stream, newChunk, prevWindow, _, windowSize, chunkSize, _,
            _) =>
        assert(stream.typ.isInstanceOf[TStream])
        val eltTyp = stream.typ.asInstanceOf[TStream].elementType
        assert(eltTyp.isInstanceOf[TStruct])
        val structTyp = eltTyp.asInstanceOf[TStruct]
        val matTyp = TNDArray(TFloat64, Nat(2))
        assert(structTyp.field(newChunk).typ == matTyp)
        assert(structTyp.field(prevWindow).typ == matTyp)
        assert(windowSize % chunkSize == 0)
      case ArrayZeros(length) =>
        assert(length.typ == TInt32)
      case MakeNDArray(data, shape, rowMajor, _) =>
        assert(data.typ.isInstanceOf[TArray] || data.typ.isInstanceOf[TStream])
        assert(shape.typ.asInstanceOf[TTuple].types.forall(t => t == TInt64))
        assert(rowMajor.typ == TBoolean)
      case NDArrayShape(nd) =>
        assert(nd.typ.isInstanceOf[TNDArray])
      case NDArrayReshape(nd, shape, _) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(shape.typ.asInstanceOf[TTuple].types.forall(t => t == TInt64))
      case x @ NDArrayConcat(nds, axis) =>
        assert(tcoerce[TArray](nds.typ).elementType.isInstanceOf[TNDArray])
        assert(axis < x.typ.nDims)
      case NDArrayRef(nd, idxs, _) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(nd.typ.asInstanceOf[TNDArray].nDims == idxs.length)
        assert(idxs.forall(_.typ == TInt64))
      case NDArraySlice(nd, slices) =>
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
      case x @ NDArrayMap(_, _, body) =>
        assert(x.elementTyp == body.typ)
      case x @ NDArrayMap2(l, r, _, _, body, _) =>
        val lTyp = tcoerce[TNDArray](l.typ)
        val rTyp = tcoerce[TNDArray](r.typ)
        assert(lTyp.nDims == rTyp.nDims)
        assert(x.elementTyp == body.typ)
      case NDArrayReindex(nd, indexExpr) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        val nInputDims = tcoerce[TNDArray](nd.typ).nDims
        val nOutputDims = indexExpr.length
        assert(nInputDims <= nOutputDims)
        assert(indexExpr.forall(i => i < nOutputDims))
        assert((0 until nOutputDims).forall(i => indexExpr.contains(i)))
      case NDArrayAgg(nd, axes) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        val nInputDims = tcoerce[TNDArray](nd.typ).nDims
        assert(axes.length <= nInputDims)
        assert(axes.forall(i => i < nInputDims))
        assert(axes.distinct.length == axes.length)
      case NDArrayWrite(nd, path) =>
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(path.typ == TString)
      case NDArrayMatMul(l, r, _) =>
        assert(l.typ.isInstanceOf[TNDArray])
        assert(r.typ.isInstanceOf[TNDArray])
        val lType = l.typ.asInstanceOf[TNDArray]
        val rType = r.typ.asInstanceOf[TNDArray]
        assert(lType.elementType == rType.elementType, "element type did not match")
        assert(lType.nDims > 0)
        assert(rType.nDims > 0)
        assert(lType.nDims == 1 || rType.nDims == 1 || lType.nDims == rType.nDims)
      case NDArrayQR(nd, _, _) =>
        val ndType = nd.typ.asInstanceOf[TNDArray]
        assert(ndType.elementType == TFloat64)
        assert(ndType.nDims == 2)
      case NDArraySVD(nd, _, _, _) =>
        val ndType = nd.typ.asInstanceOf[TNDArray]
        assert(ndType.elementType == TFloat64)
        assert(ndType.nDims == 2)
      case NDArrayEigh(nd, _, _) =>
        val ndType = nd.typ.asInstanceOf[TNDArray]
        assert(ndType.elementType == TFloat64)
        assert(ndType.nDims == 2)
      case NDArrayInv(nd, _) =>
        val ndType = nd.typ.asInstanceOf[TNDArray]
        assert(ndType.elementType == TFloat64)
        assert(ndType.nDims == 2)
      case ArraySort(a, _, _, lessThan) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(lessThan.typ == TBoolean)
      case ToSet(a) =>
        assert(a.typ.isInstanceOf[TStream], a.typ)
      case ToDict(a) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(tcoerce[TBaseStruct](tcoerce[TStream](a.typ).elementType).size == 2)
      case ToArray(a) =>
        assert(a.typ.isInstanceOf[TStream])
      case CastToArray(a) =>
        assert(a.typ.isInstanceOf[TContainer])
      case ToStream(a, _) =>
        assert(a.typ.isInstanceOf[TContainer])
      case LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val elt = tcoerce[TIterable](orderedCollection.typ).elementType
        assert(elem.typ == (if (onKey) elt match {
                              case t: TBaseStruct => t.types(0)
                              case t: TInterval => t.pointType
                            }
                            else elt))
      case x @ GroupByKey(collection) =>
        val telt = tcoerce[TBaseStruct](tcoerce[TStream](collection.typ).elementType)
        val td = tcoerce[TDict](x.typ)
        assert(td.keyType == telt.types(0))
        assert(td.valueType == TArray(telt.types(1)))
      case x @ RNGStateLiteral() =>
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
      case x @ StreamTake(a, num) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.typ == a.typ)
        assert(num.typ == TInt32)
      case x @ StreamDrop(a, num) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.typ == a.typ)
        assert(num.typ == TInt32)
      case x @ StreamGrouped(a, size) =>
        val ts = tcoerce[TStream](x.typ)
        assert(a.typ.isInstanceOf[TStream])
        assert(ts.elementType == a.typ)
        assert(size.typ == TInt32)
      case x @ StreamGroupByKey(a, key, _) =>
        val ts = tcoerce[TStream](x.typ)
        assert(ts.elementType == a.typ)
        val structType = tcoerce[TStruct](tcoerce[TStream](a.typ).elementType)
        assert(key.forall(structType.hasField))
      case x @ StreamMap(a, _, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.typ.elementType == body.typ)
      case x @ StreamZip(as, names, body, _, _) =>
        assert(as.length == names.length)
        assert(x.typ.elementType == body.typ)
        assert(as.forall(_.typ.isInstanceOf[TStream]))
      case x @ StreamZipJoin(as, key, _, _, joinF) =>
        val streamType = tcoerce[TStream](as.head.typ)
        assert(as.forall(_.typ == streamType))
        val eltType = tcoerce[TStruct](streamType.elementType)
        assert(key.forall(eltType.hasField))
        assert(x.typ.elementType == joinF.typ)
      case x @ StreamZipJoinProducers(contexts, _, makeProducer, key, _, _,
            joinF) =>
        assert(contexts.typ.isInstanceOf[TArray])
        val streamType = tcoerce[TStream](makeProducer.typ)
        val eltType = tcoerce[TStruct](streamType.elementType)
        assert(key.forall(eltType.hasField))
        assert(x.typ.elementType == joinF.typ)
      case x @ StreamMultiMerge(as, key) =>
        val streamType = tcoerce[TStream](as.head.typ)
        assert(as.forall(_.typ == streamType))
        val eltType = tcoerce[TStruct](streamType.elementType)
        assert(x.typ.elementType == eltType)
        assert(key.forall(eltType.hasField))
      case x @ StreamFilter(a, _, cond) =>
        assert(a.typ.asInstanceOf[TStream].elementType.isRealizable)
        assert(cond.typ == TBoolean, cond.typ)
        assert(x.typ == a.typ)
      case x @ StreamTakeWhile(a, _, cond) =>
        assert(a.typ.asInstanceOf[TStream].elementType.isRealizable)
        assert(cond.typ == TBoolean)
        assert(x.typ == a.typ)
      case x @ StreamDropWhile(a, _, cond) =>
        assert(a.typ.asInstanceOf[TStream].elementType.isRealizable)
        assert(cond.typ == TBoolean)
        assert(x.typ == a.typ)
      case StreamFlatMap(a, _, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(body.typ.isInstanceOf[TStream])
      case x @ StreamFold(a, zero, _, _, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(a.typ.asInstanceOf[TStream].elementType.isRealizable, Pretty(ctx, x))
        assert(body.typ == zero.typ)
        assert(x.typ == zero.typ)
      case x @ StreamFold2(a, accum, _, seq, res) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.typ == res.typ)
        assert(accum.zip(seq).forall { case ((_, z), s) => s.typ == z.typ })
      case x @ StreamScan(a, zero, _, _, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(body.typ == zero.typ)
        assert(tcoerce[TStream](x.typ).elementType == zero.typ)
        assert(zero.typ.isRealizable)
      case x @ StreamJoinRightDistinct(left, right, lKey, rKey, _, _, join, joinType) =>
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
          assert(lKey.lazyZip(rKey).forall { case (lk, rk) =>
            lEltTyp.fieldType(lk) == rEltTyp.fieldType(rk)
          })
        }
      case StreamLeftIntervalJoin(left, right, lKeyFieldName, rIntrvlName, _, _, body) =>
        assert(left.typ.isInstanceOf[TStream])
        assert(right.typ.isInstanceOf[TStream])

        val lEltTy =
          TIterable.elementType(left.typ).asInstanceOf[TStruct]

        val rPointTy =
          TIterable.elementType(right.typ)
            .asInstanceOf[TStruct]
            .fieldType(rIntrvlName)
            .asInstanceOf[TInterval]
            .pointType

        assert(lEltTy.fieldType(lKeyFieldName) == rPointTy)
        assert(body.typ.isInstanceOf[TStruct])
      case StreamFor(a, _, body) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(body.typ == TVoid)
      case StreamAgg(a, _, _) =>
        assert(a.typ.isInstanceOf[TStream])
      case x @ StreamAggScan(a, _, query) =>
        assert(a.typ.isInstanceOf[TStream])
        assert(x.typ.asInstanceOf[TStream].elementType == query.typ)
      case x @ StreamBufferedAggregate(streamChild, initAggs, newKey, seqOps, _, _, _) =>
        assert(streamChild.typ.isInstanceOf[TStream])
        assert(initAggs.typ == TVoid)
        assert(seqOps.typ == TVoid)
        assert(newKey.typ.isInstanceOf[TStruct])
        assert(x.typ.isInstanceOf[TStream])
      case x @ StreamLocalLDPrune(streamChild, r2Threshold, windowSize, maxQueueSize, nSamples) =>
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
      case x @ RunAgg(body, result, _) =>
        assert(x.typ == result.typ)
        assert(body.typ == TVoid)
      case x @ RunAggScan(array, _, init, seqs, result, _) =>
        assert(array.typ.isInstanceOf[TStream])
        assert(init.typ == TVoid)
        assert(seqs.typ == TVoid)
        assert(x.typ.asInstanceOf[TStream].elementType == result.typ)
      case x @ AggFilter(cond, aggIR, _) =>
        assert(cond.typ == TBoolean)
        assert(x.typ == aggIR.typ)
      case x @ AggExplode(array, _, aggBody, _) =>
        assert(array.typ.isInstanceOf[TStream])
        assert(x.typ == aggBody.typ)
      case x @ AggGroupBy(key, aggIR, _) =>
        assert(x.typ == TDict(key.typ, aggIR.typ))
      case x @ AggArrayPerElement(_, _, _, aggBody, knownLength, _) =>
        assert(x.typ == TArray(aggBody.typ))
        assert(knownLength.forall(_.typ == TInt32))
      case InitOp(_, args, aggSig) =>
        assert(
          args.map(_.typ) == aggSig.initOpTypes,
          s"${args.map(_.typ)} !=  ${aggSig.initOpTypes}",
        )
      case SeqOp(_, args, aggSig) =>
        assert(args.map(_.typ) == aggSig.seqOpTypes)
      case _: CombOp =>
      case _: ResultOp =>
      case AggStateValue(_, _) =>
      case CombOpValue(_, value, _) => assert(value.typ == TBinary)
      case InitFromSerializedValue(_, value, _) => assert(value.typ == TBinary)
      case _: SerializeAggs =>
      case _: DeserializeAggs =>
      case x @ ApplyAggOp(_, seqOpArgs, op) =>
        assert(x.typ == AggOp.getReturnType(op, seqOpArgs.map(_.typ)))
      case x @ ApplyScanOp(_, seqOpArgs, op) =>
        assert(x.typ == AggOp.getReturnType(op, seqOpArgs.map(_.typ)))
      case AggFold(zero, seqOp, combOp, _, _, _) =>
        assert(zero.typ == seqOp.typ)
        assert(zero.typ == combOp.typ)
      case x @ MakeStruct(fields) =>
        assert(x.typ == TStruct(fields.map { case (name, a) =>
          (name, a.typ)
        }: _*))
      case SelectFields(old, fields) =>
        assert {
          val oldfields = tcoerce[TStruct](old.typ).fieldNames.toSet
          fields.forall(id => oldfields.contains(id))
        }
      case x @ InsertFields(old, fields, fieldOrder) =>
        fieldOrder.foreach { fds =>
          val newFieldSet = fields.map(_._1).toSet
          val oldFieldNames = old.typ.asInstanceOf[TStruct].fieldNames
          val oldFieldNameSet = oldFieldNames.toSet
          assert(fds.length == x.typ.size)
          assert(fds.areDistinct())
          assert(fds.toSet.forall(f => newFieldSet.contains(f) || oldFieldNameSet.contains(f)))
        }
      case x @ GetField(o, name) =>
        val t = tcoerce[TStruct](o.typ)
        assert(t.index(name).nonEmpty, s"$name not in $t")
        assert(x.typ == t.field(name).typ)
      case x @ MakeTuple(fields) =>
        val indices = fields.map(_._1)
        assert(indices.areDistinct())
        assert(indices.isSorted)
        assert(x.typ == TTuple(fields.map { case (idx, f) => TupleField(idx, f.typ) }.toFastSeq))
      case x @ GetTupleElement(o, idx) =>
        val t = tcoerce[TTuple](o.typ)
        val fd = t.fields(t.fieldIndex(idx))
        assert(x.typ == fd.typ)
      case In(_, typ) =>
        assert(typ != null)
        typ.virtualType match {
          case stream: TStream => assert(stream.elementType.isRealizable)
          case _ =>
        }
      case Die(msg, _, _) =>
        assert(msg.typ == TString)
      case Trap(_) =>
      case ConsoleLog(msg, _) => assert(msg.typ == TString)
      case ApplyIR(_, _, _, _, _) =>
      case x: AbstractApplyNode[_] =>
        assert(x.implementation.unify(x.typeArgs, x.args.map(_.typ), x.returnType))
      case MatrixWrite(_, _) =>
      case MatrixMultiWrite(children, _) =>
        val t = children.head.typ
        assert(
          !t.rowType.hasField(MatrixReader.rowUIDFieldName) &&
            !t.colType.hasField(MatrixReader.colUIDFieldName),
          t,
        )
        assert(children.forall(_.typ == t))
      case x @ TableAggregate(_, query) =>
        assert(x.typ == query.typ)
      case x @ MatrixAggregate(_, query) =>
        assert(x.typ == query.typ)
      case RelationalLet(_, _, _) =>
      case TableWrite(_, _) =>
      case TableMultiWrite(children, _) =>
        val t = children.head.typ
        assert(children.forall(_.typ == t))
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
      case CollectDistributedArray(ctxs, _, _, _, _, dynamicID, _, _) =>
        assert(ctxs.typ.isInstanceOf[TStream])
        assert(dynamicID.typ == TString)
      case x @ ReadPartition(context, rowType, reader) =>
        assert(rowType.isRealizable)
        assert(context.typ == reader.contextType)
        assert(x.typ == TStream(rowType))
        assert(PruneDeadFields.isSupertype(rowType, reader.fullRowType))
      case x @ WritePartition(value, writeCtx, writer) =>
        assert(value.typ.isInstanceOf[TStream])
        assert(writeCtx.typ == writer.ctxType)
        assert(x.typ == writer.returnType)
      case WriteMetadata(writeAnnotations, writer) =>
        assert(writeAnnotations.typ == writer.annotationType)
      case ReadValue(path, reader, requestedType) =>
        assert(path.typ == TString)
        reader match {
          case reader: ETypeValueReader =>
            assert(reader.spec.encodedType.decodedPType(requestedType).virtualType == requestedType)
          case _ => // do nothing, we can't in general typecheck an arbitrary value reader
        }
      case WriteValue(_, path, _, stagingFile) =>
        assert(path.typ == TString)
        assert(stagingFile.forall(_.typ == TString))
      case LiftMeOut(_) =>
      case Consume(_) =>

      case TableAggregateByKey(child, _) =>
        assert(child.typ.key.nonEmpty)
      case TableExplode(child, path) =>
        assert(!child.typ.key.contains(path.head))
      case TableGen(contexts, globals, _, _, body, partitioner, _) =>
        TypeCheck.coerce[TStream]("contexts", contexts.typ): Unit
        TypeCheck.coerce[TStruct]("globals", globals.typ): Unit
        val bodyType = TypeCheck.coerce[TStream]("body", body.typ)
        val rowType = TypeCheck.coerce[TStruct]("body.elementType", bodyType.elementType)

        if (!partitioner.kType.isSubsetOf(rowType))
          throw new IllegalArgumentException(
            s"""'partitioner': key type contains fields absent from row type
               |  Key type: ${partitioner.kType}
               |  Row type: $rowType""".stripMargin
          )
      case TableJoin(left, right, _, joinKey) =>
        assert(left.typ.key.length >= joinKey)
        assert(right.typ.key.length >= joinKey)
        assert(
          left.typ.keyType.truncate(joinKey) isJoinableWith right.typ.keyType.truncate(joinKey)
        )
        assert(
          left.typ.globalType.fieldNames.toSet
            .intersect(right.typ.globalType.fieldNames.toSet)
            .isEmpty
        )
      case TableKeyBy(child, keys, _) =>
        val fields = child.typ.rowType.fieldNames.toSet
        assert(
          keys.forall(fields.contains),
          s"${keys.filter(k => !fields.contains(k)).mkString(", ")}",
        )
      case TableKeyByAndAggregate(_, expr, newKey, _, _) =>
        assert(expr.typ.isInstanceOf[TStruct])
        assert(newKey.typ.isInstanceOf[TStruct])
      case TableLeftJoinRightDistinct(left, right, _) =>
        assert(
          right.typ.keyType isPrefixOf left.typ.keyType,
          s"\n  L: ${left.typ}\n  R: ${right.typ}",
        )
      case TableMapPartitions(child, _, partitionStreamName, body, requestedKey, allowedOverlap) =>
        assert(body.typ.isInstanceOf[TStream], s"${body.typ}")
        assert(allowedOverlap >= -1)
        assert(allowedOverlap <= child.typ.key.size)
        assert(requestedKey >= 0)
        assert(requestedKey <= child.typ.key.size)
        assert(
          StreamUtils.isIterationLinear(body, partitionStreamName),
          "must iterate over the partition exactly once",
        )
        val newRowType = body.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
        child.typ.key.foreach { k =>
          if (!newRowType.hasField(k))
            throw new RuntimeException(s"prev key: ${child.typ.key}, new row: $newRowType")
        }
      case TableMapRows(child, newRow) =>
        val newFieldSet = newRow.typ.asInstanceOf[TStruct].fieldNames.toSet
        assert(child.typ.key.forall(newFieldSet.contains))
      case TableMultiWayZipJoin(childrenSeq, _, _) =>
        val first = childrenSeq.head
        val rest = childrenSeq.tail
        assert(
          rest.forall(e => e.typ.rowType == first.typ.rowType),
          "all rows must have the same type",
        )
        assert(rest.forall(e => e.typ.key == first.typ.key), "all keys must be the same")
        assert(
          rest.forall(e => e.typ.globalType == first.typ.globalType),
          "all globals must have the same type",
        )
      case TableParallelize(rowsAndGlobal, nPartitions) =>
        assert(rowsAndGlobal.typ.isInstanceOf[TStruct])
        assert(rowsAndGlobal.typ.asInstanceOf[TStruct].fieldNames.sameElements(Array(
          "rows",
          "global",
        )))
        assert(nPartitions.forall(_ > 0))
      case TableRename(child, rowMap, globalMap) =>
        assert(rowMap.keys.forall(child.typ.rowType.hasField))
        assert(globalMap.keys.forall(child.typ.globalType.hasField))
      case TableUnion(childrenSeq) =>
        assert(childrenSeq.tail.forall(_.typ.rowType == childrenSeq(0).typ.rowType))
        assert(childrenSeq.tail.forall(_.typ.key == childrenSeq(0).typ.key))
      case CastTableToMatrix(child, entriesFieldName, _, _) =>
        child.typ.rowType.fieldType(entriesFieldName) match {
          case TArray(TStruct(_)) =>
          case t => fatal(s"expected entry field to be an array of structs, found $t")
        }
      case MatrixAggregateColsByKey(child, _, _) =>
        assert(child.typ.colKey.nonEmpty)
      case MatrixAggregateRowsByKey(child, _, _) =>
        assert(child.typ.rowKey.nonEmpty)
      case MatrixAnnotateColsTable(child, _, root) =>
        assert(child.typ.colType.selfField(root).isEmpty)
      case MatrixAnnotateRowsTable(child, table, _, product) =>
        assert(
          (!product && table.typ.keyType.isPrefixOf(child.typ.rowKeyStruct)) ||
            (table.typ.keyType.size == 1 && table.typ.keyType.types(0) == TInterval(
              child.typ.rowKeyStruct.types(0)
            )),
          s"\n  L: ${child.typ}\n  R: ${table.typ}",
        )
      case MatrixKeyRowsBy(child, keys, _) =>
        val fields = child.typ.rowType.fieldNames.toSet
        assert(
          keys.forall(fields.contains),
          s"${keys.filter(k => !fields.contains(k)).mkString(", ")}",
        )
      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        assert(globalMap.keys.forall(child.typ.globalType.hasField))
        assert(colMap.keys.forall(child.typ.colType.hasField))
        assert(rowMap.keys.forall(child.typ.rowType.hasField))
        assert(entryMap.keys.forall(child.typ.entryType.hasField))
      case MatrixUnionCols(left, right, _) =>
        assert(
          left.typ.rowKeyStruct == right.typ.rowKeyStruct,
          s"${left.typ.rowKeyStruct} != ${right.typ.rowKeyStruct}",
        )
        assert(
          left.typ.colType == right.typ.colType,
          s"${left.typ.colType} != ${right.typ.colType}",
        )
        assert(
          left.typ.entryType == right.typ.entryType,
          s"${left.typ.entryType} != ${right.typ.entryType}",
        )
      case MatrixUnionRows(children) =>
        def compatible(t1: MatrixType, t2: MatrixType): Boolean =
          t1.colKeyStruct == t2.colKeyStruct &&
            t1.rowType == t2.rowType &&
            t1.rowKey == t2.rowKey &&
            t1.entryType == t2.entryType
        assert(
          children.tail.forall(c => compatible(c.typ, children.head.typ)),
          children.map(_.typ),
        )
      case BlockMatrixBroadcast(child, inIndexExpr, shape, blockSize) =>
        inIndexExpr match {
          case IndexedSeq() =>
            assert(child.typ.nRows == 1 && child.typ.nCols == 1)
          case IndexedSeq(0) => // broadcast col vector
            assert(Set(1, shape(0)) == Set(child.typ.nRows, child.typ.nCols))
          case IndexedSeq(1) => // broadcast row vector
            assert(Set(1, shape(1)) == Set(child.typ.nRows, child.typ.nCols))
          case IndexedSeq(0, 0) => // diagonal as row vector
            assert(shape(0) == 1L)
          case IndexedSeq(1, 0) => // transpose
            assert(child.typ.blockSize == blockSize)
            assert(shape(0) == child.typ.nCols && shape(1) == child.typ.nRows)
          case IndexedSeq(0, 1) =>
            assert(child.typ.blockSize == blockSize)
            assert(shape(0) == child.typ.nRows && shape(1) == child.typ.nCols)
        }
      case BlockMatrixMap(child, _, _, needsDense) =>
        assert(!(needsDense && child.typ.isSparse))
      case BlockMatrixMap2(left, right, _, _, _, _) =>
        assert(left.typ.nRows == right.typ.nRows)
        assert(left.typ.nCols == right.typ.nCols)
        assert(left.typ.blockSize == right.typ.blockSize)
      case ValueToBlockMatrix(child, _, _) =>
        assert(
          child.typ.isInstanceOf[TArray] || child.typ.isInstanceOf[TNDArray] || child.typ == TFloat64
        )
      case _ =>
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
