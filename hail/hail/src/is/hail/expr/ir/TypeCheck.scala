package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.collection.implicits.{toRichIterable, toRichOrderedSeq}
import is.hail.expr.Nat
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.JVMFunction
import is.hail.expr.ir.streams.StreamUtils
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.StackSafe._

import scala.annotation.switch
import scala.reflect.ClassTag

object TypeCheck extends Logging {
  def apply(ctx: ExecuteContext, ir: BaseIR): Unit =
    apply(ctx, ir, BindingEnv.empty)

  def apply(ctx: ExecuteContext, ir: BaseIR, env: BindingEnv[Type]): Unit =
    TimedBlock.enter {
      try
        check(ctx, ir, env).run()
      catch {
        case e: Throwable =>
          logger.fatal(
            s"Error while typechecking IR: ${Pretty(ctx, ir, allowUnboundRefs = true)}",
            e,
          )
          throw e
      }
    }

  def check(ctx: ExecuteContext, ir: BaseIR, env: BindingEnv[Type]): StackFrame[Unit] =
    for {
      _ <- ir.forEachChildWithEnvStackSafe(env)((_, t) => t) { (child, i, childEnv) =>
        check(ctx, child, childEnv).map { _ =>
          if (child.typ == TVoid) checkVoidTypedChild(ctx, ir, i)
        }
      }
    } yield checkSingleNode(ctx, ir, env)

  private def checkVoidTypedChild(ctx: ExecuteContext, ir: BaseIR, i: Int): Unit =
    ir match {
      case l: Block if i == l.bindings.length || l.body.typ == TVoid =>
      case _: StreamFor if i == 1 =>
      case _: RunAggScan if (i == 1 || i == 2) =>
      case _: StreamBufferedAggregate if (i == 1 || i == 3) =>
      case _: RunAgg if i == 0 =>
      case _: SeqOp => // let seqop checking below catch bad void arguments
      case _: InitOp => // let initop checking below catch bad void arguments
      case _: If if i != 0 =>
      case _: WriteMetadata =>
      case _ =>
        throw new RuntimeException(
          s"unexpected void-typed IR at child $i of ${ir.getClass.getSimpleName}" +
            s"\n  IR: ${Pretty(ctx, ir)}"
        )
    }

  /* Dispatch is on the node's tag, in two levels: `checkSingleNode` switches on the family, and
   * each family's method switches on the tag. Both compile to jump tables. Splitting by family also
   * keeps every method under HotSpot's 8000-byte limit (DontCompileHugeMethods), above which a
   * method is never JIT-compiled. Nodes with nothing to check fall through to `case _ =>`. */
  private def checkSingleNode(ctx: ExecuteContext, ir: BaseIR, env: BindingEnv[Type]): Unit =
    (ir.tag.family.toInt: @switch) match {
      case Family.Const => checkConst(ir)
      case Family.Core => checkCore(ir, env)
      case Family.Struct => checkStruct(ir)
      case Family.Stream => checkStream(ctx, ir)
      case Family.Collection => checkCollection(ir)
      case Family.Agg => checkAgg(ir, env)
      case Family.IO => checkIO(ir)
      case Family.NDArray => checkNDArray(ir)
      case Family.Random => checkRandom(ir)
      case Family.Relational => checkRelational(ir)
      case Family.Table => checkTable(ir)
      case Family.Matrix => checkMatrix(ir)
      case Family.BlockMatrix => checkBlockMatrix(ir)
    }

  private def checkConst(ir: BaseIR): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.NA =>
        val x = ir.asInstanceOf[NA]
        assert(x._typ != null)
      case _ =>
    }

  private def checkCore(ir: BaseIR, env: BindingEnv[Type]): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.Block =>
        val x = ir.asInstanceOf[Block]
        assert(x.typ == x.body.typ)
      case Tag.If =>
        val x = ir.asInstanceOf[If]
        assert(x.cond.typ == TBoolean)
        assert(x.typ == x.cnsq.typ && x.typ == x.altr.typ)
        x.typ match {
          case tstream: TStream => assert(tstream.elementType.isRealizable)
          case _ =>
        }
      case Tag.Switch =>
        val x = ir.asInstanceOf[Switch]
        assert(x.x.typ == TInt32)
        assert(x.cases.forall(_.typ == x.default.typ))
      case Tag.TailLoop =>
        val x = ir.asInstanceOf[TailLoop]
        assert(x.typ == x.resultType)
        assert(x.body.typ == x.resultType)
        def recurInTail(node: IR, tailPosition: Boolean): Boolean = node match {
          case r: Recur =>
            r.name != x.name || tailPosition
          case _ =>
            node.children.zipWithIndex
              .forall {
                case (c: IR, i) => recurInTail(c, tailPosition && InTailPosition(node, i))
                case _ => true
              }
        }
        assert(recurInTail(x.body, tailPosition = true))
      case Tag.Recur =>
        val x = ir.asInstanceOf[Recur]
        val TTuple(IndexedSeq(TupleField(_, argTypes), TupleField(_, rt))) = env.eval.lookup(x.name)
        assert(argTypes.asInstanceOf[TTuple].types.zip(x.args).forall { case (t, ir) =>
          t == ir.typ
        })
        assert(x._typ == rt)
      case Tag.Coalesce =>
        val x = ir.asInstanceOf[Coalesce]
        assert(x.values.tail.forall(_.typ == x.values.head.typ))
      case Tag.Die =>
        val x = ir.asInstanceOf[Die]
        assert(x.message.typ == TString)
      case Tag.Ref =>
        val x = ir.asInstanceOf[Ref]
        env.eval.lookupOption(x.name) match {
          case Some(expected) =>
            assert(
              x.typ == expected,
              s"type mismatch:\n  name: ${x.name}\n  actual: ${x.typ.parsableString()}\n  expect: ${expected.parsableString()}",
            )
          case None =>
            throw new NoSuchElementException(
              s"Ref with name ${x.name} could not be resolved in env $env"
            )
        }
      case Tag.In =>
        val x = ir.asInstanceOf[In]
        assert(x._typ != null)
        x._typ.virtualType match {
          case stream: TStream => assert(stream.elementType.isRealizable)
          case _ =>
        }
      case Tag.Cast =>
        val x = ir.asInstanceOf[Cast]
        if (!Casts.valid(x.v.typ, x._typ))
          throw new RuntimeException(s"invalid cast:\n  " +
            s"child type: ${x.v.typ.parsableString()}\n  " +
            s"cast type:  ${x._typ.parsableString()}")
      case Tag.ApplyBinaryPrimOp =>
        val x = ir.asInstanceOf[ApplyBinaryPrimOp]
        assert(x.typ == BinaryOp.getReturnType(x.op, x.l.typ, x.r.typ))
      case Tag.ApplyUnaryPrimOp =>
        val x = ir.asInstanceOf[ApplyUnaryPrimOp]
        assert(x.typ == UnaryOp.getReturnType(x.op, x.x.typ))
      case Tag.ApplyComparisonOp =>
        val x = ir.asInstanceOf[ApplyComparisonOp]
        ComparisonOp.checkCompatible(x.l.typ, x.r.typ)
        x.op match {
          case Compare => assert(x.typ == TInt32)
          case _ => assert(x.typ == TBoolean)
        }
      case Tag.Apply | Tag.ApplySpecial =>
        val x = ir.asInstanceOf[AbstractApplyNode[_ <: JVMFunction]]
        assert(x.implementation.unify(x.typeArgs, x.args.map(_.typ), x.returnType))
      case Tag.ConsoleLog =>
        val x = ir.asInstanceOf[ConsoleLog]
        assert(x.message.typ == TString)
      case _ => // ApplyIR, IsNA, Consume
    }

  private def checkStruct(ir: BaseIR): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.MakeStruct =>
        val x = ir.asInstanceOf[MakeStruct]
        assert(x.typ == TStruct(x.fields.map { case (name, a) =>
          (name, a.typ)
        }: _*))
      case Tag.SelectFields =>
        val x = ir.asInstanceOf[SelectFields]
        assert {
          val oldfields = tcoerce[TStruct](x.old.typ).fieldNames.toSet
          x.fields.forall(id => oldfields.contains(id))
        }
      case Tag.InsertFields =>
        val x = ir.asInstanceOf[InsertFields]
        x.fieldOrder.foreach { fds =>
          val newFieldSet = x.fields.map(_._1).toSet
          val oldFieldNames = x.old.typ.asInstanceOf[TStruct].fieldNames
          val oldFieldNameSet = oldFieldNames.toSet
          assert(fds.length == x.typ.size)
          assert(fds.areDistinct())
          assert(fds.toSet.forall(f => newFieldSet.contains(f) || oldFieldNameSet.contains(f)))
        }
      case Tag.GetField =>
        val x = ir.asInstanceOf[GetField]
        val t = tcoerce[TStruct](x.o.typ)
        assert(t.index(x.name).nonEmpty, s"${x.name} not in $t")
        assert(x.typ == t.field(x.name).typ)
      case Tag.MakeTuple =>
        val x = ir.asInstanceOf[MakeTuple]
        val indices = x.fields.map(_._1)
        assert(indices.areDistinct())
        assert(indices.isSorted)
        assert(x.typ == TTuple(x.fields.map { case (idx, f) => TupleField(idx, f.typ) }))
      case Tag.GetTupleElement =>
        val x = ir.asInstanceOf[GetTupleElement]
        val t = tcoerce[TTuple](x.o.typ)
        val fd = t.fields(t.fieldIndex(x.idx))
        assert(x.typ == fd.typ)
      case Tag.CastRename =>
        val x = ir.asInstanceOf[CastRename]
        if (!x.v.typ.isIsomorphicTo(x._typ))
          throw new RuntimeException(s"invalid cast:\n  " +
            s"child type: ${x.v.typ.parsableString()}\n  " +
            s"cast type:  ${x._typ.parsableString()}")
      case _ =>
    }

  private def checkStream(ctx: ExecuteContext, ir: BaseIR): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.StreamAgg =>
        val x = ir.asInstanceOf[StreamAgg]
        assert(x.a.typ.isInstanceOf[TStream])
      case Tag.StreamAggScan =>
        val x = ir.asInstanceOf[StreamAggScan]
        assert(x.a.typ.isInstanceOf[TStream])
        assert(x.typ.elementType == x.query.typ)
      case Tag.StreamBufferedAggregate =>
        val x = ir.asInstanceOf[StreamBufferedAggregate]
        assert(x.streamChild.typ.isInstanceOf[TStream])
        assert(x.initAggs.typ == TVoid)
        assert(x.seqOps.typ == TVoid)
        assert(x.newKey.typ.isInstanceOf[TStruct])
        assert(x.typ.isInstanceOf[TStream])
      case Tag.StreamDistribute =>
        val x = ir.asInstanceOf[StreamDistribute]
        assert(x.path.typ == TString)
        assert(x.child.typ.isInstanceOf[TStream])
        assert(x.pivots.typ.isInstanceOf[TArray])
        assert(x.pivots.typ.asInstanceOf[TArray].elementType.isInstanceOf[TStruct])
      case Tag.MakeStream =>
        val x = ir.asInstanceOf[MakeStream]
        assert(x._typ != null)
        assert(x._typ.elementType.isRealizable, x._typ.elementType)

        x.args.map(_.typ).zipWithIndex.foreach { case (t, i) =>
          assert(
            t == x._typ.elementType,
            s"at position $i type mismatch: ${x._typ.elementType.parsableString()} ${t.parsableString()}",
          )
        }
      case Tag.StreamIota =>
        val x = ir.asInstanceOf[StreamIota]
        assert(x.start.typ == TInt32)
        assert(x.step.typ == TInt32)
      case Tag.StreamRange =>
        val x = ir.asInstanceOf[StreamRange]
        assert(x.start.typ == TInt32)
        assert(x.stop.typ == TInt32)
        assert(x.step.typ == TInt32)
      case Tag.ToStream =>
        val x = ir.asInstanceOf[ToStream]
        assert(x.a.typ.isInstanceOf[TContainer])
      case Tag.StreamGrouped =>
        val x = ir.asInstanceOf[StreamGrouped]
        val ts = tcoerce[TStream](x.typ)
        assert(x.a.typ.isInstanceOf[TStream])
        assert(ts.elementType == x.a.typ)
        assert(x.groupSize.typ == TInt32)
      case Tag.StreamGroupByKey =>
        val x = ir.asInstanceOf[StreamGroupByKey]
        val ts = tcoerce[TStream](x.typ)
        assert(ts.elementType == x.a.typ)
        val structType = tcoerce[TStruct](tcoerce[TStream](x.a.typ).elementType)
        assert(x.key.forall(structType.hasField))
      case Tag.StreamMap =>
        val x = ir.asInstanceOf[StreamMap]
        assert(x.a.typ.isInstanceOf[TStream])
        assert(x.typ.elementType == x.body.typ)
      case Tag.StreamTakeWhile =>
        val x = ir.asInstanceOf[StreamTakeWhile]
        assert(x.a.typ.asInstanceOf[TStream].elementType.isRealizable)
        assert(x.body.typ == TBoolean)
        assert(x.typ == x.a.typ)
      case Tag.StreamDropWhile =>
        val x = ir.asInstanceOf[StreamDropWhile]
        assert(x.a.typ.asInstanceOf[TStream].elementType.isRealizable)
        assert(x.body.typ == TBoolean)
        assert(x.typ == x.a.typ)
      case Tag.StreamTake =>
        val x = ir.asInstanceOf[StreamTake]
        assert(x.a.typ.isInstanceOf[TStream])
        assert(x.typ == x.a.typ)
        assert(x.num.typ == TInt32)
      case Tag.StreamDrop =>
        val x = ir.asInstanceOf[StreamDrop]
        assert(x.a.typ.isInstanceOf[TStream])
        assert(x.typ == x.a.typ)
        assert(x.num.typ == TInt32)
      case Tag.SeqSample =>
        val x = ir.asInstanceOf[SeqSample]
        assert(x.totalRange.typ == TInt32)
        assert(x.numToSample.typ == TInt32)
        assert(x.rngState.typ == TRNGState)
      case Tag.StreamWhiten =>
        val x = ir.asInstanceOf[StreamWhiten]
        assert(x.stream.typ.isInstanceOf[TStream])
        val eltTyp = x.stream.typ.asInstanceOf[TStream].elementType
        assert(eltTyp.isInstanceOf[TStruct])
        val structTyp = eltTyp.asInstanceOf[TStruct]
        val matTyp = TNDArray(TFloat64, Nat(2))
        assert(structTyp.field(x.newChunk).typ == matTyp)
        assert(structTyp.field(x.prevWindow).typ == matTyp)
        assert(x.windowSize % x.chunkSize == 0)
      case Tag.StreamZip =>
        val x = ir.asInstanceOf[StreamZip]
        assert(x.as.length == x.names.length)
        assert(x.typ.elementType == x.body.typ)
        assert(x.as.forall(_.typ.isInstanceOf[TStream]))
      case Tag.StreamMultiMerge =>
        val x = ir.asInstanceOf[StreamMultiMerge]
        val streamType = tcoerce[TStream](x.as.head.typ)
        assert(x.as.forall(_.typ == streamType))
        val eltType = tcoerce[TStruct](streamType.elementType)
        assert(x.typ.elementType == eltType)
        assert(x.key.forall(eltType.hasField))
      case Tag.StreamZipJoinProducers =>
        val x = ir.asInstanceOf[StreamZipJoinProducers]
        assert(x.contexts.typ.isInstanceOf[TArray])
        val streamType = tcoerce[TStream](x.makeProducer.typ)
        val eltType = tcoerce[TStruct](streamType.elementType)
        assert(x.key.forall(eltType.hasField))
        assert(x.typ.elementType == x.joinF.typ)
      case Tag.StreamZipJoin =>
        val x = ir.asInstanceOf[StreamZipJoin]
        val streamType = tcoerce[TStream](x.as.head.typ)
        assert(x.as.forall(_.typ == streamType))
        val eltType = tcoerce[TStruct](streamType.elementType)
        assert(x.key.forall(eltType.hasField))
        assert(x.typ.elementType == x.joinF.typ)
      case Tag.StreamFilter =>
        val x = ir.asInstanceOf[StreamFilter]
        assert(x.a.typ.asInstanceOf[TStream].elementType.isRealizable)
        assert(x.cond.typ == TBoolean, x.cond.typ)
        assert(x.typ == x.a.typ)
      case Tag.StreamFlatMap =>
        val x = ir.asInstanceOf[StreamFlatMap]
        assert(x.a.typ.isInstanceOf[TStream])
        assert(x.cond.typ.isInstanceOf[TStream])
      case Tag.StreamScan =>
        val x = ir.asInstanceOf[StreamScan]
        assert(x.a.typ.isInstanceOf[TStream])
        assert(x.body.typ == x.zero.typ)
        assert(tcoerce[TStream](x.typ).elementType == x.zero.typ)
        assert(x.zero.typ.isRealizable)
      case Tag.StreamLeftIntervalJoin =>
        val x = ir.asInstanceOf[StreamLeftIntervalJoin]
        assert(x.left.typ.isInstanceOf[TStream])
        assert(x.right.typ.isInstanceOf[TStream])

        val lEltTy =
          TIterable.elementType(x.left.typ).asInstanceOf[TStruct]

        val rPointTy =
          TIterable.elementType(x.right.typ)
            .asInstanceOf[TStruct]
            .fieldType(x.rIntervalFieldName)
            .asInstanceOf[TInterval]
            .pointType

        assert(lEltTy.fieldType(x.lKeyFieldName) == rPointTy)
        assert(x.body.typ.isInstanceOf[TStruct])
      case Tag.StreamJoinRightDistinct =>
        val x = ir.asInstanceOf[StreamJoinRightDistinct]
        val lEltTyp = tcoerce[TStruct](tcoerce[TStream](x.left.typ).elementType)
        val rEltTyp = tcoerce[TStruct](tcoerce[TStream](x.right.typ).elementType)
        assert(tcoerce[TStream](x.typ).elementType == x.joinF.typ)
        assert(x.lKey.forall(lEltTyp.hasField))
        assert(x.rKey.forall(rEltTyp.hasField))
        if (x.isIntervalJoin) {
          val lKeyTyp = lEltTyp.fieldType(x.lKey(0))
          val rKeyTyp = rEltTyp.fieldType(x.rKey(0)).asInstanceOf[TInterval]
          assert(lKeyTyp == rKeyTyp.pointType)
          assert((x.joinType == "left") || (x.joinType == "inner"))
        } else {
          assert(x.lKey.lazyZip(x.rKey).forall { case (lk, rk) =>
            lEltTyp.fieldType(lk) == rEltTyp.fieldType(rk)
          })
        }
      case Tag.StreamLocalLDPrune =>
        val x = ir.asInstanceOf[StreamLocalLDPrune]
        assert(x.child.typ.isInstanceOf[TStream])
        assert(x.r2Threshold.typ == TFloat64)
        assert(x.windowSize.typ == TInt32)
        assert(x.maxQueueSize.typ == TInt32)
        assert(x.nSamples.typ == TInt32)
        val eltType = x.child.typ.asInstanceOf[TStream].elementType
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
      case Tag.StreamLen =>
        val x = ir.asInstanceOf[StreamLen]
        assert(x.a.typ.isInstanceOf[TStream])
      case Tag.StreamFold =>
        val x = ir.asInstanceOf[StreamFold]
        assert(x.a.typ.isInstanceOf[TStream])
        assert(x.a.typ.asInstanceOf[TStream].elementType.isRealizable, Pretty(ctx, x))
        assert(x.body.typ == x.zero.typ)
        assert(x.typ == x.zero.typ)
      case Tag.StreamFold2 =>
        val x = ir.asInstanceOf[StreamFold2]
        assert(x.a.typ.isInstanceOf[TStream])
        assert(x.typ == x.result.typ)
        assert(x.accum.zip(x.seq).forall { case ((_, z), s) => s.typ == z.typ })
      case Tag.StreamFor =>
        val x = ir.asInstanceOf[StreamFor]
        assert(x.a.typ.isInstanceOf[TStream])
        assert(x.body.typ == TVoid)
      case _ =>
    }

  private def checkCollection(ir: BaseIR): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.CollectDistributedArray =>
        val x = ir.asInstanceOf[CollectDistributedArray]
        assert(x.contexts.typ.isInstanceOf[TStream])
        assert(x.dynamicID.typ == TString)
      case Tag.MakeArray =>
        val x = ir.asInstanceOf[MakeArray]
        assert(x._typ != null)
        x.args.map(_.typ).zipWithIndex.foreach { case (t, i) =>
          assert(
            t == x._typ.elementType && t.isRealizable,
            s"at position $i type mismatch: ${x._typ.parsableString()} ${t.parsableString()}",
          )
        }
      case Tag.ArrayRef =>
        val x = ir.asInstanceOf[ArrayRef]
        assert(x.i.typ == TInt32)
        assert(x.typ == tcoerce[TArray](x.a.typ).elementType)
      case Tag.ArraySlice =>
        val x = ir.asInstanceOf[ArraySlice]
        assert(x.start.typ == TInt32)
        x.stop.foreach(ir => assert(ir.typ == TInt32))
        assert(x.step.typ == TInt32)
        assert(x.typ == tcoerce[TArray](x.a.typ))
      case Tag.ArrayLen =>
        val x = ir.asInstanceOf[ArrayLen]
        assert(x.a.typ.isInstanceOf[TArray])
      case Tag.ArrayZeros =>
        val x = ir.asInstanceOf[ArrayZeros]
        assert(x.length.typ == TInt32)
      case Tag.ArrayMaximalIndependentSet =>
        val x = ir.asInstanceOf[ArrayMaximalIndependentSet]
        assert(x.edges.typ.isInstanceOf[TArray])
        val edgeType = tcoerce[TArray](x.edges.typ).elementType
        assert(edgeType.isInstanceOf[TBaseStruct])
        val Seq(leftType, rightType) = edgeType.asInstanceOf[TBaseStruct].types
        assert(leftType == rightType)
        x.tieBreaker.foreach { case (_, _, tb) => assert(tb.typ == TFloat64) }
      case Tag.ArraySort =>
        val x = ir.asInstanceOf[ArraySort]
        assert(x.a.typ.isInstanceOf[TStream])
        assert(x.lessThan.typ == TBoolean)
      case Tag.ToSet =>
        val x = ir.asInstanceOf[ToSet]
        assert(x.a.typ.isInstanceOf[TStream], x.a.typ)
      case Tag.ToDict =>
        val x = ir.asInstanceOf[ToDict]
        assert(x.a.typ.isInstanceOf[TStream])
        assert(tcoerce[TBaseStruct](tcoerce[TStream](x.a.typ).elementType).size == 2)
      case Tag.ToArray =>
        val x = ir.asInstanceOf[ToArray]
        assert(x.a.typ.isInstanceOf[TStream])
      case Tag.CastToArray =>
        val x = ir.asInstanceOf[CastToArray]
        assert(x.a.typ.isInstanceOf[TContainer])
      case Tag.GroupByKey =>
        val x = ir.asInstanceOf[GroupByKey]
        val telt = tcoerce[TBaseStruct](tcoerce[TStream](x.collection.typ).elementType)
        val td = tcoerce[TDict](x.typ)
        assert(td.keyType == telt.types(0))
        assert(td.valueType == TArray(telt.types(1)))
      case Tag.LowerBoundOnOrderedCollection =>
        val x = ir.asInstanceOf[LowerBoundOnOrderedCollection]
        val elt = tcoerce[TIterable](x.orderedCollection.typ).elementType
        assert(x.elem.typ == (if (x.onKey) elt match {
                                case t: TBaseStruct => t.types(0)
                                case t: TInterval => t.pointType
                              }
                              else elt))
      case _ =>
    }

  private def checkAgg(ir: BaseIR, env: BindingEnv[Type]): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.AggFilter =>
        val x = ir.asInstanceOf[AggFilter]
        assert(x.cond.typ == TBoolean)
        assert(x.typ == x.aggIR.typ)
      case Tag.AggExplode =>
        val x = ir.asInstanceOf[AggExplode]
        assert(x.array.typ.isInstanceOf[TStream])
        assert(x.typ == x.aggBody.typ)
      case Tag.AggGroupBy =>
        val x = ir.asInstanceOf[AggGroupBy]
        assert(x.typ == TDict(x.key.typ, x.aggIR.typ))
      case Tag.AggArrayPerElement =>
        val x = ir.asInstanceOf[AggArrayPerElement]
        assert(x.a.typ.isInstanceOf[TArray])
        assert(x.typ == TArray(x.aggBody.typ))
        assert(x.knownLength.forall(_.typ == TInt32))
      case Tag.AggFold =>
        val x = ir.asInstanceOf[AggFold]
        assert(x.zero.typ == x.seqOp.typ)
        assert(x.zero.typ == x.combOp.typ)
      case Tag.ApplyAggOp =>
        val x = ir.asInstanceOf[ApplyAggOp]
        assert(env.agg.isDefined, "ApplyAggOp outside an agg scope")
        assert(x.typ == AggOp.getReturnType(x.op, x.seqOpArgs.map(_.typ)))
      case Tag.ApplyScanOp =>
        val x = ir.asInstanceOf[ApplyScanOp]
        assert(env.scan.isDefined, "ApplyScanOp outside a scan scope")
        assert(x.typ == AggOp.getReturnType(x.op, x.seqOpArgs.map(_.typ)))
      case Tag.InitOp =>
        val x = ir.asInstanceOf[InitOp]
        assert(
          x.args.map(_.typ) == x.aggSig.initOpTypes,
          s"${x.args.map(_.typ)} !=  ${x.aggSig.initOpTypes}",
        )
      case Tag.SeqOp =>
        val x = ir.asInstanceOf[SeqOp]
        assert(
          x.args.map(_.typ) == x.aggSig.seqOpTypes,
          s"${x.args.map(_.typ)} !=  ${x.aggSig.seqOpTypes}\n${x.aggSig}",
        )
      case Tag.CombOpValue =>
        val x = ir.asInstanceOf[CombOpValue]
        assert(x.value.typ == TBinary)
      case Tag.InitFromSerializedValue =>
        val x = ir.asInstanceOf[InitFromSerializedValue]
        assert(x.value.typ == TBinary)
      case Tag.RunAgg =>
        val x = ir.asInstanceOf[RunAgg]
        assert(x.typ == x.result.typ)
        assert(x.body.typ == TVoid)
      case Tag.RunAggScan =>
        val x = ir.asInstanceOf[RunAggScan]
        assert(x.array.typ.isInstanceOf[TStream])
        assert(x.init.typ == TVoid)
        assert(x.seqs.typ == TVoid)
        assert(x.typ.asInstanceOf[TStream].elementType == x.result.typ)
      case _ =>
    }

  private def checkIO(ir: BaseIR): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.ReadPartition =>
        val x = ir.asInstanceOf[ReadPartition]
        assert(x.rowType.isRealizable)
        assert(x.context.typ == x.reader.contextType)
        assert(x.typ == TStream(x.rowType))
        assert(PruneDeadFields.isSupertype(x.rowType, x.reader.fullRowType))
      case Tag.ReadValue =>
        val x = ir.asInstanceOf[ReadValue]
        assert(x.path.typ == TString)
        x.reader match {
          case reader: ETypeValueReader =>
            assert(
              reader.spec.encodedType.decodedPType(x.requestedType).virtualType == x.requestedType
            )
          case _ => // do nothing, we can't in general typecheck an arbitrary value reader
        }
      case Tag.WritePartition =>
        val x = ir.asInstanceOf[WritePartition]
        assert(x.value.typ.isInstanceOf[TStream])
        assert(x.writeCtx.typ == x.writer.ctxType)
        assert(x.typ == x.writer.returnType)
      case Tag.WriteValue =>
        val x = ir.asInstanceOf[WriteValue]
        assert(x.path.typ == TString)
      case Tag.WriteMetadata =>
        val x = ir.asInstanceOf[WriteMetadata]
        assert(x.writeAnnotations.typ == x.writer.annotationType)
      case _ =>
    }

  private def checkNDArray(ir: BaseIR): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.NDArrayWrite =>
        val x = ir.asInstanceOf[NDArrayWrite]
        assert(x.nd.typ.isInstanceOf[TNDArray])
        assert(x.path.typ == TString)
      case Tag.MakeNDArray =>
        val x = ir.asInstanceOf[MakeNDArray]
        assert(x.data.typ.isInstanceOf[TArray] || x.data.typ.isInstanceOf[TStream])
        assert(x.shape.typ.asInstanceOf[TTuple].types.forall(t => t == TInt64))
        assert(x.rowMajor.typ == TBoolean)
      case Tag.NDArrayShape =>
        val x = ir.asInstanceOf[NDArrayShape]
        assert(x.nd.typ.isInstanceOf[TNDArray])
      case Tag.NDArrayReshape =>
        val x = ir.asInstanceOf[NDArrayReshape]
        assert(x.nd.typ.isInstanceOf[TNDArray])
        assert(x.shape.typ.asInstanceOf[TTuple].types.forall(t => t == TInt64))
      case Tag.NDArrayConcat =>
        val x = ir.asInstanceOf[NDArrayConcat]
        assert(tcoerce[TArray](x.nds.typ).elementType.isInstanceOf[TNDArray])
        assert(x.axis < x.typ.nDims)
      case Tag.NDArrayRef =>
        val x = ir.asInstanceOf[NDArrayRef]
        assert(x.nd.typ.isInstanceOf[TNDArray])
        assert(x.nd.typ.asInstanceOf[TNDArray].nDims == x.idxs.length)
        assert(x.idxs.forall(_.typ == TInt64))
      case Tag.NDArraySlice =>
        val x = ir.asInstanceOf[NDArraySlice]
        assert(x.nd.typ.isInstanceOf[TNDArray])
        val childTyp = x.nd.typ.asInstanceOf[TNDArray]
        val slicesTuple = x.slices.typ.asInstanceOf[TTuple]
        assert(slicesTuple.size == childTyp.nDims)
        assert(slicesTuple.types.forall { t =>
          (t == TTuple(TInt64, TInt64, TInt64)) || (t == TInt64)
        })
      case Tag.NDArrayFilter =>
        val x = ir.asInstanceOf[NDArrayFilter]
        val ndtyp = tcoerce[TNDArray](x.nd.typ)
        assert(ndtyp.nDims == x.keep.length)
        assert(x.keep.forall(f => tcoerce[TArray](f.typ).elementType == TInt64))
      case Tag.NDArrayMap =>
        val x = ir.asInstanceOf[NDArrayMap]
        assert(x.typ.elementType == x.body.typ)
      case Tag.NDArrayMap2 =>
        val x = ir.asInstanceOf[NDArrayMap2]
        val lTyp = tcoerce[TNDArray](x.l.typ)
        val rTyp = tcoerce[TNDArray](x.r.typ)
        assert(lTyp.nDims == rTyp.nDims)
        assert(x.typ.elementType == x.body.typ)
      case Tag.NDArrayReindex =>
        val x = ir.asInstanceOf[NDArrayReindex]
        assert(x.nd.typ.isInstanceOf[TNDArray])
        val nInputDims = tcoerce[TNDArray](x.nd.typ).nDims
        val nOutputDims = x.indexExpr.length
        assert(nInputDims <= nOutputDims)
        assert(x.indexExpr.forall(i => i < nOutputDims))
        assert((0 until nOutputDims).forall(i => x.indexExpr.contains(i)))
      case Tag.NDArrayAgg =>
        val x = ir.asInstanceOf[NDArrayAgg]
        assert(x.nd.typ.isInstanceOf[TNDArray])
        val nInputDims = tcoerce[TNDArray](x.nd.typ).nDims
        assert(x.axes.length <= nInputDims)
        assert(x.axes.forall(i => i < nInputDims))
        assert(x.axes.distinct.length == x.axes.length)
      case Tag.NDArrayMatMul =>
        val x = ir.asInstanceOf[NDArrayMatMul]
        assert(x.l.typ.isInstanceOf[TNDArray])
        assert(x.r.typ.isInstanceOf[TNDArray])
        val lType = x.l.typ.asInstanceOf[TNDArray]
        val rType = x.r.typ.asInstanceOf[TNDArray]
        assert(lType.elementType == rType.elementType, "element type did not match")
        assert(lType.nDims > 0)
        assert(rType.nDims > 0)
        assert(lType.nDims == 1 || rType.nDims == 1 || lType.nDims == rType.nDims)
      case Tag.NDArrayQR =>
        val x = ir.asInstanceOf[NDArrayQR]
        checkFloat64Matrix(x.nd)
      case Tag.NDArraySVD =>
        val x = ir.asInstanceOf[NDArraySVD]
        checkFloat64Matrix(x.nd)
      case Tag.NDArrayEigh =>
        val x = ir.asInstanceOf[NDArrayEigh]
        checkFloat64Matrix(x.nd)
      case Tag.NDArrayInv =>
        val x = ir.asInstanceOf[NDArrayInv]
        checkFloat64Matrix(x.nd)
      case _ =>
    }

  private def checkFloat64Matrix(nd: IR): Unit = {
    val ndType = nd.typ.asInstanceOf[TNDArray]
    assert(ndType.elementType == TFloat64)
    assert(ndType.nDims == 2)
  }

  private def checkRandom(ir: BaseIR): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.RNGStateLiteral =>
        assert(ir.typ == TRNGState)
      case Tag.RNGSplit =>
        val x = ir.asInstanceOf[RNGSplit]
        assert(x.state.typ == TRNGState)
        def isValid: Type => Boolean = {
          case tuple: TTuple => tuple.types.forall(_ == TInt64)
          case t => t == TInt64
        }
        assert(isValid(x.dynBitstring.typ))
      case Tag.RNGSplitStatic =>
        val x = ir.asInstanceOf[RNGSplitStatic]
        assert(x.state.typ == TRNGState)
      case _ =>
    }

  private def checkRelational(ir: BaseIR): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.TableAggregate =>
        val x = ir.asInstanceOf[TableAggregate]
        assert(x.typ == x.query.typ)
      case Tag.TableCollect =>
        val x = ir.asInstanceOf[TableCollect]
        assert(x.child.typ.key.isEmpty)
      case _ =>
    }

  private def checkTable(ir: BaseIR): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.TableMultiWrite =>
        val x = ir.asInstanceOf[TableMultiWrite]
        val t = x._children.head.typ
        assert(x._children.forall(_.typ == t))
      case Tag.TableAggregateByKey =>
        val x = ir.asInstanceOf[TableAggregateByKey]
        assert(x.child.typ.key.nonEmpty)
      case Tag.TableExplode =>
        val x = ir.asInstanceOf[TableExplode]
        assert(!x.child.typ.key.contains(x.path.head))
      case Tag.TableGen =>
        val x = ir.asInstanceOf[TableGen]
        TypeCheck.coerce[TStream]("contexts", x.contexts.typ): Unit
        TypeCheck.coerce[TStruct]("globals", x.globals.typ): Unit
        val bodyType = TypeCheck.coerce[TStream]("body", x.body.typ)
        val rowType = TypeCheck.coerce[TStruct]("body.elementType", bodyType.elementType)

        if (!x.partitioner.kType.isSubsetOf(rowType))
          throw new IllegalArgumentException(
            s"""'partitioner': key type contains fields absent from row type
               |  Key type: ${x.partitioner.kType}
               |  Row type: $rowType""".stripMargin
          )
      case Tag.TableJoin =>
        val x = ir.asInstanceOf[TableJoin]
        assert(x.left.typ.key.length >= x.joinKey)
        assert(x.right.typ.key.length >= x.joinKey)
        assert(
          x.left.typ.keyType.truncate(x.joinKey) isJoinableWith x.right.typ.keyType.truncate(
            x.joinKey
          )
        )
        assert(
          x.left.typ.globalType.fieldNames.toSet
            .intersect(x.right.typ.globalType.fieldNames.toSet)
            .isEmpty
        )
      case Tag.TableKeyBy =>
        val x = ir.asInstanceOf[TableKeyBy]
        val fields = x.child.typ.rowType.fieldNames.toSet
        assert(
          x.keys.forall(fields.contains),
          s"${x.keys.filter(k => !fields.contains(k)).mkString(", ")}",
        )
      case Tag.TableKeyByAndAggregate =>
        val x = ir.asInstanceOf[TableKeyByAndAggregate]
        assert(x.expr.typ.isInstanceOf[TStruct])
        assert(x.newKey.typ.isInstanceOf[TStruct])
      case Tag.TableLeftJoinRightDistinct =>
        val x = ir.asInstanceOf[TableLeftJoinRightDistinct]
        assert(
          x.right.typ.keyType isPrefixOf x.left.typ.keyType,
          s"\n  L: ${x.left.typ}\n  R: ${x.right.typ}",
        )
      case Tag.TableMapPartitions =>
        val x = ir.asInstanceOf[TableMapPartitions]
        assert(x.body.typ.isInstanceOf[TStream], s"${x.body.typ}")
        assert(x.allowedOverlap >= -1)
        assert(x.allowedOverlap <= x.child.typ.key.size)
        assert(x.requestedKey >= 0)
        assert(x.requestedKey <= x.child.typ.key.size)
        assert(
          StreamUtils.isIterationLinear(x.body, x.partitionStreamName),
          "must iterate over the partition exactly once",
        )
        val newRowType = x.body.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
        x.child.typ.key.foreach { k =>
          if (!newRowType.hasField(k))
            throw new RuntimeException(s"prev key: ${x.child.typ.key}, new row: $newRowType")
        }
      case Tag.TableMapRows =>
        val x = ir.asInstanceOf[TableMapRows]
        val newFieldSet = x.newRow.typ.asInstanceOf[TStruct].fieldNames.toSet
        assert(x.child.typ.key.forall(newFieldSet.contains))
      case Tag.TableMultiWayZipJoin =>
        val x = ir.asInstanceOf[TableMultiWayZipJoin]
        val first = x.childrenSeq.head
        val rest = x.childrenSeq.tail
        assert(
          rest.forall(e => e.typ.rowType == first.typ.rowType),
          "all rows must have the same type",
        )
        assert(rest.forall(e => e.typ.key == first.typ.key), "all keys must be the same")
        assert(
          rest.forall(e => e.typ.globalType == first.typ.globalType),
          "all globals must have the same type",
        )
      case Tag.TableParallelize =>
        val x = ir.asInstanceOf[TableParallelize]
        assert(x.rowsAndGlobal.typ.isInstanceOf[TStruct])
        assert(x.rowsAndGlobal.typ.asInstanceOf[TStruct].fieldNames.sameElements(Array(
          "rows",
          "global",
        )))
        assert(x.nPartitions.forall(_ > 0))
      case Tag.TableRename =>
        val x = ir.asInstanceOf[TableRename]
        assert(x.rowMap.keys.forall(x.child.typ.rowType.hasField))
        assert(x.globalMap.keys.forall(x.child.typ.globalType.hasField))
      case Tag.TableUnion =>
        val x = ir.asInstanceOf[TableUnion]
        assert(x.childrenSeq.tail.forall(_.typ.rowType == x.childrenSeq(0).typ.rowType))
        assert(x.childrenSeq.tail.forall(_.typ.key == x.childrenSeq(0).typ.key))
      case _ =>
    }

  private def checkMatrix(ir: BaseIR): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.MatrixAggregate =>
        val x = ir.asInstanceOf[MatrixAggregate]
        assert(x.typ == x.query.typ)
      case Tag.MatrixMultiWrite =>
        val x = ir.asInstanceOf[MatrixMultiWrite]
        val t = x._children.head.typ
        assert(
          !t.rowType.hasField(MatrixReader.rowUIDFieldName) &&
            !t.colType.hasField(MatrixReader.colUIDFieldName),
          t,
        )
        assert(
          x._children.forall(_.typ == t),
          "MatrixMultiWrite requires MatrixTable inputs of the same type",
        )
      case Tag.CastTableToMatrix =>
        val x = ir.asInstanceOf[CastTableToMatrix]
        x.child.typ.rowType.fieldType(x.entriesFieldName) match {
          case TArray(TStruct(_)) =>
          case t => fatal(s"expected entry field to be an array of structs, found $t")
        }
      case Tag.MatrixAggregateColsByKey =>
        val x = ir.asInstanceOf[MatrixAggregateColsByKey]
        assert(x.child.typ.colKey.nonEmpty)
      case Tag.MatrixAggregateRowsByKey =>
        val x = ir.asInstanceOf[MatrixAggregateRowsByKey]
        assert(x.child.typ.rowKey.nonEmpty)
      case Tag.MatrixAnnotateColsTable =>
        val x = ir.asInstanceOf[MatrixAnnotateColsTable]
        assert(x.child.typ.colType.selfField(x.root).isEmpty)
      case Tag.MatrixAnnotateRowsTable =>
        val x = ir.asInstanceOf[MatrixAnnotateRowsTable]
        assert(
          (!x.product && x.table.typ.keyType.isPrefixOf(x.child.typ.rowKeyStruct)) ||
            (x.table.typ.keyType.size == 1 && x.table.typ.keyType.types(0) == TInterval(
              x.child.typ.rowKeyStruct.types(0)
            )),
          s"\n  L: ${x.child.typ}\n  R: ${x.table.typ}",
        )
      case Tag.MatrixKeyRowsBy =>
        val x = ir.asInstanceOf[MatrixKeyRowsBy]
        val fields = x.child.typ.rowType.fieldNames.toSet
        assert(
          x.keys.forall(fields.contains),
          s"${x.keys.filter(k => !fields.contains(k)).mkString(", ")}",
        )
      case Tag.MatrixRename =>
        val x = ir.asInstanceOf[MatrixRename]
        assert(x.globalMap.keys.forall(x.child.typ.globalType.hasField))
        assert(x.colMap.keys.forall(x.child.typ.colType.hasField))
        assert(x.rowMap.keys.forall(x.child.typ.rowType.hasField))
        assert(x.entryMap.keys.forall(x.child.typ.entryType.hasField))
      case Tag.MatrixUnionCols =>
        val x = ir.asInstanceOf[MatrixUnionCols]
        assert(
          x.left.typ.rowKeyStruct == x.right.typ.rowKeyStruct,
          s"${x.left.typ.rowKeyStruct} != ${x.right.typ.rowKeyStruct}",
        )
        assert(
          x.left.typ.colType == x.right.typ.colType,
          s"${x.left.typ.colType} != ${x.right.typ.colType}",
        )
        assert(
          x.left.typ.entryType == x.right.typ.entryType,
          s"${x.left.typ.entryType} != ${x.right.typ.entryType}",
        )
      case Tag.MatrixUnionRows =>
        val x = ir.asInstanceOf[MatrixUnionRows]
        def compatible(t1: MatrixType, t2: MatrixType): Boolean =
          t1.colKeyStruct == t2.colKeyStruct &&
            t1.rowType == t2.rowType &&
            t1.rowKey == t2.rowKey &&
            t1.entryType == t2.entryType
        assert(
          x.childrenSeq.tail.forall(c => compatible(c.typ, x.childrenSeq.head.typ)),
          x.childrenSeq.map(_.typ),
        )
      case _ =>
    }

  private def checkBlockMatrix(ir: BaseIR): Unit =
    (ir.tag.toInt: @switch) match {
      case Tag.BlockMatrixBroadcast =>
        val x = ir.asInstanceOf[BlockMatrixBroadcast]
        x.inIndexExpr match {
          case IndexedSeq() =>
            assert(x.child.typ.nRows == 1 && x.child.typ.nCols == 1)
          case IndexedSeq(0) => // broadcast col vector
            assert(Set(1, x.shape(0)) == Set(x.child.typ.nRows, x.child.typ.nCols))
          case IndexedSeq(1) => // broadcast row vector
            assert(Set(1, x.shape(1)) == Set(x.child.typ.nRows, x.child.typ.nCols))
          case IndexedSeq(0, 0) => // diagonal as row vector
            assert(x.shape(0) == 1L)
          case IndexedSeq(1, 0) => // transpose
            assert(x.child.typ.blockSize == x.blockSize)
            assert(x.shape(0) == x.child.typ.nCols && x.shape(1) == x.child.typ.nRows)
          case IndexedSeq(0, 1) =>
            assert(x.child.typ.blockSize == x.blockSize)
            assert(x.shape(0) == x.child.typ.nRows && x.shape(1) == x.child.typ.nCols)
        }
      case Tag.BlockMatrixMap =>
        val x = ir.asInstanceOf[BlockMatrixMap]
        assert(!(x.needsDense && x.child.typ.isSparse))
      case Tag.BlockMatrixMap2 =>
        val x = ir.asInstanceOf[BlockMatrixMap2]
        assert(x.left.typ.nRows == x.right.typ.nRows)
        assert(x.left.typ.nCols == x.right.typ.nCols)
        assert(x.left.typ.blockSize == x.right.typ.blockSize)
      case Tag.ValueToBlockMatrix =>
        val x = ir.asInstanceOf[ValueToBlockMatrix]
        assert(
          x.child.typ.isInstanceOf[TArray] || x.child.typ.isInstanceOf[TNDArray] || x.child.typ == TFloat64
        )
      case _ =>
    }

  def coerce[A <: Type](argname: String, typ: Type)(implicit tag: ClassTag[A]): A =
    if (tag.runtimeClass.isInstance(typ)) typ.asInstanceOf[A]
    else throw new IllegalArgumentException(
      s"""'$argname': Type mismatch.
         |  Expected: ${tag.runtimeClass.getName}
         |    Actual: ${typ.getClass.getName}""".stripMargin
    )

}
