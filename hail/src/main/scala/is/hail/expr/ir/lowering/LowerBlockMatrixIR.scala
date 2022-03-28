package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.Nat
import is.hail.expr.ir._
import is.hail.expr.ir.functions.GetElement
import is.hail.rvd.RVDPartitioner
import is.hail.types.{BlockMatrixSparsity, BlockMatrixType, TypeWithRequiredness}
import is.hail.types.virtual._
import is.hail.utils._

object BlockMatrixStage {
  def empty(eltType: Type): BlockMatrixStage =
    EmptyBlockMatrixStage(eltType)

  def broadcastVector(vector: IR, typ: BlockMatrixType, asRowVector: Boolean): BlockMatrixStage = {
    val v = Ref(genUID(), vector.typ)
    new BlockMatrixStage(IndexedSeq(), Array(v.name -> vector), TStruct("start" -> TInt32, "shape" -> TTuple(TInt32, TInt32))) {
      def blockContext(idx: (Int, Int)): IR = {
        val (i, j) = typ.blockShape(idx._1, idx._2)
        val start = (if (asRowVector) idx._2 else idx._1) * typ.blockSize
        makestruct("start" -> start, "shape" -> MakeTuple.ordered(FastSeq[IR](i.toInt, j.toInt)))
      }

      def blockBody(ctxRef: Ref): IR = {
        bindIR(GetField(ctxRef, "shape")) { shape =>
          val len = if (asRowVector) GetTupleElement(shape, 1) else GetTupleElement(shape, 0)
          val nRep = if (asRowVector) GetTupleElement(shape, 0) else GetTupleElement(shape, 1)
          val start = GetField(ctxRef, "start")
          bindIR(NDArrayReshape(
            NDArraySlice(v, MakeTuple.ordered(FastSeq(MakeTuple.ordered(FastSeq(start.toL, (start + len).toL, 1L))))),
            MakeTuple.ordered(if (asRowVector) FastSeq[IR](1L, len.toL) else FastSeq[IR](len.toL, 1L)), ErrorIDs.NO_ERROR)) { sliced =>
            NDArrayConcat(ToArray(mapIR(rangeIR(nRep))(_ => sliced)), if (asRowVector) 0 else 1)
          }
        }
      }
    }
  }
}

case class EmptyBlockMatrixStage(eltType: Type) extends BlockMatrixStage(IndexedSeq(), Array(), TInt32) {
  def blockContext(idx: (Int, Int)): IR =
    throw new LowererUnsupportedOperation("empty stage has no block contexts!")

  def blockBody(ctxRef: Ref): IR = NA(TNDArray(eltType, Nat(2)))

  override def collectBlocks(relationalBindings: Map[String, IR])(f: (IR, IR) => IR, blocksToCollect: Array[(Int, Int)]): IR = {
    assert(blocksToCollect.isEmpty)
    MakeArray(FastSeq(), TArray(f(Ref("x", ctxType), blockBody(Ref("x", ctxType))).typ))
  }
}

// Scope structure:
// letBindings are available in blockContext and broadcastVals.
// broadcastVals are available in the blockContext and the blockBody
abstract class BlockMatrixStage(val letBindings: IndexedSeq[(String, IR)], val broadcastVals: Array[(String, IR)], val ctxType: Type) {
  def blockContext(idx: (Int, Int)): IR

  def blockBody(ctxRef: Ref): IR

  def wrapLetsAndBroadcasts(ctxIR: IR): IR  = {
    (letBindings ++ broadcastVals).foldRight[IR](ctxIR) { case ((f, v), accum) => Let(f, v, accum) }
  }

  def collectBlocks(relationalBindings: Map[String, IR])(f: (IR, IR) => IR, blocksToCollect: Array[(Int, Int)]): IR = {
    val ctxRef = Ref(genUID(), ctxType)
    val body = f(ctxRef, blockBody(ctxRef))
    val ctxs = MakeStream(blocksToCollect.map(idx => blockContext(idx)), TStream(ctxRef.typ))
    val bodyFreeVars = FreeVariables(body, supportsAgg = false, supportsScan = false)
    val bcFields = broadcastVals.filter { case (f, _) => bodyFreeVars.eval.lookupOption(f).isDefined }
    val bcVals = MakeStruct(bcFields.map { case (f, v) => f -> Ref(f, v.typ) })
    val bcRef = Ref(genUID(), bcVals.typ)
    val wrappedBody = bcFields.foldLeft(body) { case (accum, (f, _)) =>
      Let(f, GetField(bcRef, f), accum)
    }
    val collect = wrapLetsAndBroadcasts(CollectDistributedArray(ctxs, bcVals, ctxRef.name, bcRef.name, wrappedBody))
    LowerToCDA.substLets(collect, relationalBindings)
  }

  def collectLocal(relationalBindings: Map[String, IR], typ: BlockMatrixType): IR = {
    val blocksRowMajor = Array.range(0, typ.nRowBlocks).flatMap { i =>
      Array.tabulate(typ.nColBlocks)(j => i -> j).filter(typ.hasBlock)
    }
    val cda = collectBlocks(relationalBindings)((_, b) => b, blocksRowMajor)
    val blockResults = Ref(genUID(), cda.typ)

    val rows = if (typ.isSparse) {
      val blockMap = blocksRowMajor.zipWithIndex.toMap
      MakeArray(Array.tabulate[IR](typ.nRowBlocks) { i =>
        NDArrayConcat(MakeArray(Array.tabulate[IR](typ.nColBlocks) { j =>
          if (blockMap.contains(i -> j))
            ArrayRef(blockResults, i * typ.nColBlocks + j)
          else {
            val (nRows, nCols) = typ.blockShape(i, j)
            MakeNDArray.fill(zero(typ.elementType), FastIndexedSeq(nRows, nCols), True())
          }
        }, coerce[TArray](cda.typ)), 1)
      }, coerce[TArray](cda.typ))
    } else {
      ToArray(mapIR(rangeIR(I32(typ.nRowBlocks))){ rowIdxRef =>
        val blocksInOneRow = ToArray(mapIR(rangeIR(I32(typ.nColBlocks))) { colIdxRef =>
          ArrayRef(blockResults, rowIdxRef * typ.nColBlocks + colIdxRef)
        })
        NDArrayConcat(blocksInOneRow, 1)
      })
    }

    Let(blockResults.name, cda, NDArrayConcat(rows, 0))
  }

  def addLets(newLets: (String, IR)*): BlockMatrixStage = {
    val outer = this
    new BlockMatrixStage(outer.letBindings ++ newLets, outer.broadcastVals, ctxType) {
      override def blockContext(idx: (Int, Int)): IR = outer.blockContext(idx)
      override def blockBody(ctxRef: Ref): IR = outer.blockBody(ctxRef)
    }
  }

  def addGlobals(newGlobals: (String, IR)*): BlockMatrixStage = {
    val outer = this
    new BlockMatrixStage(outer.letBindings, broadcastVals ++ newGlobals, ctxType) {
      def blockContext(idx: (Int, Int)): IR = outer.blockContext(idx)
      def blockBody(ctxRef: Ref): IR = outer.blockBody(ctxRef)
    }
  }

  def addContext(newTyp: Type)(newCtx: ((Int, Int)) => IR): BlockMatrixStage = {
    val outer = this
    val newCtxType = TStruct("old" -> ctxType, "new" -> newTyp)
    new BlockMatrixStage(outer.letBindings, broadcastVals, newCtxType) {
      def blockContext(idx: (Int, Int)): IR =
        makestruct("old" -> outer.blockContext(idx), "new" -> newCtx(idx))

      def blockBody(ctxRef: Ref): IR = bindIR(GetField(ctxRef, "old"))(outer.blockBody)
    }
  }
  def mapBody(f: (IR, IR) => IR): BlockMatrixStage = {
    val outer = this
    new BlockMatrixStage(outer.letBindings, broadcastVals, outer.ctxType) {
      def blockContext(idx: (Int, Int)): IR = outer.blockContext(idx)

      def blockBody(ctxRef: Ref): IR = f(ctxRef, outer.blockBody(ctxRef))
    }
  }

  def condenseBlocks(typ: BlockMatrixType, rowBlocks: Array[Array[Int]], colBlocks: Array[Array[Int]]): BlockMatrixStage = {
    val outer = this
    val ctxType = TArray(TArray(TTuple(TTuple(TInt64, TInt64), outer.ctxType)))
    new BlockMatrixStage(outer.letBindings, outer.broadcastVals, ctxType) {
      def blockContext(idx: (Int, Int)): IR = {
        val i = idx._1
        val j = idx._2
        MakeArray(rowBlocks(i).map { ii =>
          MakeArray(colBlocks(j).map { jj =>
            val idx2 = ii -> jj
            if (typ.hasBlock(idx2))
              MakeTuple.ordered(FastSeq(NA(TTuple(TInt64, TInt64)), outer.blockContext(idx2)))
            else {
              val (nRows, nCols) = typ.blockShape(ii, jj)
              MakeTuple.ordered(FastSeq(MakeTuple.ordered(FastSeq(nRows, nCols)), NA(outer.ctxType)))
            }
          }: _*)
        }: _*)
      }

      def blockBody(ctxRef: Ref): IR = {
        NDArrayConcat(ToArray(mapIR(ToStream(ctxRef)) { ctxRows =>
          NDArrayConcat(ToArray(mapIR(ToStream(ctxRows)) { shapeOrCtx =>
            bindIR(GetTupleElement(shapeOrCtx, 1)) { ctx =>
              If(IsNA(ctx),
                bindIR(GetTupleElement(shapeOrCtx, 0)) { shape =>
                  MakeNDArray(
                    ToArray(mapIR(
                      rangeIR((GetTupleElement(shape, 0) * GetTupleElement(shape, 1)).toI)
                    )(_ => zero(typ.elementType))),
                    shape, False(), ErrorIDs.NO_ERROR)
                },
                outer.blockBody(ctx))
            }
          }), 1)
        }), 0)
      }
    }
  }
}

object LowerBlockMatrixIR {
  def apply(node: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: Analyses, relationalLetsAbove: Map[String, IR]): IR = {

    def lower(bmir: BlockMatrixIR) = LowerBlockMatrixIR.lower(bmir, typesToLower, ctx, analyses, relationalLetsAbove)

    node match {
      case BlockMatrixCollect(child) =>
        lower(child).collectLocal(relationalLetsAbove, child.typ)
      case BlockMatrixToValueApply(child, GetElement(IndexedSeq(i, j))) =>
        val rowBlock = child.typ.getBlockIdx(i)
        val colBlock = child.typ.getBlockIdx(j)

        val iInBlock = i - rowBlock * child.typ.blockSize
        val jInBlock = j - colBlock * child.typ.blockSize

        val lowered = lower(child)

        val elt = bindIR(lowered.blockContext(rowBlock -> colBlock)) { ctx =>
          NDArrayRef(lowered.blockBody(ctx), FastIndexedSeq(I64(iInBlock), I64(jInBlock)), -1)
        }

        lowered.wrapLetsAndBroadcasts(elt)
      case BlockMatrixWrite(child, writer) =>
        writer.lower(ctx, lower(child), child, relationalLetsAbove, TypeWithRequiredness(child.typ.elementType)) //FIXME: BlockMatrixIR is currently ignored in Requiredness inference since all eltTypes are +TFloat64
      case BlockMatrixMultiWrite(blockMatrices, writer) => unimplemented(ctx, node)
      case node if node.children.exists(_.isInstanceOf[BlockMatrixIR]) =>
        throw new LowererUnsupportedOperation(s"IR nodes with BlockMatrixIR children need explicit rules: \n${ Pretty(ctx, node) }")

      case node =>
        throw new LowererUnsupportedOperation(s"Value IRs with no BlockMatrixIR children must be lowered through LowerIR: \n${ Pretty(ctx, node) }")
    }
  }

  // This lowers a BlockMatrixIR to an unkeyed TableStage with rows of (blockRow, blockCol, block)
  def lowerToTableStage(
    bmir: BlockMatrixIR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext,
    analyses: Analyses, relationalLetsAbove: Map[String, IR]
  ): TableStage = {
    val bms = lower(bmir, typesToLower, ctx, analyses, relationalLetsAbove)
    val typ = bmir.typ
    val bmsWithCtx = bms.addContext(TTuple(TInt32, TInt32)){ case (i, j) => MakeTuple(Seq(0 -> i, 1 -> j))}
    val blocksRowMajor = Array.range(0, typ.nRowBlocks).flatMap { i =>
      Array.tabulate(typ.nColBlocks)(j => i -> j).filter(typ.hasBlock)
    }
    val emptyGlobals = MakeStruct(Seq())
    val globalsId = genUID()
    val letBindings = bmsWithCtx.letBindings ++ bmsWithCtx.broadcastVals :+ globalsId -> emptyGlobals
    val contextsIR = MakeStream(blocksRowMajor.map{ case (i, j) =>  bmsWithCtx.blockContext((i, j)) }, TStream(bmsWithCtx.ctxType))

    val ctxRef = Ref(genUID(), bmsWithCtx.ctxType)
    val body = bmsWithCtx.blockBody(ctxRef)
    val bodyFreeVars = FreeVariables(body, supportsAgg = false, supportsScan = false)
    val bcFields = bmsWithCtx.broadcastVals.filter { case (f, _) => bodyFreeVars.eval.lookupOption(f).isDefined } :+ globalsId -> Ref(globalsId, emptyGlobals.typ)

    def tsPartitionFunction(ctxRef: Ref): IR = {
      val s = MakeStruct(Seq("blockRow" -> GetTupleElement(GetField(ctxRef, "new"), 0), "blockCol" -> GetTupleElement(GetField(ctxRef, "new"), 1), "block" -> bmsWithCtx.blockBody(ctxRef)))
      MakeStream(Seq(
        s
      ), TStream(s.typ))
    }
    val ts = TableStage(letBindings, bcFields, Ref(globalsId, emptyGlobals.typ), RVDPartitioner.unkeyed(blocksRowMajor.size), TableStageDependency.none, contextsIR, tsPartitionFunction)
    ts
  }

  private def unimplemented[T](ctx: ExecuteContext, node: BaseIR): T =
    throw new LowererUnsupportedOperation(s"unimplemented: \n${ Pretty(ctx, node) }")

  def lower(bmir: BlockMatrixIR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: Analyses, relationalLetsAbove: Map[String, IR]): BlockMatrixStage = {
    if (!DArrayLowering.lowerBM(typesToLower))
      throw new LowererUnsupportedOperation("found BlockMatrixIR in lowering; lowering only TableIRs.")
    bmir.children.foreach {
      case c: BlockMatrixIR if c.typ.blockSize != bmir.typ.blockSize =>
        throw new LowererUnsupportedOperation(s"Can't lower node with mismatched block sizes: ${ bmir.typ.blockSize } vs child ${ c.typ.blockSize }\n\n ${ Pretty(ctx, bmir) }")
      case _ =>
    }
    if (bmir.typ.nDefinedBlocks == 0)
      BlockMatrixStage.empty(bmir.typ.elementType)
    else lowerNonEmpty(bmir, typesToLower, ctx, analyses, relationalLetsAbove)
  }

  def lowerNonEmpty(bmir: BlockMatrixIR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: Analyses, relationalLetsAbove: Map[String, IR]): BlockMatrixStage = {
    def lower(ir: BlockMatrixIR) = LowerBlockMatrixIR.lower(ir, typesToLower, ctx, analyses, relationalLetsAbove)

    def lowerIR(node: IR): IR = LowerToCDA.lower(node, typesToLower, ctx, analyses, relationalLetsAbove: Map[String, IR])

    bmir match {
      case BlockMatrixRead(reader) => reader.lower(ctx)
      case x@BlockMatrixRandom(seed, gaussian, shape, blockSize) =>
        val generator = invokeSeeded(if (gaussian) "rand_norm" else "rand_unif", seed, TFloat64, F64(0.0), F64(1.0))
        new BlockMatrixStage(IndexedSeq(), Array(), TTuple(TInt64, TInt64)) {
          def blockContext(idx: (Int, Int)): IR = {
            val (i, j) = x.typ.blockShape(idx._1, idx._2)
            MakeTuple.ordered(FastSeq(i, j))
          }
          def blockBody(ctxRef: Ref): IR = {
            val len = (GetTupleElement(ctxRef, 0) * GetTupleElement(ctxRef, 1)).toI
            MakeNDArray(ToArray(mapIR(rangeIR(len))(_ => generator)), ctxRef, True(), ErrorIDs.NO_ERROR)
          }
        }
      case BlockMatrixMap(child, eltName, f, _) =>
        lower(child).mapBody { (_, body) =>
          NDArrayMap(body, eltName, f)
        }
      case BlockMatrixMap2(left, right, lname, rname, f, sparsityStrategy) =>
        val loweredLeft = lower(left)
        val loweredRight = lower(right)
        loweredLeft
          .addLets(loweredRight.letBindings: _*)
          .addGlobals(loweredRight.broadcastVals: _*)
          .addContext(loweredRight.ctxType)(loweredRight.blockContext).mapBody { (ctx, leftBody) =>
          NDArrayMap2(leftBody, bindIR(GetField(ctx, "new"))(loweredRight.blockBody), lname, rname, f, ErrorIDs.NO_ERROR)
        }

      case x@BlockMatrixBroadcast(child, IndexedSeq(), _, _) =>
        val lowered = lower(child)
        val eltValue = lowered.wrapLetsAndBroadcasts(bindIR(lowered.blockContext(0 -> 0)) { ctx =>
          NDArrayRef(lowered.blockBody(ctx), FastIndexedSeq(I64(0L), I64(0L)), -1)
        })

        val elt = Ref(genUID(), eltValue.typ)
        new BlockMatrixStage(lowered.letBindings, Array(elt.name -> eltValue), TTuple(TInt64, TInt64)) {
          def blockContext(idx: (Int, Int)): IR = {
            val (i, j) = x.typ.blockShape(idx._1, idx._2)
            MakeTuple.ordered(FastSeq(I64(i.toInt), I64(j.toInt)))
          }
          def blockBody(ctxRef: Ref): IR =
            MakeNDArray(ToArray(mapIR(rangeIR(Cast(GetTupleElement(ctxRef, 0) * GetTupleElement(ctxRef, 1), TInt32)))(_ => elt)),
              ctxRef, True(), ErrorIDs.NO_ERROR)
        }
      case x@BlockMatrixBroadcast(child, IndexedSeq(axis), _, _) =>
        val len = child.typ.shape.max
        val vector = NDArrayReshape(lower(child).collectLocal(relationalLetsAbove, child.typ), MakeTuple.ordered(FastSeq(I64(len))), ErrorIDs.NO_ERROR)
        BlockMatrixStage.broadcastVector(vector, x.typ, asRowVector = axis == 1)

      case x@BlockMatrixBroadcast(child, IndexedSeq(axis, axis2), _, _) if (axis == axis2) => // diagonal as row/col vector
        val nBlocks = java.lang.Math.min(child.typ.nRowBlocks, child.typ.nColBlocks)
        val idxs = Array.tabulate(nBlocks) { i => (i, i) }
        val getDiagonal = { (ctx: IR, block: IR) =>
          bindIR(block)(b => ToArray(mapIR(rangeIR(GetField(ctx, "new")))(i => NDArrayRef(b, FastIndexedSeq(Cast(i, TInt64), Cast(i, TInt64)), ErrorIDs.NO_ERROR))))
        }

        val loweredWithContext = lower(child).addContext(TInt32){ case (i, j) => {
          val (rows, cols) = x.typ.blockShape(i, j)
          I32(math.max(rows.toInt, cols.toInt))
        } }

        val vector = bindIR(loweredWithContext.collectBlocks(relationalLetsAbove)(getDiagonal, idxs.filter(child.typ.hasBlock))) { existing =>
          var i = -1
          val vectorBlocks = idxs.map { idx =>
            if (child.typ.hasBlock(idx)) {
              i += 1
              ArrayRef(existing, i)
            } else {
              val (i, j) = child.typ.blockShape(idx._1, idx._2)
              ToArray(mapIR(rangeIR(java.lang.Math.min(i, j)))(_ => zero(x.typ.elementType)))
            }
          }
          MakeNDArray(ToArray(flatten(MakeStream(vectorBlocks, TStream(TArray(x.typ.elementType))))),
            MakeTuple.ordered(FastSeq(I64(java.lang.Math.min(child.typ.nRows, child.typ.nCols)))), true, ErrorIDs.NO_ERROR)
        }
        BlockMatrixStage.broadcastVector(vector, x.typ, asRowVector = axis == 0)

      case BlockMatrixBroadcast(child, IndexedSeq(1, 0), _, _) => //transpose
        val lowered = lower(child)
        new BlockMatrixStage(lowered.letBindings, lowered.broadcastVals, lowered.ctxType) {
          def blockContext(idx: (Int, Int)): IR = lowered.blockContext(idx.swap)
          def blockBody(ctxRef: Ref): IR = NDArrayReindex(lowered.blockBody(ctxRef), FastIndexedSeq(1, 0))
        }
      case BlockMatrixBroadcast(child, IndexedSeq(0, 1), _, _) =>
        lower(child)

      case a@BlockMatrixAgg(child, axesToSumOut) =>
        val loweredChild = lower(child)
        axesToSumOut match {
          case IndexedSeq(0, 1)  =>
            val summedChild = loweredChild.mapBody { (ctx, body) =>
              NDArrayReshape(NDArrayAgg(body, IndexedSeq(0, 1)), MakeTuple.ordered(Seq(I64(1), I64(1))), ErrorIDs.NO_ERROR)
            }
            val summedChildType = BlockMatrixType(child.typ.elementType, IndexedSeq[Long](child.typ.nRowBlocks, child.typ.nColBlocks), child.typ.nRowBlocks == 1, 1, BlockMatrixSparsity.dense)
            val res = NDArrayAgg(summedChild.collectLocal(relationalLetsAbove, summedChildType), IndexedSeq[Int](0, 1))
            new BlockMatrixStage(loweredChild.letBindings, summedChild.broadcastVals, TStruct.empty) {
              override def blockContext(idx: (Int, Int)): IR = makestruct()
              override def blockBody(ctxRef: Ref): IR = NDArrayReshape(res, MakeTuple.ordered(Seq(I64(1L), I64(1L))), ErrorIDs.NO_ERROR)
            }
          case IndexedSeq(0) => { // Number of rows goes to 1. Number of cols remains the same.
            new BlockMatrixStage(loweredChild.letBindings, loweredChild.broadcastVals, TArray(loweredChild.ctxType)) {
              override def blockContext(idx: (Int, Int)): IR = {
                val (row, col) = idx
                assert(row == 0, s"Asked for idx ${idx}")
                MakeArray(
                  (0 until child.typ.nRowBlocks).map(childRow => loweredChild.blockContext((childRow, col))),
                  TArray(loweredChild.ctxType)
                )
              }
              override def blockBody(ctxRef: Ref): IR = {
                val summedChildBlocks = mapIR(ToStream(ctxRef))(singleChildCtx => {
                  bindIR(NDArrayAgg(loweredChild.blockBody(singleChildCtx), axesToSumOut))(aggedND => NDArrayReshape(aggedND, MakeTuple.ordered(Seq(I64(1), GetTupleElement(NDArrayShape(aggedND), 0))), ErrorIDs.NO_ERROR))
                })
                val aggVar = genUID()
                StreamAgg(summedChildBlocks, aggVar, ApplyAggOp(NDArraySum())(Ref(aggVar, summedChildBlocks.typ.asInstanceOf[TStream].elementType)))
              }
            }
          }
          case IndexedSeq(1) => { // Number of cols goes to 1. Number of rows remains the same.
            new BlockMatrixStage(loweredChild.letBindings, loweredChild.broadcastVals, TArray(loweredChild.ctxType)) {
              override def blockContext(idx: (Int, Int)): IR = {
                val (row, col) = idx
                assert(col == 0, s"Asked for idx ${idx}")
                MakeArray(
                  (0 until child.typ.nColBlocks).map(childCol => loweredChild.blockContext((row, childCol))),
                  TArray(loweredChild.ctxType)
                )
              }
              override def blockBody(ctxRef: Ref): IR = {
                val summedChildBlocks = mapIR(ToStream(ctxRef))(singleChildCtx => {
                  bindIR(NDArrayAgg(loweredChild.blockBody(singleChildCtx), axesToSumOut))(aggedND => NDArrayReshape(aggedND, MakeTuple(Seq((0, GetTupleElement(NDArrayShape(aggedND), 0)), (1, I64(1)))), ErrorIDs.NO_ERROR))
                })
                val aggVar = genUID()
                StreamAgg(summedChildBlocks, aggVar, ApplyAggOp(NDArraySum())(Ref(aggVar, summedChildBlocks.typ.asInstanceOf[TStream].elementType)))
              }
            }
          }
        }

      case x@BlockMatrixFilter(child, keep) =>
        val rowDependents = x.rowBlockDependents
        val colDependents = x.colBlockDependents

        lower(child).condenseBlocks(child.typ, rowDependents, colDependents)
          .addContext(TStruct("rows" -> TArray(TInt64), "cols" -> TArray(TInt64))) { idx =>
            val i = idx._1
            val j = idx._2
            val rowStartIdx = rowDependents(i).head.toLong * x.typ.blockSize
            val colStartIdx = colDependents(j).head.toLong * x.typ.blockSize
            val rows = if (keep(0).isEmpty) null else x.keepRowPartitioned(i).map(k => k - rowStartIdx).toFastIndexedSeq
            val cols = if (keep(1).isEmpty) null else x.keepColPartitioned(j).map(k => k - colStartIdx).toFastIndexedSeq
            makestruct("rows" -> Literal.coerce(TArray(TInt64), rows), "cols" -> Literal.coerce(TArray(TInt64), cols))
          }.mapBody { (ctx, body) =>
          bindIR(GetField(GetField(ctx, "new"), "rows")) { rows =>
            bindIR(GetField(GetField(ctx, "new"), "cols")) { cols =>
              NDArrayFilter(body, FastIndexedSeq(rows, cols))
            }
          }
        }
      case x@BlockMatrixSlice(child, IndexedSeq(IndexedSeq(rStart, rEnd, rStep), IndexedSeq(cStart, cEnd, cStep))) =>
        val rowDependents = x.rowBlockDependents
        val colDependents = x.colBlockDependents

        lower(child).condenseBlocks(child.typ, rowDependents, colDependents)
          .addContext(TTuple(TTuple(TInt64, TInt64, TInt64), TTuple(TInt64, TInt64, TInt64))) { idx =>
            val (i, j) = idx

            // Aligned with the edges of blocks in child BM.
            val blockAlignedRowStartIdx = rowDependents(i).head.toLong * x.typ.blockSize
            val blockAlignedColStartIdx = colDependents(j).head.toLong * x.typ.blockSize
            val blockAlignedRowEndIdx = math.min(child.typ.nRows, (rowDependents(i).last + 1L) * x.typ.blockSize * rStep)
            val blockAlignedColEndIdx = math.min(child.typ.nCols, (colDependents(j).last + 1L) * x.typ.blockSize * cStep)

            // condenseBlocks can give the same data to multiple partitions. Need to make sure we don't use data
            // that's already included in an earlier block.
            val rStartPlusSeenAlready = rStart + i * x.typ.blockSize * rStep
            val cStartPlusSeenAlready = cStart + j * x.typ.blockSize * cStep

            val rowTrueStart = rStartPlusSeenAlready - blockAlignedRowStartIdx
            val rowTrueEnd = math.min(math.min(rEnd, blockAlignedRowEndIdx) - blockAlignedRowStartIdx, rowTrueStart + x.typ.blockSize * rStep)
            val rows = MakeTuple.ordered(FastSeq[IR](
              rowTrueStart,
              rowTrueEnd,
              rStep))

            val colTrueStart = cStartPlusSeenAlready - blockAlignedColStartIdx
            val colTrueEnd = math.min(java.lang.Math.min(cEnd, blockAlignedColEndIdx) - blockAlignedColStartIdx, colTrueStart + x.typ.blockSize * cStep)
            val cols = MakeTuple.ordered(FastSeq[IR](
              colTrueStart,
              colTrueEnd,
              cStep))
            MakeTuple.ordered(FastSeq(rows, cols))
          }.mapBody { (ctx, body) => NDArraySlice(body, GetField(ctx, "new")) }

      // Both densify and sparsify change the sparsity pattern tracked on the BlockMatrixType.
      case BlockMatrixDensify(child) => lower(child)
      case BlockMatrixSparsify(child, sparsifier) => lower(child)

      case RelationalLetBlockMatrix(name, value, body) => unimplemented(ctx, bmir)

      case ValueToBlockMatrix(child, shape, blockSize) if !child.typ.isInstanceOf[TArray] && !child.typ.isInstanceOf[TNDArray] => {
        val element = lowerIR(child)
        new BlockMatrixStage(IndexedSeq(), Array(), TStruct()) {
          override def blockContext(idx: (Int, Int)): IR = MakeStruct(Seq())

          override def blockBody(ctxRef: Ref): IR = MakeNDArray(MakeArray(element), MakeTuple(Seq((0, I64(1)), (1, I64(1)))), False(), ErrorIDs.NO_ERROR)
        }
      }
      case x@ValueToBlockMatrix(child, _, blockSize) =>
        val nd = child.typ match {
          case _: TArray => MakeNDArray(lowerIR(child), MakeTuple.ordered(FastSeq(I64(x.typ.nRows), I64(x.typ.nCols))), True(), ErrorIDs.NO_ERROR)
          case _: TNDArray => lowerIR(child)
        }
        val v = Ref(genUID(), nd.typ)
        new BlockMatrixStage(IndexedSeq(v.name -> nd), Array(), nd.typ) {
          def blockContext(idx: (Int, Int)): IR = {
            val (r, c) = idx
            NDArraySlice(v, MakeTuple.ordered(FastSeq(
              MakeTuple.ordered(FastSeq(I64(r.toLong * blockSize), I64(java.lang.Math.min((r.toLong + 1) * blockSize, x.typ.nRows)), I64(1))),
              MakeTuple.ordered(FastSeq(I64(c.toLong * blockSize), I64(java.lang.Math.min((c.toLong + 1) * blockSize, x.typ.nCols)), I64(1))))))
          }

          def blockBody(ctxRef: Ref): IR = ctxRef
        }
      case x@BlockMatrixDot(leftIR, rightIR) =>
        val left = lower(leftIR)
        val right = lower(rightIR)
        val newCtxType = TArray(TTuple(left.ctxType, right.ctxType))
        new BlockMatrixStage(left.letBindings ++ right.letBindings, left.broadcastVals ++ right.broadcastVals, newCtxType) {
          def blockContext(idx: (Int, Int)): IR = {
            val (i, j) = idx
            MakeArray(Array.tabulate[Option[IR]](leftIR.typ.nColBlocks) { k =>
              if (leftIR.typ.hasBlock(i -> k) && rightIR.typ.hasBlock(k -> j))
                Some(MakeTuple.ordered(FastSeq(
                  left.blockContext(i -> k), right.blockContext(k -> j))))
              else None
            }.flatten[IR], newCtxType)
          }

          def blockBody(ctxRef: Ref): IR = {
            val tupleNDArrayStream = ToStream(ctxRef)
            val streamElementName = genUID()
            val streamElementRef = Ref(streamElementName, tupleNDArrayStream.typ.asInstanceOf[TStream].elementType)
            val leftName = genUID()
            val rightName = genUID()
            val leftRef = Ref(leftName, tupleNDArrayStream.typ.asInstanceOf[TStream].elementType.asInstanceOf[TTuple].types(0))
            val rightRef = Ref(rightName, tupleNDArrayStream.typ.asInstanceOf[TStream].elementType.asInstanceOf[TTuple].types(1))
            StreamAgg(tupleNDArrayStream, streamElementName, {
              AggLet(leftName, GetTupleElement(streamElementRef, 0),
                AggLet(rightName, GetTupleElement(streamElementRef, 1),
              ApplyAggOp(NDArrayMultiplyAdd())(left.blockBody(leftRef),
                right.blockBody(rightRef)), isScan=false), isScan=false)
            })
          }
        }
    }
  }
}
