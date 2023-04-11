package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.Nat
import is.hail.expr.ir._
import is.hail.expr.ir.functions.GetElement
import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual._
import is.hail.types.{BlockMatrixSparsity, BlockMatrixType, TypeWithRequiredness, tcoerce}
import is.hail.utils._

object BlockMatrixStage {
  def empty(eltType: Type): BlockMatrixStage =
    EmptyBlockMatrixStage(eltType)
}

case class EmptyBlockMatrixStage(eltType: Type) extends BlockMatrixStage(FastIndexedSeq(), TInt32) {
  def blockContext(idx: (Int, Int)): IR =
    throw new LowererUnsupportedOperation("empty stage has no block contexts!")

  def blockBody(ctxRef: Ref): IR = NA(TNDArray(eltType, Nat(2)))

  override def collectBlocks(staticID: String, dynamicID: IR = NA(TString))(f: (IR, IR) => IR, blocksToCollect: Array[(Int, Int)]): IR = {
    assert(blocksToCollect.isEmpty)
    MakeArray(FastSeq(), TArray(f(Ref("x", ctxType), blockBody(Ref("x", ctxType))).typ))
  }
}

abstract class BlockMatrixStage(val broadcastVals: IndexedSeq[Ref], val ctxType: Type) {
  def blockContext(idx: (Int, Int)): IR

  def blockBody(ctxRef: Ref): IR

  def collectBlocks(staticID: String, dynamicID: IR = NA(TString))(f: (IR, IR) => IR, blocksToCollect: Array[(Int, Int)]): IR = {
    val ctxRef = Ref(genUID(), ctxType)
    val body = f(ctxRef, blockBody(ctxRef))
    val ctxs = MakeStream(blocksToCollect.map(idx => blockContext(idx)), TStream(ctxRef.typ))
    val bodyFreeVars = FreeVariables(body, supportsAgg = false, supportsScan = false)
    val bcFields = broadcastVals.filter { ref => bodyFreeVars.eval.lookupOption(ref.name).isDefined }
    val bcVals = MakeStruct(bcFields.map { ref => ref.name -> ref })
    val bcRef = Ref(genUID(), bcVals.typ)
    val wrappedBody = bcFields.foldLeft(body) { case (accum, Ref(f, _)) =>
      Let(f, GetField(bcRef, f), accum)
    }
    CollectDistributedArray(ctxs, bcVals, ctxRef.name, bcRef.name, wrappedBody, dynamicID, staticID)
  }

  def collectLocal(typ: BlockMatrixType, staticID: String, dynamicID: IR = NA(TString)): IR = {
    val blocksRowMajor = Array.range(0, typ.nRowBlocks).flatMap { i =>
      Array.tabulate(typ.nColBlocks)(j => i -> j).filter(typ.hasBlock)
    }
    val cda = collectBlocks(staticID, dynamicID)((_, b) => b, blocksRowMajor)
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
        }, tcoerce[TArray](cda.typ)), 1)
      }, tcoerce[TArray](cda.typ))
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

  def addContext(newTyp: Type)(newCtx: ((Int, Int)) => IR): BlockMatrixStage = {
    val outer = this
    val newCtxType = TStruct("old" -> ctxType, "new" -> newTyp)
    new BlockMatrixStage(broadcastVals, newCtxType) {
      def blockContext(idx: (Int, Int)): IR =
        makestruct("old" -> outer.blockContext(idx), "new" -> newCtx(idx))

      def blockBody(ctxRef: Ref): IR = bindIR(GetField(ctxRef, "old"))(outer.blockBody)
    }
  }
  def mapBody(f: (IR, IR) => IR): BlockMatrixStage = {
    val outer = this
    new BlockMatrixStage(broadcastVals, outer.ctxType) {
      def blockContext(idx: (Int, Int)): IR = outer.blockContext(idx)

      def blockBody(ctxRef: Ref): IR = f(ctxRef, outer.blockBody(ctxRef))
    }
  }

  def condenseBlocks(typ: BlockMatrixType, rowBlocks: Array[Array[Int]], colBlocks: Array[Array[Int]]): BlockMatrixStage = {
    val outer = this
    val ctxType = TArray(TArray(TTuple(TTuple(TInt64, TInt64), outer.ctxType)))
    new BlockMatrixStage(outer.broadcastVals, ctxType) {
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

object BlockMatrixStage2 {
  def fromOldBMS(bms: BlockMatrixStage, typ: BlockMatrixType, ib: IRBuilder): BlockMatrixStage2 = {
    val blocks = typ.allBlocksColMajor
    val ctxsArray = MakeArray(blocks.map(idx => bms.blockContext(idx)), TArray(bms.ctxType))

    BlockMatrixStage2(bms.broadcastVals, typ, BMSContexts(typ, ctxsArray, ib), bms.blockBody)
  }

  def empty(eltType: Type, ib: IRBuilder) = {
    val ctxType = TNDArray(eltType, Nat(2))
    BlockMatrixStage2(
      IndexedSeq(),
      BlockMatrixType.dense(eltType, 0, 0, 0),
      DenseContexts(0, 0, ib.memoize(MakeArray(FastIndexedSeq(), TArray(ctxType)))),
      _ => NA(ctxType))
  }

  def broadcastVector(vector: IR, ib: IRBuilder, typ: BlockMatrixType, asRowVector: Boolean): BlockMatrixStage2 = {
    val v: Ref = ib.strictMemoize(vector)
    val contexts = BMSContexts.tabulate(typ, ib) { (i, j) =>
      val (m, n) = typ.blockShapeIR(i, j)
      val start = (if (asRowVector) j.toL else i.toL) * typ.blockSize.toLong
      makestruct("start" -> start, "shape" -> MakeTuple.ordered(FastSeq[IR](m, n)))
    }
    BlockMatrixStage2(
      FastIndexedSeq(v),
      typ,
      contexts,
      ctx => {
        bindIRs(GetField(ctx, "shape"), GetField(ctx, "start")) { case Seq(shape, start) =>
          bindIRs(
            if (asRowVector) GetTupleElement(shape, 1) else GetTupleElement(shape, 0),
            if (asRowVector) GetTupleElement(shape, 0) else GetTupleElement(shape, 1)
          ) { case Seq(len, nRep) =>
            bindIR(
              NDArrayReshape(
                NDArraySlice(v, maketuple(maketuple(start, start + len, 1L))),
                if (asRowVector) maketuple(1L, len) else maketuple(len.toL, 1L),
                ErrorIDs.NO_ERROR)
            ) { sliced =>
              NDArrayConcat(ToArray(mapIR(rangeIR(nRep.toI))(_ => sliced)), if (asRowVector) 0 else 1)
            }
          }
        }
      })
    }

  def apply(
    broadcastVals: IndexedSeq[Ref],
    typ: BlockMatrixType,
    contexts: BMSContexts,
    _blockIR: Ref => IR
  ): BlockMatrixStage2 = {
    val ctxRef = Ref(genUID(), contexts.elementType)
    val blockIR = _blockIR(ctxRef)

    new BlockMatrixStage2(broadcastVals, typ, contexts, ctxRef.name, blockIR)
  }
}

object BMSContexts {
  def apply(typ: BlockMatrixType, contexts: IR, ib: IRBuilder): BMSContexts = {
    if (typ.isSparse) {
      val (rowPos, rowIdx) = typ.sparsity.definedBlocksCSCIR(typ.nColBlocks).get
      val rowPosRef = ib.memoize(rowPos)
      val rowIdxRef = ib.memoize(rowIdx)
      SparseContexts(typ.nRowBlocks, typ.nColBlocks, rowPosRef, rowIdxRef, ib.memoize(contexts))
    } else {
      DenseContexts(typ.nRowBlocks, typ.nColBlocks, ib.memoize(contexts))
    }
  }

  def tabulate(typ: BlockMatrixType, ib: IRBuilder)(f: (IR, IR) => IR): BMSContexts = {
    val contexts = ib.memoize(ToArray(mapIR(typ.sparsity.allBlocksColMajorIR(typ.nRowBlocks, typ.nColBlocks)) { coords =>
      bindIRs(GetTupleElement(coords, 0), GetTupleElement(coords, 1)) { case Seq(i, j) =>
        f(i, j)
      }
    }))
    typ.sparsity.definedBlocksCSCIR(typ.nColBlocks) match {
      case Some((pos, idx)) =>
        SparseContexts(typ.nRowBlocks, typ.nColBlocks, ib.memoize(pos), ib.memoize(idx), contexts)
      case None =>
        DenseContexts(typ.nRowBlocks, typ.nColBlocks, contexts)
    }
  }

  def transpose(contexts: BMSContexts, ib: IRBuilder, typ: BlockMatrixType): BMSContexts = contexts match {
    case dense: DenseContexts => DenseContexts(
      dense.nCols, dense.nRows,
      ib.memoize(ToArray(flatMapIR(rangeIR(dense.nRows)) { i =>
        mapIR(rangeIR(dense.nCols)) { j =>
          ArrayRef(dense.contexts, (j * dense.nRows) + i)
        }
      })))
    case sparse: SparseContexts =>
      val (staticRowPos, staticRowIdx) = typ.sparsity.definedBlocksCSC(typ.nColBlocks).get
      val (newRowPos, newRowIdx, newToOldPos) =
        BlockMatrixSparsity.transposeCSCSparsityIR(typ.nRowBlocks, typ.nColBlocks, staticRowPos, staticRowIdx)
      val newContexts = ToArray(mapIR(ToStream(newToOldPos)) { oldPos =>
        ArrayRef(contexts.contexts, oldPos)
      })
      SparseContexts(sparse.nCols, sparse.nRows, ib.memoize(newRowPos), ib.memoize(newRowIdx), ib.memoize(newContexts))
  }
}

abstract class BMSContexts {
  def elementType: Type

  def nRows: IR

  def nCols: IR

  def contexts: IR

  def irValue: IR

  def apply(row: IR, col: IR): IR

  // body args: (rowIdx, colIdx, position, old context)
  def map(ib: IRBuilder)(f: (IR, IR, IR, IR) => IR): BMSContexts

  def zip(other: BMSContexts, ib: IRBuilder): BMSContexts

  def grouped(rowDeps: IndexedSeq[IndexedSeq[Int]], colDeps: IndexedSeq[IndexedSeq[Int]], typ: BlockMatrixType, ib: IRBuilder): BMSContexts

  def collect(makeBlock: (Ref, Ref, Ref) => IR): IR

  def print(ctx: ExecuteContext): Unit
}

object DenseContexts {
  def apply(irValue: IR, ib: IRBuilder): DenseContexts = {
    val irValueRef = ib.memoize(irValue)
    DenseContexts(
      ib.memoize(GetField(irValueRef, "nRows")),
      ib.memoize(GetField(irValueRef, "nCols")),
      ib.memoize(GetField(irValueRef, "contexts")),
    )
  }
}

case class DenseContexts(nRows: TrivialIR, nCols: TrivialIR, contexts: TrivialIR) extends BMSContexts {
  val elementType = contexts.typ.asInstanceOf[TArray].elementType

  def irValue: IR = makestruct("nRows" -> nRows, "nCols" -> nCols, "contexts" -> contexts)

  def print(ctx: ExecuteContext): Unit = {
    println(s"DenseContexts:\n  nRows = ${Pretty(ctx, nRows)}\n  nCols = ${Pretty(ctx, nCols)}\n  contexts = ${Pretty(ctx, contexts)}")
  }

  def apply(row: IR, col: IR): IR = ArrayRef(contexts, (col * nRows) + row)

  def map(ib: IRBuilder)(f: (IR, IR, IR, IR) => IR): DenseContexts = {
    DenseContexts(nRows, nCols,
      ib.memoize(ToArray(flatMapIR(rangeIR(nCols)) { j =>
        mapIR(rangeIR(nRows)) { i =>
          bindIR((j * nRows) + i) { pos =>
            bindIR(ArrayRef(contexts, pos)) { old =>
              f(i, j, pos, old)
            }
          }
        }
      })))
  }

  def zip(other: BMSContexts, ib: IRBuilder): BMSContexts = {
    val newContexts = ib.memoize(ToArray(zip2(
      ToStream(this.contexts), ToStream(other.contexts), ArrayZipBehavior.AssertSameLength
    ) { (l, r) => maketuple(l, r) }))
    DenseContexts(nRows, nCols, newContexts)
  }

  def sparsify(rowPos: TrivialIR, rowIdx: TrivialIR, ib: IRBuilder): BMSContexts = {
    val newContexts = ib.memoize(ToArray(flatMapIR(rangeIR(nCols)) { j =>
      bindIRs(ArrayRef(rowPos, j), ArrayRef(rowPos, j + 1), j * nRows) { case Seq(start, end, basePos) =>
        mapIR(rangeIR(start, end)) { pos =>
          bindIR(ArrayRef(rowIdx, pos)) { i =>
            ArrayRef(contexts, basePos + i)
          }
        }
      }
    }))
    SparseContexts(nRows, nCols, rowPos, rowIdx, newContexts)
  }

  def grouped(rowDeps: IndexedSeq[IndexedSeq[Int]], colDeps: IndexedSeq[IndexedSeq[Int]], typ: BlockMatrixType, ib: IRBuilder): DenseContexts = {
    val rowDepsLit = Literal(TArray(TArray(TInt32)), rowDeps)
    val colDepsLit = Literal(TArray(TArray(TInt32)), colDeps)
    assert(rowDeps.nonEmpty || colDeps.nonEmpty)
    if (rowDeps.isEmpty) {
      val newContexts = ToArray(flatMapIR(ToStream(colDepsLit)) { localColDeps =>
        mapIR(rangeIR(nRows)) { i =>
          IRBuilder.scoped { ib =>
            val localContexts = ToArray(mapIR(ToStream(localColDeps)) { jl =>
              this (i, jl)
            })
            DenseContexts(1, ib.memoize(ArrayLen(localColDeps)), ib.memoize(localContexts)).irValue
          }
        }
      })
      return DenseContexts(nRows, colDeps.length, ib.memoize(newContexts))
    }
    if (colDeps.isEmpty) {
      val newContexts = ToArray(flatMapIR(rangeIR(nCols)) { j =>
        mapIR(ToStream(rowDepsLit)) { localRowDeps =>
          IRBuilder.scoped { ib =>
            val localContexts = ToArray(mapIR(ToStream(localRowDeps)) { il =>
              this (il, j)
            })
            DenseContexts(ib.memoize(ArrayLen(localRowDeps)), 1, ib.memoize(localContexts)).irValue
          }
        }
      })
      return DenseContexts(rowDeps.length, nCols, ib.memoize(newContexts))
    }
    val newContexts = ToArray(flatMapIR(ToStream(colDepsLit)) { localColDeps =>
      mapIR(ToStream(rowDepsLit)) { localRowDeps =>
        IRBuilder.scoped { ib =>
          val localContexts = ToArray(flatMapIR(ToStream(localColDeps)) { jl =>
            mapIR(ToStream(localRowDeps)) { il =>
              this(il, jl)
            }
          })
          DenseContexts(ib.memoize(ArrayLen(localRowDeps)), ib.memoize(ArrayLen(localColDeps)), ib.memoize(localContexts)).irValue
        }
      }
    })
    DenseContexts(rowDeps.length, colDeps.length, ib.memoize(newContexts))
  }

  def collect(makeBlock: (Ref, Ref, Ref) => IR): IR = {
    NDArrayConcat(ToArray(mapIR(rangeIR(nCols)) { j =>
      val colBlocks = mapIR(rangeIR(nRows)) { i =>
        bindIR(ArrayRef(contexts, j * nRows + i)) { ctx =>
          makeBlock(i, j, ctx)
        }
      }
      NDArrayConcat(ToArray(colBlocks), 0)
    }), 1)
  }
}

object SparseContexts {
  def apply(irValue: IR, ib: IRBuilder): SparseContexts = {
    val irValueRef = ib.memoize(irValue)
    SparseContexts(
      ib.memoize(GetField(irValueRef, "nRows")),
      ib.memoize(GetField(irValueRef, "nCols")),
      ib.memoize(GetField(irValueRef, "rowPos")),
      ib.memoize(GetField(irValueRef, "rowIdx")),
      ib.memoize(GetField(irValueRef, "contexts")),
    )
  }
}

case class SparseContexts(nRows: TrivialIR, nCols: TrivialIR, rowPos: TrivialIR, rowIdx: TrivialIR, contexts: TrivialIR) extends BMSContexts {
  val elementType = contexts.typ.asInstanceOf[TArray].elementType

  def irValue: IR = makestruct("nRows" -> nRows, "nCols" -> nCols, "contexts" -> contexts)

  def print(ctx: ExecuteContext): Unit = {
    println(s"SparseContexts:\n  nRows = ${ Pretty(ctx, nRows) }\n  nCols = ${ Pretty(ctx, nCols) }\n  contexts = ${ Pretty(ctx, contexts) }")
  }

  def apply(row: IR, col: IR): IR = {
    val startPos = ArrayRef(rowPos, col)
    val endPos = ArrayRef(rowPos, col + 1)
    bindIR(
      Apply("lowerBound", Seq(), FastSeq(rowIdx, row, startPos, endPos), TInt32, ErrorIDs.NO_ERROR)
    ) { pos =>
      If(ArrayRef(rowIdx, pos).ceq(row),
        ArrayRef(contexts, pos),
        Die(strConcat("Internal Error, tried to load missing BlockMatrix context: (row = ", row, ", col = ",
          col, ", pos = ", pos, ", rowPos = ", rowPos, ", rowIdx = ", rowIdx, ")"),
          elementType,
          ErrorIDs.NO_ERROR))
    }
  }

  def map(ib: IRBuilder)(f: (IR, IR, IR, IR) => IR): SparseContexts = {
    SparseContexts(nRows, nCols, rowPos, rowIdx,
      ib.memoize(ToArray(flatMapIR(rangeIR(nCols)) { j =>
        bindIRs(ArrayRef(rowPos, j), ArrayRef(rowPos, j + 1)) { case Seq(start, end) =>
          mapIR(rangeIR(start, end)) { pos =>
            bindIRs(ArrayRef(rowIdx, pos), ArrayRef(contexts, pos)) { case Seq(i, old) =>
              f(i, j, pos, old)
            }
          }
        }
      })))
  }

  def mapDense(ib: IRBuilder)(f: (IR, IR, IR) => IR): DenseContexts = {
    val newContexts = ib.memoize(flatMapIR(rangeIR(nCols)) { j =>
      bindIRs(ArrayRef(rowPos, j), ArrayRef(rowPos, j + 1)) { case Seq(start, end) =>
        val allIdxs = mapIR(rangeIR(nRows)) { i => makestruct("idx" -> i) }
        val idxedExisting = mapIR(rangeIR(start, end)) { pos =>
          makestruct("idx" -> ArrayRef(rowIdx, pos), "context" -> ArrayRef(contexts, pos))
        }
        joinRightDistinctIR(allIdxs, idxedExisting, FastIndexedSeq("idx"), FastIndexedSeq("idx"), "left") { (idx, struct) =>
          val i = GetField(idx, "idx")
          val context = GetField(struct, "context")
          f(i, j, context)
        }
      }
    })
    DenseContexts(nRows, nCols, newContexts)
  }

  def mapWithNewSparsity(newRowPos: TrivialIR, newRowIdx: TrivialIR, ib: IRBuilder)(f: (IR, IR, IR) => IR): SparseContexts = {
    val newContexts = ib.memoize(ToArray(flatMapIR(rangeIR(nCols)) { j =>
      bindIRs(
        ArrayRef(rowPos, j), ArrayRef(rowPos, j + 1),
        ArrayRef(newRowPos, j), ArrayRef(newRowPos, j + 1)
      ) { case Seq(oldStart, oldEnd, newStart, newEnd) =>
        val newIdxs = mapIR(rangeIR(newStart, newEnd)) { pos =>
          makestruct("idx" -> ArrayRef(newRowIdx, pos))
        }
        val idxedExisting = mapIR(rangeIR(oldStart, oldEnd)) { pos =>
          makestruct("idx" -> ArrayRef(rowIdx, pos), "context" -> ArrayRef(contexts, pos))
        }
        joinRightDistinctIR(newIdxs, idxedExisting, FastIndexedSeq("idx"), FastIndexedSeq("idx"), "left") { (idx, struct) =>
          val i = GetField(idx, "idx")
          val context = GetField(struct, "context")
          f(i, j, context)
        }
      }
    }))
    SparseContexts(nRows, nCols, newRowPos, newRowIdx, newContexts)
  }

  def zip(other: BMSContexts, ib: IRBuilder): BMSContexts = {
    val newContexts = ib.memoize(zip2(
      this.contexts, other.contexts, ArrayZipBehavior.AssertSameLength
    ) { (l, r) => maketuple(l, r) })
    SparseContexts(nRows, nCols, rowPos, rowIdx, newContexts)
  }

  def grouped(rowDeps: IndexedSeq[IndexedSeq[Int]], colDeps: IndexedSeq[IndexedSeq[Int]], typ: BlockMatrixType, ib: IRBuilder): SparseContexts = {
    val newNRows = rowDeps.length
    val newNCols = colDeps.length
    val rowBlockSizes = Literal(TArray(TInt32), rowDeps.map(_.length))
    val colBlockSizes = Literal(TArray(TInt32), colDeps.map(_.length))
    val (staticRowPos, staticRowIdx) = typ.sparsity.definedBlocksCSC(typ.nColBlocks).get
    val (newRowPos, newRowIdx, nestedSparsities) =
      BlockMatrixSparsity.groupedCSCSparsityIR(staticRowPos, staticRowIdx, rowDeps, colDeps)
    val newContexts = ToArray(flatMapIR(rangeIR(newNCols)) { j =>
      bindIRs(ArrayRef(newRowPos, j), ArrayRef(newRowPos, j + 1)) { case Seq(start, end) =>
        mapIR(rangeIR(start, end)) { pos =>
          IRBuilder.scoped { ib =>
            val i = ib.memoize(ArrayRef(newRowIdx, pos))
            val nested = ib.memoize(ArrayRef(nestedSparsities, pos))
            val nRows = ib.memoize(ArrayRef(rowBlockSizes, i))
            val nCols = ib.memoize(ArrayRef(colBlockSizes, j))
            val nestedRowPos = ib.memoize(GetTupleElement(nested, 0))
            val nestedRowIdx = ib.memoize(GetTupleElement(nested, 1))
            val nestedContexts = ib.memoize(mapIR(ToStream(GetTupleElement(nested, 2))) { oldPos =>
              ArrayRef(contexts, oldPos)
            })
            SparseContexts(nRows, nCols, nestedRowPos, nestedRowIdx, nestedContexts).irValue
          }
        }
      }
    })

    SparseContexts(newNRows, newNCols, ib.memoize(newRowPos), ib.memoize(newRowIdx), ib.memoize(newContexts))
  }

  def collect(makeBlock: (Ref, Ref, Ref) => IR): IR = {
    NDArrayConcat(ToArray(mapIR(rangeIR(nCols)) { j =>
      val allIdxs = mapIR(rangeIR(nRows)) { i => makestruct("idx" -> i) }
      val startPos = ArrayRef(rowPos, j)
      val endPos = ArrayRef(rowPos, j + 1)
      val idxedExisting = mapIR(rangeIR(startPos, endPos)) { pos =>
        makestruct("idx" -> ArrayRef(rowIdx, pos), "ctx" -> ArrayRef(contexts, pos))
      }
      val colBlocks = joinRightDistinctIR(allIdxs, idxedExisting, FastIndexedSeq("idx"), FastIndexedSeq("idx"), "left") { (l, struct) =>
        bindIRs(GetField(l, "idx"), GetField(struct, "ctx")) { case Seq(i, ctx) =>
          makeBlock(i, j, ctx)
        }
      }
      NDArrayConcat(ToArray(colBlocks), 0)
    }), 1)
  }
}

class BlockMatrixStage2 private (
  val broadcastVals: IndexedSeq[Ref],
  val typ: BlockMatrixType,
  val contexts: BMSContexts,
  private val ctxRefName: String,
  private val _blockIR: IR
) {
  assert {
    def literalOrRef(x: IR) = x.isInstanceOf[Literal] || x.isInstanceOf[Ref]
    contexts.contexts match {
      case x: MakeStruct => x.fields.forall(f => literalOrRef(f._2))
      case x => literalOrRef(x)
    }
  }

  def print(ctx: ExecuteContext): Unit = {
    println(s"contexts:\n${contexts.print(ctx)}\nbody(${ctxRefName}) = ${Pretty(ctx, _blockIR)}")
  }

  def blockIR(ctx: Ref): IR = {
    if (ctx.name == ctxRefName)
      _blockIR
    else
      Let(ctxRefName, ctx, _blockIR)
  }

  def ctxType: Type = contexts.elementType

  def ctxRef: IR = Ref(ctxRefName, ctxType)

  def toOldBMS: BlockMatrixStage = {
    new BlockMatrixStage(broadcastVals, ctxType) {
      override def blockContext(idx: (Int, Int)): IR = contexts(idx._1, idx._2)

      override def blockBody(ctxRef: Ref): IR =
        Let(ctxRefName, ctxRef, _blockIR)
    }
  }

  def getBlock(i: IR, j: IR): IR = Let(ctxRefName, contexts(i, j), _blockIR)

  def getElement(i: IR, j: IR): IR = {
    assert(i.typ == TInt64)
    assert(j.typ == TInt64)
    val blockSize = typ.blockSize.toLong
    bindIR(i floorDiv blockSize) { rowBlock =>
      bindIR(j floorDiv blockSize) { colBlock =>
        val iInBlock = i - rowBlock * blockSize
        val jInBlock = j - colBlock * blockSize

        NDArrayRef(getBlock(rowBlock.toI, colBlock.toI), FastIndexedSeq(iInBlock, jInBlock), -1)
      }
    }
  }

  def transposed(ib: IRBuilder): BlockMatrixStage2 = {
    val newBlockIR = NDArrayReindex(_blockIR, FastIndexedSeq(1, 0))
    new BlockMatrixStage2(broadcastVals, typ.transpose, BMSContexts.transpose(contexts, ib, typ), ctxRefName, newBlockIR)
  }

  def densify(ib: IRBuilder): BlockMatrixStage2 = contexts match {
    case _: DenseContexts => this
    case contexts: SparseContexts =>
      val newContexts = contexts.mapDense(ib) { (i, j, oldContext) =>
        val (m, n) = typ.blockShapeIR(i, j)
        makestruct("oldContext" -> oldContext, "nRows" -> m, "nCols" -> n)
      }
      def newBlock(context: Ref): IR = {
        bindIR(GetField(context, "oldContext")) { oldContext =>
          If(IsNA(oldContext),
            MakeNDArray.fill(
              zero(typ.elementType),
              FastIndexedSeq(GetField(oldContext, "nRows"), GetField(oldContext, "nCols")),
              False()),
            blockIR(oldContext))
        }
      }
      BlockMatrixStage2(broadcastVals, typ, newContexts, newBlock)
  }

  def withSparsity(rowPos: TrivialIR, rowIdx: TrivialIR, ib: IRBuilder, newType: BlockMatrixType, isSubset: Boolean = false): BlockMatrixStage2 = {
    if (newType.sparsity.definedBlocksColMajor == typ.sparsity.definedBlocksColMajor) return this

    contexts match {
      case contexts: SparseContexts =>
        if (isSubset) {
          val newContexts = contexts.mapWithNewSparsity(rowPos, rowIdx, ib) { (i, j, oldContext) =>
            oldContext
          }
          BlockMatrixStage2(broadcastVals, newType, newContexts, blockIR)
        } else {
          val newContexts = contexts.mapWithNewSparsity(rowPos, rowIdx, ib) { (i, j, oldContext) =>
            val (m, n) = typ.blockShapeIR(i, j)
            makestruct("oldContext" -> oldContext, "nRows" -> m, "nCols" -> n)
          }

          def newBlock(context: Ref): IR = {
            bindIR(GetField(context, "oldContext")) { oldContext =>
              If(IsNA(oldContext),
                MakeNDArray.fill(
                  zero(typ.elementType),
                  FastIndexedSeq(GetField(context, "nRows"), GetField(context, "nCols")),
                  False()),
                blockIR(oldContext))
            }
          }

          BlockMatrixStage2(broadcastVals, newType, newContexts, newBlock)
        }

      case contexts: DenseContexts =>
        val newContexts = contexts.sparsify(rowPos, rowIdx, ib)
        BlockMatrixStage2(broadcastVals, newType, newContexts, blockIR)
    }
  }

  def mapBody(f: IR => IR): BlockMatrixStage2 = {
    val blockRef = Ref(genUID(), _blockIR.typ)
    val newBlockIR = Let(blockRef.name, _blockIR, f(blockRef))
    val newType = typ.copy(elementType = newBlockIR.typ.asInstanceOf[TNDArray].elementType)

    new BlockMatrixStage2(broadcastVals, newType, contexts, ctxRefName, newBlockIR)
  }

  def mapBody2(
    other: BlockMatrixStage2,
    ib: IRBuilder,
    sparsityStrategy: SparsityStrategy
  )(f: (IR, IR) => IR
  ): BlockMatrixStage2 = {
    val (alignedLeft, alignedRight) = (contexts, other.contexts, sparsityStrategy) match {
      case (_: DenseContexts, _: DenseContexts, _) =>
        (this, other)
      case (_: DenseContexts, _: SparseContexts, UnionBlocks) =>
        (this, other.densify(ib))
      case (_: SparseContexts, _: DenseContexts, UnionBlocks) =>
        (this.densify(ib), other)
      case (_: SparseContexts, _: SparseContexts, UnionBlocks) =>
        val newType = typ.copy(sparsity = UnionBlocks.mergeSparsity(typ.sparsity, other.typ.sparsity))
        val (unionPos, unionIdx) = newType.sparsity.definedBlocksCSCIR(newType.nColBlocks).get
        val unionPosRef = ib.memoize(unionPos)
        val unionIdxRef = ib.memoize(unionIdx)
        (this.withSparsity(unionPosRef, unionIdxRef, ib, newType), other.withSparsity(unionPosRef, unionIdxRef, ib, newType))
      case (_: DenseContexts, sparse: SparseContexts, IntersectionBlocks) =>
        (this.withSparsity(sparse.rowPos, sparse.rowIdx, ib, other.typ), other)
      case (sparse: SparseContexts, _: DenseContexts, IntersectionBlocks) =>
        (this, other.withSparsity(sparse.rowPos, sparse.rowIdx, ib, this.typ))
      case (_: SparseContexts, _: SparseContexts, IntersectionBlocks) =>
        val newType = typ.copy(sparsity = IntersectionBlocks.mergeSparsity(typ.sparsity, other.typ.sparsity))
        val (unionPos, unionIdx) = newType.sparsity.definedBlocksCSCIR(newType.nColBlocks).get
        val unionPosRef = ib.memoize(unionPos)
        val unionIdxRef = ib.memoize(unionIdx)
        (this.withSparsity(unionPosRef, unionIdxRef, ib, newType), other.withSparsity(unionPosRef, unionIdxRef, ib, newType))
    }

    alignedLeft.mapBody2Aligned(alignedRight, ib)(f)
  }

  private def mapBody2Aligned(other: BlockMatrixStage2, ib: IRBuilder)(f: (IR, IR) => IR): BlockMatrixStage2 = {
    val newContexts = contexts.zip(other.contexts, ib)
    val ctxRef = Ref(genUID(), newContexts.elementType)
    val newBlockIR =
      bindIRs(GetTupleElement(ctxRef, 0), GetTupleElement(ctxRef, 1)) { case Seq(l, r) =>
        f(this.blockIR(l), other.blockIR(r))
      }
    val newType = typ.copy(elementType = newBlockIR.typ.asInstanceOf[TNDArray].elementType)
    new BlockMatrixStage2(
      broadcastVals ++ other.broadcastVals,
      newType, newContexts, ctxRef.name, newBlockIR)
  }

  def filter(keepRows: IndexedSeq[Long], keepCols: IndexedSeq[Long], typ: BlockMatrixType, ib: IRBuilder): BlockMatrixStage2 = {
    val rowBlockDependents = keepRows.grouped(typ.blockSize).map(_.map(i => (i / typ.blockSize).toInt).distinct).toFastIndexedSeq
    val colBlockDependents = keepCols.grouped(typ.blockSize).map(_.map(i => (i / typ.blockSize).toInt).distinct).toFastIndexedSeq

    def localIndices(idxs: IndexedSeq[Long]): IndexedSeq[IndexedSeq[Long]] = {
      val result = new AnyRefArrayBuilder[IndexedSeq[Long]]()
      val builder = new LongArrayBuilder()
      var curBlock = idxs.head / typ.blockSize
      for (i <- idxs) {
        val nextBlock = i / typ.blockSize
        if (nextBlock != curBlock) {
          curBlock = nextBlock
          result += builder.result()
          builder.clear()
        }
        builder += i % typ.blockSize
      }
      result += builder.result()
      result.result()
    }

    val groupedKeepRows: IndexedSeq[IndexedSeq[IndexedSeq[Long]]] =
      keepRows.grouped(typ.blockSize).map(localIndices).toFastIndexedSeq
    val groupedKeepCols: IndexedSeq[IndexedSeq[IndexedSeq[Long]]] =
      keepCols.grouped(typ.blockSize).map(localIndices).toFastIndexedSeq
    val t = TArray(TArray(TArray(TInt64)))
    val groupedKeepRowsLit = if (keepRows.isEmpty) NA(t) else Literal(t, groupedKeepRows)
    val groupedKeepColsLit = if (keepCols.isEmpty) NA(t) else Literal(t, groupedKeepCols)
    val groupedContexts: BMSContexts = contexts.grouped(rowBlockDependents, colBlockDependents, typ, ib)
    val groupedContextsWithIndices = groupedContexts.map(ib) { (i, j, pos, context) =>
      maketuple(context, ArrayRef(groupedKeepRowsLit, i), ArrayRef(groupedKeepColsLit, j))
    }

    def newBody(ctx: Ref): IR = {
      IRBuilder.scoped { ib =>
        val localContexts = contexts match {
          case _: DenseContexts => DenseContexts(GetTupleElement(ctx, 0), ib)
          case _: SparseContexts => SparseContexts(GetTupleElement(ctx, 0), ib)
        }
        val localKeepRows = GetTupleElement(ctx, 1)
        val localKeepCols = GetTupleElement(ctx, 2)
        localContexts.collect { (i, j, localContext) =>
          bindIRs(ArrayRef(localKeepRows, i), ArrayRef(localKeepCols, j)) { case Seq(rows, cols) =>
            Coalesce(FastSeq(
              NDArrayFilter(blockIR(localContext), FastIndexedSeq(rows, cols)),
              MakeNDArray.fill(zero(typ.elementType), FastIndexedSeq(ArrayLen(rows).toL, ArrayLen(cols).toL), False())))
          }
        }
      }
    }

    BlockMatrixStage2(broadcastVals, typ, groupedContextsWithIndices, newBody)
  }

  def zeroBand(lower: Long, upper: Long, typ: BlockMatrixType, ib: IRBuilder): BlockMatrixStage2 = {
    val ctxs = contexts.map(ib) { (i, j, _, context) =>
      maketuple(context, i, j)
    }

    def newBody(ctx: Ref): IR = IRBuilder.scoped { ib =>
      val oldCtx = GetTupleElement(ctx, 0)
      val i = GetTupleElement(ctx, 1)
      val j = GetTupleElement(ctx, 2)
      val diagIndex = (j - i).toL * typ.blockSize.toLong
      bindIRs(diagIndex, oldCtx) { case Seq(diagIndex, oldCtx) =>
        val localLower = I64(lower) - diagIndex
        val localUpper = I64(upper) - diagIndex
        val (nRowsInBlock, nColsInBlock) = typ.blockShapeIR(i, j)
        val block = blockIR(oldCtx)
        If(-localLower >= (nRowsInBlock - 1L) && localUpper >= (nColsInBlock - 1L),
          block,
          invoke("zero_band", TNDArray(TFloat64, Nat(2)), block, localLower, localUpper)
        )
      }
    }

    BlockMatrixStage2(broadcastVals, typ, ctxs, newBody)
  }

  def zeroRowIntervals(starts: IndexedSeq[Long], stops: IndexedSeq[Long], typ: BlockMatrixType, ib: IRBuilder): BlockMatrixStage2 = {
    val t = TArray(TArray(TInt64))
    val startsGrouped = Literal(t, starts.grouped(typ.blockSize).toIndexedSeq)
    val stopsGrouped = Literal(t, stops.grouped(typ.blockSize).toIndexedSeq)

    val ctxs = contexts.map(ib) { (i, j, _, context) =>
      maketuple(context, i, j, ArrayRef(startsGrouped, i), ArrayRef(stopsGrouped, i))
    }

    def newBody(ctx: Ref): IR = {
      val oldCtx = GetTupleElement(ctx, 0)
      val i = GetTupleElement(ctx, 1)
      val j = GetTupleElement(ctx, 2)
      val (_, nCols) = typ.blockShapeIR(i, j)
      val starts = ToArray(mapIR(ToStream(GetTupleElement(ctx, 3))) { s => minIR(maxIR(s - j.toL * typ.blockSize.toLong, 0L), nCols) })
      val stops = ToArray(mapIR(ToStream(GetTupleElement(ctx, 4))) { s => minIR(maxIR(s - j.toL * typ.blockSize.toLong, 0L), nCols) })
      bindIRs(oldCtx) { case Seq(oldCtx) =>
        invoke("zero_row_intervals", TNDArray(TFloat64, Nat(2)), blockIR(oldCtx), starts, stops)
      }
    }

    BlockMatrixStage2(broadcastVals, typ, ctxs, newBody)
  }

  def collectBlocks(
    ib: IRBuilder,
    staticID: String,
    dynamicID: IR = NA(TString)
  )(f: (IR, IR, IR) => IR // (ctx, pos, block)
  ): IR = {
    val posRef = Ref(genUID(), TInt32)
    val newCtxRef = Ref(genUID(), TTuple(TInt32, ctxType))
    val body = Let(posRef.name, GetTupleElement(newCtxRef, 0),
      Let(ctxRefName, GetTupleElement(newCtxRef, 1),
        f(ctxRef, posRef, _blockIR)))
    val bodyFreeVars = FreeVariables(body, supportsAgg = false, supportsScan = false)
    val bcFields = broadcastVals.filter { case Ref(f, _) => bodyFreeVars.eval.lookupOption(f).isDefined }
    val bcVals = MakeStruct(bcFields.map { ref => ref.name -> ref })
    val bcRef = Ref(genUID(), bcVals.typ)
    val wrappedBody = bcFields.foldLeft(body) { case (accum, Ref(f, _)) =>
      Let(f, GetField(bcRef, f), accum)
    }

    val cdaContexts = ToStream(contexts.map(ib) { (rowIdx, colIdx, pos, oldContext) =>
      maketuple(pos, oldContext)
    }.contexts)

    CollectDistributedArray(cdaContexts, bcVals, newCtxRef.name, bcRef.name, wrappedBody, dynamicID, staticID)
  }

  def collectLocal(ib: IRBuilder, staticID: String, dynamicID: IR = NA(TString)): IR = {
    val blockResults = ib.memoize(collectBlocks(ib, staticID, dynamicID)((_, _, b) => b))
    val blocks = contexts match {
      case x: DenseContexts => DenseContexts(x.nRows, x.nCols, blockResults)
      case x: SparseContexts => SparseContexts(x.nRows, x.nCols, x.rowPos, x.rowIdx, blockResults)
    }

    blocks.collect { (i, j, block) =>
      val (m, n) = typ.blockShapeIR(i, j)
      val zeroBlock: IR = MakeNDArray.fill(zero(typ.elementType), FastIndexedSeq(m, n), False())
      Coalesce(FastSeq(block, zeroBlock))
    }
  }
}

object LowerBlockMatrixIR {
  def apply(node: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: LoweringAnalyses): IR = {

    def lower(bmir: BlockMatrixIR, ib: IRBuilder) =
      LowerBlockMatrixIR.lower(bmir, ib, typesToLower, ctx, analyses)

    IRBuilder.scoped { ib =>
      node match {
        case BlockMatrixCollect(child) =>
          lower(child, ib).collectLocal(ib, "block_matrix_collect")
        case BlockMatrixToValueApply(child, GetElement(IndexedSeq(i, j))) =>
          lower(child, ib).getElement(i, j)
        case BlockMatrixWrite(child, writer) =>
          writer.lower(ctx, lower(child, ib), ib, TypeWithRequiredness(child.typ.elementType)) //FIXME: BlockMatrixIR is currently ignored in Requiredness inference since all eltTypes are +TFloat64
        case BlockMatrixMultiWrite(blockMatrices, writer) => unimplemented(ctx, node)
        case node if node.children.exists(_.isInstanceOf[BlockMatrixIR]) =>
          throw new LowererUnsupportedOperation(s"IR nodes with BlockMatrixIR children need explicit rules: \n${ Pretty(ctx, node) }")

        case node =>
          throw new LowererUnsupportedOperation(s"Value IRs with no BlockMatrixIR children must be lowered through LowerIR: \n${ Pretty(ctx, node) }")
      }
    }
  }

  // This lowers a BlockMatrixIR to an unkeyed TableStage with rows of (blockRow, blockCol, block)
  def lowerToTableStage(
    bmir: BlockMatrixIR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext,
    analyses: LoweringAnalyses
  ): TableStage = {
    val ib = new IRBuilder()
    val bms = lower(bmir, ib, typesToLower, ctx, analyses).toOldBMS
    val typ = bmir.typ
    val bmsWithCtx = bms.addContext(TTuple(TInt32, TInt32)){ case (i, j) => MakeTuple(FastIndexedSeq(0 -> i, 1 -> j))}
    val blocksRowMajor = Array.range(0, typ.nRowBlocks).flatMap { i =>
      Array.tabulate(typ.nColBlocks)(j => i -> j).filter(typ.hasBlock)
    }
    val emptyGlobals = MakeStruct(FastIndexedSeq())
    val globalsId = genUID()
    val letBindings = ib.getBindings :+ globalsId -> emptyGlobals
    val contextsIR = MakeStream(blocksRowMajor.map{ case (i, j) =>  bmsWithCtx.blockContext((i, j)) }, TStream(bmsWithCtx.ctxType))

    val ctxRef = Ref(genUID(), bmsWithCtx.ctxType)
    val body = bmsWithCtx.blockBody(ctxRef)
    val bodyFreeVars = FreeVariables(body, supportsAgg = false, supportsScan = false)
    val bcFields = (bmsWithCtx.broadcastVals
      .filter { case Ref(f, _) => bodyFreeVars.eval.lookupOption(f).isDefined }
      .map { ref => ref.name -> ref }
      :+ globalsId -> Ref(globalsId, emptyGlobals.typ))

    def tsPartitionFunction(ctxRef: Ref): IR = {
      val s = MakeStruct(FastIndexedSeq("blockRow" -> GetTupleElement(GetField(ctxRef, "new"), 0), "blockCol" -> GetTupleElement(GetField(ctxRef, "new"), 1), "block" -> bmsWithCtx.blockBody(ctxRef)))
      MakeStream(FastIndexedSeq(s), TStream(s.typ))
    }
    val ts = TableStage(letBindings, bcFields, Ref(globalsId, emptyGlobals.typ), RVDPartitioner.unkeyed(ctx.stateManager, blocksRowMajor.size), TableStageDependency.none, contextsIR, tsPartitionFunction)
    ts
  }

  private def unimplemented[T](ctx: ExecuteContext, node: BaseIR): T =
    throw new LowererUnsupportedOperation(s"unimplemented: \n${ Pretty(ctx, node) }")

  def lower(bmir: BlockMatrixIR, ib: IRBuilder, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: LoweringAnalyses): BlockMatrixStage2 = {
    if (!DArrayLowering.lowerBM(typesToLower))
      throw new LowererUnsupportedOperation("found BlockMatrixIR in lowering; lowering only TableIRs.")
    bmir.children.foreach {
      case c: BlockMatrixIR if c.typ.blockSize != bmir.typ.blockSize =>
        throw new LowererUnsupportedOperation(s"Can't lower node with mismatched block sizes: ${ bmir.typ.blockSize } vs child ${ c.typ.blockSize }\n\n ${ Pretty(ctx, bmir) }")
      case _ =>
    }
    if (bmir.typ.matrixShape == 0L -> 0L)
      BlockMatrixStage2.empty(bmir.typ.elementType, ib)
    else lowerNonEmpty2(bmir, ib, typesToLower, ctx, analyses)
  }

  def lowerNonEmpty2(
    bmir: BlockMatrixIR,
    ib: IRBuilder,
    typesToLower: DArrayLowering.Type,
    ctx: ExecuteContext,
    analyses: LoweringAnalyses
  ): BlockMatrixStage2 = {

    def lower(ir: BlockMatrixIR, ib: IRBuilder = ib): BlockMatrixStage2 =
      LowerBlockMatrixIR.lower(ir, ib, typesToLower, ctx, analyses)

    bmir match {
      case BlockMatrixRead(reader) => reader.lower(ctx, ib)

      case x@BlockMatrixRandom(staticUID, gaussian, shape, blockSize) =>
        val contexts = BMSContexts.tabulate(x.typ, ib) { (rowIdx, colIdx) =>
          val (m, n) = x.typ.blockShapeIR(rowIdx, colIdx)
          MakeTuple.ordered(FastSeq(m, n, rowIdx * x.typ.nColBlocks + colIdx))
        }

        def bodyIR(ctx: Ref): IR = {
          val m = GetTupleElement(ctx, 0)
          val n = GetTupleElement(ctx, 1)
          val i = GetTupleElement(ctx, 2)
          val f = if (gaussian) "rand_norm_nd" else "rand_unif_nd"
          val rngState = RNGSplit(RNGStateLiteral(), Cast(i, TInt64))
          invokeSeeded(f, staticUID, TNDArray(TFloat64, Nat(2)), rngState, m, n, F64(0.0), F64(1.0))
        }

        BlockMatrixStage2(FastIndexedSeq(), x.typ, contexts, bodyIR)

      case BlockMatrixMap(child, eltName, f, _) =>
        lower(child).mapBody { body =>
          NDArrayMap(body, eltName, f)
        }

      case BlockMatrixMap2(left, right, lname, rname, f, sparsityStrategy) =>
        val loweredLeft = lower(left)
        val loweredRight = lower(right)
        loweredLeft.mapBody2(loweredRight, ib, sparsityStrategy) { (lBody, rBody) =>
          NDArrayMap2(lBody, rBody, lname, rname, f, ErrorIDs.NO_ERROR)
        }

      case x@BlockMatrixBroadcast(child, IndexedSeq(), _, _) =>
        val elt = ib.strictMemoize(IRBuilder.scoped { ib =>
          val lowered = lower(child, ib)
          lowered.getElement(0L, 0L)
        })

        val contexts = BMSContexts.tabulate(x.typ, ib) { (rowIdx, colIdx) =>
          val (i, j) = x.typ.blockShapeIR(rowIdx, colIdx)
          maketuple(i, j)
        }

        BlockMatrixStage2(
          FastIndexedSeq(elt),
          x.typ,
          contexts,
          (ctxRef: Ref) =>
            MakeNDArray.fill(elt, FastIndexedSeq(GetTupleElement(ctxRef, 0), GetTupleElement(ctxRef, 1)), True()))

      case x@BlockMatrixBroadcast(child, IndexedSeq(axis), _, _) =>
        val len = child.typ.shape.max
        val vector = NDArrayReshape(
          IRBuilder.scoped { ib =>
            lower(child, ib).collectLocal(ib, "block_matrix_broadcast_single_axis")
          },
          MakeTuple.ordered(FastSeq(I64(len))),
          ErrorIDs.NO_ERROR)
        BlockMatrixStage2.broadcastVector(vector, ib, x.typ, asRowVector = axis == 1)

      case x@BlockMatrixBroadcast(child, IndexedSeq(axis, axis2), _, _) if (axis == axis2) => // diagonal as row/col vector
        val diagLen = math.min(child.typ.nRowBlocks, child.typ.nColBlocks)
        val diagType = x.typ.copy(sparsity = BlockMatrixSparsity(Some(IndexedSeq.tabulate(diagLen)(i => (i, i)))))
        val rowPos = if (child.typ.nColBlocks > diagLen)
          ToArray(mapIR(rangeIR(child.typ.nColBlocks + 1))(i => minIR(i, diagLen)))
        else
          ToArray(rangeIR(child.typ.nColBlocks + 1))

        val diagArray = IRBuilder.scoped { ib =>
          lower(child, ib)
            .withSparsity(ib.memoize(rowPos), ib.memoize(ToArray(rangeIR(diagLen))), ib, diagType)
            .collectBlocks(ib, "block_matrix_broadcast_diagonal") { (ctx, idx, block) =>
              bindIR(NDArrayShape(block)) { shape =>
                val blockDiagLen = minIR(GetTupleElement(shape, 0), GetTupleElement(shape, 1))
                ToArray(mapIR(rangeIR(blockDiagLen.toI)) { i =>
                  NDArrayRef(block, FastIndexedSeq(Cast(i, TInt64), Cast(i, TInt64)), ErrorIDs.NO_ERROR)
                })
              }
            }
        }

        val diagVector = MakeNDArray(ToArray(flatten(diagArray)), maketuple(math.min(child.typ.nRows, child.typ.nCols)), true, ErrorIDs.NO_ERROR)
        BlockMatrixStage2.broadcastVector(diagVector, ib, x.typ, asRowVector = axis == 0)

      case x@BlockMatrixBroadcast(child, IndexedSeq(1, 0), _, _) => //transpose
        lower(child).transposed(ib)

      case BlockMatrixBroadcast(child, IndexedSeq(0, 1), _, _) =>
        lower(child)

      case x@BlockMatrixFilter(child, keep) =>
        val Array(keepRow, keepCol) = keep
        lower(child).filter(keepRow, keepCol, x.typ, ib)

      case BlockMatrixDensify(child) =>
        lower(child).densify(ib)
      case x@BlockMatrixSparsify(child, sparsifier) =>
        val Some((rowPos, rowIdx)) = x.typ.sparsity.definedBlocksCSCIR(x.typ.nColBlocks)
        val loweredChild = lower(child).withSparsity(ib.memoize(rowPos), ib.memoize(rowIdx), ib, x.typ, isSubset = true)
        sparsifier match {
          // these cases are all handled at the type level
          case BandSparsifier(blocksOnly, _, _) if (blocksOnly) => loweredChild
          case RowIntervalSparsifier(blocksOnly, _, _) if (blocksOnly) => loweredChild
          case PerBlockSparsifier(_) | RectangleSparsifier(_) => loweredChild

          case BandSparsifier(_, l, u) => loweredChild.zeroBand(l, u, x.typ, ib)
          case RowIntervalSparsifier(_, starts, stops) => loweredChild.zeroRowIntervals(starts, stops, x.typ, ib)
        }
      case _ =>
        BlockMatrixStage2.fromOldBMS(lowerNonEmpty(bmir, ib, typesToLower, ctx, analyses), bmir.typ, ib)
    }
  }

  def lowerNonEmpty(bmir: BlockMatrixIR, ib: IRBuilder, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: LoweringAnalyses): BlockMatrixStage = {
    def lower(ir: BlockMatrixIR, ib: IRBuilder = ib) =
      LowerBlockMatrixIR.lower(ir, ib, typesToLower, ctx, analyses).toOldBMS

    def lowerIR(node: IR): IR = LowerToCDA.lower(node, typesToLower, ctx, analyses)

    bmir match {

      case a@BlockMatrixAgg(child, axesToSumOut) =>
        val loweredChild = lower(child)
        axesToSumOut match {
          case IndexedSeq(0, 1)  =>
            val summedChild = loweredChild.mapBody { (ctx, body) =>
              NDArrayReshape(NDArrayAgg(body, IndexedSeq(0, 1)), MakeTuple.ordered(FastIndexedSeq(I64(1), I64(1))), ErrorIDs.NO_ERROR)
            }
            val summedChildType = BlockMatrixType(child.typ.elementType, IndexedSeq[Long](child.typ.nRowBlocks, child.typ.nColBlocks), child.typ.nRowBlocks == 1, 1, BlockMatrixSparsity.dense)
            val res = NDArrayAgg(summedChild.collectLocal(summedChildType, "block_matrix_agg"), IndexedSeq[Int](0, 1))
            new BlockMatrixStage(summedChild.broadcastVals, TStruct.empty) {
              override def blockContext(idx: (Int, Int)): IR = makestruct()
              override def blockBody(ctxRef: Ref): IR = NDArrayReshape(res, MakeTuple.ordered(FastIndexedSeq(I64(1L), I64(1L))), ErrorIDs.NO_ERROR)
            }
          case IndexedSeq(0) => { // Number of rows goes to 1. Number of cols remains the same.
            new BlockMatrixStage(loweredChild.broadcastVals, TArray(loweredChild.ctxType)) {
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
                  bindIR(NDArrayAgg(loweredChild.blockBody(singleChildCtx), axesToSumOut))(aggedND => NDArrayReshape(aggedND, MakeTuple.ordered(FastIndexedSeq(I64(1), GetTupleElement(NDArrayShape(aggedND), 0))), ErrorIDs.NO_ERROR))
                })
                val aggVar = genUID()
                StreamAgg(summedChildBlocks, aggVar, ApplyAggOp(NDArraySum())(Ref(aggVar, summedChildBlocks.typ.asInstanceOf[TStream].elementType)))
              }
            }
          }
          case IndexedSeq(1) => { // Number of cols goes to 1. Number of rows remains the same.
            new BlockMatrixStage(loweredChild.broadcastVals, TArray(loweredChild.ctxType)) {
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
                  bindIR(NDArrayAgg(loweredChild.blockBody(singleChildCtx), axesToSumOut)) {
                    aggedND => NDArrayReshape(aggedND, MakeTuple(FastIndexedSeq(0 -> GetTupleElement(NDArrayShape(aggedND), 0), 1 -> I64(1))), ErrorIDs.NO_ERROR)
                  }
                })
                val aggVar = genUID()
                StreamAgg(summedChildBlocks, aggVar, ApplyAggOp(NDArraySum())(Ref(aggVar, summedChildBlocks.typ.asInstanceOf[TStream].elementType)))
              }
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

      case RelationalLetBlockMatrix(name, value, body) => unimplemented(ctx, bmir)

      case ValueToBlockMatrix(child, shape, blockSize) if !child.typ.isInstanceOf[TArray] && !child.typ.isInstanceOf[TNDArray] => {
        val element = lowerIR(child)
        new BlockMatrixStage(FastIndexedSeq(), TStruct()) {
          override def blockContext(idx: (Int, Int)): IR = MakeStruct(FastIndexedSeq())

          override def blockBody(ctxRef: Ref): IR = MakeNDArray(MakeArray(element), MakeTuple(FastIndexedSeq((0, I64(1)), (1, I64(1)))), False(), ErrorIDs.NO_ERROR)
        }
      }
      case x@ValueToBlockMatrix(child, _, blockSize) =>
        val nd = ib.memoize(child.typ match {
          case _: TArray => MakeNDArray(lowerIR(child), MakeTuple.ordered(FastSeq(I64(x.typ.nRows), I64(x.typ.nCols))), True(), ErrorIDs.NO_ERROR)
          case _: TNDArray => lowerIR(child)
        })
        new BlockMatrixStage(FastIndexedSeq(), nd.typ) {
          def blockContext(idx: (Int, Int)): IR = {
            val (r, c) = idx
            NDArraySlice(nd, MakeTuple.ordered(FastSeq(
              MakeTuple.ordered(FastSeq(I64(r.toLong * blockSize), I64(java.lang.Math.min((r.toLong + 1) * blockSize, x.typ.nRows)), I64(1))),
              MakeTuple.ordered(FastSeq(I64(c.toLong * blockSize), I64(java.lang.Math.min((c.toLong + 1) * blockSize, x.typ.nCols)), I64(1))))))
          }

          def blockBody(ctxRef: Ref): IR = ctxRef
        }
      case x@BlockMatrixDot(leftIR, rightIR) =>
        val left = lower(leftIR)
        val right = lower(rightIR)
        val newCtxType = TArray(TTuple(left.ctxType, right.ctxType))
        new BlockMatrixStage(left.broadcastVals ++ right.broadcastVals, newCtxType) {
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
