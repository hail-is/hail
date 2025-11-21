package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.Nat
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.GetElement
import is.hail.linalg.MatrixSparsity
import is.hail.rvd.RVDPartitioner
import is.hail.types.{tcoerce, TypeWithRequiredness}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.compat.immutable.ArraySeq

import org.apache.spark.sql.Row

abstract class BlockMatrixStage(val broadcastVals: IndexedSeq[Ref], val ctxType: Type) {
  def blockContext(idx: (Int, Int)): IR

  def blockBody(ctxRef: Ref): IR

  def collectBlocks(
    staticID: String,
    dynamicID: IR = NA(TString),
  )(
    f: (IR, IR) => IR,
    blocksToCollect: Array[(Int, Int)],
  ): IR = {
    val ctxRef = Ref(freshName(), ctxType)
    val body = f(ctxRef, blockBody(ctxRef))
    val ctxs = MakeStream(blocksToCollect.map(idx => blockContext(idx)), TStream(ctxRef.typ))
    val bodyFreeVars = FreeVariables(body, supportsAgg = false, supportsScan = false)
    val bcFields = broadcastVals.filter { ref =>
      bodyFreeVars.eval.lookupOption(ref.name).isDefined
    }
    val bcVals = MakeStruct(bcFields.map(ref => ref.name.str -> ref))
    val bcRef = Ref(freshName(), bcVals.typ)
    val wrappedBody = Let(bcFields.map(ref => ref.name -> GetField(bcRef, ref.name.str)), body)
    CollectDistributedArray(ctxs, bcVals, ctxRef.name, bcRef.name, wrappedBody, dynamicID, staticID)
  }

  def collectLocal(typ: BlockMatrixType, staticID: String, dynamicID: IR = NA(TString)): IR = {
    val blocksRowMajor = Array.range(0, typ.nRowBlocks).flatMap { i =>
      Array.tabulate(typ.nColBlocks)(j => i -> j).filter((typ.hasBlock _).tupled)
    }
    val cda = collectBlocks(staticID, dynamicID)((_, b) => b, blocksRowMajor)
    val blockResults = Ref(freshName(), cda.typ)

    val rows = if (typ.isSparse) {
      val blockMap = blocksRowMajor.zipWithIndex.toMap
      MakeArray(
        Array.tabulate[IR](typ.nRowBlocks) { i =>
          NDArrayConcat(
            MakeArray(
              Array.tabulate[IR](typ.nColBlocks) { j =>
                if (blockMap.contains(i -> j))
                  ArrayRef(blockResults, i * typ.nColBlocks + j)
                else {
                  val (nRows, nCols) = typ.blockShape(i, j)
                  MakeNDArray.fill(zero(typ.elementType), FastSeq(nRows, nCols), True())
                }
              },
              tcoerce[TArray](cda.typ),
            ),
            1,
          )
        },
        tcoerce[TArray](cda.typ),
      )
    } else {
      ToArray(mapIR(rangeIR(I32(typ.nRowBlocks))) { rowIdxRef =>
        val blocksInOneRow = ToArray(mapIR(rangeIR(I32(typ.nColBlocks))) { colIdxRef =>
          ArrayRef(blockResults, rowIdxRef * typ.nColBlocks + colIdxRef)
        })
        NDArrayConcat(blocksInOneRow, 1)
      })
    }

    Let(FastSeq(blockResults.name -> cda), NDArrayConcat(rows, 0))
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

  def condenseBlocks(
    typ: BlockMatrixType,
    rowBlocks: IndexedSeq[IndexedSeq[Int]],
    colBlocks: IndexedSeq[IndexedSeq[Int]],
  ): BlockMatrixStage = {
    val outer = this
    val ctxType = TArray(TArray(TTuple(TTuple(TInt64, TInt64), outer.ctxType)))
    new BlockMatrixStage(outer.broadcastVals, ctxType) {
      def blockContext(idx: (Int, Int)): IR = {
        val i = idx._1
        val j = idx._2
        MakeArray(rowBlocks(i).map { ii =>
          MakeArray(colBlocks(j).map { jj =>
            if (typ.hasBlock(ii, jj))
              MakeTuple.ordered(FastSeq(NA(TTuple(TInt64, TInt64)), outer.blockContext(ii -> jj)))
            else {
              val (nRows, nCols) = typ.blockShape(ii, jj)
              MakeTuple.ordered(FastSeq(
                MakeTuple.ordered(FastSeq(nRows, nCols)),
                NA(outer.ctxType),
              ))
            }
          }: _*)
        }: _*)
      }

      def blockBody(ctxRef: Ref): IR = {
        NDArrayConcat(
          ToArray(mapIR(ToStream(ctxRef)) { ctxRows =>
            NDArrayConcat(
              ToArray(mapIR(ToStream(ctxRows)) { shapeOrCtx =>
                bindIR(GetTupleElement(shapeOrCtx, 1)) { ctx =>
                  If(
                    IsNA(ctx),
                    bindIR(GetTupleElement(shapeOrCtx, 0)) { shape =>
                      MakeNDArray(
                        ToArray(mapIR(
                          rangeIR((GetTupleElement(shape, 0) * GetTupleElement(shape, 1)).toI)
                        )(_ => zero(typ.elementType))),
                        shape,
                        False(),
                        ErrorIDs.NO_ERROR,
                      )
                    },
                    outer.blockBody(ctx),
                  )
                }
              }),
              1,
            )
          }),
          0,
        )
      }
    }
  }
}

abstract class DynamicBMSContexts {
  def apply(row: IR, col: IR): IR
  def map(ib: IRBuilder)(f: (IR, IR, IR, IR) => IR): DynamicBMSContexts
  def collect(makeBlock: (Ref, Ref, Ref) => IR): IR
}

object DynamicDenseContexts {
  def apply(ib: IRBuilder, irValue: IR): DynamicDenseContexts = {
    val irValueRef = ib.memoize(irValue)
    DynamicDenseContexts(
      ib.memoize(GetField(irValueRef, "nRows")),
      ib.memoize(GetField(irValueRef, "nCols")),
      ib.memoize(GetField(irValueRef, "contexts")),
    )
  }
}

case class DynamicDenseContexts(nRows: TrivialIR, nCols: TrivialIR, contexts: TrivialIR)
    extends DynamicBMSContexts {
  def irValue: IR = makestruct("nRows" -> nRows, "nCols" -> nCols, "contexts" -> contexts)

  override def apply(row: IR, col: IR): IR = ArrayRef(contexts, (col * nRows) + row)

  override def map(ib: IRBuilder)(f: (IR, IR, IR, IR) => IR): DynamicDenseContexts = {
    DynamicDenseContexts(
      nRows,
      nCols,
      ib.memoize(ToArray(flatMapIR(rangeIR(nCols)) { j =>
        mapIR(rangeIR(nRows)) { i =>
          bindIR((j * nRows) + i) { pos =>
            bindIR(ArrayRef(contexts, pos))(old => f(i, j, pos, old))
          }
        }
      })),
    )
  }

  override def collect(makeBlock: (Ref, Ref, Ref) => IR): IR = {
    NDArrayConcat(
      ToArray(mapIR(rangeIR(nCols)) { j =>
        val colBlocks = mapIR(rangeIR(nRows)) { i =>
          bindIR(ArrayRef(contexts, j * nRows + i))(ctx => makeBlock(i, j, ctx))
        }
        NDArrayConcat(ToArray(colBlocks), 0)
      }),
      1,
    )
  }
}

object DynamicSparseContexts {
  def apply(ib: IRBuilder, irValue: IR): DynamicSparseContexts = {
    val irValueRef = ib.memoize(irValue)
    new DynamicSparseContexts(
      ib.memoize(GetField(irValueRef, "nRows")),
      ib.memoize(GetField(irValueRef, "nCols")),
      ib.memoize(GetField(irValueRef, "rowPos")),
      ib.memoize(GetField(irValueRef, "rowIdx")),
      ib.memoize(GetField(irValueRef, "contexts")),
    )
  }

  def apply(ib: IRBuilder, sparsity: MatrixSparsity.Sparse, contexts: IR): DynamicSparseContexts = {
    val csc = sparsity.toCSC
    val rowPosIR = Literal(TArray(TInt32), csc.rowPos)
    val rowIdxIR = Literal(TArray(TInt32), csc.rowIdx)
    new DynamicSparseContexts(
      sparsity.nRows,
      sparsity.nCols,
      ib.memoize(rowPosIR),
      ib.memoize(rowIdxIR),
      ib.memoize(contexts),
    )
  }
}

class DynamicSparseContexts(
  nRows: TrivialIR,
  nCols: TrivialIR,
  rowPos: TrivialIR,
  rowIdx: TrivialIR,
  val contexts: TrivialIR,
) extends DynamicBMSContexts {
  val elementType: Type = contexts.typ.asInstanceOf[TArray].elementType

  def withNewContexts(newContexts: TrivialIR): DynamicSparseContexts =
    new DynamicSparseContexts(nRows, nCols, rowPos, rowIdx, newContexts)

  override def apply(row: IR, col: IR): IR = {
    val startPos = ArrayRef(rowPos, col)
    val endPos = ArrayRef(rowPos, col + 1)
    bindIR(
      Apply("lowerBound", Seq(), FastSeq(rowIdx, row, startPos, endPos), TInt32, ErrorIDs.NO_ERROR)
    ) { pos =>
      If(
        ArrayRef(rowIdx, pos).ceq(row),
        ArrayRef(contexts, pos),
        Die(
          strConcat("Internal Error, tried to load missing BlockMatrix context: (row = ", row,
            ", col = ",
            col, ", pos = ", pos, ", rowPos = ", rowPos, ", rowIdx = ", rowIdx, ")"),
          elementType,
          ErrorIDs.NO_ERROR,
        ),
      )
    }
  }

  override def map(ib: IRBuilder)(f: (IR, IR, IR, IR) => IR): DynamicSparseContexts = {
    val colIdx = flatMapIR(rangeIR(nCols)) { j =>
      mapIR(rangeIR(ArrayRef(rowPos, j), ArrayRef(rowPos, j + 1)))(_ => j)
    }
    val newContexts = ToArray(zipIR(
      ArraySeq(ToStream(rowIdx), colIdx, iota(0, 1), ToStream(contexts)),
      ArrayZipBehavior.TakeMinLength,
    ) { case Seq(i, j, pos, oldCtx) =>
      f(i, j, pos, oldCtx)
    })
    new DynamicSparseContexts(nRows, nCols, rowPos, rowIdx, ib.memoize(newContexts))
  }

  override def collect(makeBlock: (Ref, Ref, Ref) => IR): IR = {
    NDArrayConcat(
      ToArray(mapIR(rangeIR(nCols)) { j =>
        val allIdxs = mapIR(rangeIR(nRows))(i => makestruct("idx" -> i))
        val startPos = ArrayRef(rowPos, j)
        val endPos = ArrayRef(rowPos, j + 1)
        val idxedExisting = mapIR(rangeIR(startPos, endPos)) { pos =>
          makestruct("idx" -> ArrayRef(rowIdx, pos), "ctx" -> ArrayRef(contexts, pos))
        }
        val colBlocks =
          joinRightDistinctIR(allIdxs, idxedExisting, FastSeq("idx"), FastSeq("idx"), "left") {
            (l, struct) =>
              bindIRs(GetField(l, "idx"), GetField(struct, "ctx")) { case Seq(i, ctx) =>
                makeBlock(i, j, ctx)
              }
          }
        NDArrayConcat(ToArray(colBlocks), 0)
      }),
      1,
    )
  }
}

object BMSContexts {
  def apply(ib: IRBuilder, sparsity: MatrixSparsity, contexts: IR): BMSContexts =
    sparsity match {
      case sparse: MatrixSparsity.Sparse => SparseContexts(ib, sparse, contexts)
      case dense: MatrixSparsity.Dense => DenseContexts(ib, dense, contexts)
    }

  def tabulate(ib: IRBuilder, sparsity: MatrixSparsity)(f: (IR, IR) => IR): BMSContexts = {
    val definedCoordsIR = sparsity match {
      case MatrixSparsity.Dense(nRows, nCols) =>
        flatMapIR(rangeIR(nCols))(j => mapIR(rangeIR(nRows))(i => maketuple(i, j)))
      case sparsity: MatrixSparsity.Sparse =>
        ToStream(Literal(TArray(TTuple(TInt32, TInt32)), sparsity.definedCoords.map(Row.fromTuple)))
    }
    val contexts =
      ToArray(mapIR(definedCoordsIR) {
        coords =>
          bindIRs(GetTupleElement(coords, 0), GetTupleElement(coords, 1)) { case Seq(i, j) =>
            f(i, j)
          }
      })
    BMSContexts(ib, sparsity, contexts)
  }
}

abstract class BMSContexts {
  def sparsity: MatrixSparsity

  def dynamic: DynamicBMSContexts

  def elementType: Type

  def nRows: Int = sparsity.nRows

  def nCols: Int = sparsity.nCols

  def contexts: IR

  def transpose(ib: IRBuilder): BMSContexts

  def withNewSparsity(ib: IRBuilder, newSparsity: MatrixSparsity): BMSContexts

  // body args: (rowIdx, colIdx, position, old context)
  def map(ib: IRBuilder)(f: (IR, IR, IR, IR) => IR): BMSContexts

  def zip(ib: IRBuilder, other: BMSContexts): BMSContexts

  def grouped(
    ib: IRBuilder,
    rowDeps: IndexedSeq[IndexedSeq[Int]],
    colDeps: IndexedSeq[IndexedSeq[Int]],
    typ: BlockMatrixType,
  ): BMSContexts

  def groupedByCol(ib: IRBuilder): BMSContexts

  def collect(makeBlock: (Ref, Ref, Ref) => IR): IR

  def print(ctx: ExecuteContext): Unit
}

object DenseContexts {
  def apply(ib: IRBuilder, sparsity: MatrixSparsity.Dense, contexts: IR): DenseContexts =
    DenseContexts(
      sparsity,
      DynamicDenseContexts(sparsity.nRows, sparsity.nCols, ib.memoize(contexts)),
    )
}

case class DenseContexts(sparsity: MatrixSparsity.Dense, dynamic: DynamicDenseContexts)
    extends BMSContexts {
  override val elementType: Type = contexts.typ.asInstanceOf[TArray].elementType

  override def contexts: TrivialIR = dynamic.contexts

  override def print(ctx: ExecuteContext): Unit =
    println(
      s"DenseContexts:\n  nRows = ${Pretty(ctx, nRows)}\n  nCols = ${Pretty(ctx, nCols)}\n  contexts = ${Pretty(ctx, contexts)}"
    )

  override def transpose(ib: IRBuilder): DenseContexts = DenseContexts(
    ib,
    sparsity.transpose,
    ToArray(flatMapIR(rangeIR(nRows)) { i =>
      mapIR(rangeIR(nCols))(j => ArrayRef(contexts, (j * nRows) + i))
    }),
  )

  override def map(ib: IRBuilder)(f: (IR, IR, IR, IR) => IR): DenseContexts =
    DenseContexts(sparsity, dynamic.map(ib)(f))

  override def zip(ib: IRBuilder, other: BMSContexts): BMSContexts = {
    val newContexts = ToArray(zip2(
      ToStream(this.contexts),
      ToStream(other.contexts),
      ArrayZipBehavior.AssertSameLength,
    )((l, r) => maketuple(l, r)))
    DenseContexts(ib, sparsity, newContexts)
  }

  def withNewSparsity(ib: IRBuilder, newSparsity: MatrixSparsity): BMSContexts = {
    require(newSparsity.nRows == sparsity.nRows && newSparsity.nCols == sparsity.nCols)
    newSparsity match {
      case sparse: MatrixSparsity.Sparse =>
        val newToOld = Literal(TArray(TInt32), sparsity.newToOldPos(sparse))
        val newContexts = ToArray(mapIR(ToStream(newToOld))(ArrayRef(contexts, _)))
        SparseContexts(ib, sparse, newContexts)
      case _: MatrixSparsity.Dense =>
        this
    }
  }

  def groupedByCol(ib: IRBuilder): DenseContexts = {
    val groupedContexts = ToArray(mapIR(rangeIR(nCols)) { col =>
      sliceArrayIR(contexts, col * nRows, (col + 1) * nRows)
    })
    DenseContexts(ib, MatrixSparsity.dense(1, nCols), ib.memoize(groupedContexts))
  }

  override def grouped(
    ib: IRBuilder,
    rowDeps: IndexedSeq[IndexedSeq[Int]],
    colDeps: IndexedSeq[IndexedSeq[Int]],
    typ: BlockMatrixType,
  ): DenseContexts = {
    val rowDepsLit = Literal(TArray(TArray(TInt32)), rowDeps)
    val colDepsLit = Literal(TArray(TArray(TInt32)), colDeps)
    assert(rowDeps.nonEmpty || colDeps.nonEmpty)
    if (rowDeps.isEmpty) {
      val newContexts = ToArray(flatMapIR(ToStream(colDepsLit)) { localColDeps =>
        mapIR(rangeIR(nRows)) { i =>
          IRBuilder.scoped { ib =>
            val localContexts = ToArray(mapIR(ToStream(localColDeps))(jl => dynamic(i, jl)))
            DynamicDenseContexts(
              1,
              ib.memoize(ArrayLen(localColDeps)),
              ib.memoize(localContexts),
            ).irValue
          }
        }
      })
      return DenseContexts(ib, sparsity.copy(nCols = colDeps.length), newContexts)
    }
    if (colDeps.isEmpty) {
      val newContexts = ToArray(flatMapIR(rangeIR(nCols)) { j =>
        mapIR(ToStream(rowDepsLit)) { localRowDeps =>
          IRBuilder.scoped { ib =>
            val localContexts = ToArray(mapIR(ToStream(localRowDeps))(il => dynamic(il, j)))
            DynamicDenseContexts(
              ib.memoize(ArrayLen(localRowDeps)),
              1,
              ib.memoize(localContexts),
            ).irValue
          }
        }
      })
      return DenseContexts(ib, sparsity.copy(nRows = rowDeps.length), newContexts)
    }
    val newContexts = ToArray(flatMapIR(ToStream(colDepsLit)) { localColDeps =>
      mapIR(ToStream(rowDepsLit)) { localRowDeps =>
        IRBuilder.scoped { ib =>
          val localContexts = ToArray(flatMapIR(ToStream(localColDeps)) { jl =>
            mapIR(ToStream(localRowDeps))(il => dynamic(il, jl))
          })
          DynamicDenseContexts(
            ib.memoize(ArrayLen(localRowDeps)),
            ib.memoize(ArrayLen(localColDeps)),
            ib.memoize(localContexts),
          ).irValue
        }
      }
    })
    DenseContexts(ib, MatrixSparsity.Dense(rowDeps.length, colDeps.length), newContexts)
  }

  def collect(makeBlock: (Ref, Ref, Ref) => IR): IR = {
    NDArrayConcat(
      ToArray(mapIR(rangeIR(nCols)) { j =>
        val colBlocks = mapIR(rangeIR(nRows)) { i =>
          bindIR(ArrayRef(contexts, j * nRows + i))(ctx => makeBlock(i, j, ctx))
        }
        NDArrayConcat(ToArray(colBlocks), 0)
      }),
      1,
    )
  }
}

object SparseContexts {
  def apply(ib: IRBuilder, sparsity: MatrixSparsity.Sparse, contexts: IR): SparseContexts =
    new SparseContexts(sparsity, DynamicSparseContexts(ib, sparsity, contexts))
}

case class SparseContexts(
  sparsity: MatrixSparsity.Sparse,
  dynamic: DynamicSparseContexts,
) extends BMSContexts {
  override val elementType: Type = contexts.typ.asInstanceOf[TArray].elementType

  override def contexts: TrivialIR = dynamic.contexts

  override def print(ctx: ExecuteContext): Unit =
    println(
      s"SparseContexts:\n  nRows = ${Pretty(ctx, nRows)}\n  nCols = ${Pretty(ctx, nCols)}\n  contexts = ${Pretty(ctx, contexts)}"
    )

  override def transpose(ib: IRBuilder): SparseContexts = {
    val newToOldPos = sparsity.transposeNewToOld
    val newToOldPosIR = Literal(TArray(TInt32), newToOldPos)
    val newContexts =
      ToArray(mapIR(ToStream(newToOldPosIR))(oldPos => ArrayRef(contexts, oldPos)))
    SparseContexts(ib, sparsity.transpose, newContexts)
  }

  override def map(ib: IRBuilder)(f: (IR, IR, IR, IR) => IR): SparseContexts =
    new SparseContexts(sparsity, dynamic.map(ib)(f))

  def withNewSparsity(ib: IRBuilder, newSparsity: MatrixSparsity): BMSContexts = {
    require(newSparsity.nRows == sparsity.nRows && newSparsity.nCols == sparsity.nCols)
    newSparsity match {
      case sparse: MatrixSparsity.Sparse =>
        val newToOld = Literal(TArray(TInt32), sparsity.newToOldPosNonSubset(sparse))
        val newContexts = ToArray(mapIR(ToStream(newToOld))(ArrayRef(contexts, _)))
        SparseContexts(ib, sparse, newContexts)
      case dense: MatrixSparsity.Dense =>
        val indices = Literal(TArray(TInt32), sparsity.definedBlocksColMajorLinear)
        val contextType = contexts.typ.asInstanceOf[TContainer].elementType
        val scatteredContexts = invoke(
          "scatter",
          TArray(contextType),
          typeArgs = ArraySeq(contextType),
          contexts,
          indices,
          sparsity.nRows * sparsity.nCols,
        )
        DenseContexts(ib, dense, scatteredContexts)
    }
  }

  override def zip(ib: IRBuilder, other: BMSContexts): BMSContexts = {
    assert(sparsity == other.sparsity)
    val newContexts = ib.memoize(ToArray(zip2(
      this.contexts,
      other.contexts,
      ArrayZipBehavior.AssertSameLength,
    )((l, r) => maketuple(l, r))))
    new SparseContexts(sparsity, dynamic.withNewContexts(newContexts))
  }

  override def groupedByCol(ib: IRBuilder): SparseContexts = {
    val rowPos = sparsity.toDCSC.rowPos
    val rowPosIR = Literal(TArray(TInt32), rowPos)
    val groupedContexts = ToArray(mapIR(rangeIR(rowPos.length - 1)) { colPos =>
      sliceArrayIR(contexts, ArrayRef(rowPosIR, colPos), ArrayRef(rowPosIR, colPos + 1))
    })
    SparseContexts(ib, sparsity.condenseCols, groupedContexts)
  }

  override def grouped(
    ib: IRBuilder,
    rowDeps: IndexedSeq[IndexedSeq[Int]],
    colDeps: IndexedSeq[IndexedSeq[Int]],
    typ: BlockMatrixType,
  ): SparseContexts = {
    val newSparsity = typ.sparsity.asInstanceOf[MatrixSparsity.Sparse]
    val blockSparsities = newSparsity.definedCoords.map { coords =>
      sparsity.filter(rowDeps(coords._1), colDeps(coords._2))
    }
    val blockSparsitiesTuples = blockSparsities.map { sparsity =>
      val csc = sparsity.toCSC
      Row(sparsity.nRows, sparsity.nCols, csc.rowPos, csc.rowIdx)
    }
    val blockNewToOld = blockSparsities.map(sparsity.newToOldPos)
    val blockSparsityType = TStruct(
      "nRows" -> TInt32,
      "nCols" -> TInt32,
      "rowPos" -> TArray(TInt32),
      "rowIdx" -> TArray(TInt32),
    )
    val blockSparsitiesIR = Literal(TArray(blockSparsityType), blockSparsitiesTuples)
    val blockNewToOldIR = Literal(TArray(TArray(TInt32)), blockNewToOld)
    val newContexts = ToArray(zipIR(
      ArraySeq(ToStream(blockSparsitiesIR), ToStream(blockNewToOldIR)),
      ArrayZipBehavior.AssertSameLength,
    ) { case Seq(blockSparsity, newToOld) =>
      InsertFields(
        blockSparsity,
        ArraySeq("contexts" -> mapIR(ToStream(newToOld))(ArrayRef(contexts, _))),
      )
    })

    new SparseContexts(newSparsity, DynamicSparseContexts(ib, newSparsity, newContexts))
  }

  override def collect(makeBlock: (Ref, Ref, Ref) => IR): IR =
    dynamic.collect(makeBlock)
}

object BlockMatrixStage2 {
  def fromOldBMS(bms: BlockMatrixStage, typ: BlockMatrixType, ib: IRBuilder): BlockMatrixStage2 = {
    val blocks = typ.sparsity.definedCoords
    val ctxsArray = MakeArray(blocks.map(idx => bms.blockContext(idx)), TArray(bms.ctxType))

    BlockMatrixStage2(
      bms.broadcastVals,
      typ,
      BMSContexts(ib, typ.sparsity, ctxsArray),
      bms.blockBody,
    )
  }

  def empty(ib: IRBuilder, eltType: Type): BlockMatrixStage2 = {
    val ctxType = TNDArray(eltType, Nat(2))
    BlockMatrixStage2(
      IndexedSeq(),
      BlockMatrixType.dense(eltType, 0, 0, 0),
      DenseContexts(ib, MatrixSparsity.Dense(0, 0), MakeArray(FastSeq(), TArray(ctxType))),
      _ => NA(ctxType),
    )
  }

  def broadcastVector(ib: IRBuilder, vector: IR, typ: BlockMatrixType, asRowVector: Boolean)
    : BlockMatrixStage2 = {
    val v: Ref = ib.strictMemoize(vector)
    val contexts = BMSContexts.tabulate(ib, typ.sparsity)({ (i, j) =>
      val (m, n) = typ.blockShapeIR(i, j)
      val start = (if (asRowVector) j.toL else i.toL) * typ.blockSize.toLong
      makestruct("start" -> start, "shape" -> MakeTuple.ordered(FastSeq[IR](m, n)))
    })
    BlockMatrixStage2(
      FastSeq(v),
      typ,
      contexts,
      ctx => {
        bindIRs(GetField(ctx, "shape"), GetField(ctx, "start")) { case Seq(shape, start) =>
          bindIRs(
            if (asRowVector) GetTupleElement(shape, 1) else GetTupleElement(shape, 0),
            if (asRowVector) GetTupleElement(shape, 0) else GetTupleElement(shape, 1),
          ) { case Seq(len, nRep) =>
            bindIR(
              NDArrayReshape(
                NDArraySlice(v, maketuple(maketuple(start, start + len, 1L))),
                if (asRowVector) maketuple(1L, len) else maketuple(len.toL, 1L),
                ErrorIDs.NO_ERROR,
              )
            ) { sliced =>
              NDArrayConcat(
                ToArray(mapIR(rangeIR(nRep.toI))(_ => sliced)),
                if (asRowVector) 0 else 1,
              )
            }
          }
        }
      },
    )
  }

  def apply(
    broadcastVals: IndexedSeq[Ref],
    typ: BlockMatrixType,
    contexts: BMSContexts,
    _blockIR: Ref => IR,
  ): BlockMatrixStage2 = {
    val ctxRef = Ref(freshName(), contexts.elementType)
    val blockIR = _blockIR(ctxRef)

    new BlockMatrixStage2(broadcastVals, typ, contexts, ctxRef.name, blockIR)
  }
}

class BlockMatrixStage2 private (
  val broadcastVals: IndexedSeq[Ref],
  val typ: BlockMatrixType,
  val contexts: BMSContexts,
  private val ctxRefName: Name,
  private val _blockIR: IR,
) {
  assert(typ.sparsity == contexts.sparsity, s"${typ.sparsity}\n${contexts.sparsity}")

  assert {
    def literalOrRef(x: IR) = x.isInstanceOf[Literal] || x.isInstanceOf[Ref]
    contexts.contexts match {
      case x: MakeStruct => x.fields.forall(f => literalOrRef(f._2))
      case x => literalOrRef(x)
    }
  }

  def print(ctx: ExecuteContext): Unit =
    println(s"contexts:\n${contexts.print(ctx)}\nbody($ctxRefName) = ${Pretty(ctx, _blockIR)}")

  def blockIR(ctx: Ref): IR =
    if (ctx.name == ctxRefName)
      _blockIR
    else
      Let(FastSeq(ctxRefName -> ctx), _blockIR)

  private def ctxType: Type = contexts.elementType

  def toOldBMS: BlockMatrixStage = {
    new BlockMatrixStage(broadcastVals, ctxType) {
      override def blockContext(idx: (Int, Int)): IR = contexts.dynamic(idx._1, idx._2)

      override def blockBody(ctxRef: Ref): IR =
        Let(FastSeq(ctxRefName -> ctxRef), _blockIR)
    }
  }

  private def getBlock(i: IR, j: IR): IR =
    Let(FastSeq(ctxRefName -> contexts.dynamic(i, j)), _blockIR)

  def getElement(i: IR, j: IR): IR = {
    assert(i.typ == TInt64)
    assert(j.typ == TInt64)
    val blockSize = typ.blockSize.toLong
    bindIR(i floorDiv blockSize) { rowBlock =>
      bindIR(j floorDiv blockSize) { colBlock =>
        val iInBlock = i - rowBlock * blockSize
        val jInBlock = j - colBlock * blockSize

        NDArrayRef(getBlock(rowBlock.toI, colBlock.toI), FastSeq(iInBlock, jInBlock), -1)
      }
    }
  }

  def transposed(ib: IRBuilder): BlockMatrixStage2 = {
    val newBlockIR = NDArrayReindex(_blockIR, FastSeq(1, 0))
    new BlockMatrixStage2(
      broadcastVals,
      typ.transpose,
      contexts.transpose(ib),
      ctxRefName,
      newBlockIR,
    )
  }

  def densify(ib: IRBuilder): BlockMatrixStage2 = contexts match {
    case _: DenseContexts => this
    case contexts: SparseContexts =>
      val newContexts = contexts
        .withNewSparsity(ib, contexts.sparsity.toDense)
        .map(ib) { (i, j, pos, oldContext) =>
          val (m, n) = typ.blockShapeIR(i, j)
          makestruct("oldContext" -> oldContext, "nRows" -> m, "nCols" -> n)
        }
      def newBlock(context: Ref): IR = {
        bindIR(GetField(context, "oldContext")) { oldContext =>
          If(
            IsNA(oldContext),
            MakeNDArray.fill(
              zero(typ.elementType),
              FastSeq(GetField(context, "nRows"), GetField(context, "nCols")),
              False(),
            ),
            blockIR(oldContext),
          )
        }
      }
      BlockMatrixStage2(broadcastVals, typ.densify, newContexts, newBlock)
  }

  def withSparsity(
    ib: IRBuilder,
    newSparsity: MatrixSparsity.Sparse,
    isSubset: Boolean = false,
  ): BlockMatrixStage2 = {
    if (newSparsity == typ.sparsity)
      return this

    val newType = typ.copy(sparsity = newSparsity)
    if (newSparsity.isSubsetOf(contexts.sparsity)) {
      val newContexts = contexts.withNewSparsity(ib, newSparsity)
      new BlockMatrixStage2(broadcastVals, newType, newContexts, ctxRefName, _blockIR)
    } else {
      val newContexts =
        contexts.withNewSparsity(ib, newSparsity).map(ib) { (i, j, pos, oldContext) =>
          val (m, n) = typ.blockShapeIR(i, j)
          makestruct("oldContext" -> oldContext, "nRows" -> m, "nCols" -> n)
        }

      def newBlock(context: Ref): IR = {
        bindIR(GetField(context, "oldContext")) { oldContext =>
          If(
            IsNA(oldContext),
            MakeNDArray.fill(
              zero(typ.elementType),
              FastSeq(GetField(context, "nRows"), GetField(context, "nCols")),
              False(),
            ),
            blockIR(oldContext),
          )
        }
      }

      BlockMatrixStage2(broadcastVals, newType, newContexts, newBlock)
    }
  }

  def mapBody(f: IR => IR): BlockMatrixStage2 = {
    val newBlockIR = bindIR(_blockIR)(f)
    val newType = typ.copy(elementType = newBlockIR.typ.asInstanceOf[TNDArray].elementType)
    new BlockMatrixStage2(broadcastVals, newType, contexts, ctxRefName, newBlockIR)
  }

  def mapBody2(
    ib: IRBuilder,
    other: BlockMatrixStage2,
    sparsityStrategy: SparsityStrategy,
  )(
    f: (IR, IR) => IR
  ): BlockMatrixStage2 = {
    val (alignedLeft, alignedRight) = (contexts, other.contexts, sparsityStrategy) match {
      case (_: DenseContexts, _: DenseContexts, _) =>
        (this, other)
      case (_: DenseContexts, _: SparseContexts, UnionBlocks) =>
        (this, other.densify(ib))
      case (_: SparseContexts, _: DenseContexts, UnionBlocks) =>
        (this.densify(ib), other)
      case (_: SparseContexts, _: SparseContexts, UnionBlocks) =>
        val newSparsity = UnionBlocks.mergeSparsity(typ.sparsity, other.typ.sparsity)
          .asInstanceOf[MatrixSparsity.Sparse]
        (this.withSparsity(ib, newSparsity), other.withSparsity(ib, newSparsity))
      case (_: DenseContexts, sparse: SparseContexts, IntersectionBlocks) =>
        (this.withSparsity(ib, sparse.sparsity), other)
      case (sparse: SparseContexts, _: DenseContexts, IntersectionBlocks) =>
        (this, other.withSparsity(ib, sparse.sparsity))
      case (_: SparseContexts, _: SparseContexts, IntersectionBlocks) =>
        val newSparsity = IntersectionBlocks.mergeSparsity(typ.sparsity, other.typ.sparsity)
          .asInstanceOf[MatrixSparsity.Sparse]
        (this.withSparsity(ib, newSparsity), other.withSparsity(ib, newSparsity))
    }

    alignedLeft.mapBody2Aligned(ib, alignedRight)(f)
  }

  private def mapBody2Aligned(ib: IRBuilder, other: BlockMatrixStage2)(f: (IR, IR) => IR) = {
    val newContexts = contexts.zip(ib, other.contexts)
    val ctxRef = Ref(freshName(), newContexts.elementType)
    val newBlockIR =
      bindIRs(GetTupleElement(ctxRef, 0), GetTupleElement(ctxRef, 1)) { case Seq(l, r) =>
        f(this.blockIR(l), other.blockIR(r))
      }
    val newType = typ.copy(elementType = newBlockIR.typ.asInstanceOf[TNDArray].elementType)
    new BlockMatrixStage2(
      broadcastVals ++ other.broadcastVals,
      newType,
      newContexts,
      ctxRef.name,
      newBlockIR,
    )
  }

  def filter(
    keepRows: IndexedSeq[Long],
    keepCols: IndexedSeq[Long],
    typ: BlockMatrixType,
    ib: IRBuilder,
  ): BlockMatrixStage2 = {
    val rowBlockDependents =
      BlockMatrixType.getBlockDependencies(keepRows.grouped(typ.blockSize), typ.blockSize)
    val colBlockDependents =
      BlockMatrixType.getBlockDependencies(keepCols.grouped(typ.blockSize), typ.blockSize)

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
      keepRows.grouped(typ.blockSize).map(localIndices).toFastSeq
    val groupedKeepCols: IndexedSeq[IndexedSeq[IndexedSeq[Long]]] =
      keepCols.grouped(typ.blockSize).map(localIndices).toFastSeq
    val t = TArray(TArray(TArray(TInt64)))
    val groupedKeepRowsLit = if (keepRows.isEmpty) NA(t) else Literal(t, groupedKeepRows)
    val groupedKeepColsLit = if (keepCols.isEmpty) NA(t) else Literal(t, groupedKeepCols)
    val groupedContexts: BMSContexts =
      contexts.grouped(ib, rowBlockDependents, colBlockDependents, typ)
    val groupedContextsWithIndices = groupedContexts.map(ib) { (i, j, pos, context) =>
      maketuple(context, ArrayRef(groupedKeepRowsLit, i), ArrayRef(groupedKeepColsLit, j))
    }

    def newBody(ctx: Ref): IR = {
      IRBuilder.scoped { ib =>
        val localContexts = contexts match {
          case _: DenseContexts => DynamicDenseContexts(ib, GetTupleElement(ctx, 0))
          case _: SparseContexts => DynamicSparseContexts(ib, GetTupleElement(ctx, 0))
        }
        val localKeepRows = GetTupleElement(ctx, 1)
        val localKeepCols = GetTupleElement(ctx, 2)
        localContexts.collect { (i, j, localContext) =>
          bindIRs(ArrayRef(localKeepRows, i), ArrayRef(localKeepCols, j)) { case Seq(rows, cols) =>
            Coalesce(FastSeq(
              // FIXME: assumes blockIR is strict (preserves missing)
              NDArrayFilter(blockIR(localContext), FastSeq(rows, cols)),
              MakeNDArray.fill(
                zero(typ.elementType),
                FastSeq(ArrayLen(rows).toL, ArrayLen(cols).toL),
                False(),
              ),
            ))
          }
        }
      }
    }

    BlockMatrixStage2(broadcastVals, typ, groupedContextsWithIndices, newBody)
  }

  def zeroBand(ib: IRBuilder, lower: Long, upper: Long, typ: BlockMatrixType): BlockMatrixStage2 = {
    val ctxs = contexts.map(ib)((i, j, _, context) => maketuple(context, i, j))

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
        If(
          -localLower >= (nRowsInBlock - 1L) && localUpper >= (nColsInBlock - 1L),
          block,
          invoke("zero_band", TNDArray(TFloat64, Nat(2)), block, localLower, localUpper),
        )
      }
    }

    BlockMatrixStage2(broadcastVals, typ, ctxs, newBody)
  }

  def zeroRowIntervals(
    ib: IRBuilder,
    starts: IndexedSeq[Long],
    stops: IndexedSeq[Long],
    typ: BlockMatrixType,
  ): BlockMatrixStage2 = {
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
      val starts = ToArray(mapIR(ToStream(GetTupleElement(ctx, 3))) { s =>
        minIR(maxIR(s - j.toL * typ.blockSize.toLong, 0L), nCols)
      })
      val stops = ToArray(mapIR(ToStream(GetTupleElement(ctx, 4))) { s =>
        minIR(maxIR(s - j.toL * typ.blockSize.toLong, 0L), nCols)
      })
      bindIRs(oldCtx) { case Seq(oldCtx) =>
        invoke("zero_row_intervals", TNDArray(TFloat64, Nat(2)), blockIR(oldCtx), starts, stops)
      }
    }

    BlockMatrixStage2(broadcastVals, typ, ctxs, newBody)
  }

  def toTableStage(ib: IRBuilder, ctx: ExecuteContext, bmTyp: BlockMatrixType): TableStage = {
    val bodyFreeVars = FreeVariables(_blockIR, supportsAgg = false, supportsScan = false)
    val bcFields = broadcastVals.filter { case Ref(f, _) =>
      bodyFreeVars.eval.lookupOption(f).isDefined
    }

    val contextsIR = ToStream(contexts.map(ib) { (rowIdx, colIdx, pos, oldContext) =>
      maketuple(rowIdx, colIdx, oldContext)
    }.contexts)

    val emptyGlobals = MakeStruct(FastSeq())
    val globalsId = freshName()
    val letBindings = ib.getBindings :+ globalsId -> emptyGlobals

    def tsPartitionFunction(newCtxRef: Ref): IR = {
      val s = makestruct(
        "blockRow" -> GetTupleElement(newCtxRef, 0),
        "blockCol" -> GetTupleElement(newCtxRef, 1),
        "block" -> Let(FastSeq(ctxRefName -> GetTupleElement(newCtxRef, 2)), _blockIR),
      )
      MakeStream(FastSeq(s), TStream(s.typ))
    }

    TableStage(
      letBindings,
      bcFields.map(ref => ref.name -> ref) :+ globalsId -> Ref(globalsId, emptyGlobals.typ),
      Ref(globalsId, emptyGlobals.typ),
      RVDPartitioner.unkeyed(ctx.stateManager, bmTyp.nDefinedBlocks),
      TableStageDependency.none,
      contextsIR,
      tsPartitionFunction,
    )
  }

  def collectBlocks(
    ib: IRBuilder,
    staticID: String,
    dynamicID: IR = NA(TString),
  )(
    f: (TrivialIR, TrivialIR, TrivialIR) => IR // (ctx, pos, block)
  ): IR = {
    val newCtxRef = Ref(freshName(), TTuple(TInt32, ctxType))
    val body = IRBuilder.scoped { bodyIB =>
      val pos = bodyIB.memoize(GetTupleElement(newCtxRef, 0))
      val ctx = bodyIB.strictMemoize(GetTupleElement(newCtxRef, 1), ctxRefName)
      val block = bodyIB.memoize(_blockIR)
      f(ctx, pos, block)
    }

    val bodyFreeVars = FreeVariables(body, supportsAgg = false, supportsScan = false)
    val bcFields = broadcastVals.filter { case Ref(f, _) =>
      bodyFreeVars.eval.lookupOption(f).isDefined
    }
    val bcVals = MakeStruct(bcFields.map(ref => ref.name.str -> ref))
    val bcRef = Ref(freshName(), bcVals.typ)
    val wrappedBody = Let(bcFields.map(ref => ref.name -> GetField(bcRef, ref.name.str)), body)

    val cdaContexts = ToStream(contexts.map(ib) { (rowIdx, colIdx, pos, oldContext) =>
      maketuple(pos, oldContext)
    }.contexts)

    CollectDistributedArray(
      cdaContexts,
      bcVals,
      newCtxRef.name,
      bcRef.name,
      wrappedBody,
      dynamicID,
      staticID,
    )
  }

  def collectLocal(ib: IRBuilder, staticID: String, dynamicID: IR = NA(TString)): IR = {
    val blockResults = collectBlocks(ib, staticID, dynamicID)((_, _, b) => b)
    val blocks = contexts match {
      case x: DenseContexts => DenseContexts(ib, x.sparsity, blockResults)
      case x: SparseContexts => SparseContexts(ib, x.sparsity, blockResults)
    }

    blocks.collect { (i, j, block) =>
      val (m, n) = typ.blockShapeIR(i, j)
      val zeroBlock: IR = MakeNDArray.fill(zero(typ.elementType), FastSeq(m, n), False())
      Coalesce(FastSeq(block, zeroBlock))
    }
  }
}

object LowerBlockMatrixIR {
  def apply(
    node: IR,
    typesToLower: DArrayLowering.Type,
    ctx: ExecuteContext,
    analyses: LoweringAnalyses,
  ): IR = {

    def lower(bmir: BlockMatrixIR, ib: IRBuilder) =
      LowerBlockMatrixIR.lower(ib, bmir, typesToLower, ctx, analyses)

    IRBuilder.scoped { ib =>
      node match {
        case BlockMatrixCollect(child) =>
          lower(child, ib).collectLocal(ib, "block_matrix_collect")
        case BlockMatrixToValueApply(child, GetElement(IndexedSeq(i, j))) =>
          lower(child, ib).getElement(i, j)
        case BlockMatrixWrite(child, writer) =>
          writer.lower(
            ctx,
            lower(child, ib),
            ib,
            TypeWithRequiredness(child.typ.elementType),
          ) // FIXME: BlockMatrixIR is currently ignored in Requiredness inference since all eltTypes are +TFloat64
        case BlockMatrixMultiWrite(_, _) => unimplemented(ctx, node)
        case node if node.children.exists(_.isInstanceOf[BlockMatrixIR]) =>
          throw new LowererUnsupportedOperation(
            s"IR nodes with BlockMatrixIR children need explicit rules: \n${Pretty(ctx, node)}"
          )

        case node =>
          throw new LowererUnsupportedOperation(
            s"Value IRs with no BlockMatrixIR children must be lowered through LowerIR: \n${Pretty(ctx, node)}"
          )
      }
    }
  }

  // This lowers a BlockMatrixIR to an unkeyed TableStage with rows of (blockRow, blockCol, block)
  def lowerToTableStage(
    bmir: BlockMatrixIR,
    typesToLower: DArrayLowering.Type,
    ctx: ExecuteContext,
    analyses: LoweringAnalyses,
  ): TableStage = {
    val ib = new IRBuilder()
    val bms = lower(ib, bmir, typesToLower, ctx, analyses)
    bms.toTableStage(ib, ctx, bmir.typ)
  }

  private def unimplemented[T](ctx: ExecuteContext, node: BaseIR): T =
    throw new LowererUnsupportedOperation(s"unimplemented: \n${Pretty(ctx, node)}")

  def lower(
    ib: IRBuilder,
    bmir: BlockMatrixIR,
    typesToLower: DArrayLowering.Type,
    ctx: ExecuteContext,
    analyses: LoweringAnalyses,
  ): BlockMatrixStage2 = {
    if (!DArrayLowering.lowerBM(typesToLower))
      throw new LowererUnsupportedOperation(
        "found BlockMatrixIR in lowering; lowering only TableIRs."
      )
    bmir.children.foreach {
      case c: BlockMatrixIR if c.typ.blockSize != bmir.typ.blockSize =>
        throw new LowererUnsupportedOperation(
          s"Can't lower node with mismatched block sizes: ${bmir.typ.blockSize} vs child ${c.typ.blockSize}\n\n ${Pretty(ctx, bmir)}"
        )
      case _ =>
    }
    if (bmir.typ.nRows == 0L && bmir.typ.nCols == 0L)
      BlockMatrixStage2.empty(ib, bmir.typ.elementType)
    else lowerNonEmpty2(ib, bmir, typesToLower, ctx, analyses)
  }

  def lowerNonEmpty2(
    ib: IRBuilder,
    bmir: BlockMatrixIR,
    typesToLower: DArrayLowering.Type,
    ctx: ExecuteContext,
    analyses: LoweringAnalyses,
  ): BlockMatrixStage2 = {

    def lower(ir: BlockMatrixIR, ib: IRBuilder = ib): BlockMatrixStage2 =
      LowerBlockMatrixIR.lower(ib, ir, typesToLower, ctx, analyses)

    bmir match {
      case BlockMatrixRead(reader) => reader.lower(ctx, ib)

      case ValueToBlockMatrix(child, _, _)
          if !child.typ.isInstanceOf[TArray] && !child.typ.isInstanceOf[TNDArray] =>
        val element = LowerToCDA.lower(child, typesToLower, ctx, analyses)
        val contextsIR = MakeArray(MakeStruct(FastSeq()))

        def blockIR(ctxRef: Ref): IR = MakeNDArray(
          MakeArray(element),
          MakeTuple(FastSeq((0, I64(1)), (1, I64(1)))),
          False(),
          ErrorIDs.NO_ERROR,
        )

        BlockMatrixStage2(
          FastSeq(),
          bmir.typ,
          BMSContexts(ib, bmir.typ.sparsity, contextsIR),
          blockIR,
        )

      case x @ ValueToBlockMatrix(child, _, _) =>
        val lowered = LowerToCDA.lower(child, typesToLower, ctx, analyses)
        val nd = ib.memoize(child.typ match {
          case _: TArray => MakeNDArray(
              lowered,
              MakeTuple.ordered(FastSeq(I64(x.typ.nRows), I64(x.typ.nCols))),
              True(),
              ErrorIDs.NO_ERROR,
            )
          case _: TNDArray => lowered
        })
        val contexts = BMSContexts.tabulate(ib, x.typ.sparsity)({ (rowBlockIdx, colBlockIdx) =>
          val rowStartIdx = rowBlockIdx.toL * I64(x.typ.blockSize.toLong)
          val colStartIdx = colBlockIdx.toL * I64(x.typ.blockSize.toLong)
          val (numRowsBlock, numColsBlock) = x.typ.blockShapeIR(rowBlockIdx, colBlockIdx)
          bindIRs(rowStartIdx, colStartIdx) { case Seq(rowStartIdx, colStartIdx) =>
            NDArraySlice(
              nd,
              MakeTuple.ordered(FastSeq(
                MakeTuple.ordered(FastSeq(
                  rowStartIdx,
                  rowStartIdx + numRowsBlock,
                  I64(1),
                )),
                MakeTuple.ordered(FastSeq(
                  colStartIdx,
                  colStartIdx + numColsBlock,
                  I64(1),
                )),
              )),
            )
          }
        })

        BlockMatrixStage2(FastSeq(), x.typ, contexts, identity)

      case x @ BlockMatrixRandom(staticUID, gaussian, _, _) =>
        val contexts = BMSContexts.tabulate(ib, x.typ.sparsity)({ (rowIdx, colIdx) =>
          val (m, n) = x.typ.blockShapeIR(rowIdx, colIdx)
          MakeTuple.ordered(FastSeq(m, n, rowIdx * x.typ.nColBlocks + colIdx))
        })

        def bodyIR(ctx: Ref): IR = {
          val m = GetTupleElement(ctx, 0)
          val n = GetTupleElement(ctx, 1)
          val i = GetTupleElement(ctx, 2)
          val f = if (gaussian) "rand_norm_nd" else "rand_unif_nd"
          val rngState = RNGSplit(RNGStateLiteral(), Cast(i, TInt64))
          invokeSeeded(f, staticUID, TNDArray(TFloat64, Nat(2)), rngState, m, n, F64(0.0), F64(1.0))
        }

        BlockMatrixStage2(FastSeq(), x.typ, contexts, bodyIR)

      case BlockMatrixAgg(child, IndexedSeq(0, 1) /* axesToSumOut */ ) =>
        // Reduce to a single row and column
        val summedChild = lower(child).mapBody { body =>
          NDArrayReshape(
            NDArrayAgg(body, IndexedSeq(0, 1)),
            MakeTuple.ordered(FastSeq(I64(1), I64(1))),
            ErrorIDs.NO_ERROR,
          )
        }
        val blockResults =
          summedChild.collectBlocks(ib, "block_matrix_agg_axes_0_1")((_, _, block) => block)
        val ndArrayResults = NDArrayConcat(blockResults, 0)
        val aggResult = NDArrayAgg(ndArrayResults, IndexedSeq(0, 1))
        val newBlockIR =
          NDArrayReshape(aggResult, MakeTuple.ordered(FastSeq(I64(1), I64(1))), ErrorIDs.NO_ERROR)
        val contexts = BMSContexts.tabulate(ib, bmir.typ.sparsity)((_, _) => newBlockIR)

        BlockMatrixStage2(FastSeq(), bmir.typ, contexts, identity)

      case BlockMatrixAgg(child, IndexedSeq(0) /* axesToSumOut */ ) =>
        // Number of rows goes to 1. Number of cols remains the same.
        val loweredChild = lower(child)
        val contexts = loweredChild.contexts.groupedByCol(ib)

        BlockMatrixStage2(
          loweredChild.broadcastVals,
          bmir.typ,
          contexts,
          (ctx) =>
            streamAggIR(mapIR(ToStream(ctx)) { childCtx =>
              bindIR(NDArrayAgg(loweredChild.blockIR(childCtx), IndexedSeq(0))) { aggedND =>
                NDArrayReshape(
                  aggedND,
                  MakeTuple.ordered(FastSeq(
                    I64(1),
                    GetTupleElement(NDArrayShape(aggedND), 0),
                  )),
                  ErrorIDs.NO_ERROR,
                )
              }
            })(block => ApplyAggOp(NDArraySum())(block)),
        )

      case BlockMatrixAgg(child, IndexedSeq(1) /* axesToSumOut */ ) =>
        // Number of cols goes to 1. Number of rows remains the same.
        val loweredChild = lower(child)

        val contexts = loweredChild.contexts.transpose(ib).groupedByCol(ib).transpose(ib)

        BlockMatrixStage2(
          loweredChild.broadcastVals,
          bmir.typ,
          contexts,
          (ctx) =>
            streamAggIR(mapIR(ToStream(ctx)) { childCtx =>
              bindIR(NDArrayAgg(loweredChild.blockIR(childCtx), IndexedSeq(1))) { aggedND =>
                NDArrayReshape(
                  aggedND,
                  MakeTuple.ordered(FastSeq(
                    GetTupleElement(NDArrayShape(aggedND), 0),
                    I64(1),
                  )),
                  ErrorIDs.NO_ERROR,
                )
              }
            })(block => ApplyAggOp(NDArraySum())(block)),
        )

      case BlockMatrixMap(child, eltName, f, _) =>
        lower(child).mapBody(body => NDArrayMap(body, eltName, f))

      case BlockMatrixMap2(left, right, lname, rname, f, sparsityStrategy) =>
        val loweredLeft = lower(left)
        val loweredRight = lower(right)
        loweredLeft.mapBody2(ib, loweredRight, sparsityStrategy)({ (lBody, rBody) =>
          NDArrayMap2(lBody, rBody, lname, rname, f, ErrorIDs.NO_ERROR)
        })

      case x @ BlockMatrixBroadcast(child, IndexedSeq(), _, _) =>
        val elt = ib.strictMemoize(IRBuilder.scoped { ib =>
          val lowered = lower(child, ib)
          lowered.getElement(0L, 0L)
        })

        val contexts = BMSContexts.tabulate(ib, x.typ.sparsity)({ (rowIdx, colIdx) =>
          val (i, j) = x.typ.blockShapeIR(rowIdx, colIdx)
          maketuple(i, j)
        })

        BlockMatrixStage2(
          FastSeq(elt),
          x.typ,
          contexts,
          (ctxRef: Ref) =>
            MakeNDArray.fill(
              elt,
              FastSeq(GetTupleElement(ctxRef, 0), GetTupleElement(ctxRef, 1)),
              True(),
            ),
        )

      case BlockMatrixBroadcast(child, IndexedSeq(axis), _, _) =>
        val len = scala.math.max(child.typ.nRows, child.typ.nCols)
        val vector = NDArrayReshape(
          IRBuilder.scoped { ib =>
            lower(child, ib).collectLocal(ib, "block_matrix_broadcast_single_axis")
          },
          MakeTuple.ordered(FastSeq(I64(len))),
          ErrorIDs.NO_ERROR,
        )
        BlockMatrixStage2.broadcastVector(ib, vector, bmir.typ, asRowVector = axis == 1)

      case x @ BlockMatrixBroadcast(child, IndexedSeq(axis, axis2), _, _)
          if (axis == axis2) => // diagonal as row/col vector
        val diagLen = math.min(child.typ.nRowBlocks, child.typ.nColBlocks)
        val diagSparsity = MatrixSparsity.Sparse.sorted(
          child.typ.nRowBlocks,
          child.typ.nColBlocks,
          ArraySeq.tabulate(diagLen)(i => (i, i)),
        )

        val diagArray = IRBuilder.scoped { ib =>
          lower(child, ib)
            .withSparsity(ib, diagSparsity)
            .collectBlocks(ib, "block_matrix_broadcast_diagonal") { (ctx, idx, block) =>
              bindIR(NDArrayShape(block)) { shape =>
                val blockDiagLen = minIR(GetTupleElement(shape, 0), GetTupleElement(shape, 1))
                ToArray(mapIR(rangeIR(blockDiagLen.toI)) { i =>
                  NDArrayRef(block, FastSeq(Cast(i, TInt64), Cast(i, TInt64)), ErrorIDs.NO_ERROR)
                })
              }
            }
        }

        val diagVector = MakeNDArray(
          ToArray(flatten(diagArray)),
          maketuple(math.min(child.typ.nRows, child.typ.nCols)),
          true,
          ErrorIDs.NO_ERROR,
        )
        BlockMatrixStage2.broadcastVector(ib, diagVector, x.typ, asRowVector = axis == 0)

      case BlockMatrixBroadcast(child, IndexedSeq(1, 0), _, _) => // transpose
        lower(child).transposed(ib)

      case BlockMatrixBroadcast(child, IndexedSeq(0, 1), _, _) =>
        lower(child)

      case x @ BlockMatrixFilter(child, keep) =>
        val Array(keepRow, keepCol) = keep
        lower(child).filter(keepRow, keepCol, x.typ, ib)

      case BlockMatrixDensify(child) =>
        lower(child).densify(ib)

      case x @ BlockMatrixSparsify(child, sparsifier) =>
        val loweredChild = lower(child).withSparsity(
          ib,
          x.typ.sparsity.asInstanceOf[MatrixSparsity.Sparse],
          isSubset = true,
        )
        sparsifier match {
          // these cases are all handled at the block level
          case BandSparsifier(blocksOnly, _, _) if blocksOnly => loweredChild
          case RowIntervalSparsifier(blocksOnly, _, _) if blocksOnly => loweredChild
          case PerBlockSparsifier(_) | RectangleSparsifier(_) => loweredChild

          case BandSparsifier(_, l, u) => loweredChild.zeroBand(ib, l, u, x.typ)
          case RowIntervalSparsifier(_, starts, stops) =>
            loweredChild.zeroRowIntervals(ib, starts, stops, x.typ)
        }

      case _ =>
        BlockMatrixStage2.fromOldBMS(
          lowerNonEmpty(ib, bmir, typesToLower, ctx, analyses),
          bmir.typ,
          ib,
        )
    }
  }

  def lowerNonEmpty(
    ib: IRBuilder,
    bmir: BlockMatrixIR,
    typesToLower: DArrayLowering.Type,
    ctx: ExecuteContext,
    analyses: LoweringAnalyses,
  ): BlockMatrixStage = {
    def lower(ir: BlockMatrixIR, ib: IRBuilder = ib) =
      LowerBlockMatrixIR.lower(ib, ir, typesToLower, ctx, analyses).toOldBMS

    bmir match {

      case x @ BlockMatrixSlice(
            child,
            IndexedSeq(IndexedSeq(rStart, rEnd, rStep), IndexedSeq(cStart, cEnd, cStep)),
          ) =>
        val rowDependents = x.rowBlockDependents
        val colDependents = x.colBlockDependents

        lower(child).condenseBlocks(child.typ, rowDependents, colDependents)
          .addContext(TTuple(TTuple(TInt64, TInt64, TInt64), TTuple(TInt64, TInt64, TInt64))) {
            idx =>
              val (i, j) = idx

              // Aligned with the edges of blocks in child BM.
              val blockAlignedRowStartIdx = rowDependents(i).head.toLong * x.typ.blockSize
              val blockAlignedColStartIdx = colDependents(j).head.toLong * x.typ.blockSize
              val blockAlignedRowEndIdx =
                math.min(child.typ.nRows, (rowDependents(i).last + 1L) * x.typ.blockSize * rStep)
              val blockAlignedColEndIdx =
                math.min(child.typ.nCols, (colDependents(j).last + 1L) * x.typ.blockSize * cStep)

              /* condenseBlocks can give the same data to multiple partitions. Need to make sure we
               * don't use data */
              // that's already included in an earlier block.
              val rStartPlusSeenAlready = rStart + i * x.typ.blockSize * rStep
              val cStartPlusSeenAlready = cStart + j * x.typ.blockSize * cStep

              val rowTrueStart = rStartPlusSeenAlready - blockAlignedRowStartIdx
              val rowTrueEnd = math.min(
                math.min(rEnd, blockAlignedRowEndIdx) - blockAlignedRowStartIdx,
                rowTrueStart + x.typ.blockSize * rStep,
              )
              val rows = MakeTuple.ordered(FastSeq[IR](
                rowTrueStart,
                rowTrueEnd,
                rStep,
              ))

              val colTrueStart = cStartPlusSeenAlready - blockAlignedColStartIdx
              val colTrueEnd = math.min(
                java.lang.Math.min(cEnd, blockAlignedColEndIdx) - blockAlignedColStartIdx,
                colTrueStart + x.typ.blockSize * cStep,
              )
              val cols = MakeTuple.ordered(FastSeq[IR](
                colTrueStart,
                colTrueEnd,
                cStep,
              ))
              MakeTuple.ordered(FastSeq(rows, cols))
          }.mapBody((ctx, body) => NDArraySlice(body, GetField(ctx, "new")))

      case BlockMatrixDot(leftIR, rightIR) =>
        val left = lower(leftIR)
        val right = lower(rightIR)
        val newCtxType = TArray(TTuple(left.ctxType, right.ctxType))
        new BlockMatrixStage(left.broadcastVals ++ right.broadcastVals, newCtxType) {
          def blockContext(idx: (Int, Int)): IR = {
            val (i, j) = idx
            MakeArray(
              Array.tabulate[Option[IR]](leftIR.typ.nColBlocks) { k =>
                if (leftIR.typ.hasBlock(i, k) && rightIR.typ.hasBlock(k, j))
                  Some(MakeTuple.ordered(FastSeq(
                    left.blockContext(i -> k),
                    right.blockContext(k -> j),
                  )))
                else None
              }.flatten[IR],
              newCtxType,
            )
          }

          def blockBody(ctxRef: Ref): IR = {
            val tupleNDArrayStream = ToStream(ctxRef)
            val streamElementName = freshName()
            val streamElementRef =
              Ref(streamElementName, tupleNDArrayStream.typ.asInstanceOf[TStream].elementType)
            val leftName = freshName()
            val rightName = freshName()
            val leftRef = Ref(
              leftName,
              tupleNDArrayStream.typ.asInstanceOf[TStream].elementType.asInstanceOf[TTuple].types(0),
            )
            val rightRef = Ref(
              rightName,
              tupleNDArrayStream.typ.asInstanceOf[TStream].elementType.asInstanceOf[TTuple].types(1),
            )
            StreamAgg(
              tupleNDArrayStream,
              streamElementName,
              AggLet(
                leftName,
                GetTupleElement(streamElementRef, 0),
                AggLet(
                  rightName,
                  GetTupleElement(streamElementRef, 1),
                  ApplyAggOp(NDArrayMultiplyAdd())(
                    left.blockBody(leftRef),
                    right.blockBody(rightRef),
                  ),
                  isScan = false,
                ),
                isScan = false,
              ),
            )
          }
        }
    }
  }
}
