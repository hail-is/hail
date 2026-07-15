package is.hail.expr.ir.lowering

import is.hail.annotations.RowSeq
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.toRichIterator
import is.hail.expr.Nat
import is.hail.expr.ir._
import is.hail.expr.ir.{Memoized => M}
import is.hail.expr.ir.Scope.EVAL
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.GetElement
import is.hail.linalg.MatrixSparsity
import is.hail.rvd.RVDPartitioner
import is.hail.types.TypeWithRequiredness
import is.hail.types.virtual._

import org.apache.spark.sql.Row

abstract class BlockMatrixStage(val broadcastVals: IndexedSeq[Ref], val ctxType: Type) {
  def blockContext(idx: (Int, Int)): IR
  def blockBody(ctxRef: Atom): IR
}

abstract class DynamicBMSContexts {
  def apply(row: Atom, col: Atom): IR
  def map(ib: IRBuilder)(f: (Atom, Atom, Atom, Atom) => IR): DynamicBMSContexts
  def collect(makeBlock: (Atom, Atom, Atom) => IR): IR
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

case class DynamicDenseContexts(nRows: Atom, nCols: Atom, contexts: Atom)
    extends DynamicBMSContexts {
  def irValue: IR = makestruct("nRows" -> nRows, "nCols" -> nCols, "contexts" -> contexts)

  override def apply(row: Atom, col: Atom): IR = ArrayRef(contexts, (col * nRows) + row)

  override def map(ib: IRBuilder)(f: (Atom, Atom, Atom, Atom) => IR): DynamicDenseContexts = {
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

  override def collect(makeBlock: (Atom, Atom, Atom) => IR): IR = {
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
  nRows: Atom,
  nCols: Atom,
  rowPos: Atom,
  rowIdx: Atom,
  val contexts: Atom,
) extends DynamicBMSContexts {
  val elementType: Type = contexts.typ.asInstanceOf[TArray].elementType

  def withNewContexts(newContexts: Atom): DynamicSparseContexts =
    new DynamicSparseContexts(nRows, nCols, rowPos, rowIdx, newContexts)

  override def apply(row: Atom, col: Atom): IR = {
    val startPos = ArrayRef(rowPos, col)
    val endPos = ArrayRef(rowPos, col + 1)
    bindIR(
      Apply(
        "lowerBound",
        ArraySeq(),
        ArraySeq(rowIdx, row, startPos, endPos),
        TInt32,
        ErrorIDs.NO_ERROR,
      )
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

  override def map(ib: IRBuilder)(f: (Atom, Atom, Atom, Atom) => IR): DynamicSparseContexts = {
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

  override def collect(makeBlock: (Atom, Atom, Atom) => IR): IR = {
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

  def tabulate(ib: IRBuilder, sparsity: MatrixSparsity)(f: (Atom, Atom) => IR): BMSContexts = {
    val definedCoordsIR =
      sparsity match {
        case MatrixSparsity.Dense(nRows, nCols) =>
          flatMapIR(rangeIR(nCols))(j => mapIR(rangeIR(nRows))(i => maketuple(i, j)))
        case sparsity: MatrixSparsity.Sparse =>
          ToStream(Literal(
            TArray(TTuple(TInt32, TInt32)),
            sparsity.definedCoords.map(RowSeq.fromTuple),
          ))
      }

    val contexts =
      ToArray(mapIR(definedCoordsIR) { coords =>
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

  def contexts: Atom

  def transpose(ib: IRBuilder): BMSContexts

  def withNewSparsity(ib: IRBuilder, newSparsity: MatrixSparsity): BMSContexts

  // body args: (rowIdx, colIdx, position, old context)
  def map(ib: IRBuilder)(f: (Atom, Atom, Atom, Atom) => IR): BMSContexts

  def zip(ib: IRBuilder, other: BMSContexts): BMSContexts

  def grouped(
    ib: IRBuilder,
    rowDeps: IndexedSeq[IndexedSeq[Int]],
    colDeps: IndexedSeq[IndexedSeq[Int]],
    typ: BlockMatrixType,
  ): BMSContexts

  def groupedByCol(ib: IRBuilder): BMSContexts

  def collect(makeBlock: (Atom, Atom, Atom) => IR): IR

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

  override def contexts: Atom = dynamic.contexts

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

  override def map(ib: IRBuilder)(f: (Atom, Atom, Atom, Atom) => IR): DenseContexts =
    DenseContexts(sparsity, dynamic.map(ib)(f))

  override def zip(ib: IRBuilder, other: BMSContexts): BMSContexts = {
    val newContexts = ToArray(zip2(
      ToStream(this.contexts),
      ToStream(other.contexts),
      ArrayZipBehavior.AssertSameLength,
    )((l, r) => maketuple(l, r)))
    DenseContexts(ib, sparsity, newContexts)
  }

  override def withNewSparsity(ib: IRBuilder, newSparsity: MatrixSparsity): BMSContexts = {
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

  override def groupedByCol(ib: IRBuilder): DenseContexts = {
    val groupedContexts = ToArray(mapIR(rangeIR(nCols)) { col =>
      bindIR(col * nRows)(start => sliceArrayIR(contexts, start, start + nRows))
    })
    DenseContexts(ib, MatrixSparsity.dense(1, nCols), ib.memoize(groupedContexts))
  }

  override def grouped(
    ib: IRBuilder,
    rowDeps: IndexedSeq[IndexedSeq[Int]],
    colDeps: IndexedSeq[IndexedSeq[Int]],
    typ: BlockMatrixType,
  ): DenseContexts = {
    assert(rowDeps.nonEmpty || colDeps.nonEmpty)
    val depsArrayType = TArray(TArray(TInt32))
    val rowDepsLit = ib.memoize(Literal(depsArrayType, rowDeps))
    val colDepsLit = ib.memoize(Literal(depsArrayType, colDeps))
    if (rowDeps.isEmpty) {
      val newContexts =
        ToArray(flatMapIR(ToStream(colDepsLit)) { localColDeps =>
          bindIR(ArrayLen(localColDeps)) { nLocalColDeps =>
            mapIR(rangeIR(nRows)) { i =>
              bindIR(ToArray(mapIR(ToStream(localColDeps))(jl => dynamic(i, jl)))) {
                localContexts => DynamicDenseContexts(1, nLocalColDeps, localContexts).irValue
              }
            }
          }
        })

      DenseContexts(ib, sparsity.copy(nCols = colDeps.length), newContexts)
    } else if (colDeps.isEmpty) {
      val newContexts =
        ToArray(flatMapIR(rangeIR(nCols)) { j =>
          mapIR(ToStream(rowDepsLit)) { localRowDeps =>
            M.eval {
              for {
                nLocalRowDeps <- ArrayLen(localRowDeps)
                localContexts <- ToArray(mapIR(ToStream(localRowDeps))(il => dynamic(il, j)))
              } yield DynamicDenseContexts(nLocalRowDeps, 1, localContexts).irValue
            }
          }
        })

      DenseContexts(ib, sparsity.copy(nRows = rowDeps.length), newContexts)
    } else {
      val newContexts =
        ToArray(flatMapIR(ToStream(colDepsLit)) { localColDeps =>
          bindIR(ArrayLen(localColDeps)) { nLocalColDeps =>
            mapIR(ToStream(rowDepsLit)) { localRowDeps =>
              M.eval {
                for {
                  nLocalRowDeps <- ArrayLen(localRowDeps)
                  localContexts <-
                    ToArray(flatMapIR(ToStream(localColDeps)) { jl =>
                      mapIR(ToStream(localRowDeps))(il => dynamic(il, jl))
                    })
                } yield DynamicDenseContexts(nLocalRowDeps, nLocalColDeps, localContexts).irValue
              }
            }
          }
        })

      DenseContexts(ib, MatrixSparsity.Dense(rowDeps.length, colDeps.length), newContexts)
    }
  }

  override def collect(makeBlock: (Atom, Atom, Atom) => IR): IR = {
    NDArrayConcat(
      ToArray(mapIR(rangeIR(nCols)) { j =>
        bindIR(j * nRows) { offset =>
          NDArrayConcat(
            ToArray(mapIR(rangeIR(nRows)) { i =>
              bindIR(ArrayRef(contexts, offset + i))(ctx => makeBlock(i, j, ctx))
            }),
            0,
          )
        }
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

  override def contexts: Atom = dynamic.contexts

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

  override def map(ib: IRBuilder)(f: (Atom, Atom, Atom, Atom) => IR): SparseContexts =
    new SparseContexts(sparsity, dynamic.map(ib)(f))

  override def withNewSparsity(ib: IRBuilder, newSparsity: MatrixSparsity): BMSContexts = {
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
    val rowPosIR = ib.memoize(Literal(TArray(TInt32), rowPos))
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
      RowSeq(sparsity.nRows, sparsity.nCols, csc.rowPos, csc.rowIdx)
    }
    val coordToPos = sparsity.definedCoords.zipWithIndex.toMap
    val blockNewToOld = newSparsity.definedCoords.zip(blockSparsities).map {
      case (coords, filteredSparsity) =>
        val rows = rowDeps(coords._1)
        val cols = colDeps(coords._2)
        filteredSparsity.definedCoords.map { case (li, lj) =>
          coordToPos((rows(li), cols(lj)))
        }
    }
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
        ArraySeq("contexts" -> mapArray(newToOld)(ArrayRef(contexts, _))),
      )
    })

    new SparseContexts(newSparsity, DynamicSparseContexts(ib, newSparsity, newContexts))
  }

  override def collect(makeBlock: (Atom, Atom, Atom) => IR): IR =
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
      FastSeq(),
      BlockMatrixType.dense(eltType, 0, 0, 0),
      DenseContexts(ib, MatrixSparsity.Dense(0, 0), MakeArray(FastSeq(), TArray(ctxType))),
      _ => NA(ctxType),
    )
  }

  def broadcastVector(ib: IRBuilder, vector: IR, typ: BlockMatrixType, asRowVector: Boolean)
    : BlockMatrixStage2 = {
    val v = ib.strictMemoize(vector)
    val contexts = BMSContexts.tabulate(ib, typ.sparsity) { (i, j) =>
      val (m, n) = typ.blockShapeIR(i, j)
      val start = (if (asRowVector) j.toL else i.toL) * typ.blockSize.toLong
      makestruct("start" -> start, "shape" -> maketuple(m, n))
    }
    BlockMatrixStage2(
      FastSeq(v.ir.asInstanceOf[Ref]),
      typ,
      contexts,
      ctx =>
        M.eval {
          for {
            shape <- GetField(ctx, "shape")
            start <- GetField(ctx, "start")
            len <- GetTupleElement(shape, if (asRowVector) 1 else 0)
            axis = if (asRowVector) 0 else 1
            nRep <- GetTupleElement(shape, axis).toI
            sliced <-
              NDArrayReshape(
                NDArraySlice(v, maketuple(maketuple(start, start + len, 1L))),
                if (asRowVector) maketuple(1L, len) else maketuple(len, 1L),
                ErrorIDs.NO_ERROR,
              )
          } yield NDArrayConcat(
            ToArray(mapIR(rangeIR(nRep))(_ => sliced)),
            axis,
          )
        },
    )
  }

  def apply(
    broadcastVals: IndexedSeq[Ref],
    typ: BlockMatrixType,
    contexts: BMSContexts,
    _blockIR: Atom => IR,
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

  def print(ctx: ExecuteContext): Unit =
    println(s"contexts:\n${contexts.print(ctx)}\nbody($ctxRefName) = ${Pretty(ctx, _blockIR)}")

  def blockIR(ctx: IR): IR =
    Let(FastSeq(ctxRefName -> ctx), _blockIR)

  private def ctxType: Type = contexts.elementType

  def toOldBMS: BlockMatrixStage =
    new BlockMatrixStage(broadcastVals, ctxType) {
      override def blockContext(idx: (Int, Int)): IR = contexts.dynamic(idx._1, idx._2)
      override def blockBody(ctxRef: Atom): IR = blockIR(ctxRef)
    }

  def getElement(i: Atom, j: Atom): IR =
    M.eval {
      assert(i.typ == TInt64)
      assert(j.typ == TInt64)
      val blockSize = typ.blockSize.toLong
      for {
        rowBlock <- i floorDiv blockSize
        colBlock <- j floorDiv blockSize

        rowBlockI <- rowBlock.toI
        colBlockI <- colBlock.toI

        iInBlock <- i - rowBlock * blockSize
        jInBlock <- j - colBlock * blockSize

        _ <- ctxRefName -> contexts.dynamic(rowBlockI, colBlockI)
      } yield NDArrayRef(_blockIR, FastSeq(iInBlock, jInBlock), -1)
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
        .map(ib) { (i, j, _, oldContext) =>
          val (m, n) = typ.blockShapeIR(i, j)
          makestruct("oldContext" -> oldContext, "nRows" -> m, "nCols" -> n)
        }

      def newBlock(context: Atom): IR =
        bindIR(GetField(context, "oldContext")) { oldContext =>
          If(
            IsNA(oldContext),
            bindIRs(context.get("nRows"), context.get("nCols")) { shape =>
              MakeNDArray.fill(zero(typ.elementType), shape, False())
            },
            blockIR(oldContext),
          )
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
        contexts.withNewSparsity(ib, newSparsity).map(ib) { (i, j, _, oldContext) =>
          val (m, n) = typ.blockShapeIR(i, j)
          makestruct("oldContext" -> oldContext, "nRows" -> m, "nCols" -> n)
        }

      def newBlock(context: Atom): IR =
        bindIR(GetField(context, "oldContext")) { oldContext =>
          If(
            IsNA(oldContext),
            bindIRs(context.get("nRows"), context.get("nCols")) { shape =>
              MakeNDArray.fill(zero(typ.elementType), shape, False())
            },
            blockIR(oldContext),
          )
        }

      BlockMatrixStage2(broadcastVals, newType, newContexts, newBlock)
    }
  }

  def mapBody(f: Atom => IR): BlockMatrixStage2 = {
    val newBlockIR = bindIR(_blockIR)(f)
    val newType = typ.copy(elementType = newBlockIR.typ.asInstanceOf[TNDArray].elementType)
    new BlockMatrixStage2(broadcastVals, newType, contexts, ctxRefName, newBlockIR)
  }

  def mapBody2(
    ib: IRBuilder,
    other: BlockMatrixStage2,
    sparsityStrategy: SparsityStrategy,
  )(
    f: (Atom, Atom) => IR
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

  private def mapBody2Aligned(ib: IRBuilder, other: BlockMatrixStage2)(f: (Atom, Atom) => IR) = {
    val newContexts = contexts.zip(ib, other.contexts)
    val newCtxName = freshName()
    val newBlockIR =
      M.eval {
        val ctx: Atom =
          Ref(newCtxName, newContexts.elementType)

        for {
          lBlockIr <- (ctxRefName -> GetTupleElement(ctx, 0)) >> _blockIR
          rBlockIr <- (other.ctxRefName -> GetTupleElement(ctx, 1)) >> other._blockIR
        } yield f(lBlockIr, rBlockIr)
      }
    val newType = typ.copy(elementType = newBlockIR.typ.asInstanceOf[TNDArray].elementType)
    new BlockMatrixStage2(
      broadcastVals ++ other.broadcastVals,
      newType,
      newContexts,
      newCtxName,
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
      val result = ArraySeq.newBuilder[IndexedSeq[Long]]
      val builder = ArraySeq.newBuilder[Long]
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
    val groupedKeepRowsLit =
      ib.memoize(if (keepRows.isEmpty) NA(t) else Literal(t, groupedKeepRows))
    val groupedKeepColsLit =
      ib.memoize(if (keepCols.isEmpty) NA(t) else Literal(t, groupedKeepCols))
    val groupedContexts: BMSContexts =
      contexts.grouped(ib, rowBlockDependents, colBlockDependents, typ)
    val groupedContextsWithIndices = groupedContexts.map(ib) { (i, j, pos, context) =>
      maketuple(context, ArrayRef(groupedKeepRowsLit, i), ArrayRef(groupedKeepColsLit, j))
    }

    def newBody(ctx: Atom): IR = {
      IRBuilder.scoped { ib =>
        val localContexts = contexts match {
          case _: DenseContexts => DynamicDenseContexts(ib, GetTupleElement(ctx, 0))
          case _: SparseContexts => DynamicSparseContexts(ib, GetTupleElement(ctx, 0))
        }
        val localKeepRows = ib.memoize(GetTupleElement(ctx, 1))
        val localKeepCols = ib.memoize(GetTupleElement(ctx, 2))
        localContexts.collect { (i, j, localContext) =>
          bindIRs(ArrayRef(localKeepRows, i), ArrayRef(localKeepCols, j)) { case Seq(rows, cols) =>
            Coalesce(FastSeq(
              // FIXME: assumes blockIR is strict (preserves missing)
              NDArrayFilter(blockIR(localContext), FastSeq(rows, cols)),
              bindIRs(rows.len.toL, cols.len.toL) { shape =>
                MakeNDArray.fill(zero(typ.elementType), shape, False())
              },
            ))
          }
        }
      }
    }

    BlockMatrixStage2(broadcastVals, typ, groupedContextsWithIndices, newBody)
  }

  def zeroBand(ib: IRBuilder, lower: Long, upper: Long, typ: BlockMatrixType): BlockMatrixStage2 = {
    val ctxs = contexts.map(ib)((i, j, _, context) => maketuple(context, i, j))

    def newBody(ctx: Atom): IR =
      M.eval {
        for {
          i <- GetTupleElement(ctx, 1)
          j <- GetTupleElement(ctx, 2)
          diagIndex <- (j - i).toL * typ.blockSize.toLong
          localLower <- I64(lower) - diagIndex
          localUpper <- I64(upper) - diagIndex
          (nRowsInBlock, nColsInBlock) = typ.blockShapeIR(i, j)
          block <- (ctxRefName -> GetTupleElement(ctx, 0)) >> _blockIR
        } yield If(
          -localLower >= (nRowsInBlock - 1L) && localUpper >= (nColsInBlock - 1L),
          block,
          invoke("zero_band", TNDArray(TFloat64, Nat(2)), block, localLower, localUpper),
        )
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
    val startsGrouped = ib.memoize(Literal(t, starts.grouped(typ.blockSize).to(ArraySeq)))
    val stopsGrouped = ib.memoize(Literal(t, stops.grouped(typ.blockSize).to(ArraySeq)))

    val ctxs = contexts.map(ib) { (i, j, _, context) =>
      maketuple(context, i, j, ArrayRef(startsGrouped, i), ArrayRef(stopsGrouped, i))
    }

    def newBody(ctx: Atom): IR =
      M.eval {
        for {
          block <- (ctxRefName -> GetTupleElement(ctx, 0)) >> _blockIR

          i <- GetTupleElement(ctx, 1)
          j <- GetTupleElement(ctx, 2)

          (_, nCols) = typ.blockShapeIR(i, j)
          nColsRef <- nCols
          colOffset <- j.toL * typ.blockSize.toLong

          starts <-
            ToArray(mapIR(ToStream(GetTupleElement(ctx, 3))) { s =>
              minIR(maxIR(s - colOffset, 0L), nColsRef)
            })

          stops <-
            ToArray(mapIR(ToStream(GetTupleElement(ctx, 4))) { s =>
              minIR(maxIR(s - colOffset, 0L), nColsRef)
            })
        } yield invoke(
          "zero_row_intervals",
          TNDArray(TFloat64, Nat(2)),
          block,
          starts,
          stops,
        )
      }

    BlockMatrixStage2(broadcastVals, typ, ctxs, newBody)
  }

  def toTableStage(ib: IRBuilder, ctx: ExecuteContext, bmTyp: BlockMatrixType): TableStage = {
    val bodyFreeVars = FreeVariables(_blockIR, supportsAgg = false, supportsScan = false)
    val bcFields = broadcastVals.filter { case Ref(f, _) =>
      bodyFreeVars.eval.lookupOption(f).isDefined
    }

    val contextsIR = ToStream(contexts.map(ib) { (rowIdx, colIdx, _, oldContext) =>
      maketuple(rowIdx, colIdx, oldContext)
    }.contexts)

    val empty = makestruct()
    val globals = Ref(freshName(), empty.typ)
    val letBindings =
      ib.getBindings
        .map { b => assert(b.scope == EVAL, b.name); b.name -> b.value } :+
        globals.name -> empty

    TableStage(
      letBindings,
      bcFields.map(ref => ref.name -> ref) :+ globals.name -> globals.ir,
      globals,
      RVDPartitioner.unkeyed(ctx.stateManager, bmTyp.nDefinedBlocks),
      TableStageDependency.none,
      contextsIR,
      newCtxRef =>
        MakeStream(
          makestruct(
            "blockRow" -> GetTupleElement(newCtxRef, 0),
            "blockCol" -> GetTupleElement(newCtxRef, 1),
            "block" -> Let(FastSeq(ctxRefName -> GetTupleElement(newCtxRef, 2)), _blockIR),
          )
        ),
    )
  }

  def collectBlocks(
    ib: IRBuilder,
    staticID: String,
    dynamicID: IR = NA(TString),
  )(
    f: (Atom, Atom, Atom) => IR // (ctx, pos, block)
  ): IR = {

    val newContexts =
      ToStream(
        contexts
          .map(ib)((_, _, pos, oldContext) => maketuple(pos, oldContext))
          .contexts
      )

    val globals =
      MakeStruct(broadcastVals.map(ref => ref.name.str -> ref))

    cdaIR(newContexts, globals, staticID, dynamicID) { case (ctxRef, globals) =>
      IRBuilder.scoped { ib =>
        broadcastVals.foreach { case Ref(name, _) =>
          ib.strictMemoize(GetField(globals, name.str), name)
        }

        val pos = ib.memoize(GetTupleElement(ctxRef, 0))
        val ctx = ib.strictMemoize(GetTupleElement(ctxRef, 1), name = ctxRefName)
        val block = ib.memoize(_blockIR)
        f(ctx, pos, block)
      }
    }
  }

  def collectLocal(ib: IRBuilder, staticID: String, dynamicID: IR = NA(TString)): IR = {
    val blockResults = collectBlocks(ib, staticID, dynamicID)((_, _, b) => b)
    val blocks = contexts match {
      case x: DenseContexts => DenseContexts(ib, x.sparsity, blockResults)
      case x: SparseContexts => SparseContexts(ib, x.sparsity, blockResults)
    }

    blocks.collect { (i, j, block) =>
      val (m, n) = typ.blockShapeIR(i, j)
      val zeroBlock =
        bindIRs(m, n)(shape => MakeNDArray.fill(zero(typ.elementType), shape, False()))
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

    val lowered =
      IRBuilder.scoped { ib =>
        node match {
          case BlockMatrixCollect(child) =>
            lower(child, ib).collectLocal(ib, "block_matrix_collect")
          case BlockMatrixToValueApply(child, GetElement(Seq(i, j))) =>
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

    NormalizeNames()(ctx, lowered)
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
        val contextsIR = MakeArray(makestruct())

        BlockMatrixStage2(
          FastSeq(),
          bmir.typ,
          BMSContexts(ib, bmir.typ.sparsity, contextsIR),
          _ =>
            MakeNDArray(
              MakeArray(element),
              maketuple(1L, 1L),
              False(),
              ErrorIDs.NO_ERROR,
            ),
        )

      case x @ ValueToBlockMatrix(child, _, _) =>
        val lowered = LowerToCDA.lower(child, typesToLower, ctx, analyses)
        val nd = ib.memoize(child.typ match {
          case _: TArray => MakeNDArray(
              lowered,
              maketuple(x.typ.nRows, x.typ.nCols),
              True(),
              ErrorIDs.NO_ERROR,
            )
          case _: TNDArray => lowered
        })
        val contexts = BMSContexts.tabulate(ib, x.typ.sparsity) { (rowBlockIdx, colBlockIdx) =>
          val rowStartIdx = rowBlockIdx.toL * I64(x.typ.blockSize.toLong)
          val colStartIdx = colBlockIdx.toL * I64(x.typ.blockSize.toLong)
          val (numRowsBlock, numColsBlock) = x.typ.blockShapeIR(rowBlockIdx, colBlockIdx)
          bindIRs(rowStartIdx, colStartIdx) { case Seq(rowStartIdx, colStartIdx) =>
            NDArraySlice(
              nd,
              maketuple(
                maketuple(
                  rowStartIdx,
                  rowStartIdx + numRowsBlock,
                  1L,
                ),
                maketuple(
                  colStartIdx,
                  colStartIdx + numColsBlock,
                  1L,
                ),
              ),
            )
          }
        }

        BlockMatrixStage2(FastSeq(), x.typ, contexts, identity)

      case x @ BlockMatrixRandom(staticUID, gaussian, _, _) =>
        val contexts = BMSContexts.tabulate(ib, x.typ.sparsity) { (rowIdx, colIdx) =>
          val (m, n) = x.typ.blockShapeIR(rowIdx, colIdx)
          maketuple(m, n, rowIdx * x.typ.nColBlocks + colIdx)
        }

        def bodyIR(ctx: Atom): IR = {
          val m = GetTupleElement(ctx, 0)
          val n = GetTupleElement(ctx, 1)
          val i = GetTupleElement(ctx, 2)
          val f = if (gaussian) "rand_norm_nd" else "rand_unif_nd"
          val rngState = RNGSplit(RNGSplitStatic(RNGStateLiteral(), staticUID), Cast(i, TInt64))
          invoke(f, TNDArray(TFloat64, Nat(2)), rngState, m, n, F64(0.0), F64(1.0))
        }

        BlockMatrixStage2(FastSeq(), x.typ, contexts, bodyIR)

      case BlockMatrixAgg(child, Seq(0, 1) /* axesToSumOut */ ) =>
        // Reduce to a single row and column
        val summedChild = lower(child).mapBody { body =>
          NDArrayReshape(
            NDArrayAgg(body, FastSeq(0, 1)),
            maketuple(1L, 1L),
            ErrorIDs.NO_ERROR,
          )
        }
        val blockResults =
          summedChild.collectBlocks(ib, "block_matrix_agg_axes_0_1")((_, _, block) => block)
        val ndArrayResults = NDArrayConcat(blockResults, 0)
        val aggResult = NDArrayAgg(ndArrayResults, FastSeq(0, 1))
        val newBlockIR = ib.memoize(NDArrayReshape(aggResult, maketuple(1L, 1L), ErrorIDs.NO_ERROR))
        val contexts = BMSContexts.tabulate(ib, bmir.typ.sparsity)((_, _) => newBlockIR)

        BlockMatrixStage2(FastSeq(), bmir.typ, contexts, identity)

      case BlockMatrixAgg(child, Seq(0) /* axesToSumOut */ ) =>
        // Number of rows goes to 1. Number of cols remains the same.
        val loweredChild = lower(child)
        val contexts = loweredChild.contexts.groupedByCol(ib)

        BlockMatrixStage2(
          loweredChild.broadcastVals,
          bmir.typ,
          contexts,
          ctx =>
            streamAggIR(mapIR(ToStream(ctx)) { childCtx =>
              bindIR(NDArrayAgg(loweredChild.blockIR(childCtx), FastSeq(0))) { aggedND =>
                NDArrayReshape(
                  aggedND,
                  maketuple(1L, GetTupleElement(NDArrayShape(aggedND), 0)),
                  ErrorIDs.NO_ERROR,
                )
              }
            })(block => ApplyAggOp(NDArraySum())(block)),
        )

      case BlockMatrixAgg(child, Seq(1) /* axesToSumOut */ ) =>
        // Number of cols goes to 1. Number of rows remains the same.
        val loweredChild = lower(child)

        val contexts = loweredChild.contexts.transpose(ib).groupedByCol(ib).transpose(ib)

        BlockMatrixStage2(
          loweredChild.broadcastVals,
          bmir.typ,
          contexts,
          ctx =>
            streamAggIR(mapIR(ToStream(ctx)) { childCtx =>
              bindIR(NDArrayAgg(loweredChild.blockIR(childCtx), FastSeq(1))) { aggedND =>
                NDArrayReshape(
                  aggedND,
                  maketuple(GetTupleElement(NDArrayShape(aggedND), 0), 1L),
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

      case x @ BlockMatrixBroadcast(child, Seq(), _, _) =>
        val elt = ib.strictMemoize(IRBuilder.scoped(ib => lower(child, ib).getElement(0L, 0L)))

        val contexts = BMSContexts.tabulate(ib, x.typ.sparsity) { (rowIdx, colIdx) =>
          val (i, j) = x.typ.blockShapeIR(rowIdx, colIdx)
          maketuple(i, j)
        }

        BlockMatrixStage2(
          FastSeq(elt.ir.asInstanceOf[Ref]),
          x.typ,
          contexts,
          ctxRef =>
            bindIRs(ctxRef.get(0), ctxRef.get(1))(shape => MakeNDArray.fill(elt, shape, True())),
        )

      case BlockMatrixBroadcast(child, Seq(axis), _, _) =>
        val vector = NDArrayReshape(
          IRBuilder.scoped { ib =>
            lower(child, ib).collectLocal(ib, "block_matrix_broadcast_single_axis")
          },
          maketuple(math.max(child.typ.nRows, child.typ.nCols)),
          ErrorIDs.NO_ERROR,
        )
        BlockMatrixStage2.broadcastVector(ib, vector, bmir.typ, asRowVector = axis == 1)

      case x @ BlockMatrixBroadcast(child, Seq(axis, axis2), _, _)
          if axis == axis2 => // diagonal as row/col vector
        val diagLen = math.min(child.typ.nRowBlocks, child.typ.nColBlocks)
        val diagSparsity = MatrixSparsity.Sparse.sorted(
          child.typ.nRowBlocks,
          child.typ.nColBlocks,
          ArraySeq.tabulate(diagLen)(i => (i, i)),
        )

        val diagArray = IRBuilder.scoped { ib =>
          lower(child, ib)
            .withSparsity(ib, diagSparsity)
            .collectBlocks(ib, "block_matrix_broadcast_diagonal") { (_, _, block) =>
              bindIR(NDArrayShape(block)) { shape =>
                val blockDiagLen = minIR(GetTupleElement(shape, 0), GetTupleElement(shape, 1))
                ToArray(mapIR(rangeIR(blockDiagLen.toI)) { i =>
                  bindIR(i.toL)(i => NDArrayRef(block, FastSeq(i, i), ErrorIDs.NO_ERROR))
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

      case BlockMatrixBroadcast(child, Seq(1, 0), _, _) => // transpose
        lower(child).transposed(ib)

      case BlockMatrixBroadcast(child, Seq(0, 1), _, _) =>
        lower(child)

      case x @ BlockMatrixFilter(child, keep) =>
        val Seq(keepRow, keepCol) = keep
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

      case x @ BlockMatrixSlice(
            child,
            Seq(Seq(rStart, _, rStep), Seq(cStart, _, cStep)),
          ) =>
        val rowDependents = x.rowBlockDependents
        val colDependents = x.colBlockDependents
        val childBMS = lower(child, ib)

        // Precompute per-dependent-block slice bounds so we can slice each
        // child block individually before concatenation, avoiding a large
        // intermediate. The slice end is not needed here; it's already
        // accounted for in the output type's nRows/nCols which cap how many
        // elements each block produces. Blocks that contribute zero elements
        // are excluded from the dependents by BlockMatrixSlice.
        def computePerBlockSlices(
          dependents: IndexedSeq[IndexedSeq[Int]],
          start: Long,
          step: Long,
          outputDimSize: Long,
          childDimSize: Long,
        ): IndexedSeq[IndexedSeq[Row]] =
          dependents.zipWithIndex.map { case (deps, i) =>
            val outputBlockStart = i.toLong * x.typ.blockSize
            val outputBlockElems =
              math.min((i + 1L) * x.typ.blockSize, outputDimSize) - outputBlockStart
            var outputProduced = 0L
            deps.map { d =>
              val childBlockStart = d.toLong * x.typ.blockSize
              val childBlockElems =
                math.min((d + 1L) * x.typ.blockSize, childDimSize) - childBlockStart
              val nextGlobalPos = start + (outputBlockStart + outputProduced) * step
              val localStart = nextGlobalPos - childBlockStart
              assert(
                localStart >= 0 && localStart < childBlockElems,
                s"localStart=$localStart out of range [0, $childBlockElems)",
              )
              val maxFromBlock = (childBlockElems - localStart + step - 1) / step
              val nSelected = math.min(maxFromBlock, outputBlockElems - outputProduced)
              assert(nSelected > 0, s"block d=$d should contribute at least one element")
              outputProduced += nSelected
              RowSeq(localStart, math.min(childBlockElems, localStart + nSelected * step), step)
            }
          }

        val perBlockSlicesType = TArray(TArray(TTuple(TInt64, TInt64, TInt64)))
        val perBlockRowSlices =
          ib.memoize(Literal(
            perBlockSlicesType,
            computePerBlockSlices(
              rowDependents,
              rStart,
              rStep,
              x.typ.nRows,
              child.typ.nRows,
            ),
          ))

        val perBlockColSlices =
          ib.memoize(Literal(
            perBlockSlicesType,
            computePerBlockSlices(
              colDependents,
              cStart,
              cStep,
              x.typ.nCols,
              child.typ.nCols,
            ),
          ))

        val groupedContexts = childBMS.contexts.grouped(ib, rowDependents, colDependents, x.typ)
        val groupedContextsWithSlices = groupedContexts.map(ib) { (i, j, _, context) =>
          maketuple(
            context,
            ArrayRef(perBlockRowSlices, i),
            ArrayRef(perBlockColSlices, j),
          )
        }

        def sliceLen(slice: Atom): M[EVAL.type] =
          for {
            start <- GetTupleElement(slice, 0)
            stop <- GetTupleElement(slice, 1)
            step <- GetTupleElement(slice, 2)
          } yield (stop - start + step - 1L) floorDiv step

        def newBody(ctxRef: Atom): IR =
          IRBuilder.scoped { ib =>
            val localContexts = childBMS.contexts match {
              case _: DenseContexts => DynamicDenseContexts(ib, ctxRef.get(0))
              case _: SparseContexts => DynamicSparseContexts(ib, ctxRef.get(0))
            }
            val localRowSlices = ib.memoize(ctxRef.get(1))
            val localColSlices = ib.memoize(ctxRef.get(2))

            localContexts.collect { (localI, localJ, localContext) =>
              bindIRs(localRowSlices.at(localI), localColSlices.at(localJ)) {
                case Seq(rowSlice, colSlice) =>
                  Coalesce(FastSeq(
                    NDArraySlice(childBMS.blockIR(localContext), maketuple(rowSlice, colSlice)),
                    M.eval {
                      for {
                        m <- sliceLen(rowSlice)
                        n <- sliceLen(colSlice)
                      } yield MakeNDArray.fill(zero(child.typ.elementType), FastSeq(m, n), False())
                    },
                  ))
              }
            }
          }

        BlockMatrixStage2(childBMS.broadcastVals, x.typ, groupedContextsWithSlices, newBody)

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
      case BlockMatrixDot(leftIR, rightIR) =>
        val left = lower(leftIR)
        val right = lower(rightIR)
        val newCtxType = TArray(TTuple(left.ctxType, right.ctxType))
        new BlockMatrixStage(left.broadcastVals ++ right.broadcastVals, newCtxType) {
          override def blockContext(idx: (Int, Int)): IR = {
            val (i, j) = idx

            val nColBlocks = leftIR.typ.nColBlocks
            val ctxs = ArraySeq.newBuilder[IR]
            ctxs.sizeHint(nColBlocks)

            for (k <- 0 until nColBlocks)
              if (leftIR.typ.hasBlock(i, k) && rightIR.typ.hasBlock(k, j))
                ctxs += maketuple(left.blockContext(i -> k), right.blockContext(k -> j))

            MakeArray(ctxs.result(), newCtxType)
          }

          override def blockBody(ctxRef: Atom): IR =
            streamAggIR(ToStream(ctxRef)) { elem =>
              M.agg {
                for {
                  lCtx <- GetTupleElement(elem, 0)
                  rCtx <- GetTupleElement(elem, 1)
                } yield ApplyAggOp(NDArrayMultiplyAdd())(
                  left.blockBody(lCtx),
                  right.blockBody(rCtx),
                )
              }
            }
        }
    }
  }
}
