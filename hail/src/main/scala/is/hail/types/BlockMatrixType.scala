package is.hail.types

import is.hail.expr.ir._
import is.hail.linalg.BlockMatrix
import is.hail.types.virtual._
import is.hail.utils._

import org.apache.spark.sql.Row

object BlockMatrixSparsity {
  private val builder: BoxedArrayBuilder[(Int, Int)] = new BoxedArrayBuilder[(Int, Int)]

  val dense: BlockMatrixSparsity = new BlockMatrixSparsity(None: Option[IndexedSeq[(Int, Int)]])

  def apply(definedBlocks: IndexedSeq[(Int, Int)]): BlockMatrixSparsity =
    BlockMatrixSparsity(Some(definedBlocks))

  def constructFromShapeAndFunction(nRows: Int, nCols: Int)(exists: (Int, Int) => Boolean)
    : BlockMatrixSparsity = {
    var i = 0
    builder.clear()
    while (i < nRows) {
      var j = 0
      while (j < nCols) {
        if (exists(i, j))
          builder += i -> j
        j += 1
      }
      i += 1
    }
    BlockMatrixSparsity(Some(builder.result().toFastSeq))
  }

  def fromLinearBlocks(
    nCols: Long,
    nRows: Long,
    blockSize: Int,
    definedBlocks: Option[IndexedSeq[Int]],
  ): BlockMatrixSparsity = {
    val nColBlocks = BlockMatrixType.numBlocks(nCols, blockSize)
    definedBlocks.map { blocks =>
      BlockMatrixSparsity(blocks.map { linearIdx =>
        java.lang.Math.floorDiv(linearIdx, nColBlocks) -> linearIdx % nColBlocks
      })
    }.getOrElse(dense)
  }

  def transposeCSCSparsity(
    nRows: Int,
    nCols: Int,
    rowPos: IndexedSeq[Int],
    rowIdx: IndexedSeq[Int],
  ): (IndexedSeq[Int], IndexedSeq[Int], IndexedSeq[Int]) = {
    val newRowPos = Array.ofDim[Int](nRows + 1)
    val newRowIdx = Array.ofDim[Int](rowIdx.length)
    val newToOldPos = Array.ofDim[Int](rowIdx.length)

    // count size of each row
    var curPos = 0
    while (curPos < rowIdx.length) {
      newRowPos(rowIdx(curPos)) += 1
      curPos += 1
    }

    // compute prefix sum over row sizes
    var i = 0
    var prefixSum = 0
    while (i < nRows) {
      val curSum = prefixSum
      prefixSum += newRowPos(i)
      newRowPos(i) = curSum
      i += 1
    }
    newRowPos(i) = prefixSum

    // fill in newRowIdx and newToOldPos
    val curRowPositions = newRowPos.clone()
    var j = 0
    while (j < nCols) {
      curPos = rowPos(j)
      val endPos = rowPos(j + 1)
      while (curPos < endPos) {
        val i = rowIdx(curPos)
        val newPos = curRowPositions(i)
        curRowPositions(i) += 1
        newRowIdx(newPos) = j
        newToOldPos(newPos) = curPos
        curPos += 1
      }
      j += 1
    }

    (newRowPos, newRowIdx, newToOldPos)
  }

  def transposeCSCSparsityIR(
    nRows: Int,
    nCols: Int,
    rowPos: IndexedSeq[Int],
    rowIdx: IndexedSeq[Int],
  ): (IR, IR, IR) = {
    val (newRowPos, newRowIdx, newToOldPos) = transposeCSCSparsity(nRows, nCols, rowPos, rowIdx)
    val t = TArray(TInt32)
    (Literal(t, newRowPos), Literal(t, newRowIdx), Literal(t, newToOldPos))
  }

  def filterCSCSparsity(
    rowPos: IndexedSeq[Int],
    rowIdx: IndexedSeq[Int],
    rowDeps: IndexedSeq[Int],
    colDeps: IndexedSeq[Int],
  ): (IndexedSeq[Int], IndexedSeq[Int], IndexedSeq[Int]) = {
    val newRowPos = new IntArrayBuilder()
    val newRowIdx = new IntArrayBuilder()
    val newToOldPos = new IntArrayBuilder()

    var curOutPos = 0
    for (j <- colDeps) {
      newRowPos += curOutPos
      var curLPos = rowPos(j)
      val endLPos = rowPos(j + 1)
      var curRPos = 0
      while (curLPos < endLPos && curRPos < rowDeps.length) {
        val curLIdx = rowIdx(curLPos)
        val curRIdx = rowDeps(curRPos)
        if (curLIdx == curRIdx) {
          newRowIdx += curLIdx
          newToOldPos += curOutPos
          curLPos += 1
          curRPos += 1
          curOutPos += 1
        } else {
          val c = curLIdx < curRIdx
          curLPos += c.toInt
          curRPos += (!c).toInt
        }
      }
    }
    newRowPos += curOutPos

    (newRowPos.result(), newRowIdx.result(), newToOldPos.result())
  }

  def groupedCSCSparsity(
    rowPos: IndexedSeq[Int],
    rowIdx: IndexedSeq[Int],
    rowDeps: IndexedSeq[IndexedSeq[Int]],
    colDeps: IndexedSeq[IndexedSeq[Int]],
  ): (
    IndexedSeq[Int],
    IndexedSeq[Int],
    IndexedSeq[(IndexedSeq[Int], IndexedSeq[Int], IndexedSeq[Int])],
  ) = {
    val newRowPos = new IntArrayBuilder()
    val newRowIdx = new IntArrayBuilder()
    val nestedSparsities =
      new AnyRefArrayBuilder[(IndexedSeq[Int], IndexedSeq[Int], IndexedSeq[Int])]()

    var curOutPos = 0
    var j = 0
    while (j < colDeps.length) {
      newRowPos += curOutPos
      var i = 0
      while (i < rowDeps.length) {
        val nested = filterCSCSparsity(rowPos, rowIdx, rowDeps(i), colDeps(j))
        if (nested._2.nonEmpty) {
          newRowIdx += i
          nestedSparsities += nested
          curOutPos += 1
        }
        i += 1
      }
      j += 1
    }
    newRowPos += curOutPos

    (newRowPos.result(), newRowIdx.result(), nestedSparsities.result())
  }

  def groupedCSCSparsityIR(
    rowPos: IndexedSeq[Int],
    rowIdx: IndexedSeq[Int],
    rowDeps: IndexedSeq[IndexedSeq[Int]],
    colDeps: IndexedSeq[IndexedSeq[Int]],
  ): (IR, IR, IR) = {
    val (newRowPos, newRowIdx, nestedSparsities) =
      groupedCSCSparsity(rowPos, rowIdx, rowDeps, colDeps)
    val t = TArray(TInt32)
    (
      Literal(t, newRowPos),
      Literal(t, newRowIdx),
      Literal(TArray(TTuple(t, t, t)), nestedSparsities.map(Row.fromTuple)),
    )
  }
}

case class BlockMatrixSparsity(definedBlocks: Option[IndexedSeq[(Int, Int)]]) {
  lazy val definedBlocksColMajor: Option[IndexedSeq[(Int, Int)]] = definedBlocks.map { blocks =>
    blocks.sortWith { case ((i1, j1), (i2, j2)) =>
      j1 < j2 || (j1 == j2 && i1 < i2)
    }
  }

  def definedBlocksCSC(nCols: Int): Option[(IndexedSeq[Int], IndexedSeq[Int])] =
    definedBlocksColMajor.map { blocks =>
      var curColIdx = 0
      var curPos = 0
      val pos = new Array[Int](nCols + 1)
      val rowIdx = new IntArrayBuilder()

      pos(0) = 0
      for ((i, j) <- blocks) {
        while (curColIdx < j) {
          pos(curColIdx + 1) = curPos
          curColIdx += 1
        }
        rowIdx += i
        curPos += 1
      }
      while (curColIdx < nCols) {
        pos(curColIdx + 1) = curPos
        curColIdx += 1
      }
      (pos, rowIdx.result())
    }

  def definedBlocksCSCIR(nCols: Int): Option[(IR, IR)] =
    definedBlocksCSC(nCols).map { case (rowPos, rowIdx) =>
      val t = TArray(TInt32)
      (Literal(t, rowPos), Literal(t, rowIdx))
    }

  def definedBlocksColMajorIR: Option[IR] = definedBlocksColMajor.map { blocks =>
    ToStream(Literal(TArray(TTuple(TInt32, TInt32)), blocks.map(Row.fromTuple)))
  }

  lazy val definedBlocksRowMajor: Option[IndexedSeq[(Int, Int)]] = definedBlocks.map { blocks =>
    blocks.sortWith { case ((i1, j1), (i2, j2)) =>
      i1 < i2 || (i1 == i2 && j1 < j2)
    }
  }

  def definedBlocksRowMajorIR: Option[IR] = definedBlocksRowMajor.map { blocks =>
    ToStream(Literal(TArray(TTuple(TInt32, TInt32)), blocks.map(Row.fromTuple)))
  }

  def isSparse: Boolean = definedBlocks.isDefined
  lazy val blockSet: Set[(Int, Int)] = definedBlocks.get.toSet
  def hasBlock(idx: (Int, Int)): Boolean = definedBlocks.isEmpty || blockSet.contains(idx)

  def condense(blockOverlaps: => (Array[Array[Int]], Array[Array[Int]])): BlockMatrixSparsity = {
    definedBlocks.map { _ =>
      val (ro, co) = blockOverlaps
      BlockMatrixSparsity.constructFromShapeAndFunction(ro.length, co.length) { (i, j) =>
        ro(i).exists(ii => co(j).exists(jj => hasBlock(ii -> jj)))
      }
    }.getOrElse(BlockMatrixSparsity.dense)
  }

  def allBlocksColMajor(nRowBlocks: Int, nColBlocks: Int): IndexedSeq[(Int, Int)] = {
    definedBlocksColMajor.getOrElse {
      val foo = Array.fill[(Int, Int)](nRowBlocks * nColBlocks)(null)
      var idx = 0
      var j = 0
      while (j < nColBlocks) {
        var i = 0
        while (i < nRowBlocks) {
          foo(idx) = i -> j
          i += 1
          idx += 1
        }
        j += 1
      }
      foo
    }
  }

  def allBlocksColMajorIR(nRowBlocks: Int, nColBlocks: Int): IR =
    definedBlocksColMajorIR.getOrElse {
      flatMapIR(rangeIR(nColBlocks))(j => mapIR(rangeIR(nRowBlocks))(i => maketuple(i, j)))
    }

  def allBlocksRowMajor(nRowBlocks: Int, nColBlocks: Int): IndexedSeq[(Int, Int)] = {
    (definedBlocksRowMajor).getOrElse {
      val foo = Array.fill[(Int, Int)](nRowBlocks * nColBlocks)(null)
      var idx = 0
      var i = 0
      while (i < nRowBlocks) {
        var j = 0
        while (j < nColBlocks) {
          foo(idx) = i -> j
          j += 1
          idx += 1
        }
        i += 1
      }
      foo
    }
  }

  def allBlocksRowMajorIR(nRowBlocks: Int, nColBlocks: Int): IR =
    definedBlocksRowMajorIR.getOrElse {
      flatMapIR(rangeIR(nRowBlocks))(i => mapIR(rangeIR(nColBlocks))(j => maketuple(i, j)))
    }

  def transpose: BlockMatrixSparsity =
    BlockMatrixSparsity(definedBlocks.map(_.map { case (i, j) => (j, i) }))

  override def toString: String =
    definedBlocks.map { blocks =>
      blocks.map { case (i, j) => s"($i,$j)" }.mkString("[", ",", "]")
    }.getOrElse("None")
}

object BlockMatrixType {
  def tensorToMatrixShape(shape: IndexedSeq[Long], isRowVector: Boolean): (Long, Long) =
    shape match {
      case IndexedSeq() => (1, 1)
      case IndexedSeq(vectorLength) => if (isRowVector) (1, vectorLength) else (vectorLength, 1)
      case IndexedSeq(numRows, numCols) => (numRows, numCols)
    }

  def matrixToTensorShape(nRows: Long, nCols: Long): (IndexedSeq[Long], Boolean) = {
    (nRows, nCols) match {
      case (1, 1) => (FastSeq(), false)
      case (_, 1) => (FastSeq(nRows), false)
      case (1, _) => (FastSeq(nCols), true)
      case _ => (FastSeq(nRows, nCols), false)
    }
  }

  def numBlocks(n: Long, blockSize: Int): Int =
    java.lang.Math.floorDiv(n - 1, blockSize).toInt + 1

  def getBlockIdx(i: Long, blockSize: Int): Int = java.lang.Math.floorDiv(i, blockSize).toInt

  def dense(elementType: Type, nRows: Long, nCols: Long, blockSize: Int): BlockMatrixType = {
    val (shape, isRowVector) = matrixToTensorShape(nRows, nCols)
    BlockMatrixType(elementType, shape, isRowVector, blockSize, BlockMatrixSparsity.dense)
  }

  def fromBlockMatrix(value: BlockMatrix): BlockMatrixType = {
    val sparsity = BlockMatrixSparsity.fromLinearBlocks(
      value.nRows,
      value.nCols,
      value.blockSize,
      value.gp.partitionIndexToBlockIndex,
    )
    val (shape, isRowVector) = matrixToTensorShape(value.nRows, value.nCols)
    BlockMatrixType(TFloat64, shape, isRowVector, value.blockSize, sparsity)
  }
}

case class BlockMatrixType(
  elementType: Type,
  shape: IndexedSeq[Long],
  isRowVector: Boolean,
  blockSize: Int,
  sparsity: BlockMatrixSparsity,
) extends BaseType {
  require(blockSize >= 0)
  lazy val (nRows: Long, nCols: Long) = BlockMatrixType.tensorToMatrixShape(shape, isRowVector)

  def matrixShape: (Long, Long) = nRows -> nCols

  lazy val nRowBlocks: Int = if (blockSize == 0) 0 else BlockMatrixType.numBlocks(nRows, blockSize)
  lazy val nColBlocks: Int = if (blockSize == 0) 0 else BlockMatrixType.numBlocks(nCols, blockSize)
  lazy val defaultBlockShape: (Int, Int) = (nRowBlocks, nColBlocks)

  def densify: BlockMatrixType = copy(sparsity = BlockMatrixSparsity(None))

  def getBlockIdx(i: Long): Int = java.lang.Math.floorDiv(i, blockSize).toInt
  def isSparse: Boolean = sparsity.isSparse

  def nDefinedBlocks: Int =
    if (isSparse) sparsity.definedBlocks.get.length else nRowBlocks * nColBlocks

  def hasBlock(idx: (Int, Int)): Boolean =
    if (isSparse) sparsity.hasBlock(idx)
    else idx._1 >= 0 && idx._1 < nRowBlocks && idx._2 >= 0 && idx._2 < nColBlocks

  def transpose: BlockMatrixType = {
    val newShape = shape match {
      case Seq() => IndexedSeq()
      case Seq(m) => IndexedSeq(m)
      case Seq(m, n) => IndexedSeq(n, m)
    }
    val newIsRowVector = (shape.length == 1) && !isRowVector
    BlockMatrixType(elementType, newShape, newIsRowVector, blockSize, sparsity.transpose)
  }

  def allBlocksColMajor: IndexedSeq[(Int, Int)] = sparsity.allBlocksColMajor(nRowBlocks, nColBlocks)
  def allBlocksColMajorIR: IR = sparsity.allBlocksColMajorIR(nRowBlocks, nColBlocks)
  def allBlocksRowMajor: IndexedSeq[(Int, Int)] = sparsity.allBlocksRowMajor(nRowBlocks, nColBlocks)
  def allBlocksRowMajorIR: IR = sparsity.allBlocksRowMajorIR(nRowBlocks, nColBlocks)

  lazy val linearizedDefinedBlocks: Option[IndexedSeq[Int]] = sparsity.definedBlocksColMajor.map {
    blocks => blocks.map { case (i, j) => i + j * nRowBlocks }
  }

  def blockShape(i: Int, j: Int): (Long, Long) = {
    val r = if (i == nRowBlocks - 1) nRows - (i * blockSize) else blockSize
    val c = if (j == nColBlocks - 1) nCols - (j * blockSize) else blockSize
    r -> c
  }

  def blockShapeIR(i: IR, j: IR): (IR, IR) = {
    assert(i.typ == TInt32)
    assert(j.typ == TInt32)
    val r = If(i.ceq(nRowBlocks - 1), I64(nRows) - (i.toL * blockSize.toLong), blockSize.toLong)
    val c = If(j.ceq(nColBlocks - 1), I64(nCols) - (j.toL * blockSize.toLong), blockSize.toLong)
    r -> c
  }

  private[this] def getBlockDependencies(keep: Array[Array[Long]]): Array[Array[Int]] =
    keep.map(keeps =>
      Array.range(
        BlockMatrixType.getBlockIdx(keeps.head, blockSize),
        BlockMatrixType.getBlockIdx(keeps.last, blockSize) + 1,
      )
    ).toArray

  def rowBlockDependents(keepRows: Array[Array[Long]]): Array[Array[Int]] = if (keepRows.isEmpty)
    Array.tabulate(nRowBlocks)(i => Array(i))
  else
    getBlockDependencies(keepRows)

  def colBlockDependents(keepCols: Array[Array[Long]]): Array[Array[Int]] = if (keepCols.isEmpty)
    Array.tabulate(nColBlocks)(i => Array(i))
  else
    getBlockDependencies(keepCols)

  override def pretty(sb: StringBuilder, indent0: Int, compact: Boolean): Unit = {
    var indent = indent0

    val space: String = if (compact) "" else " "

    def newline(): Unit =
      if (!compact) {
        sb += '\n'
        sb.append(" " * indent)
      }

    sb.append(s"BlockMatrix$space{")
    indent += 4
    newline()

    sb.append(s"elementType:$space")
    elementType.pretty(sb, indent, compact)
    sb += ','
    newline()

    sb.append(s"shape:$space[")
    shape.foreachBetween(dimSize => sb.append(dimSize))(sb.append(s",$space"))
    sb += ']'
    sb += ','
    newline()

    sb.append(s"isRowVector:$space")
    sb.append(isRowVector)
    sb += ','
    newline()

    sb.append(s"blockSize:$space")
    sb.append(blockSize)
    sb += ','
    newline()

    sb.append(s"sparsity:$space")
    sb.append(sparsity.toString)
    sb += ','
    newline()

    indent -= 4
    newline()
    sb += '}'
  }
}
