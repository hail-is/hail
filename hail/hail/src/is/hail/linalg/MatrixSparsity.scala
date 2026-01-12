package is.hail.linalg

import is.hail.collection.compat._
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.toRichOrderedSeq

import scala.collection.Searching._
import scala.collection.compat._

object MatrixSparsity {
  def dense(nRows: Int, nCols: Int): MatrixSparsity.Dense =
    Dense(nRows, nCols)

  def apply(nRows: Int, nCols: Int, definedBlocks: IterableOnce[(Int, Int)])
    : MatrixSparsity.Sparse =
    Sparse(nRows, nCols, definedBlocks)

  def union(left: MatrixSparsity, right: MatrixSparsity): MatrixSparsity = {
    assert(left.nRows == right.nRows && left.nCols == right.nCols)
    (left, right) match {
      case (left: MatrixSparsity.Sparse, right: MatrixSparsity.Sparse) =>
        Sparse(left.nRows, left.nCols, left.blockSet.union(right.blockSet))
      case _ => MatrixSparsity.dense(left.nRows, left.nCols)
    }
  }

  def intersection(left: MatrixSparsity, right: MatrixSparsity): MatrixSparsity = {
    assert(left.nRows == right.nRows && left.nCols == right.nCols)
    (left, right) match {
      case (left: MatrixSparsity.Sparse, right: MatrixSparsity.Sparse) =>
        Sparse(
          left.nRows,
          left.nCols,
          left.blockSet.intersect(right.blockSet),
        )
      case (_, right: MatrixSparsity.Sparse) => right
      case _ => left
    }
  }

  def constructFromShapeAndFunction(nRows: Int, nCols: Int)(exists: (Int, Int) => Boolean)
    : Sparse = {
    val builder = ArraySeq.newBuilder[(Int, Int)]
    for {
      j <- 0 until nCols
      i <- 0 until nRows
      if exists(i, j)
    } builder += i -> j
    Sparse.sorted(nRows, nCols, builder.result())
  }

  def fromLinearCoords(nRows: Int, nCols: Int, definedBlocks: Option[IndexedSeq[Int]])
    : MatrixSparsity = {
    definedBlocks.map { blocks =>
      val blocksCoord = blocks.map { linearIdx =>
        java.lang.Math.floorDiv(linearIdx, nCols) -> linearIdx % nCols
      }
      Sparse(nRows, nCols, blocksCoord)
    }.getOrElse(dense(nRows, nCols))
  }

  case class CSC(nRows: Int, nCols: Int, rowPos: IndexedSeq[Int], rowIdx: IndexedSeq[Int])

  case class DCSC(
    nRows: Int,
    nCols: Int,
    colIdx: IndexedSeq[Int],
    rowPos: IndexedSeq[Int],
    rowIdx: IndexedSeq[Int],
  )

  object Sparse {
    def apply(nRows: Int, nCols: Int, definedBlocks: IterableOnce[(Int, Int)]): Sparse = {
      val a = Array.from(definedBlocks)
      new Sparse(nRows, nCols, ArraySeq.unsafeWrapArray(a.sortInPlaceBy(_.swap).array))
    }

    def apply(nRows: Int, nCols: Int, definedBlocks: ArraySeq[(Int, Int)]): Sparse =
      Sparse(nRows, nCols, definedBlocks: IterableOnce[(Int, Int)])

    def sorted(nRows: Int, nCols: Int, definedBlocks: ArraySeq[(Int, Int)]): Sparse = {
      require(definedBlocks.isSorted(Ordering.by(_.swap)))
      new Sparse(nRows, nCols, definedBlocks)
    }
  }

  // Constructor is kept private to enforce the invariant that definedCoords is
  // in column major order.
  final case class Sparse private[MatrixSparsity] (
    nRows: Int,
    nCols: Int,
    definedCoords: ArraySeq[(Int, Int)],
  ) extends MatrixSparsity {

    override def numDefined: Int = definedCoords.length
    override def isSparse: Boolean = true
    def nonEmpty: Boolean = definedCoords.nonEmpty
    def isEmpty: Boolean = definedCoords.isEmpty

    def toDense: Dense = Dense(nRows, nCols)

    def copy(
      nRows: Int = nRows,
      nCols: Int = nCols,
      definedCoords: ArraySeq[(Int, Int)] = definedCoords,
    ): Sparse =
      MatrixSparsity(nRows, nCols, definedCoords)

    private[MatrixSparsity] def blockSet: Set[(Int, Int)] = definedCoords.toSet

    override def contains(row: Int, col: Int): Boolean =
      definedCoords.search(row -> col)(Ordering.by(_.swap)) match {
        case Found(_) => true
        case _ => false
      }

    override def newToOldPos(newSparsity: Sparse): IndexedSeq[Int] = {
      if (newSparsity.isEmpty) return ArraySeq.empty
      var cur =
        definedCoords.search(newSparsity.definedCoords.head)(Ordering.by(_.swap)).insertionPoint
      newSparsity.definedCoords.map { coords =>
        val i = definedCoords.indexOf(coords, cur)
        assert(i >= 0, "newToOld: other sparsity must be a subset of this")
        cur = i + 1
        i
      }
    }

    override def newToOldPosNonSubset(newSparsity: Sparse): IndexedSeq[Integer] = {
      if (newSparsity.isEmpty) return ArraySeq.empty
      var cur =
        definedCoords.search(newSparsity.definedCoords.head)(Ordering.by(_.swap)).insertionPoint
      val ord: Ordering[(Int, Int)] = Ordering.by(_.swap)
      newSparsity.definedCoords.map { coords =>
        cur = cur + definedCoords.segmentLength(ord.lteq(_, coords), cur)
        if (cur > 0 && definedCoords(cur - 1) == coords) Int.box(cur - 1) else null
      }
    }

    def filter(rows: IndexedSeq[Int], cols: IndexedSeq[Int]): Sparse = {
      require(rows.isSorted)
      require(cols.isSorted)
      val newDefined = for {
        j <- cols.indices.view
        i <- rows.indices
        if contains(rows(i), cols(j))
      } yield i -> j
      Sparse.sorted(rows.length, cols.length, newDefined.to(ArraySeq))
    }

    override def condense(
      rowOverlaps: IndexedSeq[IndexedSeq[Int]],
      colOverlaps: IndexedSeq[IndexedSeq[Int]],
    ): Sparse =
      MatrixSparsity.constructFromShapeAndFunction(rowOverlaps.length, colOverlaps.length) {
        (i, j) => rowOverlaps(i).exists(ii => colOverlaps(j).exists(jj => contains(ii, jj)))
      }

    override def condenseCols: Sparse =
      MatrixSparsity.constructFromShapeAndFunction(1, nCols) { (_, j) =>
        (0 until nRows).exists(i => contains(i, j))
      }

    override def condenseRows: Sparse =
      MatrixSparsity.constructFromShapeAndFunction(nRows, 1) { (i, _) =>
        (0 until nCols).exists(j => contains(i, j))
      }

    override lazy val transpose: Sparse =
      Sparse(nCols, nRows, definedCoords.map { case (i, j) => (j, i) })

    def transposeNewToOld: IndexedSeq[Int] =
      definedCoords.zipWithIndex.sortBy(_._1).map(_._2)

    override def isSubsetOf(other: MatrixSparsity): Boolean = other match {
      case _: Dense => true
      case other: Sparse =>
        if (isEmpty) return true
        var cur = other.definedCoords.search(definedCoords.head)(Ordering.by(_.swap)).insertionPoint
        definedCoords.forall { coords =>
          val i = other.definedCoords.indexOf(coords, cur)
          cur = i + 1
          i > 0
        }
    }

    def toCSC: CSC = {
      val rowPos = (0 to nCols).view.scanLeft(0) { (curPos, j) =>
        val nextPos = definedCoords.indexWhere(_._2 >= j, curPos)
        if (nextPos == -1) numDefined else nextPos
      }.tail.to(ArraySeq)
      val rowIdx = definedCoords.map(_._1)
      CSC(nRows, nCols, rowPos, rowIdx)
    }

    def toDCSC: DCSC = {
      val rowPos = (0 to numDefined).filter { pos =>
        pos == 0 || pos == numDefined || definedCoords(pos)._2 != definedCoords(pos - 1)._2
      }
      val colIdx = rowPos.view.init.map(definedCoords(_)._2).to(ArraySeq)
      val rowIdx = definedCoords.map(_._1)
      DCSC(nRows, nCols, colIdx, rowPos, rowIdx)
    }

    def definedBlocksColMajorLinear: IndexedSeq[Int] =
      definedCoords.map { case (i, j) => i + j * nRows }
  }

  final case class Dense(nRows: Int, nCols: Int) extends MatrixSparsity {
    override def numDefined: Int = nRows * nCols
    override def isSparse: Boolean = false

    override def contains(row: Int, col: Int): Boolean =
      row >= 0 && row < nRows && col >= 0 && col < nCols

    override def condense(
      rowOverlaps: IndexedSeq[IndexedSeq[Int]],
      colOverlaps: IndexedSeq[IndexedSeq[Int]],
    ): Dense =
      MatrixSparsity.dense(rowOverlaps.length, colOverlaps.length)

    override def condenseCols: Dense =
      MatrixSparsity.dense(1, nCols)

    override def condenseRows: Dense =
      MatrixSparsity.dense(nRows, 1)

    override def newToOldPos(newSparsity: Sparse): IndexedSeq[Int] =
      newSparsity.definedCoords.map { case (i, j) =>
        j * nRows + i
      }

    override def newToOldPosNonSubset(newSparsity: Sparse): IndexedSeq[Integer] =
      newToOldPos(newSparsity).map(Int.box)

    override def transpose: Dense = Dense(nCols, nRows)

    override def isSubsetOf(other: MatrixSparsity): Boolean = other match {
      case _: Dense => true
      case other: Sparse => other.numDefined == nRows * nCols
    }

    override def definedCoords: IndexedSeq[(Int, Int)] =
      for {
        j <- 0 until nCols
        i <- 0 until nRows
      } yield (i, j)
  }
}

abstract class MatrixSparsity {
  def nRows: Int
  def nCols: Int
  def numDefined: Int
  def isSparse: Boolean
  def contains(row: Int, col: Int): Boolean
  def newToOldPos(newSparsity: MatrixSparsity.Sparse): IndexedSeq[Int]
  def newToOldPosNonSubset(newSparsity: MatrixSparsity.Sparse): IndexedSeq[Integer]
  def transpose: MatrixSparsity

  def condense(rowOverlaps: IndexedSeq[IndexedSeq[Int]], colOverlaps: IndexedSeq[IndexedSeq[Int]])
    : MatrixSparsity

  def condenseCols: MatrixSparsity
  def condenseRows: MatrixSparsity

  def isSubsetOf(other: MatrixSparsity): Boolean

  def definedCoords: IndexedSeq[(Int, Int)]
}
