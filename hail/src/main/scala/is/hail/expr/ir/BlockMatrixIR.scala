package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.types.BlockMatrixType
import is.hail.expr.types.virtual.{TArray, TFloat64}
import is.hail.linalg.BlockMatrix
import is.hail.utils._
import breeze.linalg.DenseMatrix
import is.hail.expr.types.virtual.Type

import scala.collection.mutable.ArrayBuffer
import is.hail.utils.richUtils.RichDenseMatrixDouble
import org.json4s.{DefaultFormats, Formats, ShortTypeHints}

import scala.collection.immutable.NumericRange

object BlockMatrixIR {
  def checkFitsIntoArray(nRows: Long, nCols: Long) {
    require(nRows <= Int.MaxValue, s"Number of rows exceeds Int.MaxValue: $nRows")
    require(nCols <= Int.MaxValue, s"Number of columns exceeds Int.MaxValue: $nCols")
    require(nRows * nCols <= Int.MaxValue, s"Number of values exceeds Int.MaxValue: ${ nRows * nCols }")
  }

  def toBlockMatrix(
    hc: HailContext,
    nRows: Int,
    nCols: Int,
    data: Array[Double],
    blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix = {

    BlockMatrix.fromBreezeMatrix(hc.sc,
      new DenseMatrix[Double](nRows, nCols, data, 0, nCols, isTranspose = true), blockSize)
  }

  def matrixShapeToTensorShape(nRows: Long,  nCols: Long): (IndexedSeq[Long], Boolean) = {
    (nRows, nCols) match {
      case (1, 1) => (FastIndexedSeq(), false)
      case (_, 1) => (FastIndexedSeq(nRows), false)
      case (1, _) => (FastIndexedSeq(nCols), true)
      case _ => (FastIndexedSeq(nRows, nCols), false)
    }
  }

  def tensorShapeToMatrixShape(bmir: BlockMatrixIR): (Long, Long) = {
    val shape = bmir.typ.shape
    val isRowVector = bmir.typ.isRowVector

    assert(shape.length <= 2)
    shape match {
      case IndexedSeq() => (1, 1)
      case IndexedSeq(len) => if (isRowVector) (1, len) else (len, 1)
      case IndexedSeq(r, c) => (r, c)
    }
  }
}

abstract sealed class BlockMatrixIR extends BaseIR {
  def typ: BlockMatrixType

  protected[ir] def execute(hc: HailContext): BlockMatrix =
    fatal("tried to execute unexecutable IR:\n" + Pretty(this))

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR
}

case class BlockMatrixRead(reader: BlockMatrixReader) extends BlockMatrixIR {
  override def typ: BlockMatrixType = reader.fullType

  lazy val children: IndexedSeq[BaseIR] = Array.empty[BlockMatrixIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixRead = {
    assert(newChildren.isEmpty)
    BlockMatrixRead(reader)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    reader(hc)
  }
}

object BlockMatrixReader {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(
      List(classOf[BlockMatrixNativeReader], classOf[BlockMatrixBinaryReader]))
    override val typeHintFieldName: String = "name"
  }
}

abstract class BlockMatrixReader {
  def apply(hc: HailContext): BlockMatrix
  def fullType: BlockMatrixType
}

case class BlockMatrixNativeReader(path: String) extends BlockMatrixReader {
  override def fullType: BlockMatrixType = {
    val metadata = BlockMatrix.readMetadata(HailContext.get, path)
    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(metadata.nRows, metadata.nCols)

    BlockMatrixType(TFloat64(), tensorShape, isRowVector, metadata.blockSize)
  }

  override def apply(hc: HailContext): BlockMatrix = BlockMatrix.read(hc, path)
}

case class BlockMatrixBinaryReader(path: String, shape: IndexedSeq[Long], blockSize: Int) extends BlockMatrixReader {
  val IndexedSeq(nRows, nCols) = shape
  BlockMatrixIR.checkFitsIntoArray(nRows, nCols)

  override def fullType: BlockMatrixType = {
    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(nRows, nCols)

    BlockMatrixType(TFloat64(), tensorShape, isRowVector, blockSize)
  }

  override def apply(hc: HailContext): BlockMatrix = {
    val breezeMatrix = RichDenseMatrixDouble.importFromDoubles(hc, path, nRows.toInt, nCols.toInt, rowMajor = true)
    BlockMatrix.fromBreezeMatrix(hc.sc, breezeMatrix, blockSize)
  }
}

class BlockMatrixLiteral(value: BlockMatrix) extends BlockMatrixIR {
  override def typ: BlockMatrixType = {
    val (shape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(value.nRows, value.nCols)
    BlockMatrixType(TFloat64(), shape, isRowVector, value.blockSize)
  }

  lazy val children: IndexedSeq[BaseIR] = Array.empty[BlockMatrixIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixLiteral = {
    assert(newChildren.isEmpty)
    new BlockMatrixLiteral(value)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = value
}

case class BlockMatrixMap(child: BlockMatrixIR, f: IR) extends BlockMatrixIR {
  assert(f.isInstanceOf[ApplyUnaryPrimOp] || f.isInstanceOf[Apply])

  override def typ: BlockMatrixType = child.typ

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixMap = {
    assert(newChildren.length == 1)
    BlockMatrixMap(newChildren(0).asInstanceOf[BlockMatrixIR], f)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    f match {
      case ApplyUnaryPrimOp(Negate(), _) => child.execute(hc).unary_-()
      case Apply("abs", _) => child.execute(hc).abs()
      case Apply("log", _) => child.execute(hc).log()
      case Apply("sqrt", _) => child.execute(hc).sqrt()
      case Apply("ceil", _) => child.execute(hc).ceil()
      case Apply("floor", _) => child.execute(hc).floor()
      case _ => fatal(s"Unsupported operation on BlockMatrices: ${Pretty(f)}")
    }
  }
}

case class BlockMatrixMap2(left: BlockMatrixIR, right: BlockMatrixIR, f: IR) extends BlockMatrixIR {
  assert(f.isInstanceOf[ApplyBinaryPrimOp] || f.isInstanceOf[Apply])

  override def typ: BlockMatrixType = left.typ

  lazy val children: IndexedSeq[BaseIR] = Array(left, right, f)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixMap2 = {
    assert(newChildren.length == 3)
    BlockMatrixMap2(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[BlockMatrixIR],
      newChildren(2).asInstanceOf[IR])
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    left match {
      case BlockMatrixBroadcast(scalarIR: BlockMatrixIR, IndexedSeq(), _, _) =>
        scalarOnLeft(hc, coerceToScalar(hc, scalarIR), right, f)
      case BlockMatrixBroadcast(vectorIR: BlockMatrixIR,inIndexExpr, _, _) =>
        val vector = coerceToVector(hc, vectorIR)
        inIndexExpr match {
          case IndexedSeq(1) => rowVectorOnLeft(hc, vector, right, f)
          case IndexedSeq(0) => colVectorOnLeft(hc, vector, right, f)
        }
      case _ =>
        matrixOnLeft(hc, left.execute(hc), right, f)
    }
  }

  private def scalarOnLeft(hc: HailContext, scalar: Double, right: BlockMatrixIR, f: IR): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(vectorIR: BlockMatrixIR, IndexedSeq(_), _, _) =>
        opWithScalar(vectorIR.execute(hc), scalar, f, reverse = true)
      case _ =>
        opWithScalar(right.execute(hc), scalar, f, reverse = true)
    }
  }

  private def rowVectorOnLeft(hc: HailContext, rowVector: Array[Double], right: BlockMatrixIR, f: IR): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(scalarIR, IndexedSeq(), _, _) =>
        val rightAsScalar = coerceToScalar(hc, scalarIR)
        val rowVectorAsBm = BlockMatrixIR.toBlockMatrix(hc, 1, rowVector.length, rowVector)
        opWithScalar(rowVectorAsBm, rightAsScalar, f, reverse = false)
      case _ =>
        opWithRowVector(right.execute(hc), rowVector, f, reverse = true)
    }
  }

  private def colVectorOnLeft(hc: HailContext, colVector: Array[Double], right: BlockMatrixIR, f: IR): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(scalarIR, IndexedSeq(), _, _) =>
        val rightAsScalar = coerceToScalar(hc, scalarIR)
        val colVectorAsBm = BlockMatrixIR.toBlockMatrix(hc, colVector.length, 1, colVector)
        opWithScalar(colVectorAsBm, rightAsScalar, f, reverse = false)
      case _ =>
        opWithColVector(right.execute(hc), colVector, f, reverse = true)
    }
  }

  private def matrixOnLeft(hc: HailContext, matrix: BlockMatrix, right: BlockMatrixIR, f: IR): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(scalarIR, IndexedSeq(), _, _) =>
        val rightAsScalar = coerceToScalar(hc, scalarIR)
        opWithScalar(matrix, rightAsScalar, f, reverse = false)
      case BlockMatrixBroadcast(vectorIR, inIndexExpr, _, _) =>
        inIndexExpr match {
          case IndexedSeq(1) =>
            val rightAsRowVec = coerceToVector(hc, vectorIR)
            opWithRowVector(matrix, rightAsRowVec, f, reverse = false)
          case IndexedSeq(0) =>
            val rightAsColVec = coerceToVector(hc, vectorIR)
            opWithColVector(matrix, rightAsColVec, f, reverse = false)
        }
      case _ =>
        opWithTwoBlockMatrices(matrix, right.execute(hc), f)
    }
  }

  private def coerceToScalar(hc: HailContext, ir: BlockMatrixIR): Double = {
    ir match {
      case ValueToBlockMatrix(child, _, _) =>
        Interpret[Any](child) match {
          case scalar: Double => scalar
          case oneElementArray: IndexedSeq[_] => oneElementArray.head.asInstanceOf[Double]
        }
      case _ => ir.execute(hc).getElement(row = 0, col = 0)
    }
  }

  private def coerceToVector(hc: HailContext, ir: BlockMatrixIR): Array[Double] = {
    ir match {
      case ValueToBlockMatrix(child, _, _) =>
        Interpret[Any](child) match {
          case vector: IndexedSeq[_] => vector.asInstanceOf[IndexedSeq[Double]].toArray
        }
      case _ => ir.execute(hc).toBreezeMatrix().data
    }
  }

  private def opWithScalar(left: BlockMatrix, right: Double, f: IR, reverse: Boolean): BlockMatrix = {
    (f: @unchecked) match {
      case ApplyBinaryPrimOp(Add(), _, _) => left.scalarAdd(right)
      case ApplyBinaryPrimOp(Multiply(), _, _) => left.scalarMul(right)
      case ApplyBinaryPrimOp(Subtract(), _, _) =>
        if (reverse) left.reverseScalarSub(right) else left.scalarSub(right)
      case ApplyBinaryPrimOp(FloatingPointDivide(), _, _) =>
        if (reverse) left.reverseScalarDiv(right) else left.scalarDiv(right)
      case Apply("**", _) => left.pow(right)
    }
  }

  private def opWithRowVector(left: BlockMatrix, right: Array[Double], f: IR, reverse: Boolean): BlockMatrix = {
    (f: @unchecked) match {
      case ApplyBinaryPrimOp(Add(), _, _) => left.rowVectorAdd(right)
      case ApplyBinaryPrimOp(Multiply(), _, _) => left.rowVectorMul(right)
      case ApplyBinaryPrimOp(Subtract(), _, _) =>
        if (reverse) left.reverseRowVectorSub(right) else left.rowVectorSub(right)
      case ApplyBinaryPrimOp(FloatingPointDivide(), _, _) =>
        if (reverse) left.reverseRowVectorDiv(right) else left.rowVectorDiv(right)
    }
  }

  private def opWithColVector(left: BlockMatrix, right: Array[Double], f: IR, reverse: Boolean): BlockMatrix = {
    (f: @unchecked) match {
      case ApplyBinaryPrimOp(Add(), _, _) => left.colVectorAdd(right)
      case ApplyBinaryPrimOp(Multiply(), _, _) => left.colVectorMul(right)
      case ApplyBinaryPrimOp(Subtract(), _, _) =>
        if (reverse) left.reverseColVectorSub(right) else left.colVectorSub(right)
      case ApplyBinaryPrimOp(FloatingPointDivide(), _, _) =>
        if (reverse) left.reverseColVectorDiv(right) else left.colVectorDiv(right)
    }
  }

  private def opWithTwoBlockMatrices(left: BlockMatrix, right: BlockMatrix, f: IR): BlockMatrix = {
    (f: @unchecked) match {
      case ApplyBinaryPrimOp(Add(), _, _) => left.add(right)
      case ApplyBinaryPrimOp(Multiply(), _, _) => left.mul(right)
      case ApplyBinaryPrimOp(Subtract(), _, _) => left.sub(right)
      case ApplyBinaryPrimOp(FloatingPointDivide(), _, _) => left.div(right)
      case Apply("**", _) =>
        assert(right.nRows == 1 && right.nCols == 1)
        // BlockMatrix does not currently support elem-wise pow and this case would
        // only get hit when left and right are both 1x1
        left.pow(right.getElement(row = 0, col = 0))
    }
  }
}

case class BlockMatrixDot(left: BlockMatrixIR, right: BlockMatrixIR) extends BlockMatrixIR {

  override def typ: BlockMatrixType = {
    val (lRows, lCols) = BlockMatrixIR.tensorShapeToMatrixShape(left)
    val (rRows, rCols) = BlockMatrixIR.tensorShapeToMatrixShape(right)
    assert(lCols == rRows)

    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(lRows, rCols)
    BlockMatrixType(left.typ.elementType, tensorShape, isRowVector, left.typ.blockSize)
  }

  lazy val children: IndexedSeq[BaseIR] = Array(left, right)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixDot = {
    assert(newChildren.length == 2)
    BlockMatrixDot(newChildren(0).asInstanceOf[BlockMatrixIR], newChildren(1).asInstanceOf[BlockMatrixIR])
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    left.execute(hc).dot(right.execute(hc))
  }
}

case class BlockMatrixBroadcast(
  child: BlockMatrixIR,
  inIndexExpr: IndexedSeq[Int],
  shape: IndexedSeq[Long],
  blockSize: Int) extends BlockMatrixIR {

  assert(shape.length == 2)
  assert(inIndexExpr.length <= 2 && inIndexExpr.forall(x => x == 0 || x == 1))

  val (nRows, nCols) = BlockMatrixIR.tensorShapeToMatrixShape(child)
  val childMatrixShape = IndexedSeq(nRows, nCols)
  assert(inIndexExpr.zipWithIndex.forall({ case (out: Int, in: Int) =>
    !child.typ.shape.contains(in) || childMatrixShape(in) == shape(out)
  }))

  override def typ: BlockMatrixType = {
    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(shape(0), shape(1))
    BlockMatrixType(child.typ.elementType, tensorShape, isRowVector, blockSize)
  }

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixBroadcast = {
    assert(newChildren.length == 1)
    BlockMatrixBroadcast(newChildren(0).asInstanceOf[BlockMatrixIR], inIndexExpr, shape, blockSize)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val childBm = child.execute(hc)
    val nRows = shape(0)
    val nCols = shape(1)

    inIndexExpr match {
      case IndexedSeq() =>
        val scalar = childBm.getElement(row = 0, col = 0)
        BlockMatrix.fill(hc, nRows, nCols, scalar, blockSize)
      case IndexedSeq(0) =>
        BlockMatrixIR.checkFitsIntoArray(nRows, nCols)
        broadcastColVector(hc, childBm.toBreezeMatrix().data, nRows.toInt, nCols.toInt)
      case IndexedSeq(1) =>
        BlockMatrixIR.checkFitsIntoArray(nRows, nCols)
        broadcastRowVector(hc, childBm.toBreezeMatrix().data, nRows.toInt, nCols.toInt)
      case IndexedSeq(0, 0) =>
        BlockMatrixIR.checkFitsIntoArray(nRows, nCols)
        BlockMatrixIR.toBlockMatrix(hc, nRows.toInt, nCols.toInt, childBm.diagonal(), blockSize)
      case IndexedSeq(1, 0) => childBm.transpose()
      case IndexedSeq(0, 1) => childBm
    }
  }

  private def broadcastRowVector(hc: HailContext, vec: Array[Double], nRows: Int, nCols: Int): BlockMatrix = {
    val data = ArrayBuffer[Double]()
    data.sizeHint(nRows * nCols)
    (0 until nRows).foreach(_ => data ++= vec)
    BlockMatrixIR.toBlockMatrix(hc, nRows, nCols, data.toArray, blockSize)
  }

  private def broadcastColVector(hc: HailContext, vec: Array[Double], nRows: Int, nCols: Int): BlockMatrix = {
    val data = ArrayBuffer[Double]()
    data.sizeHint(nRows * nCols)
    (0 until nRows).foreach(row => (0 until nCols).foreach(_ => data += vec(row)))
    BlockMatrixIR.toBlockMatrix(hc, nRows, nCols, data.toArray, blockSize)
  }
}

case class BlockMatrixAgg(
  child: BlockMatrixIR,
  outIndexExpr: IndexedSeq[Int]) extends BlockMatrixIR {

  assert(outIndexExpr.length < 2)

  override def typ: BlockMatrixType = {
    val shape = outIndexExpr.map({ i: Int => child.typ.shape(i) }).toFastIndexedSeq
    val isRowVector = outIndexExpr == FastIndexedSeq(1)

    BlockMatrixType(child.typ.elementType, shape, isRowVector, child.typ.blockSize)
  }

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixAgg = {
    assert(newChildren.length == 1)
    BlockMatrixAgg(newChildren(0).asInstanceOf[BlockMatrixIR], outIndexExpr)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val childBm = child.execute(hc)

    outIndexExpr match {
      case IndexedSeq() => BlockMatrixIR.toBlockMatrix(hc, nRows = 1, nCols = 1, Array(childBm.sum()), typ.blockSize)
      case IndexedSeq(1) => childBm.rowSum()
      case IndexedSeq(0) => childBm.colSum()
    }
  }
}

case class BlockMatrixFilter(
  child: BlockMatrixIR,
  indices: Array[Array[Long]]) extends BlockMatrixIR {

  assert(indices.length == 2)

  override def typ: BlockMatrixType = {
    val matrixShape = indices.zipWithIndex.map({ case (dim, i) =>
      if (dim.isEmpty) child.typ.shape(i) else dim.length
    })

    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(matrixShape(0), matrixShape(1))
    BlockMatrixType(child.typ.elementType, tensorShape, isRowVector, child.typ.blockSize)
  }

  override def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixFilter = {
    assert(newChildren.length == 1)
    BlockMatrixFilter(newChildren(0).asInstanceOf[BlockMatrixIR], indices)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val Array(rowIndices, colIndices) = indices
    val bm = child.execute(hc)

    if (rowIndices.isEmpty) {
      bm.filterCols(colIndices)
    } else if (colIndices.isEmpty) {
      bm.filterRows(rowIndices)
    } else {
      bm.filter(rowIndices, colIndices)
    }
  }
}

case class BlockMatrixSlice(child: BlockMatrixIR, slices: IndexedSeq[IndexedSeq[Long]]) extends BlockMatrixIR {
  assert(slices.length == 2)
  assert(slices.forall(_.length == 3))

  override def typ: BlockMatrixType = {
    val matrixShape: IndexedSeq[Long] = slices.map { s =>
      val IndexedSeq(start, stop, step) = s
      1 + (stop - start - 1) / step
    }

    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(matrixShape(0), matrixShape(1))
    BlockMatrixType(child.typ.elementType, tensorShape, isRowVector, child.typ.blockSize)
  }

  override def children: IndexedSeq[BaseIR] = Array(child)

  override def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR = {
    assert(newChildren.length == 1)
    BlockMatrixSlice(newChildren(0).asInstanceOf[BlockMatrixIR], slices)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val bm = child.execute(hc)
    val IndexedSeq(rowKeep, colKeep) = slices.map { s =>
      val IndexedSeq(start, stop, step) = s
      start until stop by step
    }

    val (childNRows, childNCols) = BlockMatrixIR.tensorShapeToMatrixShape(child)
    if (isFullRange(rowKeep, childNRows)) {
      bm.filterCols(colKeep.toArray)
    } else if (isFullRange(colKeep, childNCols)) {
      bm.filterRows(rowKeep.toArray)
    } else {
      bm.filter(rowKeep.toArray, colKeep.toArray)
    }
  }

  private def isFullRange(r: NumericRange[Long], dimLength: Long): Boolean = {
    r.start == 0 && r.end == dimLength && r.step == 1
  }
}

case class ValueToBlockMatrix(
  child: IR,
  shape: IndexedSeq[Long],
  blockSize: Int) extends BlockMatrixIR {

  assert(shape.length == 2)

  override def typ: BlockMatrixType = {
    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(shape(0), shape(1))
    BlockMatrixType(elementType(child.typ), tensorShape, isRowVector, blockSize)
  }

  private def elementType(childType: Type): Type = {
    childType match {
      case array: TArray => array.elementType
      case _ => childType
    }
  }

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): ValueToBlockMatrix = {
    assert(newChildren.length == 1)
    ValueToBlockMatrix(newChildren(0).asInstanceOf[IR], shape, blockSize)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val IndexedSeq(nRows, nCols) = shape
    BlockMatrixIR.checkFitsIntoArray(nRows, nCols)

    Interpret[Any](child) match {
      case scalar: Double =>
        assert(nRows == 1 && nCols == 1)
        BlockMatrix.fill(hc, nRows, nCols, scalar, blockSize)
      case data: IndexedSeq[_] =>
        BlockMatrixIR.toBlockMatrix(hc, nRows.toInt, nCols.toInt, data.asInstanceOf[IndexedSeq[Double]].toArray, blockSize)
    }
  }
}

case class BlockMatrixRandom(
  seed: Int,
  gaussian: Boolean,
  shape: IndexedSeq[Long],
  blockSize: Int) extends BlockMatrixIR {

  assert(shape.length == 2)

  override def typ: BlockMatrixType = {
    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(shape(0), shape(1))
    BlockMatrixType(TFloat64(), tensorShape, isRowVector, blockSize)
  }

  lazy val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixRandom = {
    assert(newChildren.isEmpty)
    BlockMatrixRandom(seed, gaussian, shape, blockSize)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    BlockMatrix.random(hc, shape(0), shape(1), blockSize, seed, gaussian)
  }
}

case class RelationalLetBlockMatrix(name: String, value: IR, body: BlockMatrixIR) extends BlockMatrixIR {
  def typ: BlockMatrixType = body.typ

  def children: IndexedSeq[BaseIR] = Array(value, body)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR = {
    val IndexedSeq(newValue: IR, newBody: BlockMatrixIR) = newChildren
    RelationalLetBlockMatrix(name, newValue, newBody)
  }
}