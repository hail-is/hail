package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.types.BlockMatrixType
import is.hail.expr.types.virtual.{TArray, TBaseStruct, TFloat64, TInt64, Type}
import is.hail.linalg.BlockMatrix
import is.hail.utils._
import breeze.linalg
import breeze.linalg.DenseMatrix
import breeze.numerics
import is.hail.annotations.Region

import scala.collection.mutable.ArrayBuffer
import is.hail.utils.richUtils.RichDenseMatrixDouble
import org.apache.spark.sql.Row
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

  def pyExecute(): BlockMatrix = {
    ExecuteContext.scoped { ctx =>
      Interpret(this, ctx, optimize = true)
    }
  }

  protected[ir] def execute(ctx: ExecuteContext): BlockMatrix =
    fatal("tried to execute unexecutable IR:\n" + Pretty(this))

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR

  def blockCostIsLinear: Boolean
}

case class BlockMatrixRead(reader: BlockMatrixReader) extends BlockMatrixIR {
  override lazy val typ: BlockMatrixType = reader.fullType

  lazy val children: IndexedSeq[BaseIR] = Array.empty[BlockMatrixIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixRead = {
    assert(newChildren.isEmpty)
    BlockMatrixRead(reader)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    reader(HailContext.get)
  }

  val blockCostIsLinear: Boolean = true
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
  override lazy val fullType = {
    val metadata = BlockMatrix.readMetadata(HailContext.get, path)
    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(metadata.nRows, metadata.nCols)

    BlockMatrixType(TFloat64(), tensorShape, isRowVector, metadata.blockSize)
  }

  override def apply(hc: HailContext): BlockMatrix = BlockMatrix.read(hc, path)
}

case class BlockMatrixBinaryReader(path: String, shape: IndexedSeq[Long], blockSize: Int) extends BlockMatrixReader {
  val IndexedSeq(nRows, nCols) = shape
  BlockMatrixIR.checkFitsIntoArray(nRows, nCols)

  override lazy val fullType: BlockMatrixType = {
    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(nRows, nCols)

    BlockMatrixType(TFloat64(), tensorShape, isRowVector, blockSize)
  }

  override def apply(hc: HailContext): BlockMatrix = {
    val breezeMatrix = RichDenseMatrixDouble.importFromDoubles(hc, path, nRows.toInt, nCols.toInt, rowMajor = true)
    BlockMatrix.fromBreezeMatrix(hc.sc, breezeMatrix, blockSize)
  }
}

class BlockMatrixLiteral(value: BlockMatrix) extends BlockMatrixIR {
  override lazy val typ: BlockMatrixType = {
    val (shape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(value.nRows, value.nCols)
    BlockMatrixType(TFloat64(), shape, isRowVector, value.blockSize)
  }

  lazy val children: IndexedSeq[BaseIR] = Array.empty[BlockMatrixIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixLiteral = {
    assert(newChildren.isEmpty)
    new BlockMatrixLiteral(value)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = value

  val blockCostIsLinear: Boolean = true // not guaranteed
}

case class BlockMatrixMap(child: BlockMatrixIR, eltName: String, f: IR) extends BlockMatrixIR {
  assert(f.isInstanceOf[ApplyUnaryPrimOp] || f.isInstanceOf[Apply] || f.isInstanceOf[ApplyBinaryPrimOp])

  override lazy val typ: BlockMatrixType = child.typ

  lazy val children: IndexedSeq[BaseIR] = Array(child, f)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixMap = {
    val IndexedSeq(newChild: BlockMatrixIR, newF: IR) = newChildren
    BlockMatrixMap(newChild, eltName, newF)
  }

  val blockCostIsLinear: Boolean = child.blockCostIsLinear

  private def evalIR(ctx: ExecuteContext, ir: IR): Double = {
    val res: Any = CompileAndEvaluate(ctx, ir)
    if (res == null)
      fatal("can't perform BlockMatrix operation on missing values!")
    res.asInstanceOf[Double]
  }

  private def binaryOp(scalar: Double, f: (DenseMatrix[Double], Double) => DenseMatrix[Double]): DenseMatrix[Double] => DenseMatrix[Double] =
    f(_, scalar)

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val prev = child.execute(ctx)

    val (name, breezeF, reqDense): (String, DenseMatrix[Double] => DenseMatrix[Double], Boolean) = f match {
      case ApplyUnaryPrimOp(Negate(), _) => ("negate", BlockMatrix.negationOp, false)
      case Apply("abs", _, _) => ("abs", numerics.abs(_), false)
      case Apply("log", _, _) => ("log", numerics.log(_), true)
      case Apply("sqrt", _, _) => ("sqrt", numerics.sqrt(_), false)
      case Apply("ceil", _, _) => ("ceil", numerics.ceil(_), false)
      case Apply("floor", _, _) => ("floor", numerics.floor(_), false)

      case Apply("**", Seq(Ref(`eltName`, _), r), _) if !Mentions(r, eltName) =>
        ("**", binaryOp(evalIR(ctx, r), numerics.pow(_, _)), false)
      case ApplyBinaryPrimOp(Add(), Ref(`eltName`, _), r) if !Mentions(r, eltName) =>
        ("+", binaryOp(evalIR(ctx, r), _ + _), true)
      case ApplyBinaryPrimOp(Add(), l, Ref(`eltName`, _)) if !Mentions(l, eltName) =>
        ("+", binaryOp(evalIR(ctx, l), _ + _), true)
      case ApplyBinaryPrimOp(Multiply(), Ref(`eltName`, _), r) if !Mentions(r, eltName) =>
        val i = evalIR(ctx, r)
        ("*", binaryOp(i, _ *:* _), i.isNaN | i.isInfinity)
      case ApplyBinaryPrimOp(Multiply(), l, Ref(`eltName`, _)) if !Mentions(l, eltName) =>
        val i = evalIR(ctx, l)
        ("*", binaryOp(i, _ *:* _), i.isNaN | i.isInfinity)
      case ApplyBinaryPrimOp(Subtract(), Ref(`eltName`, _), r) if !Mentions(r, eltName) =>
        ("-", binaryOp(evalIR(ctx, r), (m, s) => m - s), true)
      case ApplyBinaryPrimOp(Subtract(), l, Ref(`eltName`, _)) if !Mentions(l, eltName) =>
        ("-", binaryOp(evalIR(ctx, l), (m, s) => s - m), true)
      case ApplyBinaryPrimOp(FloatingPointDivide(), Ref(`eltName`, _), r) if !Mentions(r, eltName) =>
        val i = evalIR(ctx, r)
        ("/", binaryOp(evalIR(ctx, r), (m, s) => m /:/ s), i == 0.0 | i.isNaN | i.isInfinity)
      case ApplyBinaryPrimOp(FloatingPointDivide(), l, Ref(`eltName`, _)) if !Mentions(l, eltName) =>
        val i = evalIR(ctx, l)
        ("/", binaryOp(evalIR(ctx, l), BlockMatrix.reverseScalarDiv), i == 0.0 | i.isNaN | i.isInfinity)

      case _ => fatal(s"Unsupported operation on BlockMatrices: ${Pretty(f)}")
    }

    if (reqDense)
      prev.densify().blockMap(breezeF, name, reqDense = true)
    else
      prev.blockMap(breezeF, name, reqDense = false)
  }
}

case class BlockMatrixMap2(left: BlockMatrixIR, right: BlockMatrixIR, leftName: String, rightName: String, f: IR) extends BlockMatrixIR {
  assert(f.isInstanceOf[ApplyBinaryPrimOp] || f.isInstanceOf[Apply])

  override def typ: BlockMatrixType = left.typ

  lazy val children: IndexedSeq[BaseIR] = Array(left, right, f)

  val blockCostIsLinear: Boolean = left.blockCostIsLinear && right.blockCostIsLinear

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixMap2 = {
    assert(newChildren.length == 3)
    BlockMatrixMap2(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[BlockMatrixIR],
      leftName, rightName,
      newChildren(2).asInstanceOf[IR])
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    left match {
      case BlockMatrixBroadcast(vectorIR: BlockMatrixIR, IndexedSeq(x), _, _) =>
        val vector = coerceToVector(ctx , vectorIR)
        x match {
          case 1 => rowVectorOnLeft(ctx, vector, right, f)
          case 0 => colVectorOnLeft(ctx, vector, right, f)
        }
      case _ =>
        matrixOnLeft(ctx, left.execute(ctx), right, f)
    }
  }

  private def rowVectorOnLeft(ctx: ExecuteContext, rowVector: Array[Double], right: BlockMatrixIR, f: IR): BlockMatrix =
    opWithRowVector(right.execute(ctx), rowVector, f, reverse = true)

  private def colVectorOnLeft(ctx: ExecuteContext, colVector: Array[Double], right: BlockMatrixIR, f: IR): BlockMatrix =
    opWithColVector(right.execute(ctx), colVector, f, reverse = true)

  private def matrixOnLeft(ctx: ExecuteContext, matrix: BlockMatrix, right: BlockMatrixIR, f: IR): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(vectorIR, IndexedSeq(x), _, _) =>
        x match {
          case 1 =>
            val rightAsRowVec = coerceToVector(ctx, vectorIR)
            opWithRowVector(matrix, rightAsRowVec, f, reverse = false)
          case 0 =>
            val rightAsColVec = coerceToVector(ctx, vectorIR)
            opWithColVector(matrix, rightAsColVec, f, reverse = false)
        }
      case _ =>
        opWithTwoBlockMatrices(matrix, right.execute(ctx), f)
    }
  }

  private def coerceToVector(ctx: ExecuteContext, ir: BlockMatrixIR): Array[Double] = {
    ir match {
      case ValueToBlockMatrix(child, _, _) =>
        Interpret[Any](ctx, child) match {
          case vector: IndexedSeq[_] => vector.asInstanceOf[IndexedSeq[Double]].toArray
        }
      case _ => ir.execute(ctx).toBreezeMatrix().data
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

  val blockCostIsLinear: Boolean = false

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    var leftBM = left.execute(ctx)
    var rightBM = right.execute(ctx)
    val hc = HailContext.get
    if (!left.blockCostIsLinear) {
      val path = hc.getTemporaryFile(suffix = Some("bm"))
      info(s"BlockMatrix multiply: writing left input with ${ leftBM.nRows } rows and ${ leftBM.nCols } cols " +
        s"(${ leftBM.gp.nBlocks } blocks of size ${ leftBM.blockSize }) to temporary file $path")
      leftBM.write(hc.sFS, path)
      leftBM = BlockMatrixNativeReader(path).apply(hc)
    }
    if (!right.blockCostIsLinear) {
      val path = hc.getTemporaryFile(suffix = Some("bm"))
      info(s"BlockMatrix multiply: writing right input with ${ rightBM.nRows } rows and ${ rightBM.nCols } cols " +
        s"(${ rightBM.gp.nBlocks } blocks of size ${ rightBM.blockSize }) to temporary file $path")
      rightBM.write(hc.sFS, path)
      rightBM = BlockMatrixNativeReader(path).apply(hc)
    }
    leftBM.dot(rightBM)
  }
}

case class BlockMatrixBroadcast(
  child: BlockMatrixIR,
  inIndexExpr: IndexedSeq[Int],
  shape: IndexedSeq[Long],
  blockSize: Int) extends BlockMatrixIR {

  val blockCostIsLinear: Boolean = child.blockCostIsLinear

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

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val hc = HailContext.get
    val childBm = child.execute(ctx)
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

  val blockCostIsLinear: Boolean = child.blockCostIsLinear

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

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val childBm = child.execute(ctx)

    outIndexExpr match {
      case IndexedSeq() => BlockMatrixIR.toBlockMatrix(HailContext.get, nRows = 1, nCols = 1, Array(childBm.sum()), typ.blockSize)
      case IndexedSeq(1) => childBm.rowSum()
      case IndexedSeq(0) => childBm.colSum()
    }
  }
}

case class BlockMatrixFilter(
  child: BlockMatrixIR,
  indices: Array[Array[Long]]) extends BlockMatrixIR {

  assert(indices.length == 2)

  val blockCostIsLinear: Boolean = child.blockCostIsLinear

  override lazy val typ: BlockMatrixType = {
    val childTensorShape = child.typ.shape
    val childMatrixShape = (childTensorShape, child.typ.isRowVector) match {
      case (IndexedSeq(vectorLength), true) => IndexedSeq(1, vectorLength)
      case (IndexedSeq(vectorLength), false) => IndexedSeq(vectorLength, 1)
      case (IndexedSeq(numRows, numCols), false) => IndexedSeq(numRows, numCols)
    }

    val matrixShape = indices.zipWithIndex.map({ case (dim, i) =>
      if (dim.isEmpty) childMatrixShape(i) else dim.length
    })

    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(matrixShape(0), matrixShape(1))
    BlockMatrixType(child.typ.elementType, tensorShape, isRowVector, child.typ.blockSize)
  }

  override def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixFilter = {
    assert(newChildren.length == 1)
    BlockMatrixFilter(newChildren(0).asInstanceOf[BlockMatrixIR], indices)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val Array(rowIndices, colIndices) = indices
    val bm = child.execute(ctx)

    if (rowIndices.isEmpty) {
      bm.filterCols(colIndices)
    } else if (colIndices.isEmpty) {
      bm.filterRows(rowIndices)
    } else {
      bm.filter(rowIndices, colIndices)
    }
  }
}

case class BlockMatrixDensify(child: BlockMatrixIR) extends BlockMatrixIR {
  def typ: BlockMatrixType = child.typ

  def blockCostIsLinear: Boolean = child.blockCostIsLinear

  val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR = {
    val IndexedSeq(newChild: BlockMatrixIR) = newChildren
    BlockMatrixDensify(newChild)
  }

  override def execute(ctx: ExecuteContext): BlockMatrix =
    child.execute(ctx).densify()
}

sealed abstract class BlockMatrixSparsifier {
  def typecheck(typ: Type): Unit
  def sparsify(ctx: ExecuteContext, bm: BlockMatrix, value: IR): BlockMatrix
}

case class BandSparsifier(blocksOnly: Boolean) extends BlockMatrixSparsifier {
  def typecheck(typ: Type): Unit = {
    val sTyp = coerce[TBaseStruct](typ)
    val Array(start, stop) = sTyp.types
    assert(start.isOfType(TInt64()))
    assert(stop.isOfType(TInt64()))
  }
  def sparsify(ctx: ExecuteContext, bm: BlockMatrix, value: IR): BlockMatrix = {
    val Row(l: Long, u: Long) = CompileAndEvaluate[Row](ctx, value, optimize = true)
    bm.filterBand(l, u, blocksOnly)
  }
}

case class RowIntervalSparsifier(blocksOnly: Boolean) extends BlockMatrixSparsifier {
  def typecheck(typ: Type): Unit = {
    val sTyp = coerce[TBaseStruct](typ)
    val Array(start, stop) = sTyp.types
    assert(start.isOfType(TArray(TInt64())))
    assert(stop.isOfType(TArray(TInt64())))
  }

  def sparsify(ctx: ExecuteContext, bm: BlockMatrix, value: IR): BlockMatrix = {
    val Row(starts: IndexedSeq[Long @unchecked], stops: IndexedSeq[Long @unchecked]) = CompileAndEvaluate[Row](ctx, value, optimize = true)
    bm.filterRowIntervals(starts.toArray, stops.toArray, blocksOnly)
  }
}

case object RectangleSparsifier extends BlockMatrixSparsifier {
  def typecheck(typ: Type): Unit = {
    assert(typ.isOfType(TArray(TInt64())))
  }
  def sparsify(ctx: ExecuteContext, bm: BlockMatrix, value: IR): BlockMatrix = {
    val rectangles = CompileAndEvaluate[IndexedSeq[Long]](ctx, value, optimize = true)
    bm.filterRectangles(rectangles.toArray)
  }
}

case class BlockMatrixSparsify(
  child: BlockMatrixIR,
  value: IR,
  sparsifier: BlockMatrixSparsifier
) extends BlockMatrixIR {
  def typ: BlockMatrixType = child.typ
  sparsifier.typecheck(value.typ)

  def blockCostIsLinear: Boolean = child.blockCostIsLinear

  val children: IndexedSeq[BaseIR] = Array(child, value)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR = {
    val IndexedSeq(newChild: BlockMatrixIR, newValue: IR) = newChildren
    BlockMatrixSparsify(newChild, newValue, sparsifier)
  }

  override def execute(ctx: ExecuteContext): BlockMatrix =
    sparsifier.sparsify(ctx, child.execute(ctx), value)
}

case class BlockMatrixSlice(child: BlockMatrixIR, slices: IndexedSeq[IndexedSeq[Long]]) extends BlockMatrixIR {
  assert(slices.length == 2)
  assert(slices.forall(_.length == 3))

  val blockCostIsLinear: Boolean = child.blockCostIsLinear

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

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val bm = child.execute(ctx)
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

  val blockCostIsLinear: Boolean = true

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

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val IndexedSeq(nRows, nCols) = shape
    BlockMatrixIR.checkFitsIntoArray(nRows, nCols)
    val hc = HailContext.get
    Interpret[Any](ctx, child) match {
      case scalar: Double =>
        assert(nRows == 1 && nCols == 1)
        BlockMatrix.fill(hc, nRows, nCols, scalar, blockSize)
      case data: IndexedSeq[_] =>
        BlockMatrixIR.toBlockMatrix(hc, nRows.toInt, nCols.toInt, data.asInstanceOf[IndexedSeq[Double]].toArray, blockSize)
    }
  }
}

case class BlockMatrixRandom(
  seed: Long,
  gaussian: Boolean,
  shape: IndexedSeq[Long],
  blockSize: Int) extends BlockMatrixIR {

  assert(shape.length == 2)

  val blockCostIsLinear: Boolean = true

  override def typ: BlockMatrixType = {
    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(shape(0), shape(1))
    BlockMatrixType(TFloat64(), tensorShape, isRowVector, blockSize)
  }

  lazy val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixRandom = {
    assert(newChildren.isEmpty)
    BlockMatrixRandom(seed, gaussian, shape, blockSize)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    BlockMatrix.random(HailContext.get, shape(0), shape(1), blockSize, seed, gaussian)
  }
}

case class RelationalLetBlockMatrix(name: String, value: IR, body: BlockMatrixIR) extends BlockMatrixIR {
  def typ: BlockMatrixType = body.typ

  def children: IndexedSeq[BaseIR] = Array(value, body)

  val blockCostIsLinear: Boolean = body.blockCostIsLinear

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR = {
    val IndexedSeq(newValue: IR, newBody: BlockMatrixIR) = newChildren
    RelationalLetBlockMatrix(name, newValue, newBody)
  }
}