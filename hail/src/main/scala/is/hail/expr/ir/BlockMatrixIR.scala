package is.hail.expr.ir

import is.hail.HailContext
import is.hail.types.{BlockMatrixSparsity, BlockMatrixType}
import is.hail.types.virtual.{TArray, TBaseStruct, TFloat64, TInt32, TInt64, TNDArray, TString, TTuple, Type}
import is.hail.linalg.{BlockMatrix, BlockMatrixMetadata}
import is.hail.utils._
import breeze.linalg
import breeze.linalg.DenseMatrix
import breeze.numerics
import is.hail.annotations.Region
import is.hail.backend.spark.SparkBackend
import is.hail.expr.Nat
import is.hail.expr.ir.lowering.{BlockMatrixStage, LowererUnsupportedOperation}
import is.hail.io.TypedCodecSpec
import is.hail.io.fs.FS
import is.hail.types.encoded.{EBlockMatrixNDArray, EFloat64}

import scala.collection.mutable.ArrayBuffer
import is.hail.utils.richUtils.RichDenseMatrixDouble
import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}

import scala.collection.immutable.NumericRange

object BlockMatrixIR {
  def checkFitsIntoArray(nRows: Long, nCols: Long) {
    require(nRows <= Int.MaxValue, s"Number of rows exceeds Int.MaxValue: $nRows")
    require(nCols <= Int.MaxValue, s"Number of columns exceeds Int.MaxValue: $nCols")
    require(nRows * nCols <= Int.MaxValue, s"Number of values exceeds Int.MaxValue: ${ nRows * nCols }")
  }

  def toBlockMatrix(
    nRows: Int,
    nCols: Int,
    data: Array[Double],
    blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix = {

    BlockMatrix.fromBreezeMatrix(
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

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = reader(ctx)

  val blockCostIsLinear: Boolean = true
}

object BlockMatrixReader {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(
      List(classOf[BlockMatrixNativeReader], classOf[BlockMatrixBinaryReader], classOf[BlockMatrixPersistReader]))
    override val typeHintFieldName: String = "name"
  }

  def fromJValue(ctx: ExecuteContext, jv: JValue): BlockMatrixReader = {
    (jv \ "name").extract[String] match {
      case "BlockMatrixNativeReader" => BlockMatrixNativeReader.fromJValue(ctx.fs, jv)
      case _ => jv.extract[BlockMatrixReader]
    }
  }
}


abstract class BlockMatrixReader {
  def pathsUsed: Seq[String]
  def apply(ctx: ExecuteContext): BlockMatrix
  def lower(ctx: ExecuteContext): BlockMatrixStage =
    throw new LowererUnsupportedOperation(s"BlockMatrixReader not implemented: ${ this.getClass }")
  def fullType: BlockMatrixType
  def toJValue: JValue = {
    Extraction.decompose(this)(BlockMatrixReader.formats)
  }
}

object BlockMatrixNativeReader {
  def apply(fs: FS, path: String): BlockMatrixNativeReader =
    BlockMatrixNativeReader(fs, BlockMatrixNativeReaderParameters(path))

  def apply(fs: FS, params: BlockMatrixNativeReaderParameters): BlockMatrixNativeReader = {
    val metadata = BlockMatrix.readMetadata(fs, params.path)
    new BlockMatrixNativeReader(params, metadata)
  }

  def fromJValue(fs: FS, jv: JValue): BlockMatrixNativeReader = {
    implicit val formats: Formats = BlockMatrixReader.formats
    val params = jv.extract[BlockMatrixNativeReaderParameters]
    BlockMatrixNativeReader(fs, params)
  }
}

case class BlockMatrixNativeReaderParameters(path: String)

class BlockMatrixNativeReader(
  val params: BlockMatrixNativeReaderParameters,
  val metadata: BlockMatrixMetadata) extends BlockMatrixReader {
  def pathsUsed: Seq[String] = Array(params.path)

  lazy val fullType: BlockMatrixType = {
    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(metadata.nRows, metadata.nCols)

    val sparsity = BlockMatrixSparsity.fromLinearBlocks(metadata.nRows, metadata.nCols, metadata.blockSize, metadata.maybeFiltered)
    BlockMatrixType(TFloat64, tensorShape, isRowVector, metadata.blockSize, sparsity)
  }

  def apply(ctx: ExecuteContext): BlockMatrix = BlockMatrix.read(ctx.fs, params.path)

  override def lower(ctx: ExecuteContext): BlockMatrixStage = {
    val blockFiles = fullType.sparsity.definedBlocksColMajor.map { blocks =>
      blocks.zipWithIndex.toMap.mapValues { i => metadata.partFiles(i) }
    }
    val vType = TNDArray(fullType.elementType, Nat(2))
    val spec = TypedCodecSpec(EBlockMatrixNDArray(EFloat64(required = true), required = true), vType, BlockMatrix.bufferSpec)


    new BlockMatrixStage(Array(), TString) {
      def blockContext(idx: (Int, Int)): IR = {
        if (!fullType.hasBlock(idx))
          fatal(s"trying to read nonexistent block $idx from path ${ params.path }")
        val filename = blockFiles.map(_(idx)).getOrElse(metadata.partFiles(idx._1 + idx._2 * fullType.nRowBlocks))
        Str(s"${params.path}/parts/$filename")
      }

      def blockBody(ctxRef: Ref): IR = ReadValue(ctxRef, spec, vType)
    }
  }

  override def toJValue: JValue = {
    decomposeWithName(params, "BlockMatrixNativeReader")(BlockMatrixReader.formats)
  }

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: BlockMatrixNativeReader => params == that.params
    case _ => false
  }
}

case class BlockMatrixBinaryReader(path: String, shape: IndexedSeq[Long], blockSize: Int) extends BlockMatrixReader {
  def pathsUsed: Seq[String] = Array(path)

  val IndexedSeq(nRows, nCols) = shape
  BlockMatrixIR.checkFitsIntoArray(nRows, nCols)

  lazy val fullType: BlockMatrixType = {
    BlockMatrixType.dense(TFloat64, nRows, nCols, blockSize)
  }

  def apply(ctx: ExecuteContext): BlockMatrix = {
    val breezeMatrix = RichDenseMatrixDouble.importFromDoubles(ctx.fs, path, nRows.toInt, nCols.toInt, rowMajor = true)
    BlockMatrix.fromBreezeMatrix(breezeMatrix, blockSize)
  }
}

case class BlockMatrixPersistReader(id: String) extends BlockMatrixReader {
  lazy val bm: BlockMatrix = HailContext.sparkBackend("BlockMatrixPersistReader").bmCache.getPersistedBlockMatrix(id)
  def pathsUsed: Seq[String] = FastSeq()
  lazy val fullType: BlockMatrixType = BlockMatrixType.fromBlockMatrix(bm)
  def apply(ctx: ExecuteContext): BlockMatrix = bm
}

class BlockMatrixLiteral(value: BlockMatrix) extends BlockMatrixIR {
  override lazy val typ: BlockMatrixType =
    BlockMatrixType.fromBlockMatrix(value)

  lazy val children: IndexedSeq[BaseIR] = Array.empty[BlockMatrixIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixLiteral = {
    assert(newChildren.isEmpty)
    new BlockMatrixLiteral(value)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = value

  val blockCostIsLinear: Boolean = true // not guaranteed
}

case class BlockMatrixMap(child: BlockMatrixIR, eltName: String, f: IR, needsDense: Boolean) extends BlockMatrixIR {
  assert(f.isInstanceOf[ApplyUnaryPrimOp] || f.isInstanceOf[Apply] || f.isInstanceOf[ApplyBinaryPrimOp])

  override lazy val typ: BlockMatrixType = child.typ
  assert(!needsDense || !typ.isSparse)

  lazy val children: IndexedSeq[BaseIR] = Array(child, f)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixMap = {
    val IndexedSeq(newChild: BlockMatrixIR, newF: IR) = newChildren
    BlockMatrixMap(newChild, eltName, newF, needsDense)
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

    val (name, breezeF): (String, DenseMatrix[Double] => DenseMatrix[Double]) = f match {
      case ApplyUnaryPrimOp(Negate(), _) => ("negate", BlockMatrix.negationOp)
      case Apply("abs", _, _, _) => ("abs", numerics.abs(_))
      case Apply("log", _, _, _) => ("log", numerics.log(_))
      case Apply("sqrt", _, _, _) => ("sqrt", numerics.sqrt(_))
      case Apply("ceil", _, _, _) => ("ceil", numerics.ceil(_))
      case Apply("floor", _, _, _) => ("floor", numerics.floor(_))

      case Apply("pow", _, Seq(Ref(`eltName`, _), r), _) if !Mentions(r, eltName) =>
        ("**", binaryOp(evalIR(ctx, r), numerics.pow(_, _)))
      case ApplyBinaryPrimOp(Add(), Ref(`eltName`, _), r) if !Mentions(r, eltName) =>
        ("+", binaryOp(evalIR(ctx, r), _ + _))
      case ApplyBinaryPrimOp(Add(), l, Ref(`eltName`, _)) if !Mentions(l, eltName) =>
        ("+", binaryOp(evalIR(ctx, l), _ + _))
      case ApplyBinaryPrimOp(Multiply(), Ref(`eltName`, _), r) if !Mentions(r, eltName) =>
        val i = evalIR(ctx, r)
        ("*", binaryOp(i, _ *:* _))
      case ApplyBinaryPrimOp(Multiply(), l, Ref(`eltName`, _)) if !Mentions(l, eltName) =>
        val i = evalIR(ctx, l)
        ("*", binaryOp(i, _ *:* _))
      case ApplyBinaryPrimOp(Subtract(), Ref(`eltName`, _), r) if !Mentions(r, eltName) =>
        ("-", binaryOp(evalIR(ctx, r), (m, s) => m - s))
      case ApplyBinaryPrimOp(Subtract(), l, Ref(`eltName`, _)) if !Mentions(l, eltName) =>
        ("-", binaryOp(evalIR(ctx, l), (m, s) => s - m))
      case ApplyBinaryPrimOp(FloatingPointDivide(), Ref(`eltName`, _), r) if !Mentions(r, eltName) =>
        val i = evalIR(ctx, r)
        ("/", binaryOp(evalIR(ctx, r), (m, s) => m /:/ s))
      case ApplyBinaryPrimOp(FloatingPointDivide(), l, Ref(`eltName`, _)) if !Mentions(l, eltName) =>
        val i = evalIR(ctx, l)
        ("/", binaryOp(evalIR(ctx, l), BlockMatrix.reverseScalarDiv))

      case _ => fatal(s"Unsupported operation on BlockMatrices: ${Pretty(f)}")
    }

    prev.blockMap(breezeF, name, reqDense = needsDense)
  }
}

object SparsityStrategy {
  def fromString(s: String): SparsityStrategy = s match {
    case "union" | "Union" | "UnionBlocks" => UnionBlocks
    case "intersection" | "Intersection" | "IntersectionBlocks" => IntersectionBlocks
    case "needs_dense" | "NeedsDense" => NeedsDense
  }

}

abstract class SparsityStrategy {
  def exists(leftBlock: Boolean, rightBlock: Boolean): Boolean
  def mergeSparsity(left: BlockMatrixSparsity, right: BlockMatrixSparsity): BlockMatrixSparsity
}
case object UnionBlocks extends SparsityStrategy {
  def exists(leftBlock: Boolean, rightBlock: Boolean): Boolean = leftBlock || rightBlock
  def mergeSparsity(left: BlockMatrixSparsity, right: BlockMatrixSparsity): BlockMatrixSparsity = {
    if (left.isSparse && right.isSparse) {
      BlockMatrixSparsity(left.blockSet.union(right.blockSet).toFastIndexedSeq)
    } else BlockMatrixSparsity.dense
  }
}
case object IntersectionBlocks extends SparsityStrategy {
  def exists(leftBlock: Boolean, rightBlock: Boolean): Boolean = leftBlock && rightBlock
  def mergeSparsity(left: BlockMatrixSparsity, right: BlockMatrixSparsity): BlockMatrixSparsity = {
    if (right.isSparse) {
      if (left.isSparse)
        BlockMatrixSparsity(left.blockSet.intersect(right.blockSet).toFastIndexedSeq)
      else right
    } else left
  }
}
case object NeedsDense extends SparsityStrategy {
  def exists(leftBlock: Boolean, rightBlock: Boolean): Boolean = true
  def mergeSparsity(left: BlockMatrixSparsity, right: BlockMatrixSparsity): BlockMatrixSparsity = {
    assert(!left.isSparse && !right.isSparse)
    BlockMatrixSparsity.dense
  }
}

case class BlockMatrixMap2(left: BlockMatrixIR, right: BlockMatrixIR, leftName: String, rightName: String, f: IR, sparsityStrategy: SparsityStrategy) extends BlockMatrixIR {
  assert(f.isInstanceOf[ApplyBinaryPrimOp] || f.isInstanceOf[Apply])
  assert(
    left.typ.nRows == right.typ.nRows &&
    left.typ.nCols == right.typ.nCols &&
    left.typ.blockSize == right.typ.blockSize)

  override def typ: BlockMatrixType = left.typ.copy(sparsity = sparsityStrategy.mergeSparsity(left.typ.sparsity, right.typ.sparsity))

  lazy val children: IndexedSeq[BaseIR] = Array(left, right, f)

  val blockCostIsLinear: Boolean = left.blockCostIsLinear && right.blockCostIsLinear

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixMap2 = {
    assert(newChildren.length == 3)
    BlockMatrixMap2(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[BlockMatrixIR],
      leftName, rightName,
      newChildren(2).asInstanceOf[IR],
      sparsityStrategy)
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
    val blockSize = left.typ.blockSize
    val (lRows, lCols) = BlockMatrixIR.tensorShapeToMatrixShape(left)
    val (rRows, rCols) = BlockMatrixIR.tensorShapeToMatrixShape(right)
    assert(lCols == rRows)

    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(lRows, rCols)
    val sparsity = if (left.typ.isSparse || right.typ.isSparse)
      BlockMatrixSparsity(
        BlockMatrixType.numBlocks(lRows, blockSize),
        BlockMatrixType.numBlocks(rCols, blockSize)) { (i: Int, j: Int) =>
        Array.tabulate(BlockMatrixType.numBlocks(rCols, blockSize)) { k =>
          left.typ.hasBlock(i -> k) && right.typ.hasBlock(k -> j)
        }.reduce(_ || _)
      } else BlockMatrixSparsity.dense
    BlockMatrixType(left.typ.elementType, tensorShape, isRowVector, blockSize, sparsity)
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
    val fs = ctx.fs
    if (!left.blockCostIsLinear) {
      val path = ctx.createTmpPath("blockmatrix-dot-left", "bm")
      info(s"BlockMatrix multiply: writing left input with ${ leftBM.nRows } rows and ${ leftBM.nCols } cols " +
        s"(${ leftBM.gp.nBlocks } blocks of size ${ leftBM.blockSize }) to temporary file $path")
      leftBM.write(ctx, path)
      leftBM = BlockMatrixNativeReader(fs, path).apply(ctx)
    }
    if (!right.blockCostIsLinear) {
      val path = ctx.createTmpPath("blockmatrix-dot-right", "bm")
      info(s"BlockMatrix multiply: writing right input with ${ rightBM.nRows } rows and ${ rightBM.nCols } cols " +
        s"(${ rightBM.gp.nBlocks } blocks of size ${ rightBM.blockSize }) to temporary file $path")
      rightBM.write(ctx, path)
      rightBM = BlockMatrixNativeReader(fs, path).apply(ctx)
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

  override val typ: BlockMatrixType = {
    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(shape(0), shape(1))
    val nRowBlocks = BlockMatrixType.numBlocks(shape(0), blockSize)
    val nColBlocks = BlockMatrixType.numBlocks(shape(1), blockSize)
    val sparsity =
      if (child.typ.isSparse)
        inIndexExpr match {
          case IndexedSeq() =>
            assert(child.typ.nRows == 1 && child.typ.nCols == 1)
            BlockMatrixSparsity.dense
          case IndexedSeq(0) => // broadcast col vector
            assert(Set(1, shape(0)) == Set(child.typ.nRows, child.typ.nCols))
            BlockMatrixSparsity(nRowBlocks, nColBlocks)((i: Int, j: Int) => child.typ.hasBlock(0 -> j))
          case IndexedSeq(1) => // broadcast row vector
            assert(Set(1, shape(1)) == Set(child.typ.nRows, child.typ.nCols))
            BlockMatrixSparsity(nRowBlocks, nColBlocks)((i: Int, j: Int) => child.typ.hasBlock(i -> 0))
          case IndexedSeq(0, 0) => // diagonal as col vector
            assert(shape(0) == 1L)
            assert(shape(1) == java.lang.Math.min(child.typ.nRows, child.typ.nCols))
            BlockMatrixSparsity(nRowBlocks, nColBlocks)((_, j: Int) => child.typ.hasBlock(j -> j))
          case IndexedSeq(1, 0) => // transpose
            assert(child.typ.blockSize == blockSize)
            assert(shape(0) == child.typ.nCols && shape(1) == child.typ.nRows)
            BlockMatrixSparsity(child.typ.sparsity.definedBlocks.map(seq => seq.map { case (i, j) => (j, i)}))
          case IndexedSeq(0, 1) =>
            assert(child.typ.blockSize == blockSize)
            assert(shape(0) == child.typ.nRows && shape(1) == child.typ.nCols)
            child.typ.sparsity
        }
    else BlockMatrixSparsity.dense

    BlockMatrixType(child.typ.elementType, tensorShape, isRowVector, blockSize, sparsity)
  }

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixBroadcast = {
    assert(newChildren.length == 1)
    BlockMatrixBroadcast(newChildren(0).asInstanceOf[BlockMatrixIR], inIndexExpr, shape, blockSize)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val childBm = child.execute(ctx)
    val nRows = shape(0)
    val nCols = shape(1)

    inIndexExpr match {
      case IndexedSeq() =>
        val scalar = childBm.getElement(row = 0, col = 0)
        BlockMatrix.fill(nRows, nCols, scalar, blockSize)
      case IndexedSeq(0) =>
        BlockMatrixIR.checkFitsIntoArray(nRows, nCols)
        broadcastColVector(childBm.toBreezeMatrix().data, nRows.toInt, nCols.toInt)
      case IndexedSeq(1) =>
        BlockMatrixIR.checkFitsIntoArray(nRows, nCols)
        broadcastRowVector(childBm.toBreezeMatrix().data, nRows.toInt, nCols.toInt)
        // FIXME: I'm pretty sure this case is broken.
      case IndexedSeq(0, 0) =>
        BlockMatrixIR.checkFitsIntoArray(nRows, nCols)
        BlockMatrixIR.toBlockMatrix(nRows.toInt, nCols.toInt, childBm.diagonal(), blockSize)
      case IndexedSeq(1, 0) => childBm.transpose()
      case IndexedSeq(0, 1) => childBm
    }
  }

  private def broadcastRowVector(vec: Array[Double], nRows: Int, nCols: Int): BlockMatrix = {
    val data = ArrayBuffer[Double]()
    data.sizeHint(nRows * nCols)
    (0 until nRows).foreach(_ => data ++= vec)
    BlockMatrixIR.toBlockMatrix(nRows, nCols, data.toArray, blockSize)
  }

  private def broadcastColVector(vec: Array[Double], nRows: Int, nCols: Int): BlockMatrix = {
    val data = ArrayBuffer[Double]()
    data.sizeHint(nRows * nCols)
    (0 until nRows).foreach(row => (0 until nCols).foreach(_ => data += vec(row)))
    BlockMatrixIR.toBlockMatrix(nRows, nCols, data.toArray, blockSize)
  }
}

case class BlockMatrixAgg(
  child: BlockMatrixIR,
  outIndexExpr: IndexedSeq[Int]) extends BlockMatrixIR {

  val blockCostIsLinear: Boolean = child.blockCostIsLinear

  assert(outIndexExpr.length < 2)

  override def typ: BlockMatrixType = {
    val matrixShape = BlockMatrixIR.tensorShapeToMatrixShape(child)
    val matrixShapeArr = Array[Long](matrixShape._1, matrixShape._2)
    val shape = outIndexExpr.map({ i: Int => matrixShapeArr(i) }).toFastIndexedSeq
    val isRowVector = outIndexExpr == FastIndexedSeq(1)

    val sparsity = if (child.typ.isSparse) {
      outIndexExpr match {
        case IndexedSeq() => BlockMatrixSparsity.dense
        case IndexedSeq(1) => // col vector result; agg over row
          BlockMatrixSparsity(child.typ.nRowBlocks, 1) { (i, _) =>
            (0 until child.typ.nColBlocks).exists(j => child.typ.hasBlock(i -> j))
          }
        case IndexedSeq(0) => // row vector result; agg over col
          BlockMatrixSparsity(1, child.typ.nColBlocks) { (_, j) =>
            (0 until child.typ.nRowBlocks).exists(i => child.typ.hasBlock(i -> j))
          }
      }
    } else BlockMatrixSparsity.dense

    BlockMatrixType(child.typ.elementType, shape, isRowVector, child.typ.blockSize, sparsity)
  }

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixAgg = {
    assert(newChildren.length == 1)
    BlockMatrixAgg(newChildren(0).asInstanceOf[BlockMatrixIR], outIndexExpr)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val childBm = child.execute(ctx)

    outIndexExpr match {
      case IndexedSeq() => BlockMatrixIR.toBlockMatrix(nRows = 1, nCols = 1, Array(childBm.sum()), typ.blockSize)
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
  private[this] val Array(keepRow, keepCol) = indices
  private[this] val blockSize = child.typ.blockSize

  lazy val keepRowPartitioned: Array[Array[Long]] = keepRow.grouped(blockSize).toArray
  lazy val keepColPartitioned: Array[Array[Long]] = keepCol.grouped(blockSize).toArray

  private[this] def packOverlap(keep: Array[Array[Long]]): Array[Array[Int]] =
    keep.map(keeps => Array.range(BlockMatrixType.getBlockIdx(keeps.head, blockSize), BlockMatrixType.getBlockIdx(keeps.last, blockSize) + 1)).toArray

  lazy val rowBlockDependents: Array[Array[Int]] = if (keepRow.isEmpty) Array.tabulate(child.typ.nRowBlocks)(i => Array(i)) else packOverlap(keepRowPartitioned)
  lazy val colBlockDependents: Array[Array[Int]] = if (keepCol.isEmpty) Array.tabulate(child.typ.nColBlocks)(i => Array(i)) else packOverlap(keepColPartitioned)

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

    val IndexedSeq(nRows: Long, nCols: Long) = matrixShape.toFastIndexedSeq
    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(nRows, nCols)

    val sparsity = child.typ.sparsity.condense(rowBlockDependents -> colBlockDependents)
    BlockMatrixType(child.typ.elementType, tensorShape, isRowVector, blockSize, sparsity)
  }

  override def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixFilter = {
    assert(newChildren.length == 1)
    BlockMatrixFilter(newChildren(0).asInstanceOf[BlockMatrixIR], indices)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val bm = child.execute(ctx)
    if (keepRow.isEmpty) {
      bm.filterCols(keepCol)
    } else if (keepCol.isEmpty) {
      bm.filterRows(keepRow)
    } else {
      bm.filter(keepRow, keepCol)
    }
  }
}

case class BlockMatrixDensify(child: BlockMatrixIR) extends BlockMatrixIR {
  def typ: BlockMatrixType = BlockMatrixType.dense(
    child.typ.elementType,
    child.typ.nRows, child.typ.nCols, child.typ.blockSize)

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
  def typ: Type
  def definedBlocks(childType: BlockMatrixType): BlockMatrixSparsity
  def sparsify(bm: BlockMatrix): BlockMatrix
  def pretty(): String
}

//lower <= j - i <= upper
case class BandSparsifier(blocksOnly: Boolean, l: Long, u: Long) extends BlockMatrixSparsifier {
  val typ: Type = TTuple(TInt64, TInt64)
  def definedBlocks(childType: BlockMatrixType): BlockMatrixSparsity = {
    val leftBuffer = java.lang.Math.floorDiv(-l, childType.blockSize)
    val rightBuffer = java.lang.Math.floorDiv(u, childType.blockSize)

    BlockMatrixSparsity(childType.nRowBlocks, childType.nColBlocks) { (i, j) =>
      j >= (i - leftBuffer) && j <= (i + rightBuffer) && childType.hasBlock(i -> j)
    }
  }

  def sparsify(bm: BlockMatrix): BlockMatrix = {
    bm.filterBand(l, u, blocksOnly)
  }
  def pretty(): String =
    s"(BandSparsifier ${Pretty.prettyBooleanLiteral(blocksOnly)} $l $u)"
}

// interval per row, [start, end)
case class RowIntervalSparsifier(blocksOnly: Boolean, starts: IndexedSeq[Long], stops: IndexedSeq[Long]) extends BlockMatrixSparsifier {
  val typ: Type = TTuple(TArray(TInt64), TArray(TInt64))

  def definedBlocks(childType: BlockMatrixType): BlockMatrixSparsity = {
    val blockStarts = starts.grouped(childType.blockSize).map(idxs => childType.getBlockIdx(idxs.min)).toArray
    val blockStops = stops.grouped(childType.blockSize).map(idxs => childType.getBlockIdx(idxs.max - 1)).toArray

    BlockMatrixSparsity(childType.nRowBlocks, childType.nColBlocks) { (i, j) =>
      blockStarts(i) <= j && blockStops(i) >= j && childType.hasBlock(i -> j)
    }
  }

  def sparsify(bm: BlockMatrix): BlockMatrix = {
    bm.filterRowIntervals(starts.toArray, stops.toArray, blocksOnly)
  }

  def pretty(): String =
    s"(RowIntervalSparsifier ${ Pretty.prettyBooleanLiteral(blocksOnly) } ${ starts.mkString("(", " ", ")") } ${ stops.mkString("(", " ", ")") })"
}

//rectangle, starts/ends inclusive
case class RectangleSparsifier(rectangles: IndexedSeq[IndexedSeq[Long]]) extends BlockMatrixSparsifier {
  val typ: Type = TArray(TInt64)

  def definedBlocks(childType: BlockMatrixType): BlockMatrixSparsity = {
    val definedBlocks = rectangles.flatMap { case IndexedSeq(rowStart, rowEnd, colStart, colEnd) =>
      val rs = childType.getBlockIdx(java.lang.Math.max(rowStart, 0))
      val re = childType.getBlockIdx(java.lang.Math.min(rowEnd, childType.nRows))
      val cs = childType.getBlockIdx(java.lang.Math.max(colStart, 0))
      val ce = childType.getBlockIdx(java.lang.Math.min(colEnd, childType.nCols))
      Array.range(rs, re + 1).flatMap { i =>
        Array.range(cs, ce + 1)
          .filter { j => childType.hasBlock(i -> j) }
          .map { j => i -> j }
      }
    }.distinct

    BlockMatrixSparsity(definedBlocks)
  }

  def sparsify(bm: BlockMatrix): BlockMatrix = {
    bm.filterRectangles(rectangles.flatten.toArray)
  }

  def pretty(): String =
    s"(RectangleSparsifier ${ rectangles.flatten.mkString("(", " ", ")") })"
}

case class PerBlockSparsifier(blocks: IndexedSeq[Int]) extends BlockMatrixSparsifier {
  override def typ: Type = TArray(TInt32)

  val blockSet = blocks.toSet

  override def definedBlocks(childType: BlockMatrixType): BlockMatrixSparsity = {
    BlockMatrixSparsity(childType.nRowBlocks, childType.nColBlocks){ case(i: Int, j: Int) =>
      blockSet.contains(i + j * childType.nRowBlocks)
    }
  }

  override def sparsify(bm: BlockMatrix): BlockMatrix = bm.filterBlocks(blocks.toArray)

  override def pretty(): String = s"(PerBlockSparsifier with blocks $blocks"
}

case class BlockMatrixSparsify(
  child: BlockMatrixIR,
  sparsifier: BlockMatrixSparsifier
) extends BlockMatrixIR {
  def typ: BlockMatrixType = child.typ.copy(sparsity=sparsifier.definedBlocks(child.typ))

  def blockCostIsLinear: Boolean = child.blockCostIsLinear

  val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR = {
    val IndexedSeq(newChild: BlockMatrixIR) = newChildren
    BlockMatrixSparsify(newChild, sparsifier)
  }

  override def execute(ctx: ExecuteContext): BlockMatrix =
    sparsifier.sparsify(child.execute(ctx))
}

case class BlockMatrixSlice(child: BlockMatrixIR, slices: IndexedSeq[IndexedSeq[Long]]) extends BlockMatrixIR {
  assert(slices.length == 2)
  assert(slices.forall(_.length == 3))

  val blockCostIsLinear: Boolean = child.blockCostIsLinear

  lazy val IndexedSeq(rowBlockDependents: Array[Array[Int]], colBlockDependents: Array[Array[Int]]) = slices.map { case IndexedSeq(start, stop, step) =>
    val size = 1 + (stop - start - 1) / step
    val nBlocks = BlockMatrixType.numBlocks(size, child.typ.blockSize)
    Array.tabulate(nBlocks) { blockIdx =>
      val blockStart = start + blockIdx * child.typ.blockSize * step
      val blockEnd = java.lang.Math.min(start + ((blockIdx + 1) * child.typ.blockSize - 1) * step, stop)
      Array.range(child.typ.getBlockIdx(blockStart), child.typ.getBlockIdx(blockEnd) + 1)
    }
  }

  override def typ: BlockMatrixType = {
    val matrixShape: IndexedSeq[Long] = slices.map { s =>
      val IndexedSeq(start, stop, step) = s
      1 + (stop - start - 1) / step
    }

    val sparsity = child.typ.sparsity.condense(rowBlockDependents -> colBlockDependents)

    val (tensorShape, isRowVector) = BlockMatrixIR.matrixShapeToTensorShape(matrixShape(0), matrixShape(1))
    BlockMatrixType(child.typ.elementType, tensorShape, isRowVector, child.typ.blockSize, sparsity)
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
    BlockMatrixType.dense(elementType(child.typ), shape(0), shape(1), blockSize)
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
    Interpret[Any](ctx, child) match {
      case scalar: Double =>
        assert(nRows == 1 && nCols == 1)
        BlockMatrix.fill(nRows, nCols, scalar, blockSize)
      case data: IndexedSeq[_] =>
        BlockMatrixIR.toBlockMatrix(nRows.toInt, nCols.toInt, data.asInstanceOf[IndexedSeq[Double]].toArray, blockSize)
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

  override def typ: BlockMatrixType =
    BlockMatrixType.dense(TFloat64, shape(0), shape(1), blockSize)

  lazy val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixRandom = {
    assert(newChildren.isEmpty)
    BlockMatrixRandom(seed, gaussian, shape, blockSize)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    BlockMatrix.random(shape(0), shape(1), blockSize, seed, gaussian)
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