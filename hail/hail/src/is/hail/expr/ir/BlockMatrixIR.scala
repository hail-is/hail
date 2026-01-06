package is.hail.expr.ir

import is.hail.annotations.NDArray
import is.hail.backend.ExecuteContext
import is.hail.expr.Nat
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.{BMSContexts, BlockMatrixStage2, LowererUnsupportedOperation}
import is.hail.io.TypedCodecSpec
import is.hail.io.fs.FS
import is.hail.linalg.{BlockMatrix, BlockMatrixMetadata, MatrixSparsity}
import is.hail.types.encoded.{EBlockMatrixNDArray, EFloat64}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.compat.immutable.ArraySeq
import is.hail.utils.richUtils.RichDenseMatrixDouble

import scala.collection.immutable.NumericRange
import scala.collection.mutable.ArrayBuffer

import breeze.linalg.DenseMatrix
import breeze.numerics
import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}

object BlockMatrixIR {
  def checkFitsIntoArray(nRows: Long, nCols: Long): Unit = {
    require(nRows <= Int.MaxValue, s"Number of rows exceeds Int.MaxValue: $nRows")
    require(nCols <= Int.MaxValue, s"Number of columns exceeds Int.MaxValue: $nCols")
    require(
      nRows * nCols <= Int.MaxValue,
      s"Number of values exceeds Int.MaxValue: ${nRows * nCols}",
    )
  }

  def toBlockMatrix(
    ctx: ExecuteContext,
    nRows: Int,
    nCols: Int,
    data: Array[Double],
    blockSize: Int = BlockMatrix.defaultBlockSize,
  ): BlockMatrix =
    BlockMatrix.fromBreezeMatrix(
      ctx,
      new DenseMatrix[Double](nRows, nCols, data, 0, nCols, isTranspose = true),
      blockSize,
    )
}

sealed abstract class BlockMatrixIR extends BaseIR {
  def typ: BlockMatrixType

  protected[ir] def execute(ctx: ExecuteContext): BlockMatrix =
    fatal("tried to execute unexecutable IR:\n" + Pretty(ctx, this))

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR

  def blockCostIsLinear: Boolean
}

case class BlockMatrixRead(reader: BlockMatrixReader) extends BlockMatrixIR {
  override lazy val typ: BlockMatrixType = reader.fullType

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array.empty[BlockMatrixIR]

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BlockMatrixRead = {
    assert(newChildren.isEmpty)
    BlockMatrixRead(reader)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = reader(ctx)

  val blockCostIsLinear: Boolean = true
}

object BlockMatrixReader {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(
      List(
        classOf[BlockMatrixNativeReader],
        classOf[BlockMatrixBinaryReader],
        classOf[BlockMatrixPersistReader],
      ),
      typeHintFieldName = "name",
    )
  }

  def fromJValue(ctx: ExecuteContext, jv: JValue): BlockMatrixReader =
    (jv \ "name").extract[String] match {
      case "BlockMatrixNativeReader" => BlockMatrixNativeReader.fromJValue(ctx.fs, jv)
      case "BlockMatrixPersistReader" => BlockMatrixPersistReader.fromJValue(ctx, jv)
      case _ => jv.extract[BlockMatrixReader]
    }
}

abstract class BlockMatrixReader {
  def pathsUsed: Seq[String]
  def apply(ctx: ExecuteContext): BlockMatrix

  def lower(ctx: ExecuteContext, evalCtx: IRBuilder): BlockMatrixStage2 =
    throw new LowererUnsupportedOperation(s"BlockMatrixReader not implemented: ${this.getClass}")

  def fullType: BlockMatrixType

  def toJValue: JValue =
    Extraction.decompose(this)(BlockMatrixReader.formats)
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
  val metadata: BlockMatrixMetadata,
) extends BlockMatrixReader {
  def pathsUsed: Seq[String] = Array(params.path)

  lazy val fullType: BlockMatrixType = {
    val sparsity = MatrixSparsity.fromLinearCoords(
      BlockMatrixType.numBlocks(metadata.nRows, metadata.blockSize),
      BlockMatrixType.numBlocks(metadata.nCols, metadata.blockSize),
      metadata.maybeFiltered,
    )
    BlockMatrixType(TFloat64, metadata.nRows, metadata.nCols, metadata.blockSize, sparsity)
  }

  def apply(ctx: ExecuteContext): BlockMatrix = {
    val key = ("BlockMatrixNativeReader.apply", params.path)
    if (ctx.memo.contains(key)) {
      ctx.memo(key).asInstanceOf[BlockMatrix]
    } else {
      val bm = BlockMatrix.read(ctx, params.path)
      ctx.memo.update(key, bm)
      bm
    }

  }

  override def lower(ctx: ExecuteContext, evalCtx: IRBuilder): BlockMatrixStage2 = {
    val fileNames = Literal(TArray(TString), metadata.partFiles)

    val contexts = BMSContexts(evalCtx, fullType.sparsity, fileNames)

    val vType = TNDArray(fullType.elementType, Nat(2))
    val spec = TypedCodecSpec(
      EBlockMatrixNDArray(EFloat64(required = true), required = true),
      vType,
      BlockMatrix.bufferSpec,
    )
    val reader = ETypeValueReader(spec)

    def blockIR(ctx: IR): IR = {
      val path = Apply(
        "concat",
        FastSeq(),
        FastSeq(Str(s"${params.path}/parts/"), ctx),
        TString,
        ErrorIDs.NO_ERROR,
      )

      ReadValue(path, reader, vType)
    }

    BlockMatrixStage2(
      FastSeq(),
      fullType,
      contexts,
      blockIR,
    )
  }

  override def toJValue: JValue =
    decomposeWithName(params, "BlockMatrixNativeReader")(BlockMatrixReader.formats)

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: BlockMatrixNativeReader => params == that.params
    case _ => false
  }
}

case class BlockMatrixBinaryReader(path: String, shape: IndexedSeq[Long], blockSize: Int)
    extends BlockMatrixReader {
  def pathsUsed: Seq[String] = Array(path)

  val IndexedSeq(nRows, nCols) = shape
  BlockMatrixIR.checkFitsIntoArray(nRows, nCols)

  lazy val fullType: BlockMatrixType =
    BlockMatrixType.dense(TFloat64, nRows, nCols, blockSize)

  def apply(ctx: ExecuteContext): BlockMatrix = {
    val breezeMatrix =
      RichDenseMatrixDouble.importFromDoubles(
        ctx.fs,
        path,
        nRows.toInt,
        nCols.toInt,
        rowMajor = true,
      )
    BlockMatrix.fromBreezeMatrix(ctx, breezeMatrix, blockSize)
  }

  override def lower(ctx: ExecuteContext, evalCtx: IRBuilder): BlockMatrixStage2 = {
    val reader = NumpyBinaryValueReader(nRows, nCols)
    val nd = evalCtx.memoize(ReadValue(Str(path), reader, TNDArray(TFloat64, nDimsBase = Nat(2))))

    val typ = fullType
    val contexts = BMSContexts.tabulate(evalCtx, typ.sparsity)({ (blockRow, blockCol) =>
      NDArraySlice(
        nd,
        MakeTuple.ordered(FastSeq(
          MakeTuple.ordered(FastSeq(
            blockRow.toL * blockSize.toLong,
            minIR((blockRow + 1).toL * blockSize.toLong, nRows),
            1L,
          )),
          MakeTuple.ordered(FastSeq(
            blockCol.toL * blockSize.toLong,
            minIR((blockCol + 1).toL * blockSize.toLong, nCols),
            1L,
          )),
        )),
      )
    })

    def blockIR(ctx: IR) = ctx

    BlockMatrixStage2(FastSeq(), typ, contexts, blockIR)
  }
}

case class BlockMatrixNativePersistParameters(id: String)

object BlockMatrixPersistReader {
  def fromJValue(ctx: ExecuteContext, jv: JValue): BlockMatrixPersistReader = {
    implicit val formats: Formats = BlockMatrixReader.formats
    val params = jv.extract[BlockMatrixNativePersistParameters]
    BlockMatrixPersistReader(
      params.id,
      BlockMatrixType.fromBlockMatrix(ctx.BlockMatrixCache(params.id)),
    )
  }
}

case class BlockMatrixPersistReader(id: String, typ: BlockMatrixType) extends BlockMatrixReader {
  def pathsUsed: Seq[String] = FastSeq()
  lazy val fullType: BlockMatrixType = typ
  def apply(ctx: ExecuteContext): BlockMatrix = ctx.BlockMatrixCache(id)
}

case class BlockMatrixMap(child: BlockMatrixIR, eltName: Name, f: IR, needsDense: Boolean)
    extends BlockMatrixIR {
  override def typ: BlockMatrixType = child.typ

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, f)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BlockMatrixMap = {
    val IndexedSeq(newChild: BlockMatrixIR, newF: IR) = newChildren
    BlockMatrixMap(newChild, eltName, newF, needsDense)
  }

  val blockCostIsLinear: Boolean = child.blockCostIsLinear

  private def evalIR(ctx: ExecuteContext, ir: IR): Double = {
    val res: Any = CompileAndEvaluate[Any](ctx, ir)
    if (res == null)
      fatal("can't perform BlockMatrix operation on missing values!")
    res.asInstanceOf[Double]
  }

  private def binaryOp(scalar: Double, f: (DenseMatrix[Double], Double) => DenseMatrix[Double])
    : DenseMatrix[Double] => DenseMatrix[Double] =
    f(_, scalar)

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    assert(
      f.isInstanceOf[ApplyUnaryPrimOp]
        || f.isInstanceOf[Apply]
        || f.isInstanceOf[ApplyBinaryPrimOp]
        || f.isInstanceOf[Ref]
    )

    val prev = child.execute(ctx)

    val functionArgs = f match {
      case ApplyUnaryPrimOp(_, arg1) => IndexedSeq(arg1)
      case Apply(_, _, args, _, _) => args
      case ApplyBinaryPrimOp(_, l, r) => IndexedSeq(l, r)
      case _: Ref => IndexedSeq(f)
      case Constant(k) => IndexedSeq(k)
    }

    assert(
      functionArgs.forall(ir => IsConstant(ir) || ir.isInstanceOf[Ref]),
      "Spark backend without lowering does not support general mapping over " +
        "BlockMatrix entries. Use predefined functions like `BlockMatrix.abs`.",
    )

    val (name, breezeF): (String, DenseMatrix[Double] => DenseMatrix[Double]) = f match {
      case ApplyUnaryPrimOp(Negate, _) => ("negate", BlockMatrix.negationOp)
      case Apply("abs", _, _, _, _) => ("abs", numerics.abs(_))
      case Apply("log", _, _, _, _) => ("log", numerics.log(_))
      case Apply("sqrt", _, _, _, _) => ("sqrt", numerics.sqrt(_))
      case Apply("ceil", _, _, _, _) => ("ceil", numerics.ceil(_))
      case Apply("floor", _, _, _, _) => ("floor", numerics.floor(_))

      case Apply("pow", _, Seq(Ref(`eltName`, _), r), _, _) if !Mentions(r, eltName) =>
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
      case ApplyBinaryPrimOp(FloatingPointDivide(), Ref(`eltName`, _), r)
          if !Mentions(r, eltName) =>
        ("/", binaryOp(evalIR(ctx, r), (m, s) => m /:/ s))
      case ApplyBinaryPrimOp(FloatingPointDivide(), l, Ref(`eltName`, _))
          if !Mentions(l, eltName) =>
        ("/", binaryOp(evalIR(ctx, l), BlockMatrix.reverseScalarDiv))
      case Ref(`eltName`, _) =>
        ("identity", identity(_))
      case _: Ref | Constant(_) =>
        ("const", binaryOp(evalIR(ctx, f), (m, s) => m := s))

      case _ => fatal(s"Unsupported operation on BlockMatrices: ${Pretty(ctx, f)}")
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
  def mergeSparsity(left: MatrixSparsity, right: MatrixSparsity): MatrixSparsity
}

case object UnionBlocks extends SparsityStrategy {
  def exists(leftBlock: Boolean, rightBlock: Boolean): Boolean = leftBlock || rightBlock

  def mergeSparsity(left: MatrixSparsity, right: MatrixSparsity): MatrixSparsity =
    MatrixSparsity.union(left, right)
}

case object IntersectionBlocks extends SparsityStrategy {
  def exists(leftBlock: Boolean, rightBlock: Boolean): Boolean = leftBlock && rightBlock

  def mergeSparsity(left: MatrixSparsity, right: MatrixSparsity): MatrixSparsity =
    MatrixSparsity.intersection(left, right)
}

case object NeedsDense extends SparsityStrategy {
  def exists(leftBlock: Boolean, rightBlock: Boolean): Boolean = true

  def mergeSparsity(left: MatrixSparsity, right: MatrixSparsity): MatrixSparsity = {
    assert(left.nRows == right.nRows && left.nCols == right.nCols)
    assert(!left.isSparse && !right.isSparse)
    left
  }
}

case class BlockMatrixMap2(
  left: BlockMatrixIR,
  right: BlockMatrixIR,
  leftName: Name,
  rightName: Name,
  f: IR,
  sparsityStrategy: SparsityStrategy,
) extends BlockMatrixIR with Logging {
  override lazy val typ: BlockMatrixType =
    left.typ.copy(sparsity = sparsityStrategy.mergeSparsity(left.typ.sparsity, right.typ.sparsity))

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(left, right, f)

  val blockCostIsLinear: Boolean = left.blockCostIsLinear && right.blockCostIsLinear

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BlockMatrixMap2 = {
    assert(newChildren.length == 3)
    BlockMatrixMap2(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[BlockMatrixIR],
      leftName,
      rightName,
      newChildren(2).asInstanceOf[IR],
      sparsityStrategy,
    )
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    assert(f.isInstanceOf[ApplyBinaryPrimOp] || f.isInstanceOf[Apply])

    left match {
      case BlockMatrixBroadcast(vectorIR: BlockMatrixIR, IndexedSeq(x), _, _) =>
        val vector = coerceToVector(ctx, vectorIR)
        x match {
          case 1 => rowVectorOnLeft(ctx, vector, right, f)
          case 0 => colVectorOnLeft(ctx, vector, right, f)
        }
      case _ =>
        matrixOnLeft(ctx, left.execute(ctx), right, f)
    }
  }

  private def rowVectorOnLeft(
    ctx: ExecuteContext,
    rowVector: Array[Double],
    right: BlockMatrixIR,
    f: IR,
  ): BlockMatrix =
    opWithRowVector(ctx, right.execute(ctx), rowVector, f, reverse = true)

  private def colVectorOnLeft(
    ctx: ExecuteContext,
    colVector: Array[Double],
    right: BlockMatrixIR,
    f: IR,
  ): BlockMatrix =
    opWithColVector(ctx, right.execute(ctx), colVector, f, reverse = true)

  private def matrixOnLeft(ctx: ExecuteContext, matrix: BlockMatrix, right: BlockMatrixIR, f: IR)
    : BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(vectorIR, IndexedSeq(x), _, _) =>
        x match {
          case 1 =>
            val rightAsRowVec = coerceToVector(ctx, vectorIR)
            opWithRowVector(ctx, matrix, rightAsRowVec, f, reverse = false)
          case 0 =>
            val rightAsColVec = coerceToVector(ctx, vectorIR)
            opWithColVector(ctx, matrix, rightAsColVec, f, reverse = false)
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
          case vector: NDArray =>
            val IndexedSeq(numRows, numCols) = vector.shape
            assert(numRows == 1L || numCols == 1L)
            vector.getRowMajorElements().asInstanceOf[IndexedSeq[Double]].toArray
        }
      case _ => ir.execute(ctx).toBreezeMatrix().data
    }
  }

  private def opWithRowVector(
    ctx: ExecuteContext,
    left: BlockMatrix,
    right: Array[Double],
    f: IR,
    reverse: Boolean,
  ): BlockMatrix = {
    (f: @unchecked) match {
      case ApplyBinaryPrimOp(Add(), _, _) => left.rowVectorAdd(ctx, right)
      case ApplyBinaryPrimOp(Multiply(), _, _) => left.rowVectorMul(ctx, right)
      case ApplyBinaryPrimOp(Subtract(), _, _) =>
        if (reverse) left.reverseRowVectorSub(ctx, right)
        else left.rowVectorSub(ctx, right)
      case ApplyBinaryPrimOp(FloatingPointDivide(), _, _) =>
        if (reverse) left.reverseRowVectorDiv(ctx, right)
        else left.rowVectorDiv(ctx, right)
    }
  }

  private def opWithColVector(
    ctx: ExecuteContext,
    left: BlockMatrix,
    right: Array[Double],
    f: IR,
    reverse: Boolean,
  ): BlockMatrix = {
    (f: @unchecked) match {
      case ApplyBinaryPrimOp(Add(), _, _) => left.colVectorAdd(ctx, right)
      case ApplyBinaryPrimOp(Multiply(), _, _) => left.colVectorMul(ctx, right)
      case ApplyBinaryPrimOp(Subtract(), _, _) =>
        if (reverse) left.reverseColVectorSub(ctx, right)
        else left.colVectorSub(ctx, right)
      case ApplyBinaryPrimOp(FloatingPointDivide(), _, _) =>
        if (reverse) left.reverseColVectorDiv(ctx, right)
        else left.colVectorDiv(ctx, right)
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

case class BlockMatrixDot(left: BlockMatrixIR, right: BlockMatrixIR)
    extends BlockMatrixIR with Logging {
  override lazy val typ: BlockMatrixType = {
    val blockSize = left.typ.blockSize
    assert(left.typ.nCols == right.typ.nRows)

    val sparsity = if (left.typ.isSparse || right.typ.isSparse)
      MatrixSparsity.constructFromShapeAndFunction(left.typ.nRowBlocks, right.typ.nColBlocks) {
        (i: Int, j: Int) =>
          Array.tabulate(BlockMatrixType.numBlocks(right.typ.nCols, blockSize)) { k =>
            left.typ.hasBlock(i, k) && right.typ.hasBlock(k, j)
          }.reduce(_ || _)
      }
    else MatrixSparsity.dense(left.typ.nRowBlocks, right.typ.nColBlocks)
    BlockMatrixType(left.typ.elementType, left.typ.nRows, right.typ.nCols, blockSize, sparsity)
  }

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(left, right)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BlockMatrixDot = {
    assert(newChildren.length == 2)
    BlockMatrixDot(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[BlockMatrixIR],
    )
  }

  val blockCostIsLinear: Boolean = false

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    var leftBM = left.execute(ctx)
    var rightBM = right.execute(ctx)
    val fs = ctx.fs
    if (!left.blockCostIsLinear) {
      val path = ctx.createTmpPath("blockmatrix-dot-left", "bm")
      logger.info(
        s"BlockMatrix multiply: writing left input with ${leftBM.nRows} rows and ${leftBM.nCols} cols " +
          s"(${leftBM.gp.nBlocks} blocks of size ${leftBM.blockSize}) to temporary file $path"
      )
      leftBM.write(ctx, path)
      leftBM = BlockMatrixNativeReader(fs, path).apply(ctx)
    }
    if (!right.blockCostIsLinear) {
      val path = ctx.createTmpPath("blockmatrix-dot-right", "bm")
      logger.info(
        s"BlockMatrix multiply: writing right input with ${rightBM.nRows} rows and ${rightBM.nCols} cols " +
          s"(${rightBM.gp.nBlocks} blocks of size ${rightBM.blockSize}) to temporary file $path"
      )
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
  blockSize: Int,
) extends BlockMatrixIR {

  val blockCostIsLinear: Boolean = child.blockCostIsLinear

  assert(shape.length == 2)
  assert(inIndexExpr.length <= 2 && inIndexExpr.forall(x => x == 0 || x == 1))

  override lazy val typ: BlockMatrixType = {
    val IndexedSeq(nRows, nCols) = shape
    val nRowBlocks = BlockMatrixType.numBlocks(nRows, blockSize)
    val nColBlocks = BlockMatrixType.numBlocks(nCols, blockSize)
    val sparsity = child.typ.sparsity match {
      case sparse: MatrixSparsity.Sparse =>
        inIndexExpr match {
          case IndexedSeq() =>
            MatrixSparsity.dense(nRowBlocks, nColBlocks)
          case IndexedSeq(0) => // broadcast col vector
            MatrixSparsity.constructFromShapeAndFunction(nRowBlocks, nColBlocks)((i: Int, j: Int) =>
              sparse.contains(i, 0)
            )
          case IndexedSeq(1) => // broadcast row vector
            MatrixSparsity.constructFromShapeAndFunction(nRowBlocks, nColBlocks)((i: Int, j: Int) =>
              sparse.contains(0, j)
            )
          case IndexedSeq(0, 0) => // diagonal as row vector
            MatrixSparsity.constructFromShapeAndFunction(nRowBlocks, nColBlocks)((_, j: Int) =>
              sparse.contains(j, j)
            )
          case IndexedSeq(1, 0) => // transpose
            sparse.transpose
          case IndexedSeq(0, 1) =>
            sparse
        }
      case _ => MatrixSparsity.Dense(nRowBlocks, nColBlocks)
    }

    BlockMatrixType(child.typ.elementType, nRows, nCols, blockSize, sparsity)
  }

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : BlockMatrixBroadcast = {
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
        broadcastColVector(ctx, childBm.toBreezeMatrix().data, nRows.toInt, nCols.toInt)
      case IndexedSeq(1) =>
        BlockMatrixIR.checkFitsIntoArray(nRows, nCols)
        broadcastRowVector(ctx, childBm.toBreezeMatrix().data, nRows.toInt, nCols.toInt)
      // FIXME: I'm pretty sure this case is broken.
      case IndexedSeq(0, 0) =>
        BlockMatrixIR.checkFitsIntoArray(nRows, nCols)
        BlockMatrixIR.toBlockMatrix(ctx, nRows.toInt, nCols.toInt, childBm.diagonal(), blockSize)
      case IndexedSeq(1, 0) => childBm.transpose()
      case IndexedSeq(0, 1) => childBm
    }
  }

  private def broadcastRowVector(ctx: ExecuteContext, vec: Array[Double], nRows: Int, nCols: Int)
    : BlockMatrix = {
    val data = ArrayBuffer[Double]()
    data.sizeHint(nRows * nCols)
    (0 until nRows).foreach(_ => data ++= vec)
    BlockMatrixIR.toBlockMatrix(ctx, nRows, nCols, data.toArray, blockSize)
  }

  private def broadcastColVector(ctx: ExecuteContext, vec: Array[Double], nRows: Int, nCols: Int)
    : BlockMatrix = {
    val data = ArrayBuffer[Double]()
    data.sizeHint(nRows * nCols)
    (0 until nRows).foreach(row => (0 until nCols).foreach(_ => data += vec(row)))
    BlockMatrixIR.toBlockMatrix(ctx, nRows, nCols, data.toArray, blockSize)
  }
}

case class BlockMatrixAgg(
  child: BlockMatrixIR,
  axesToSumOut: IndexedSeq[Int],
) extends BlockMatrixIR {

  val blockCostIsLinear: Boolean = child.blockCostIsLinear

  assert(axesToSumOut.nonEmpty)

  override lazy val typ: BlockMatrixType = {
    val (nRows, nCols) = axesToSumOut match {
      case IndexedSeq(0, 1) => 1L -> 1L
      case IndexedSeq(0) => 1L -> child.typ.nCols
      case IndexedSeq(1) => child.typ.nRows -> 1L
    }
    val sparsity = if (child.typ.isSparse) {
      axesToSumOut match {
        case IndexedSeq(0, 1) => MatrixSparsity.dense(1, 1)
        case IndexedSeq(0) => // row vector result; agg over col
          child.typ.sparsity.condenseCols
        case IndexedSeq(1) => // col vector result; agg over row
          child.typ.sparsity.condenseRows
      }
    } else MatrixSparsity.dense(
      BlockMatrixType.numBlocks(nRows, child.typ.blockSize),
      BlockMatrixType.numBlocks(nCols, child.typ.blockSize),
    )

    BlockMatrixType(child.typ.elementType, nRows, nCols, child.typ.blockSize, sparsity)
  }

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BlockMatrixAgg = {
    assert(newChildren.length == 1)
    BlockMatrixAgg(newChildren(0).asInstanceOf[BlockMatrixIR], axesToSumOut)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val childBm = child.execute(ctx)

    axesToSumOut match {
      case IndexedSeq(0, 1) =>
        BlockMatrixIR.toBlockMatrix(ctx, nRows = 1, nCols = 1, Array(childBm.sum()), typ.blockSize)
      case IndexedSeq(0) => childBm.rowSum()
      case IndexedSeq(1) => childBm.colSum()
    }
  }
}

case class BlockMatrixFilter(
  child: BlockMatrixIR,
  indices: Array[Array[Long]],
) extends BlockMatrixIR {

  assert(indices.length == 2)

  val blockCostIsLinear: Boolean = child.blockCostIsLinear
  private[this] val Array(keepRow, keepCol) = indices

  override lazy val typ: BlockMatrixType = {
    val blockSize = child.typ.blockSize
    val keepRowPartitioned: IndexedSeq[IndexedSeq[Long]] =
      keepRow.grouped(blockSize).map(_.to(ArraySeq)).to(ArraySeq)
    val keepColPartitioned: IndexedSeq[IndexedSeq[Long]] =
      keepCol.grouped(blockSize).map(_.to(ArraySeq)).to(ArraySeq)

    val rowBlockDependents: IndexedSeq[IndexedSeq[Int]] = if (keepRowPartitioned.isEmpty)
      ArraySeq.tabulate(child.typ.nRowBlocks)(i => ArraySeq(i))
    else
      BlockMatrixType.getBlockDependencies(keepRowPartitioned, blockSize)

    val colBlockDependents: IndexedSeq[IndexedSeq[Int]] = if (keepColPartitioned.isEmpty)
      ArraySeq.tabulate(child.typ.nColBlocks)(i => ArraySeq(i))
    else
      BlockMatrixType.getBlockDependencies(keepColPartitioned, blockSize)

    val nRows = if (keepRow.isEmpty) child.typ.nRows else keepRow.length.toLong
    val nCols = if (keepCol.isEmpty) child.typ.nCols else keepCol.length.toLong

    val sparsity = child.typ.sparsity.condense(rowBlockDependents, colBlockDependents)
    BlockMatrixType(child.typ.elementType, nRows, nCols, blockSize, sparsity)
  }

  override def childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BlockMatrixFilter = {
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
  override lazy val typ: BlockMatrixType = child.typ.densify

  def blockCostIsLinear: Boolean = child.blockCostIsLinear

  val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR = {
    val IndexedSeq(newChild: BlockMatrixIR) = newChildren
    BlockMatrixDensify(newChild)
  }

  override def execute(ctx: ExecuteContext): BlockMatrix =
    child.execute(ctx).densify()
}

sealed abstract class BlockMatrixSparsifier {
  def typ: Type
  def definedBlocks(childType: BlockMatrixType): MatrixSparsity
  def sparsify(ctx: ExecuteContext, bm: BlockMatrix): BlockMatrix
  def pretty(): String
}

//lower <= j - i <= upper
case class BandSparsifier(blocksOnly: Boolean, l: Long, u: Long) extends BlockMatrixSparsifier {
  val typ: Type = TTuple(TInt64, TInt64)

  def definedBlocks(childType: BlockMatrixType): MatrixSparsity = {
    val lowerBlock = java.lang.Math.floorDiv(l, childType.blockSize).toInt
    val upperBlock = java.lang.Math.floorDiv(u + childType.blockSize - 1, childType.blockSize).toInt

    val blocks = for {
      j <- (0 until childType.nColBlocks).view
      i <- ((j - upperBlock) max 0) to
        ((j - lowerBlock) min (childType.nRowBlocks - 1))
      if childType.hasBlock(i, j)
    } yield i -> j
    MatrixSparsity.Sparse.sorted(
      childType.nRowBlocks,
      childType.nColBlocks,
      blocks.to(ArraySeq),
    )
  }

  override def sparsify(ctx: ExecuteContext, bm: BlockMatrix): BlockMatrix =
    bm.filterBand(l, u, blocksOnly)

  def pretty(): String =
    s"(BandSparsifier ${Pretty.prettyBooleanLiteral(blocksOnly)} $l $u)"
}

// interval per row, [start, end)
case class RowIntervalSparsifier(
  blocksOnly: Boolean,
  starts: IndexedSeq[Long],
  stops: IndexedSeq[Long],
) extends BlockMatrixSparsifier {
  val typ: Type = TTuple(TArray(TInt64), TArray(TInt64))

  def definedBlocks(childType: BlockMatrixType): MatrixSparsity = {
    val blockStarts =
      starts.grouped(childType.blockSize).map(idxs => childType.getBlockIdx(idxs.min)).toArray
    val blockStops =
      stops.grouped(childType.blockSize).map(idxs => childType.getBlockIdx(idxs.max - 1)).toArray

    MatrixSparsity.constructFromShapeAndFunction(childType.nRowBlocks, childType.nColBlocks) {
      (i, j) => blockStarts(i) <= j && blockStops(i) >= j && childType.hasBlock(i, j)
    }
  }

  override def sparsify(ctx: ExecuteContext, bm: BlockMatrix): BlockMatrix =
    bm.filterRowIntervals(ctx, starts.toArray, stops.toArray, blocksOnly)

  def pretty(): String =
    s"(RowIntervalSparsifier ${Pretty.prettyBooleanLiteral(blocksOnly)} ${starts.mkString("(", " ", ")")} ${stops.mkString("(", " ", ")")})"
}

//rectangle, starts/ends inclusive
case class RectangleSparsifier(rectangles: IndexedSeq[IndexedSeq[Long]])
    extends BlockMatrixSparsifier {
  val typ: Type = TArray(TInt64)

  def definedBlocks(childType: BlockMatrixType): MatrixSparsity = {
    val definedBlocks = rectangles.flatMap { case IndexedSeq(rowStart, rowEnd, colStart, colEnd) =>
      val rs = childType.getBlockIdx(java.lang.Math.max(rowStart, 0))
      val re = childType.getBlockIdx(java.lang.Math.min(rowEnd - 1, childType.nRows)) + 1
      val cs = childType.getBlockIdx(java.lang.Math.max(colStart, 0))
      val ce = childType.getBlockIdx(java.lang.Math.min(colEnd - 1, childType.nCols)) + 1
      Range(rs, re).flatMap { i =>
        Range(cs, ce)
          .filter(j => childType.hasBlock(i, j))
          .map(j => i -> j)
      }
    }.distinct

    MatrixSparsity.Sparse(childType.nRowBlocks, childType.nColBlocks, definedBlocks)
  }

  override def sparsify(ctx: ExecuteContext, bm: BlockMatrix): BlockMatrix =
    bm.filterRectangles(rectangles.flatten.toArray)

  def pretty(): String =
    s"(RectangleSparsifier ${rectangles.flatten.mkString("(", " ", ")")})"
}

case class PerBlockSparsifier(blocks: IndexedSeq[Int]) extends BlockMatrixSparsifier {
  override lazy val typ: Type = TArray(TInt32)

  val blockSet = blocks.toSet

  override def definedBlocks(childType: BlockMatrixType): MatrixSparsity =
    MatrixSparsity.constructFromShapeAndFunction(childType.nRowBlocks, childType.nColBlocks) {
      case (i: Int, j: Int) =>
        blockSet.contains(i + j * childType.nRowBlocks)
    }

  override def sparsify(ctx: ExecuteContext, bm: BlockMatrix): BlockMatrix =
    bm.filterBlocks(blocks.toArray)

  override def pretty(): String = s"(PerBlockSparsifier with blocks $blocks)"
}

case class BlockMatrixSparsify(
  child: BlockMatrixIR,
  sparsifier: BlockMatrixSparsifier,
) extends BlockMatrixIR {
  override lazy val typ: BlockMatrixType =
    child.typ.copy(sparsity = sparsifier.definedBlocks(child.typ))

  def blockCostIsLinear: Boolean = child.blockCostIsLinear

  val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR = {
    val IndexedSeq(newChild: BlockMatrixIR) = newChildren
    BlockMatrixSparsify(newChild, sparsifier)
  }

  override def execute(ctx: ExecuteContext): BlockMatrix =
    sparsifier.sparsify(ctx, child.execute(ctx))
}

case class BlockMatrixSlice(child: BlockMatrixIR, slices: IndexedSeq[IndexedSeq[Long]])
    extends BlockMatrixIR {
  assert(slices.length == 2)
  assert(slices.forall(_.length == 3))

  val blockCostIsLinear: Boolean = child.blockCostIsLinear

  lazy val IndexedSeq(
    rowBlockDependents: IndexedSeq[IndexedSeq[Int]],
    colBlockDependents: IndexedSeq[IndexedSeq[Int]],
  ) = slices.map { case IndexedSeq(start, stop, step) =>
    val size = 1 + (stop - start - 1) / step
    val nBlocks = BlockMatrixType.numBlocks(size, child.typ.blockSize)
    ArraySeq.tabulate(nBlocks) { blockIdx =>
      val blockStart = start + blockIdx * child.typ.blockSize * step
      val blockEnd =
        java.lang.Math.min(start + ((blockIdx + 1) * child.typ.blockSize - 1) * step, stop)
      Range(child.typ.getBlockIdx(blockStart), child.typ.getBlockIdx(blockEnd) + 1)
    }
  }

  override lazy val typ: BlockMatrixType = {
    val IndexedSeq(nRows, nCols): IndexedSeq[Long] =
      slices.map { case IndexedSeq(start, stop, step) =>
        1 + (stop - start - 1) / step
      }

    val sparsity = child.typ.sparsity.condense(rowBlockDependents, colBlockDependents)

    BlockMatrixType(child.typ.elementType, nRows, nCols, child.typ.blockSize, sparsity)
  }

  override def childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BlockMatrixIR = {
    assert(newChildren.length == 1)
    BlockMatrixSlice(newChildren(0).asInstanceOf[BlockMatrixIR], slices)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val bm = child.execute(ctx)
    val IndexedSeq(rowKeep, colKeep) = slices.map { s =>
      val IndexedSeq(start, stop, step) = s
      start until stop by step
    }

    if (isFullRange(rowKeep, child.typ.nRows)) {
      bm.filterCols(colKeep.toArray)
    } else if (isFullRange(colKeep, child.typ.nCols)) {
      bm.filterRows(rowKeep.toArray)
    } else {
      bm.filter(rowKeep.toArray, colKeep.toArray)
    }
  }

  private def isFullRange(r: NumericRange[Long], dimLength: Long): Boolean =
    r.start == 0 && r.end == dimLength && r.step == 1
}

case class ValueToBlockMatrix(
  child: IR,
  shape: IndexedSeq[Long],
  blockSize: Int,
) extends BlockMatrixIR {
  assert(shape.length == 2)

  val blockCostIsLinear: Boolean = true

  override lazy val typ: BlockMatrixType =
    BlockMatrixType.dense(elementType(child.typ), shape(0), shape(1), blockSize)

  private def elementType(childType: Type): Type =
    childType match {
      case ndarray: TNDArray => ndarray.elementType
      case array: TArray => array.elementType
      case _ => childType
    }

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : ValueToBlockMatrix = {
    assert(newChildren.length == 1)
    ValueToBlockMatrix(newChildren(0).asInstanceOf[IR], shape, blockSize)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix = {
    val IndexedSeq(nRows, nCols) = shape
    BlockMatrixIR.checkFitsIntoArray(nRows, nCols)
    CompileAndEvaluate[Any](ctx, child) match {
      case scalar: Double =>
        assert(nRows == 1 && nCols == 1)
        BlockMatrix.fill(nRows, nCols, scalar, blockSize)
      case data: IndexedSeq[_] =>
        BlockMatrixIR.toBlockMatrix(
          ctx,
          nRows.toInt,
          nCols.toInt,
          data.asInstanceOf[IndexedSeq[Double]].toArray,
          blockSize,
        )
      case ndData: NDArray =>
        BlockMatrixIR.toBlockMatrix(
          ctx,
          nRows.toInt,
          nCols.toInt,
          ndData.getRowMajorElements().asInstanceOf[IndexedSeq[Double]].toArray,
          blockSize,
        )
    }
  }
}

case class BlockMatrixRandom(
  staticUID: Long,
  gaussian: Boolean,
  shape: IndexedSeq[Long],
  blockSize: Int,
) extends BlockMatrixIR {

  assert(shape.length == 2)

  val blockCostIsLinear: Boolean = true

  override lazy val typ: BlockMatrixType =
    BlockMatrixType.dense(TFloat64, shape(0), shape(1), blockSize)

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): BlockMatrixRandom = {
    assert(newChildren.isEmpty)
    BlockMatrixRandom(staticUID, gaussian, shape, blockSize)
  }

  override protected[ir] def execute(ctx: ExecuteContext): BlockMatrix =
    BlockMatrix.random(shape(0), shape(1), blockSize, ctx.rngNonce, staticUID, gaussian)
}
