package is.hail.expr.ir

import breeze.linalg.{DenseMatrix => BDM}
import cats.syntax.all._
import is.hail.ExecStrategy.ExecStrategy
import is.hail.backend.utils._
import is.hail.expr.Nat
import is.hail.expr.ir.lowering.Lower.monadLowerInstanceForLower
import is.hail.expr.ir.lowering.{Lower, LoweringState}
import is.hail.io.TypedCodecSpec
import is.hail.linalg.BlockMatrix
import is.hail.types.encoded.{EBlockMatrixNDArray, EFloat64Required}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.{ExecStrategy, HailSuite}
import org.testng.annotations.Test

class BlockMatrixIRSuite extends HailSuite {

  val N_ROWS = 3
  val N_COLS = 3
  val BLOCK_SIZE = 10
  val shape: Array[Long] = Array[Long](N_ROWS, N_COLS)

  def toIR(bdm: BDM[Double], blockSize: Int = BLOCK_SIZE): BlockMatrixIR =
    ValueToBlockMatrix(Literal(TArray(TFloat64), bdm.t.toArray.toFastIndexedSeq),
      FastIndexedSeq(bdm.rows, bdm.cols), blockSize)

  def fill(v: Double, nRows: Int = N_ROWS, nCols: Int = N_COLS, blockSize: Int = BLOCK_SIZE) =
    toIR(BDM.fill[Double](nRows, nCols)(v), blockSize)

  val ones: BlockMatrixIR = fill(1)

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.allRelational

  def makeMap2(left: BlockMatrixIR, right: BlockMatrixIR,  op: BinaryOp, strategy: SparsityStrategy):
  BlockMatrixMap2 = {
    BlockMatrixMap2(left, right, "l", "r", ApplyBinaryPrimOp(op, Ref("l", TFloat64), Ref("r", TFloat64)), strategy)
  }

  @Test def testBlockMatrixWriteRead() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.interpretOnly
    (for {
      tempPath <- newTmpPath[Lower]("test-blockmatrix-write-read", "bm")
      _ <- Interpret(BlockMatrixWrite(ones, BlockMatrixNativeWriter(tempPath, false, false, false)))
    } yield assertBMEvalsTo(BlockMatrixRead(BlockMatrixNativeReader(fs, tempPath)), BDM.fill[Double](N_ROWS, N_COLS)(1)))
      .runA(ctx, LoweringState())
  }

  @Test def testBlockMatrixMap() {
    val sqrtIR = BlockMatrixMap(ones, "element", Apply("sqrt", FastIndexedSeq(), FastIndexedSeq(Ref("element", TFloat64)), TFloat64, ErrorIDs.NO_ERROR), false)
    val negIR = BlockMatrixMap(ones, "element", ApplyUnaryPrimOp(Negate(), Ref("element", TFloat64)), false)
    val logIR = BlockMatrixMap(ones, "element", Apply("log", FastIndexedSeq(), FastIndexedSeq(Ref("element", TFloat64)), TFloat64, ErrorIDs.NO_ERROR), true)
    val absIR = BlockMatrixMap(ones, "element", Apply("abs", FastIndexedSeq(), FastIndexedSeq(Ref("element", TFloat64)), TFloat64, ErrorIDs.NO_ERROR), false)

    assertBMEvalsTo(sqrtIR, BDM.fill[Double](3, 3)(1))
    assertBMEvalsTo(negIR, BDM.fill[Double](3, 3)(-1))
    assertBMEvalsTo(logIR, BDM.fill[Double](3, 3)(0))
    assertBMEvalsTo(absIR, BDM.fill[Double](3, 3)(1))
  }

  @Test def testBlockMatrixMap2() {
    val onesAddOnes = makeMap2(ones, ones, Add(), UnionBlocks)
    val onesSubOnes = makeMap2(ones, ones, Subtract(), UnionBlocks)
    val onesMulOnes = makeMap2(ones, ones, Multiply(), IntersectionBlocks)
    val onesDivOnes = makeMap2(ones, ones, FloatingPointDivide(), NeedsDense)

    assertBMEvalsTo(onesAddOnes, BDM.fill[Double](3, 3)(1.0 + 1.0))
    assertBMEvalsTo(onesSubOnes, BDM.fill[Double](3, 3)(1.0 - 1.0))
    assertBMEvalsTo(onesMulOnes, BDM.fill[Double](3, 3)(1.0 * 1.0))
    assertBMEvalsTo(onesDivOnes, BDM.fill[Double](3, 3)(1.0 / 1.0))
  }

  @Test def testBlockMatrixBroadcastValue_Scalars() {
    val broadcastTwo = BlockMatrixBroadcast(
      ValueToBlockMatrix(MakeArray(IndexedSeq[F64](F64(2)), TArray(TFloat64)), Array[Long](1, 1), ones.typ.blockSize),
      FastIndexedSeq(), shape, ones.typ.blockSize)

    val onesAddTwo = makeMap2(ones, broadcastTwo, Add(), UnionBlocks)
    val onesSubTwo = makeMap2(ones, broadcastTwo, Subtract(), UnionBlocks)
    val onesMulTwo = makeMap2(ones, broadcastTwo, Multiply(), IntersectionBlocks)
    val onesDivTwo = makeMap2(ones, broadcastTwo, FloatingPointDivide(), NeedsDense)

    assertBMEvalsTo(onesAddTwo, BDM.fill[Double](3, 3)(1.0 + 2.0))
    assertBMEvalsTo(onesSubTwo, BDM.fill[Double](3, 3)(1.0 - 2.0))
    assertBMEvalsTo(onesMulTwo, BDM.fill[Double](3, 3)(1.0 * 2.0))
    assertBMEvalsTo(onesDivTwo, BDM.fill[Double](3, 3)(1.0 / 2.0))
  }

  @Test def testBlockMatrixBroadcastValue_Vectors() {
    val vectorLiteral = MakeArray(IndexedSeq[F64](F64(1), F64(2), F64(3)), TArray(TFloat64))

    val broadcastRowVector = BlockMatrixBroadcast(ValueToBlockMatrix(vectorLiteral, Array[Long](1, 3),
      ones.typ.blockSize), FastIndexedSeq(1), shape, ones.typ.blockSize)
    val broadcastColVector = BlockMatrixBroadcast(ValueToBlockMatrix(vectorLiteral, Array[Long](3, 1),
      ones.typ.blockSize), FastIndexedSeq(0), shape, ones.typ.blockSize)

    val ops = Array(
      (Add(), UnionBlocks, (i: Double, j: Double) => i + j),
      (Subtract(), UnionBlocks, (i: Double, j: Double) => i - j),
      (Multiply(), IntersectionBlocks, (i: Double, j: Double) => i * j),
      (FloatingPointDivide(), NeedsDense, (i: Double, j: Double) => i / j))
    for ((op, merge, f) <- ops) {
      val rightRowOp = makeMap2(ones, broadcastRowVector, op, merge)
      val rightColOp = makeMap2(ones, broadcastColVector, op, merge)
      val leftRowOp = makeMap2(broadcastRowVector, ones, op, merge)
      val leftColOp = makeMap2(broadcastColVector, ones, op, merge)

      BDM.tabulate(3, 3){ (_, j) => f(1.0, j + 1) }

      val expectedRightRowOp = BDM.tabulate(3, 3){ (_, j) => f(1.0, j + 1) }
      val expectedRightColOp = BDM.tabulate(3, 3){ (i, _) => f(1.0, i + 1) }
      val expectedLeftRowOp = BDM.tabulate(3, 3){ (_, j) => f(j + 1, 1.0) }
      val expectedLeftColOp = BDM.tabulate(3, 3){ (i, _) => f(i + 1, 1.0) }

      assertBMEvalsTo(rightRowOp, expectedRightRowOp)
      assertBMEvalsTo(rightColOp, expectedRightColOp)
      assertBMEvalsTo(leftRowOp, expectedLeftRowOp)
      assertBMEvalsTo(leftColOp, expectedLeftColOp)
    }
  }

  @Test def testBlockMatrixFilter() {
    val nRows = 5
    val nCols = 8
    val original = BDM.tabulate[Double](nRows, nCols)((i, j) => i * nCols + j)
    val unfiltered = toIR(original, blockSize = 3)

    val keepRows = Array(0L, 1L, 4L)
    val keepCols = Array(0L, 2L, 7L)

    assertBMEvalsTo(BlockMatrixFilter(unfiltered, Array(keepRows, Array())),
      original(keepRows.map(_.toInt).toFastIndexedSeq, ::).toDenseMatrix)
    assertBMEvalsTo(BlockMatrixFilter(unfiltered, Array(Array(), keepCols)),
      original(::, keepCols.map(_.toInt).toFastIndexedSeq).toDenseMatrix)
    assertBMEvalsTo(BlockMatrixFilter(unfiltered, Array(keepRows, keepCols)),
      original(keepRows.map(_.toInt).toFastIndexedSeq, keepCols.map(_.toInt).toFastIndexedSeq).toDenseMatrix)
  }

  @Test def testBlockMatrixSlice() {
    val nRows = 12
    val nCols = 8
    val original = BDM.tabulate[Double](nRows, nCols)((i, j) => i * nCols + j)
    val unsliced = toIR(original, blockSize = 3)

    val rowSlice = FastIndexedSeq(1L, 10L, 3L)
    val colSlice = FastIndexedSeq(4L, 8L, 2L)
    assertBMEvalsTo(BlockMatrixSlice(unsliced, FastIndexedSeq(rowSlice, colSlice)),
      original(
        Array.range(rowSlice(0).toInt, rowSlice(1).toInt, rowSlice(2).toInt).toFastIndexedSeq,
        Array.range(colSlice(0).toInt, colSlice(1).toInt, colSlice(2).toInt).toFastIndexedSeq).toDenseMatrix)
  }

  @Test def testBlockMatrixDot() {
    val m1 = BDM.tabulate[Double](5, 4)((i, j) => (i + 1) * j)
    val m2 = BDM.tabulate[Double](4, 6)((i, j) => (i + 5) * (j - 2))
    assertBMEvalsTo(BlockMatrixDot(toIR(m1), toIR(m2)), m1 * m2)
  }

  @Test def testBlockMatrixRandom() {
    val gaussian = BlockMatrixRandom(0, gaussian = true, shape = Array(5L, 6L), blockSize = 3)
    val uniform = BlockMatrixRandom(0, gaussian = false, shape = Array(5L, 6L), blockSize = 3)

    assertBMEvalsTo(BlockMatrixMap2(gaussian, gaussian, "l", "r", Ref("l", TFloat64) - Ref("r", TFloat64), NeedsDense), BDM.fill(5, 6)(0.0))
    assertBMEvalsTo(BlockMatrixMap2(uniform, uniform, "l", "r", Ref("l", TFloat64) - Ref("r", TFloat64), NeedsDense), BDM.fill(5, 6)(0.0))
  }

  @Test def readBlockMatrixIR() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly
    val etype = EBlockMatrixNDArray(EFloat64Required, required = true)
    val path = "src/test/resources/blockmatrix_example/0/parts/part-0-28-0-0-0feb7ac2-ab02-6cd4-5547-bfcb94dacb33"
    val matrix = BlockMatrix.read(fs, "src/test/resources/blockmatrix_example/0").toBreezeMatrix()
    val expected = Array.tabulate(2)(i => Array.tabulate(2)(j => matrix(i, j)).toFastIndexedSeq).toFastIndexedSeq

    val typ = TNDArray(TFloat64, Nat(2))
    val spec = TypedCodecSpec(etype, typ, BlockMatrix.bufferSpec)
    val read = ReadValue(Str(path), spec, typ)
    assertNDEvals(read, expected)
    assertNDEvals(ReadValue(
      WriteValue(read, Str(ctx.createTmpPath("read-blockmatrix-ir", "hv")) + UUID4(), ETypeFileValueWriter(spec)),
      spec, typ), expected)
  }

  @Test def readWriteBlockMatrix() {
    val original = "src/test/resources/blockmatrix_example/0"
    val expected = BlockMatrix.read(ctx.fs, original).toBreezeMatrix()

    val path = ctx.createTmpPath("read-blockmatrix-ir", "bm")

    assertEvalsTo(BlockMatrixWrite(
      BlockMatrixRead(BlockMatrixNativeReader(ctx.fs, original)),
      BlockMatrixNativeWriter(path, overwrite = true, forceRowMajor = false, stageLocally = false)), ())

    assertBMEvalsTo(BlockMatrixRead(BlockMatrixNativeReader(ctx.fs, path)), expected)
  }
}
