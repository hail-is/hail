package is.hail.expr.ir

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.{ExecStrategy, HailSuite}
import is.hail.expr.types.virtual.{TArray, TFloat64}
import is.hail.utils._
import is.hail.TestUtils._
import org.testng.annotations.Test

class BlockMatrixIRSuite extends HailSuite {

  val N_ROWS = 3
  val N_COLS = 3
  val BLOCK_SIZE = 10
  val shape: Array[Long] = Array[Long](N_ROWS, N_COLS)

  def fill(v: Double, nRows: Int = N_ROWS, nCols: Int = N_COLS, blockSize: Int = BLOCK_SIZE) =
    ValueToBlockMatrix(MakeArray(Array.fill(nRows * nCols)(F64(v)).toFastIndexedSeq, TArray(TFloat64)),
      FastIndexedSeq(nRows, nCols), blockSize)

  val ones: BlockMatrixIR = fill(1)

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.interpretOnly

  def makeMap2(left: BlockMatrixIR, right: BlockMatrixIR,  op: BinaryOp, strategy: SparsityStrategy):
  BlockMatrixMap2 = {
    BlockMatrixMap2(left, right, "l", "r", ApplyBinaryPrimOp(op, Ref("l", TFloat64), Ref("r", TFloat64)), strategy)
  }

  @Test def testBlockMatrixWriteRead() {
    val tempPath = tmpDir.createLocalTempFile()
    Interpret[Unit](ctx, BlockMatrixWrite(ones,
      BlockMatrixNativeWriter(tempPath, false, false, false)))

    assertBMEvalsTo(BlockMatrixRead(BlockMatrixNativeReader(tempPath)), BDM.fill[Double](N_ROWS, N_COLS)(1))
  }

  @Test def testBlockMatrixMap() {
    val sqrtIR = BlockMatrixMap(ones, "element", Apply("sqrt", FastIndexedSeq(Ref("element", TFloat64)), TFloat64), false)
    val negIR = BlockMatrixMap(ones, "element", ApplyUnaryPrimOp(Negate(), Ref("element", TFloat64)), false)
    val logIR = BlockMatrixMap(ones, "element", Apply("log", FastIndexedSeq(Ref("element", TFloat64)), TFloat64), true)
    val absIR = BlockMatrixMap(ones, "element", Apply("abs", FastIndexedSeq(Ref("element", TFloat64)), TFloat64), false)

    assertBMEvalsTo(sqrtIR, BDM.fill[Double](3, 3)(1))
    assertBMEvalsTo(negIR, BDM.fill[Double](3, 3)(-1))
    assertBMEvalsTo(logIR, BDM.fill[Double](3, 3)(0))
    assertBMEvalsTo(absIR, BDM.fill[Double](3, 3)(1))
  }

  @Test def testBlockMatrixBroadcastValue_Scalars() {
    val broadcastTwo = BlockMatrixBroadcast(
      ValueToBlockMatrix(MakeArray(Seq[F64](F64(2)), TArray(TFloat64)), Array[Long](1, 1), ones.typ.blockSize),
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
    val vectorLiteral = MakeArray(Seq[F64](F64(1), F64(2), F64(3)), TArray(TFloat64))

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

  @Test def testBlockMatrixDot() {
    assertBMEvalsTo(BlockMatrixDot(fill(2, nRows = N_ROWS, nCols = 5), fill(3, nRows = 5, nCols = N_COLS)),
      BDM.fill[Double](N_ROWS, N_COLS)(2 * 3 * 5))
  }

  @Test def testLower() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.allRelational
    val blockSize = 3
    def value(nRows: Long, nCols: Long, data: Double*): (BlockMatrixIR, BDM[Double]) = {
      val ir = ValueToBlockMatrix(Literal(TArray(TFloat64), data),
        FastIndexedSeq(nRows, nCols), blockSize)
      val bdm = new BDM(nCols.toInt, nRows.toInt, data.toArray).t
      ir -> bdm
    }

    val (m1IR, m1) = value(5, 4, Array.tabulate(20)(i => i + 1.0): _*)
    val (m2IR, m2) = value(4, 6, Array.tabulate(24)(i => 25.0 - i): _*)
    assertBMEvalsTo(BlockMatrixDot(m1IR, m2IR), m1 * m2)
  }
}
