package is.hail.expr.ir

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.SparkSuite
import is.hail.expr.types.virtual.{TArray, TFloat64}
import is.hail.linalg.BlockMatrix
import org.testng.annotations.Test

class BlockMatrixIRSuite extends SparkSuite {

  def toBM(rows: Seq[Array[Double]]): BlockMatrix =
    toBM(rows, BlockMatrix.defaultBlockSize)

  def toBM(rows: Seq[Array[Double]], blockSize: Int): BlockMatrix = {
    val n = rows.length
    val m = if (n == 0) 0 else rows.head.length

    BlockMatrix.fromBreezeMatrix(sc, new BDM[Double](m, n, rows.flatten.toArray).t, blockSize)
  }

  def makeMatFromCol(vec: Seq[Double]): BlockMatrix = {
    toBM(vec.map(entry => Array(entry, entry, entry)))
  }

  def makeMatFromRow(vec: Seq[Double]): BlockMatrix = {
    toBM(Seq(vec.toArray, vec.toArray, vec.toArray))
  }

  def makeMap2(left: BlockMatrixIR, right: BlockMatrixIR,  op: BinaryOp):
  BlockMatrixMap2 = {
    BlockMatrixMap2(left, right, ApplyBinaryPrimOp(op, Ref("l", TFloat64()), Ref("l", TFloat64())))
  }

  def assertBmEq(actual: BlockMatrix, expected: BlockMatrix) {
    assert(actual.toBreezeMatrix() == expected.toBreezeMatrix())
  }

  val ones: BlockMatrix = BlockMatrix.fill(hc, 3, 3, 1)
  val twos: BlockMatrix = BlockMatrix.fill(hc, 3, 3, 2)
  val threes: BlockMatrix = BlockMatrix.fill(hc, 3, 3, 3)
  val fours: BlockMatrix = BlockMatrix.fill(hc, 3, 3, 4)

  @Test def testBlockMatrixWriteRead() {
    def sampleMatrix: BlockMatrix = BlockMatrix.fill(hc, 5, 5, 1)

    val tempPath = tmpDir.createLocalTempFile()
    Interpret(BlockMatrixWrite(new BlockMatrixLiteral(sampleMatrix), tempPath, false, false, false))

    val actualMatrix = BlockMatrixRead(tempPath).execute(hc)
    assertBmEq(actualMatrix, sampleMatrix)
  }

  val shape: Array[Long] = Array[Long](3, 3)

  @Test def testBlockMatrixBroadcastValue_Scalars() {
    val broadcastTwo = BlockMatrixBroadcast(
      ValueToBlockMatrix(MakeArray(Seq[F64](F64(2)), TArray(TFloat64())), Array[Long](), 0, Array[Boolean]()),
        Broadcast2D.SCALAR, shape, 0, Array[Boolean](false, false))

    val onesAddTwo = makeMap2(new BlockMatrixLiteral(ones), broadcastTwo, Add())
    val threesSubTwo = makeMap2(new BlockMatrixLiteral(threes), broadcastTwo, Subtract())
    val twosMulTwo = makeMap2(new BlockMatrixLiteral(twos), broadcastTwo, Multiply())
    val foursDivTwo = makeMap2(new BlockMatrixLiteral(fours), broadcastTwo, FloatingPointDivide())

    assertBmEq(onesAddTwo.execute(hc), threes)
    assertBmEq(threesSubTwo.execute(hc), ones)
    assertBmEq(twosMulTwo.execute(hc), fours)
    assertBmEq(foursDivTwo.execute(hc), twos)
  }

  @Test def testBlockMatrixBroadcastValue_Vectors() {
    val vectorLiteral = MakeArray(Seq[F64](F64(1), F64(2), F64(3)), TArray(TFloat64()))

    val broadcastRowVector = BlockMatrixBroadcast(ValueToBlockMatrix(vectorLiteral, Array[Long](3),
      0, Array(false)), Broadcast2D.ROW, shape, 0, Array[Boolean](false, false))
    val broadcastColVector = BlockMatrixBroadcast(ValueToBlockMatrix(vectorLiteral, Array[Long](3),
      0, Array(false)), Broadcast2D.COL, shape, 0, Array[Boolean](false, false))

    // Addition
    val actualOnesAddRowOnRight = makeMap2(new BlockMatrixLiteral(ones), broadcastRowVector, Add())
    val actualOnesAddColOnRight = makeMap2(new BlockMatrixLiteral(ones), broadcastColVector, Add())
    val actualOnesAddRowOnLeft  = makeMap2(broadcastRowVector, new BlockMatrixLiteral(ones), Add())
    val actualOnesAddColOnLeft  = makeMap2(broadcastColVector, new BlockMatrixLiteral(ones), Add())

    val expectedOnesAddRow = makeMatFromRow(Seq(2, 3, 4))
    val expectedOnesAddCol = makeMatFromCol(Seq(2, 3, 4))

    assertBmEq(actualOnesAddRowOnRight.execute(hc), expectedOnesAddRow)
    assertBmEq(actualOnesAddColOnRight.execute(hc), expectedOnesAddCol)
    assertBmEq(actualOnesAddRowOnLeft.execute(hc),  expectedOnesAddRow)
    assertBmEq(actualOnesAddColOnLeft.execute(hc),  expectedOnesAddCol)


    // Multiplication
    val actualOnesMulRowOnRight = makeMap2(new BlockMatrixLiteral(ones), broadcastRowVector, Multiply())
    val actualOnesMulColOnRight = makeMap2(new BlockMatrixLiteral(ones), broadcastColVector, Multiply())
    val actualOnesMulRowOnLeft  = makeMap2(broadcastRowVector, new BlockMatrixLiteral(ones), Multiply())
    val actualOnesMulColOnLeft  = makeMap2(broadcastColVector, new BlockMatrixLiteral(ones), Multiply())

    val expectedOnesMulRow = makeMatFromRow(Seq(1, 2, 3))
    val expectedOnesMulCol = makeMatFromCol(Seq(1, 2, 3))

    assertBmEq(actualOnesMulRowOnRight.execute(hc), expectedOnesMulRow)
    assertBmEq(actualOnesMulColOnRight.execute(hc), expectedOnesMulCol)
    assertBmEq(actualOnesMulRowOnLeft.execute(hc),  expectedOnesMulRow)
    assertBmEq(actualOnesMulColOnLeft.execute(hc),  expectedOnesMulCol)


    // Subtraction
    val actualOnesSubRowOnRight = makeMap2(new BlockMatrixLiteral(ones), broadcastRowVector, Subtract())
    val actualOnesSubColOnRight = makeMap2(new BlockMatrixLiteral(ones), broadcastColVector, Subtract())
    val actualOnesSubRowOnLeft  = makeMap2(broadcastRowVector, new BlockMatrixLiteral(ones), Subtract())
    val actualOnesSubColOnLeft  = makeMap2(broadcastColVector, new BlockMatrixLiteral(ones), Subtract())

    val expectedOnesSubRowRight = makeMatFromRow(Seq(0, -1, -2))
    val expectedOnesSubColRight = makeMatFromCol(Seq(0, -1, -2))
    val expectedOnesSubRowLeft = makeMatFromRow(Seq(0, 1, 2))
    val expectedOnesSubColLeft = makeMatFromCol(Seq(0, 1, 2))

    assertBmEq(actualOnesSubRowOnRight.execute(hc), expectedOnesSubRowRight)
    assertBmEq(actualOnesSubColOnRight.execute(hc), expectedOnesSubColRight)
    assertBmEq(actualOnesSubRowOnLeft.execute(hc),  expectedOnesSubRowLeft)
    assertBmEq(actualOnesSubColOnLeft.execute(hc),  expectedOnesSubColLeft)


    // Division
    val actualOnesDivRowOnRight = makeMap2(new BlockMatrixLiteral(ones), broadcastRowVector, FloatingPointDivide())
    val actualOnesDivColOnRight = makeMap2(new BlockMatrixLiteral(ones), broadcastColVector, FloatingPointDivide())
    val actualOnesDivRowOnLeft  = makeMap2(broadcastRowVector, new BlockMatrixLiteral(ones), FloatingPointDivide())
    val actualOnesDivColOnLeft  = makeMap2(broadcastColVector, new BlockMatrixLiteral(ones), FloatingPointDivide())

    val expectedOnesDivRowRight = makeMatFromRow(Seq(1, 1.0 / 2.0, 1.0 / 3.0))
    val expectedOnesDivColRight = makeMatFromCol(Seq(1, 1.0 / 2.0, 1.0 / 3.0))
    val expectedOnesDivRowLeft = makeMatFromRow(Seq(1, 2, 3))
    val expectedOnesDivColLeft = makeMatFromCol(Seq(1, 2, 3))

    assertBmEq(actualOnesDivRowOnRight.execute(hc), expectedOnesDivRowRight)
    assertBmEq(actualOnesDivColOnRight.execute(hc), expectedOnesDivColRight)
    assertBmEq(actualOnesDivRowOnLeft.execute(hc),  expectedOnesDivRowLeft)
    assertBmEq(actualOnesDivColOnLeft.execute(hc),  expectedOnesDivColLeft)
  }
}
