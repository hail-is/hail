package is.hail.expr.ir

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.SparkSuite
import is.hail.expr.types.virtual.{TArray, TFloat64, TInt32}
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

  val element = Ref("element", TFloat64())
  def mapOnRight(bm: BlockMatrixIR, op: BinaryOp, value: IR): BlockMatrixIR = {
    BlockMatrixBroadcastValue(bm, ApplyBinaryPrimOp(op, element, value))
  }

  def mapOnLeft(bm: BlockMatrixIR, op: BinaryOp, value: IR): BlockMatrixIR = {
    BlockMatrixBroadcastValue(bm, ApplyBinaryPrimOp(op, value, element))
  }

  def makeMatFromCol(vec: Seq[Double]): BlockMatrix = {
    toBM(vec.map(entry => Array(entry, entry, entry)))
  }

  def makeMatFromRow(vec: Seq[Double]): BlockMatrix = {
    toBM(Seq(vec.toArray, vec.toArray, vec.toArray))
  }

  def createBlockMatrixElemWiseOp(left: BlockMatrixIR, right: BlockMatrixIR,  op: BinaryOp):
  BlockMatrixElementWiseBinaryOp = {
    BlockMatrixElementWiseBinaryOp(left, right, ApplyBinaryPrimOp(op, element, element))
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
    assert(actualMatrix.toBreezeMatrix() == sampleMatrix.toBreezeMatrix())
  }

  @Test def testBlockMatrixBroadcastValue_Scalars() {
    val onesAddTwo = mapOnRight(new BlockMatrixLiteral(ones), Add(), F64(2))
    val threesSubTwo = mapOnRight(new BlockMatrixLiteral(threes), Subtract(), F64(2))
    val twosMulTwo = mapOnRight(new BlockMatrixLiteral(twos), Multiply(), F64(2))
    val foursDivTwo = mapOnRight(new BlockMatrixLiteral(fours), FloatingPointDivide(), F64(2))

    assert(onesAddTwo.execute(hc).toBreezeMatrix() == threes.toBreezeMatrix())
    assert(threesSubTwo.execute(hc).toBreezeMatrix() == ones.toBreezeMatrix())
    assert(twosMulTwo.execute(hc).toBreezeMatrix() == fours.toBreezeMatrix())
    assert(foursDivTwo.execute(hc).toBreezeMatrix() == twos.toBreezeMatrix())
  }

  @Test def testBlockMatrixBroadcastValue_Vectors() {
    val rowVectorShapeLiteral = Literal(TArray(TInt32()), Seq[Long](1, 3))
    val colVectorShapeLiteral = Literal(TArray(TInt32()), Seq[Long](3, 1))
    val vectorLiteral = Literal(TArray(TFloat64()), Seq[Double](1, 2, 3))

    val rowVector = MakeStruct(IndexedSeq(("row_vector", rowVectorShapeLiteral), ("data", vectorLiteral)))
    val colVector = MakeStruct(IndexedSeq(("row_vector", colVectorShapeLiteral), ("data", vectorLiteral)))

    // Addition
    val actualOnesAddRowOnRight = mapOnRight(new BlockMatrixLiteral(ones), Add(), rowVector)
    val actualOnesAddColOnRight = mapOnRight(new BlockMatrixLiteral(ones), Add(), colVector)
    val actualOnesAddRowOnLeft = mapOnLeft(new BlockMatrixLiteral(ones), Add(), rowVector)
    val actualOnesAddColOnLeft = mapOnLeft(new BlockMatrixLiteral(ones), Add(), colVector)

    val expectedOnesAddRow = makeMatFromRow(Seq(2, 3, 4))
    val expectedOnesAddCol = makeMatFromCol(Seq(2, 3, 4))

    assert(actualOnesAddRowOnRight.execute(hc).toBreezeMatrix() == expectedOnesAddRow.toBreezeMatrix())
    assert(actualOnesAddColOnRight.execute(hc).toBreezeMatrix() == expectedOnesAddCol.toBreezeMatrix())
    assert(actualOnesAddRowOnLeft.execute(hc).toBreezeMatrix() == expectedOnesAddRow.toBreezeMatrix())
    assert(actualOnesAddColOnLeft.execute(hc).toBreezeMatrix() == expectedOnesAddCol.toBreezeMatrix())


    // Multiplication
    val actualOnesMulRowOnRight = mapOnRight(new BlockMatrixLiteral(ones), Multiply(), rowVector)
    val actualOnesMulColOnRight = mapOnRight(new BlockMatrixLiteral(ones), Multiply(), colVector)
    val actualOnesMulRowOnLeft  = mapOnLeft(new BlockMatrixLiteral(ones), Multiply(), rowVector)
    val actualOnesMulColOnLeft  = mapOnLeft(new BlockMatrixLiteral(ones), Multiply(), colVector)

    val expectedOnesMulRow = makeMatFromRow(Seq(1, 2, 3))
    val expectedOnesMulCol = makeMatFromCol(Seq(1, 2, 3))

    assert(actualOnesMulRowOnRight.execute(hc).toBreezeMatrix() == expectedOnesMulRow.toBreezeMatrix())
    assert(actualOnesMulColOnRight.execute(hc).toBreezeMatrix() == expectedOnesMulCol.toBreezeMatrix())
    assert(actualOnesMulRowOnLeft.execute(hc).toBreezeMatrix() == expectedOnesMulRow.toBreezeMatrix())
    assert(actualOnesMulColOnLeft.execute(hc).toBreezeMatrix() == expectedOnesMulCol.toBreezeMatrix())


    // Subtraction
    val actualOnesSubRowOnRight = mapOnRight(new BlockMatrixLiteral(ones), Subtract(), rowVector)
    val actualOnesSubColOnRight = mapOnRight(new BlockMatrixLiteral(ones), Subtract(), colVector)
    val actualOnesSubRowOnLeft  = mapOnLeft(new BlockMatrixLiteral(ones), Subtract(), rowVector)
    val actualOnesSubColOnLeft  = mapOnLeft(new BlockMatrixLiteral(ones), Subtract(), colVector)

    val expectedOnesSubRowRight = makeMatFromRow(Seq(0, -1, -2))
    val expectedOnesSubColRight = makeMatFromCol(Seq(0, -1, -2))
    val expectedOnesSubRowLeft = makeMatFromRow(Seq(0, 1, 2))
    val expectedOnesSubColLeft = makeMatFromCol(Seq(0, 1, 2))

    assert(actualOnesSubRowOnRight.execute(hc).toBreezeMatrix() == expectedOnesSubRowRight.toBreezeMatrix())
    assert(actualOnesSubColOnRight.execute(hc).toBreezeMatrix() == expectedOnesSubColRight.toBreezeMatrix())
    assert(actualOnesSubRowOnLeft.execute(hc).toBreezeMatrix() == expectedOnesSubRowLeft.toBreezeMatrix())
    assert(actualOnesSubColOnLeft.execute(hc).toBreezeMatrix() == expectedOnesSubColLeft.toBreezeMatrix())


    // Division
    val actualOnesDivRowOnRight = mapOnRight(new BlockMatrixLiteral(ones), FloatingPointDivide(), rowVector)
    val actualOnesDivColOnRight = mapOnRight(new BlockMatrixLiteral(ones), FloatingPointDivide(), colVector)
    val actualOnesDivRowOnLeft  = mapOnLeft(new BlockMatrixLiteral(ones), FloatingPointDivide(), rowVector)
    val actualOnesDivColOnLeft  = mapOnLeft(new BlockMatrixLiteral(ones), FloatingPointDivide(), colVector)

    val expectedOnesDivRowRight = makeMatFromRow(Seq(1, 1.0 / 2.0, 1.0 / 3.0))
    val expectedOnesDivColRight = makeMatFromCol(Seq(1, 1.0 / 2.0, 1.0 / 3.0))
    val expectedOnesDivRowLeft = makeMatFromRow(Seq(1, 2, 3))
    val expectedOnesDivColLeft = makeMatFromCol(Seq(1, 2, 3))

    assert(actualOnesDivRowOnRight.execute(hc).toBreezeMatrix() == expectedOnesDivRowRight.toBreezeMatrix())
    assert(actualOnesDivColOnRight.execute(hc).toBreezeMatrix() == expectedOnesDivColRight.toBreezeMatrix())
    assert(actualOnesDivRowOnLeft.execute(hc).toBreezeMatrix() == expectedOnesDivRowLeft.toBreezeMatrix())
    assert(actualOnesDivColOnLeft.execute(hc).toBreezeMatrix() == expectedOnesDivColLeft.toBreezeMatrix())
  }
}
