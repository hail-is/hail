package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.expr.types.virtual.TFloat64
import is.hail.linalg.BlockMatrix
import org.testng.annotations.Test

class BlockMatrixIRSuite extends SparkSuite {

  val element = Ref("element", TFloat64(true))
  def createBroadcastValueOp(bm: BlockMatrixIR, op: BinaryOp, value: IR): BlockMatrixIR = {
    BlockMatrixBroadcastValue(bm, ApplyBinaryPrimOp(op, element, value))
  }

  def createBlockMatrixElemWiseOp(left: BlockMatrixIR, right: BlockMatrixIR,  op: BinaryOp):
  BlockMatrixElementWiseBinaryOp = {
    BlockMatrixElementWiseBinaryOp(left, right, ApplyBinaryPrimOp(op, element, element))
  }

  val ones: BlockMatrix = BlockMatrix.fill(hc, 5, 5, 1)
  val twos: BlockMatrix = BlockMatrix.fill(hc, 5, 5, 2)
  val threes: BlockMatrix = BlockMatrix.fill(hc, 5, 5, 3)
  val fours: BlockMatrix = BlockMatrix.fill(hc, 5, 5, 4)

  @Test def testBlockMatrixWriteRead() {
    def sampleMatrix: BlockMatrix = BlockMatrix.fill(hc, 5, 5, 1)

    val tempPath = tmpDir.createLocalTempFile()
    Interpret(BlockMatrixWrite(new BlockMatrixLiteral(sampleMatrix), tempPath, false, false, false))

    val actualMatrix = BlockMatrixRead(tempPath).execute(hc)
    assert(actualMatrix.toBreezeMatrix() == sampleMatrix.toBreezeMatrix())
  }

  @Test def testBlockMatrixAdd() {
    def a = BlockMatrix.fill(hc, 5, 5, 1)
    def b = BlockMatrix.fill(hc, 5, 5, 2)

    def expectedAPlusB = BlockMatrix.fill(hc, 5, 5, 3)
    val actualAPlusB = BlockMatrixAdd(new BlockMatrixLiteral(a), new BlockMatrixLiteral(b)).execute(hc)

    assert(actualAPlusB.toBreezeMatrix() == expectedAPlusB.toBreezeMatrix())
  }

  @Test def testBlockMatrixBroadcastValue_CompatibleDimensions() {
    val onesAddTwo = createBroadcastValueOp(new BlockMatrixLiteral(ones), Add(), F64(2))
    val threesSubTwo = createBroadcastValueOp(new BlockMatrixLiteral(threes), Subtract(), F64(2))
    val twosMulTwo = createBroadcastValueOp(new BlockMatrixLiteral(twos), Multiply(), F64(2))
    val foursDivTwo = createBroadcastValueOp(new BlockMatrixLiteral(fours), FloatingPointDivide(), F64(2))

    assert(onesAddTwo.execute(hc).toBreezeMatrix() == threes.toBreezeMatrix())
    assert(threesSubTwo.execute(hc).toBreezeMatrix() == ones.toBreezeMatrix())
    assert(twosMulTwo.execute(hc).toBreezeMatrix() == fours.toBreezeMatrix())
    assert(foursDivTwo.execute(hc).toBreezeMatrix() == twos.toBreezeMatrix())
  }
}
