package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.linalg.BlockMatrix
import org.testng.annotations.Test

class BlockMatrixIRSuite extends SparkSuite {

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
}
