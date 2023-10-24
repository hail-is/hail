package is.hail

import java.io.File
import java.lang.reflect.Modifier
import java.net.URI

import breeze.linalg.{DenseMatrix, DenseVector}
import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import org.testng.annotations.{DataProvider, Test}

import java.util.Base64

class TestUtilsSuite extends HailSuite {

  @Test def matrixEqualityTest() {
    val M = DenseMatrix((1d, 0d), (0d, 1d))
    val M1 = DenseMatrix((1d, 0d), (0d, 1.0001d))
    val V = DenseVector(0d, 1d)
    val V1 = DenseVector(0d, 0.5d)

    TestUtils.assertMatrixEqualityDouble(M, DenseMatrix.eye(2))
    TestUtils.assertMatrixEqualityDouble(M, M1, 0.001)
    TestUtils.assertVectorEqualityDouble(V, 2d * V1)

    intercept[Exception](TestUtils.assertVectorEqualityDouble(V, V1))
    intercept[Exception](TestUtils.assertMatrixEqualityDouble(M, M1))
  }

  @Test def constantVectorTest() {
    assert(TestUtils.isConstant(DenseVector()))
    assert(TestUtils.isConstant(DenseVector(0)))
    assert(TestUtils.isConstant(DenseVector(0, 0)))
    assert(TestUtils.isConstant(DenseVector(0, 0, 0)))
    assert(!TestUtils.isConstant(DenseVector(0, 1)))
    assert(!TestUtils.isConstant(DenseVector(0, 0, 1)))
  }

  @Test def removeConstantColsTest(): Unit = {
    val M = DenseMatrix((0, 0, 1, 1, 0),
                        (0, 1, 0, 1, 1))

    val M1 = DenseMatrix((0, 1, 0),
                         (1, 0, 1))

    assert(TestUtils.removeConstantCols(M) == M1)
  }

  @Test def decoderBugTest(): Unit = {
    val encodedB64 = scala.io.Source.fromFile("/Users/dgoldste/hail/foo2").mkString.trim
    val encodedValue = Base64.getDecoder.decode(encodedB64)

    val typ = TDict(TString, TInt32)
    val codec = TypedCodecSpec(
      EType.fromPythonTypeEncoding(typ),
      typ,
      BufferSpec.unblockedUncompressed
    )

    try {
      for (i <- 1 to 10) {
        ExecuteContext.scoped() { ctx =>
          ctx.r.getPool().scopedRegion { r =>
            val (pt, addr) = codec.decodeArrays(ctx, typ, Array(encodedValue), ctx.r)
            SafeRow.read(pt, addr)
          }
        }
      }
    } catch {
      case t: Throwable =>
        val typ = codec.encodedType
        log.info(s"The type of the bad stuff: ${codec.encodedType}")
        log.info(s"The bad encoded stuff: $encodedB64")
    }
  }
}
