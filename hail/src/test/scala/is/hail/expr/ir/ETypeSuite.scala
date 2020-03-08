package is.hail.expr.ir

import is.hail.expr.types.encoded._
import is.hail.rvd.AbstractRVDSpec
import is.hail.utils._
import org.json4s.jackson.Serialization
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class ETypeSuite extends TestNGSuite {

  @DataProvider(name="etypes")
  def etypes(): Array[Array[Any]] = {
    Array[EType](
      EInt32Required,
      EInt32Optional,
      EInt64Required,
      EFloat32Optional,
      EFloat32Required,
      EFloat64Optional,
      EFloat64Required,
      EBooleanOptional,
      EBinaryRequired,
      EBinaryOptional,
      EBinaryRequired,
      EArray(EInt32Required, required = false),
      EArray(EArray(EInt32Optional, required = true), required = true),
      EBaseStruct(FastIndexedSeq(), required = true),
      EBaseStruct(FastIndexedSeq(EField("x", EBinaryRequired, 0), EField("y", EFloat64Optional, 1)), required = true)
    ).map(t => Array(t: Any))
  }

  @Test def testDataProvider(): Unit = {
    etypes()
  }

  @Test(dataProvider="etypes")
  def testSerialization(etype: EType): Unit = {
    implicit val formats = AbstractRVDSpec.formats
    val s = Serialization.write(etype)
    assert(Serialization.read[EType](s) == etype)
  }
}
