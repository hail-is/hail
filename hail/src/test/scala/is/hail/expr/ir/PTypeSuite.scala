package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.expr.types.physical._
import is.hail.rvd.AbstractRVDSpec
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.json4s.jackson.Serialization
import org.testng.annotations.{DataProvider, Test}

class PTypeSuite extends HailSuite {

  @DataProvider(name="ptypes")
  def ptypes(): Array[Array[Any]] = {
    Array[PType](
      PInt32(true),
      PInt32(false),
      PInt64(true),
      PInt64(false),
      PFloat32(true),
      PFloat64(true),
      PBoolean(true),
      PCanonicalCall(true),
      PBinary(false),
      PCanonicalString(true),
      PCanonicalLocus(ReferenceGenome.GRCh37, false),
      PCanonicalArray(PInt32Required, true),
      PCanonicalSet(PInt32Required, false),
      PCanonicalDict(PInt32Required, PCanonicalString(true), true),
      PCanonicalInterval(PInt32Optional, false),
      PCanonicalTuple(FastIndexedSeq(PTupleField(1, PInt32Required), PTupleField(3, PCanonicalString(false))), true),
      PCanonicalStruct(FastIndexedSeq(PField("foo", PInt32Required, 0), PField("bar", PCanonicalString(false), 1)), true)
    ).map(t => Array(t: Any))
  }

  @Test def testPTypesDataProvider(): Unit = {
    ptypes()
  }

  @Test(dataProvider="ptypes")
  def testSerialization(ptype: PType): Unit = {
    implicit val formats = AbstractRVDSpec.formats
    val s = Serialization.write(ptype)
    assert(Serialization.read[PType](s) == ptype)
  }
}
