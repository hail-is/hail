package is.hail.types.physical

import is.hail.ParameterizedTest
import is.hail.TestUtils._
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.rvd.AbstractRVDSpec
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import org.apache.spark.sql.Row
import org.json4s.jackson.Serialization
import org.junit.jupiter.api.Test

class PTypeSuite {

  def ptypes() = ArraySeq[PType](
    PInt32(true),
    PInt32(false),
    PInt64(true),
    PInt64(false),
    PFloat32(true),
    PFloat64(true),
    PBoolean(true),
    PCanonicalCall(true),
    PCanonicalBinary(false),
    PCanonicalString(true),
    PCanonicalLocus(ReferenceGenome.GRCh37, false),
    PCanonicalArray(PInt32Required, true),
    PCanonicalSet(PInt32Required, false),
    PCanonicalDict(PInt32Required, PCanonicalString(true), true),
    PCanonicalInterval(PInt32Optional, false),
    PCanonicalTuple(
      FastSeq(PTupleField(1, PInt32Required), PTupleField(3, PCanonicalString(false))),
      true,
    ),
    PCanonicalStruct(
      FastSeq(PField("foo", PInt32Required, 0), PField("bar", PCanonicalString(false), 1)),
      true,
    ),
  )

  @Test def testPTypesDataProvider(): Unit = ptypes(): Unit

  @ParameterizedTest("ptypes")
  def testSerialization(ptype: PType): Unit = {
    implicit val formats = AbstractRVDSpec.formats
    val s = Serialization.write(ptype)
    assertEq(Serialization.read[PType](s), ptype)
  }

  @Test def testLiteralPType(): Unit = {
    assertEq(PType.literalPType(TInt32, 5), PInt32(true))
    assertEq(PType.literalPType(TInt32, null), PInt32())

    assertEq(PType.literalPType(TArray(TInt32), null), PCanonicalArray(PInt32(true)))
    assertEq(PType.literalPType(TArray(TInt32), FastSeq(1, null)), PCanonicalArray(PInt32(), true))
    assertEq(PType.literalPType(TArray(TInt32), FastSeq(1, 5)), PCanonicalArray(PInt32(true), true))

    assertEq(
      PType.literalPType(
        TInterval(TInt32),
        Interval(5, null, false, true),
      ),
      PCanonicalInterval(PInt32(), true),
    )

    val p = TStruct("a" -> TInt32, "b" -> TInt32)
    val d = TDict(p, p)
    assertEq(
      PType.literalPType(d, Map(Row(3, null) -> Row(null, 3))),
      PCanonicalDict(
        PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32()),
        PCanonicalStruct(true, "a" -> PInt32(), "b" -> PInt32(true)),
        true,
      ),
    )
  }
}
