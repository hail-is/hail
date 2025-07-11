package is.hail.types.physical.stypes.concrete

import is.hail.HailSuite
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.interfaces.SBaseStruct
import is.hail.types.tcoerce
import is.hail.types.virtual.{TInt32, TInt64, TStruct}
import is.hail.utils.FastSeq

import org.scalatest
import org.testng.annotations.Test

class SStructViewSuite extends HailSuite {

  val xyz: SBaseStruct =
    tcoerce[SBaseStruct](
      SType.canonical(
        TStruct(
          "x" -> TInt32,
          "y" -> TInt64,
          "z" -> TStruct("a" -> TInt32),
        )
      )
    )

  @Test def testCastRename(): scalatest.Assertion = {
    val newtype = TStruct("x" -> TStruct("b" -> TInt32))

    val expected =
      new SStructView(
        parent = xyz,
        restrict = FastSeq("z"),
        rename = newtype,
      )

    assert(SStructView.subset(FastSeq("z"), xyz).castRename(newtype) == expected)
  }

  @Test def testSubsetRenameSubset(): scalatest.Assertion = {
    val subset =
      SStructView.subset(
        FastSeq("x"),
        SStructView.subset(FastSeq("x", "z"), xyz)
          .castRename(TStruct("y" -> TInt32, "x" -> TStruct("b" -> TInt32)))
          .asInstanceOf[SBaseStruct],
      )

    val expected =
      new SStructView(
        parent = xyz,
        restrict = FastSeq("z"),
        rename = TStruct("x" -> TStruct("b" -> TInt32)),
      )

    assert(subset == expected)
  }

  @Test def testAssertIsomorphism(): scalatest.Assertion =
    assertThrows[AssertionError] {
      SStructView.subset(FastSeq("x", "y"), xyz)
        .castRename(TStruct("x" -> TInt64, "x" -> TInt32))
    }
}
