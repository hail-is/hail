package is.hail.types.physical.stypes.concrete

import is.hail.HailSuite
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.interfaces.SBaseStruct
import is.hail.types.virtual.{TInt32, TInt64, TStruct}
import is.hail.utils.FastSeq

import org.testng.annotations.Test

class SSubsetStructSuite extends HailSuite {

  @Test def testCastRename(): Unit = {
    val sparent = SType.canonical(
      TStruct(
        "x" -> TInt32,
        "y" -> TInt64,
        "z" -> TStruct("a" -> TInt32),
      )
    )

    val subset =
      new SSubsetStruct(sparent.asInstanceOf[SBaseStruct], FastSeq("z"))
        .castRename(TStruct("x" -> TStruct("b" -> TInt32)))
        .asInstanceOf[SSubsetStruct]

    val expected =
      new SSubsetStruct(
        SType.canonical(
          TStruct(
            "z" -> TInt32,
            "y" -> TInt64,
            "x" -> TStruct("b" -> TInt32),
          )
        ).asInstanceOf[SBaseStruct],
        FastSeq("x"),
      )

    assert(subset == expected)
    assert(subset.virtualType == expected.virtualType)
  }

}
