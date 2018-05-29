package is.hail.expr.ir

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class RewriteSuite extends TestNGSuite {

  @Test def testStructRewriting() {

    def check(pre: IR, post: IR): Unit = {
      assert(Optimize(pre) == post)
    }

    check(
      InsertFields(
        MakeStruct(
          Seq(
            ("a", I32(2)),
            ("b", F64(1.5))
          )),
        Seq(
          ("c", I32(5)),
          ("d", I32(6))
        )),
      MakeStruct(
        Seq(
          ("a", I32(2)),
          ("b", F64(1.5)),
          ("c", I32(5)),
          ("d", I32(6)))))

    check(
      InsertFields(
        MakeStruct(
          Seq(
            ("a", I32(2)),
            ("b", F64(1.5))
          )),
        Seq(
          ("c", I32(5)),
          ("a", I32(6)))
      ),
      MakeStruct(
        Seq(
          ("a", I32(6)),
          ("b", F64(1.5)),
          ("c", I32(5)))))

    check(
      GetField(
        MakeStruct(
          Seq(
            "a" -> I32(1),
            "b" -> I32(5)
          )
        ),
        "a"
      ),
      I32(1)
    )

    check(
      GetField(
        InsertFields(
          MakeStruct(
            Seq(
              "a" -> I32(1),
              "b" -> I32(5)
            )
          ),
          Seq(
            "c" -> F64(5.5),
            "d" -> F64(10.1)
          )),
        "a"
      ),
      I32(1)
    )

    check(
      GetField(
        InsertFields(
          MakeStruct(
            Seq(
              "a" -> I32(1),
              "b" -> I32(5)
            )
          ),
          Seq(
            "c" -> F64(5.5),
            "d" -> F64(10.1)
          )),
        "d"
      ),
      F64(10.1)
    )
  }
}