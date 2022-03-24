package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.utils.FastSeq
import org.testng.annotations.Test

class IRTraversalSuite extends HailSuite {

 @Test def testTopDownTraversal(): Unit = {
   val ir0 = MakeTuple.ordered(FastSeq(MakeArray(I32(1)), F64(0d)))

   val ab = new AnyRefArrayBuilder[BaseIR]()

   VisitIR.topDown(ir0) { x =>
     ab.add(x)
   }

   assert(ab.result().map(_.getClass.getSimpleName).toSeq == Seq("MakeTuple", "MakeArray", "F64", "I32"))
 }

  @Test def testBottomUpTraversal(): Unit = {
    val ir0 = MakeTuple.ordered(FastSeq(MakeArray(I32(1)), F64(0d)))

    val ab = new AnyRefArrayBuilder[BaseIR]()

    VisitIR.bottomUp(ir0) { x =>
      ab.add(x)
    }

    assert(ab.result().map(_.getClass.getSimpleName).toSeq == Seq("I32", "MakeArray", "F64", "MakeTuple"))
  }
}
