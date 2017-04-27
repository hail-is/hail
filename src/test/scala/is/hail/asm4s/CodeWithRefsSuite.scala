package is.hail.asm4s

import is.hail.check.{Gen, Prop}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import is.hail.asm4s.CodeWithRefs._
import is.hail.asm4s.Code._

object CodeWithRefsSuite {
  def toISeq(a: Array[Int]): IndexedSeq[Int] = a
}

class CodeWithRefsSuite extends TestNGSuite {

  @Test def sum() {
    val m = for {
      a <- newVar[Array[Int]](Code.newArray[Int](3))
      _ <- a.update(0, 1)
      _ <- a.update(1, 2)
      _ <- a.update(2, 3)
      i <- newVar[Int](0)
      s <- newVar[Int](0)
      _ <- whileLoop(i < a.length(),
        Code(s := s + a(i), i := i + 1))
    } yield s: Code[Int]

    val f = m.run(new FunctionToIBuilder)

    assert(f() == 6)
  }
}
