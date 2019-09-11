package is.hail.asm4s

import is.hail.asm4s.joinpoint._

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class JoinPointSuite extends TestNGSuite {

  def fac(mb: MethodBuilder, n: Code[Int]): Code[Int] =
    new JoinPoint.CallCC[Code[Int]](mb) {
      def apply[X](jb: JoinPointBuilder[X], ret: JoinPoint[Code[Int], X]) = {
        val loop = jb.joinPoint[(Code[Int], Code[Int])]
        loop.define { case (i, a) =>
          (i <= n).mux(
            loop((i + 1, a * i)),
            ret(a))
        }
        loop((1, 1))
      }
    }

  def fac(n: Int): Int =
    (1 to n).fold(1)(_ * _)

  @Test def testFac() {
    val f = {
      val fb = FunctionBuilder.functionBuilder[Int, Int]
      val mb = fb.apply_method
      mb.emit(fac(mb, mb.getArg[Int](1).load))
      fb.result()()
    }
    for (i <- 0 to 12)
      assert(f(i) == fac(i), s"fac($i)")
  }

}
