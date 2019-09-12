package is.hail.asm4s

import is.hail.asm4s.joinpoint._

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class JoinPointSuite extends TestNGSuite {
  private def compile1[T: TypeInfo, U: TypeInfo](f: (MethodBuilder, Code[T]) => Code[U]): T => U = {
    val fb = FunctionBuilder.functionBuilder[T, U]
    val mb = fb.apply_method
    mb.emit(f(mb, mb.getArg[T](1).load))
    val asmFn = fb.result()()
    asmFn(_)
  }

  // op(...op(op(t0, 0), 1)..., (n-1))
  def rangeFoldCPS[X, T: ParameterPack](
    jb: JoinPointBuilder[X],
    n: Code[Int],
    t0: T,
    op: (T, Code[Int], T => Code[X]) => Code[X],
    ret: T => Code[X]
  ): Code[X] = {
    val loop = jb.joinPoint[(Code[Int], T)]
    loop.define { case (i, acc) =>
      (i < n).mux(
        op(acc, i, acc2 => loop((i + 1, acc2))),
        ret(acc))
    }
    loop((0, t0))
  }

  // n! = 1 * ... * n
  def facStaged[X](
    jb: JoinPointBuilder[X],
    n: Code[Int],
    ret: Code[Int] => Code[X]
  ): Code[X] =
    rangeFoldCPS[X, Code[Int]](jb, n,
      1,
      (prod, i, k) => k(prod * (i + 1)),
      ret)

  // f(n) = 0! + ... + (n-1)!
  def sumFacStaged[X](
    jb: JoinPointBuilder[X],
    n: Code[Int],
    ret: Code[Int] => Code[X]
  ): Code[X] =
    rangeFoldCPS[X, Code[Int]](jb, n,
      0,
      (sum, i, k) => facStaged(jb, i, fac => k(sum + fac)),
      ret)

  // parity(2n, even, odd) = even
  // parity(2n+1, even, odd) = odd
  def parityStaged[X] (
    jb: JoinPointBuilder[X],
    n: Code[Int],
    even: Code[X],
    odd: Code[X]
  ): Code[X] = {
    val isEven = jb.joinPoint[Code[Int]]
    val isOdd = jb.joinPoint[Code[Int]]
    isEven.define { i => (i ceq 0).mux(even, isOdd(i - 1)) }
    isOdd.define { i => (i ceq 0).mux(odd, isEven(i - 1)) }
    isEven(n)
  }

  def whileLoop(cond: Code[Boolean], code: Code[Unit]*): Code[Unit] =
    // NOTE: we just-so-happen to not need a methodbuilder because none of the
    //       join points take any arguments. if we did take arguments then this
    //       null methodbuilder would become a problem.
    new JoinPoint.CallCC[Unit](mb = null) {
      def apply[X](jb: JoinPointBuilder[X], ret: JoinPoint[Unit, X]): Code[X] = {
        val guard = jb.joinPoint[Unit]
        val body = jb.joinPoint[Unit]
        guard.define { _ => JoinPoint.mux(cond, body, ret) }
        body.define { _ => Code(Code(code: _*), guard(())) }
        guard(())
      }
    }

  @Test def testSimpleEarlyReturn() {
    val f = compile1[Int, Boolean] { (mb, n) =>
      new JoinPoint.CallCC[Code[Boolean]](mb) {
        def apply[X](jb: JoinPointBuilder[X], ret: JoinPoint[Code[Boolean], X]): Code[X] =
          Code(ret(true), ret(false))
      }
    }
    assert(f(0) == true)
  }

  @Test def testMux() {
    val f = compile1[Boolean, Int] { (mb, arg) =>
      new JoinPoint.CallCC[Code[Int]](mb) {
        def apply[X](jb: JoinPointBuilder[X], ret: JoinPoint[Code[Int], X]): Code[X] = {
          val j1 = jb.joinPoint[Code[Int]]
          val j2 = jb.joinPoint[Code[Int]]
          j1.define { n => ret(n + 5) }
          j2.define { n => ret(n * 5) }
          JoinPoint.mux(const(100), arg, j1, j2)
        }
      }
    }
    assert(f(true) == 105)
    assert(f(false) == 500)
  }

  @Test def testFac() {
    def fac(n: Int): Int = (1 to n).fold(1)(_ * _)
    val facS = compile1[Int, Int] { (mb, n) =>
      new JoinPoint.CallCC[Code[Int]](mb) {
        def apply[X](jb: JoinPointBuilder[X], ret: JoinPoint[Code[Int], X]): Code[X] =
          facStaged(jb, n, ret)
      }
    }
    for (i <- 0 to 12)
      assert(facS(i) == fac(i), s"compute: $i!")
  }

  @Test def testNestedFac() {
    def sumFac(n: Int): Int = (0 until n).map { i => (1 to i).fold(1)(_ * _) }.sum
    val sumFacS = compile1[Int, Int] { (mb, n) =>
      new JoinPoint.CallCC[Code[Int]](mb) {
        def apply[X](jb: JoinPointBuilder[X], ret: JoinPoint[Code[Int], X]): Code[X] =
          sumFacStaged(jb, n, ret)
      }
    }
    for (i <- 1 to 12)
      assert(sumFacS(i) == sumFac(i), s"compute: 0! + ... + ${i-1}!")
  }

  @Test def testNestedCallCC() {
    def sumFac(n: Int): Int = (0 until n).map { i => (1 to i).fold(1)(_ * _) }.sum
    val sumFacS = compile1[Int, Int] { (mb, n) =>
      new JoinPoint.CallCC[Code[Int]](mb) {
        def apply[X](jb: JoinPointBuilder[X], ret: JoinPoint[Code[Int], X]): Code[X] =
          rangeFoldCPS[X, Code[Int]](jb, n,
            0,
            (acc, i, k) => k(acc +
              new JoinPoint.CallCC[Code[Int]](mb) {
                def apply[X2](jb2: JoinPointBuilder[X2], ret2: JoinPoint[Code[Int], X2]): Code[X2] =
                  facStaged(jb2, i, ret2)
              }),
            ret)
      }
    }
    for (i <- 1 to 12)
      assert(sumFacS(i) == sumFac(i), s"compute: 0! + ... + ${i-1}!")
  }

  @Test def testSelectiveSum() {
    def ssum(n: Int): Int = (0 until n).filter(_ % 2 == 0).sum
    val ssumS = compile1[Int, Int] { (mb, n) =>
      new JoinPoint.CallCC[Code[Int]](mb) {
        def apply[X](jb: JoinPointBuilder[X], ret: JoinPoint[Code[Int], X]): Code[X] =
          rangeFoldCPS[X, Code[Int]](jb, n,
            0,
            (sum, i, k) => parityStaged(jb, i, k(sum + i), k(sum)),
            ret)
      }
    }
    for (i <- 1 to 12)
      assert(ssumS(i) == ssum(i), s"compute: 0 + 2 + ... + $i")
  }

  @Test def testEarlyReturnSum() {
    def sum(n: Int): Int = (0 until n.min(10)).sum
    val sumS = compile1[Int, Int] { (mb, n) =>
      new JoinPoint.CallCC[Code[Int]](mb) {
        def apply[X](jb: JoinPointBuilder[X], ret: JoinPoint[Code[Int], X]): Code[X] =
          rangeFoldCPS[X, Code[Int]](jb, n,
            0,
            (sum, i, k) => {
              (i < 10).mux(
                k(sum + i),
                ret(sum))
            },
            ret)
      }
    }
    for (i <- 1 to 50)
      assert(sumS(i) == sum(i), s"compute: 0 + ... + min($i, 10)")
  }

  @Test def testWhileLoop() {
    def sum(n: Int): Int = (0 until n.min(10)).sum
    val sumS = compile1[Int, Int] { (mb, n) =>
      val acc = mb.newField[Int]
      val i = mb.newField[Int]
      Code(acc := 0, i := 0,
        whileLoop((i < n && i < 10),
          acc := acc + i,
          i := i + 1),
        acc)
    }
    for (i <- 1 to 50)
      assert(sumS(i) == sum(i), s"compute: 0 + ... + min($i, 10)")
  }
}
