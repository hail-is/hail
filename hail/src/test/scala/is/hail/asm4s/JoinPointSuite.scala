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
  def rangeFoldCPS[T: ParameterPack](
    mb: MethodBuilder,
    jb: JoinPointBuilder,
    n: Code[Int],
    t0: T,
    op: (T, Code[Int], T => Code[Ctrl]) => Code[Ctrl],
    ret: T => Code[Ctrl]
  ): Code[Ctrl] = {
    val loop = jb.joinPoint[(Code[Int], T)](mb)
    loop.define { case (i, acc) =>
      (i < n).mux(
        op(acc, i, acc2 => loop((i + 1, acc2))),
        ret(acc))
    }
    loop((0, t0))
  }

  // n! = 1 * ... * n
  def facStaged(
    mb: MethodBuilder,
    jb: JoinPointBuilder,
    n: Code[Int],
    ret: Code[Int] => Code[Ctrl]
  ): Code[Ctrl] =
    rangeFoldCPS[Code[Int]](mb, jb, n,
      1,
      (prod, i, k) => k(prod * (i + 1)),
      ret)

  // f(n) = 0! + ... + (n-1)!
  def sumFacStaged(
    mb: MethodBuilder,
    jb: JoinPointBuilder,
    n: Code[Int],
    ret: Code[Int] => Code[Ctrl]
  ): Code[Ctrl] =
    rangeFoldCPS[Code[Int]](mb, jb, n,
      0,
      (sum, i, k) => facStaged(mb, jb, i, fac => k(sum + fac)),
      ret)

  // parity(2n, even, odd) = even
  // parity(2n+1, even, odd) = odd
  def parityStaged (
    mb: MethodBuilder,
    jb: JoinPointBuilder,
    n: Code[Int],
    even: Code[Ctrl],
    odd: Code[Ctrl]
  ): Code[Ctrl] = {
    val isEven = jb.joinPoint[Code[Int]](mb)
    val isOdd = jb.joinPoint[Code[Int]](mb)
    isEven.define { i => (i ceq 0).mux(even, isOdd(i - 1)) }
    isOdd.define { i => (i ceq 0).mux(odd, isEven(i - 1)) }
    isEven(n)
  }

  @Test def testSimpleEarlyReturn() {
    val f = compile1[Int, Boolean] { (mb, n) =>
      JoinPoint.CallCC[Code[Boolean]] { (jb, ret) =>
        Code.concat(ret(true), ret(false))
      }
    }
    assert(f(0) == true)
  }

  @Test def testMux() {
    val f = compile1[Boolean, Int] { (mb, arg) =>
      JoinPoint.CallCC[Code[Int]] { (jb, ret) =>
        val j1 = jb.joinPoint[Code[Int]](mb)
        val j2 = jb.joinPoint[Code[Int]](mb)
        j1.define { n => ret(n + 5) }
        j2.define { n => ret(n * 5) }
        JoinPoint.mux(100: Code[Int], arg, j1, j2)
      }
    }
    assert(f(true) == 105)
    assert(f(false) == 500)
  }

  @Test def testFac() {
    def fac(n: Int): Int = (1 to n).fold(1)(_ * _)
    val facS = compile1[Int, Int] { (mb, n) =>
      JoinPoint.CallCC[Code[Int]](
        facStaged(mb, _, n, _)
      )
    }
    for (i <- 0 to 12)
      assert(facS(i) == fac(i), s"compute: $i!")
  }

  @Test def testNestedFac() {
    def sumFac(n: Int): Int = (0 until n).map { i => (1 to i).fold(1)(_ * _) }.sum
    val sumFacS = compile1[Int, Int] { (mb, n) =>
      JoinPoint.CallCC[Code[Int]](
        sumFacStaged(mb, _, n, _)
      )
    }
    for (i <- 1 to 12)
      assert(sumFacS(i) == sumFac(i), s"compute: 0! + ... + ${i-1}!")
  }

  @Test def testNestedCallCC() {
    def sumFac(n: Int): Int = (0 until n).map { i => (1 to i).fold(1)(_ * _) }.sum
    val sumFacS = compile1[Int, Int] { (mb, n) =>
      JoinPoint.CallCC[Code[Int]] { (jb, ret) =>
        rangeFoldCPS[Code[Int]](mb, jb, n,
          0,
          (acc, i, k) => k(acc +
            JoinPoint.CallCC[Code[Int]](
              facStaged(mb, _, i, _)
            )),
          ret)
      }
    }
    for (i <- 1 to 12)
      assert(sumFacS(i) == sumFac(i), s"compute: 0! + ... + ${i-1}!")
  }

  @Test def testSelectiveSum() {
    def ssum(n: Int): Int = (0 until n).filter(_ % 2 == 0).sum
    val ssumS = compile1[Int, Int] { (mb, n) =>
      JoinPoint.CallCC[Code[Int]] { (jb, ret) =>
        rangeFoldCPS[Code[Int]](mb, jb, n,
          0,
          (sum, i, k) => parityStaged(mb, jb, i, k(sum + i), k(sum)),
          ret)
      }
    }
    for (i <- 1 to 12)
      assert(ssumS(i) == ssum(i), s"compute: 0 + 2 + ... + $i")
  }

  @Test def testEarlyReturnSum() {
    def sum(n: Int): Int = (0 until n.min(10)).sum
    val sumS = compile1[Int, Int] { (mb, n) =>
      JoinPoint.CallCC[Code[Int]] { (jb, ret) =>
        rangeFoldCPS[Code[Int]](mb, jb, n,
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

  @Test def testBadCallCCEscape() {
    intercept[JoinPoint.EmitLongJumpError] {
      compile1[Int, Int] { (mb, n) =>
        JoinPoint.CallCC[Code[Int]] { (jb1, ret1) =>
          ret1(const(1) + JoinPoint.CallCC[Code[Int]] { (jb2, ret2) =>
            ret1(const(2))
          })
        }
      }
    }
  }

  @Test def testDuplicateCallCC() {
    val f = compile1[Int, Int] { (mb, arg) =>
      val num: Code[Int] =
        JoinPoint.CallCC[Code[Int]] { (jb, ret) =>
          val j = jb.joinPoint()
          j.define { _ => ret(arg) }
          j(())
        }
      (num + num)
    }
    for (i <- 1 to 50)
      assert(f(i) == i * 2)
  }
}
