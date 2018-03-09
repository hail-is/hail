package is.hail.expr.ir

import java.io.PrintWriter

import is.hail.annotations._
import ScalaToRegionValue._
import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.ir._
import is.hail.expr.types._
import org.testng.annotations.Test
import is.hail.expr.{EvalContext, Parser}
import org.apache.spark.sql.Row

class FunctionSuite {

  val ec = EvalContext()
  val region = Region()

  def fromHailString(hql: String): IR = Parser.parseToAST(hql, ec).toIR().get

  def toF[R: TypeInfo](ir: IR): AsmFunction1[Region, R] = {
    Infer(ir)
    val fb = FunctionBuilder.functionBuilder[Region, R]
    Emit(ir, fb)
    fb.result(Some(new PrintWriter(System.out)))()
  }

  @Test def testRange() {
    val ir = fromHailString("range(0, 5, 2)")
    val f = toF[Long](ir)
    val t = coerce[TArray](ir.typ)
    val off = f(region)
    assert(t.loadLength(region, off) == 3)
    Array(0, 1, 2).foreach { i =>
      val actual = region.loadInt(t.loadElement(region, off, i))
      assert(2 * i == actual, s"expected $i but got $actual")
    }
  }

  @Test def testRange2() {
    val ir = fromHailString("range(0, 5, 1)")
    val f = toF[Long](ir)
    val t = coerce[TArray](ir.typ)
    val off = f(region)
    assert(t.loadLength(region, off) == 5)
    Array(0, 1, 2, 3, 4).foreach { i =>
      val actual = region.loadInt(t.loadElement(region, off, i))
      assert(i == actual, s"expected $i but got $actual")
    }
  }

  @Test def testFilter() {
    region.clear()
    val ir = fromHailString("range(0, 10, 1).filter(g => g < 3)")
    val f = toF[Long](ir)
    val t = coerce[TArray](ir.typ)
    val off = f(region)
    assert(t.loadLength(region, off) == 3)
    Array(0, 1, 2).foreach { i =>
      val actual = region.loadInt(t.loadElement(region, off, i))
      assert(i == actual, s"expected $i but got $actual")
    }
  }



}
