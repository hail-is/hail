package is.hail.asm4s

import is.hail.asm4s._
import is.hail.asm4s.Code._
import is.hail.check.{Gen, Prop}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import is.hail.asm4s.FunctionBuilder._

class StagedBitSetSuite extends TestNGSuite {
  def withOneBit(f : SettableBit => Code[Boolean]): Boolean = {
    val fb = functionBuilder[Boolean]
    val bs = new StagedBitSet(fb)
    val x = bs.newBit()
    fb.emit(f(x))
    fb.result()()()
  }

  def withTwoBits(f : (SettableBit, SettableBit) => Code[Boolean]): Boolean = {
    val fb = functionBuilder[Boolean]
    val bs = new StagedBitSet(fb)
    val x = bs.newBit()
    val y = bs.newBit()
    fb.emit(f(x, y))
    fb.result()()()
  }

  @Test
  def testSetOneBit() {
    assert(withOneBit(x => Code(x := true, x)))
  }

  @Test
  def testSetTwoBits1() {
    assert(withTwoBits((x, y) =>
      Code(x := true, y := false, x)))
  }

  @Test
  def testSetTwoBits2() {
    assert(!withTwoBits((x, y) =>
      Code(x := true, y := false, y)))
  }

  @Test
  def testSetTwoBits3() {
    assert(withTwoBits((x, y) =>
      Code(x := true, y := true, x && y)))
  }

  @Test
  def testSetBitTwice1() {
    assert(!withOneBit(x =>
      Code(x := true, x := false, x)))
  }

  @Test
  def testSetBitThrice() {
    assert(withOneBit(x =>
      Code(x := true, x := false, x := true, x)))
  }

  @Test
  def testSetBitFromBit1() {
    assert(!withTwoBits((x, y) =>
      Code(x := true, y := false, x := y, x)))
  }

  @Test
  def testSetBitFromBit2() {
    assert(!withTwoBits((x, y) =>
      Code(x := true, y := false, y := x, y)))
  }

  @Test
  def testSetBitFromBit3() {
    assert(withTwoBits((x, y) =>
      Code(x := true, y := false, y := x, x)))
  }
}
