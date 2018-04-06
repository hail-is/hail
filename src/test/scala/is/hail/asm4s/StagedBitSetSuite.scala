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
    val bs = new LocalBitSet(fb)
    val x = bs.newBit()
    fb.emit(f(x))
    fb.result()()()
  }

  def withTwoBits(f : (SettableBit, SettableBit) => Code[Boolean]): Boolean = {
    val fb = functionBuilder[Boolean]
    val bs = new LocalBitSet(fb)
    val x = bs.newBit()
    val y = bs.newBit()
    fb.emit(f(x, y))
    fb.result()()()
  }

  def withNBits(n: Int)(f: Array[SettableBit] => Code[Boolean]): Boolean = {
    val fb = functionBuilder[Boolean]
    val bs = new LocalBitSet(fb)
    val x = Array.tabulate(n)(i => bs.newBit())
    fb.emit(f(x))
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
    assert(withTwoBits((x, y) =>
      Code(x := true, y := false, y := x, y)))
  }

  @Test
  def testSetBitFromBit3() {
    assert(withTwoBits((x, y) =>
      Code(x := true, y := false, y := x, x)))
  }

  @Test
  def moreThan64Bits() {
    for (i <- 0 until 66) {
      val b = withNBits(66) { bits =>
        Code(
          Code(bits.zipWithIndex.map { case (b,i) => b := const(i % 3 == 0) }:_*),
          bits(i))
      }
      assert(b == (i % 3 == 0), s"at position $i, $b was not ${(i % 3 == 0)}")
    }
  }

  @Test
  def moreThan64Bits2() {
    def gadget(ret: Array[SettableBit] => Code[Boolean]) = withNBits(66) { bits =>
      val b63 = bits(63)
      val b64 = bits(64)
      val b65 = bits(65)
      Code(
        b65 := false,
        b63 := true,
        b64 := false,
        b65 := true,
        ret(bits))
    }

    assert(!gadget(_(64)))
    assert(gadget(_(65)))
    assert(gadget(_(63)))
    assert(gadget(!_(64)))
    assert(!gadget(!_(65)))
    assert(!gadget(!_(63)))
    assert(!gadget(bs => bs(63) && bs(64)))
    assert(gadget(bs => bs(63) && !bs(64)))
  }
}
