package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder, SafeRow}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.utils._

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import org.testng.Assert._

class StagedBlockLinkedListSuite extends TestNGSuite {

  def compile[T: TypeInfo](emit: EmitRegion => Code[T]): (Region => T) = {
    val fb = EmitFunctionBuilder[Region, T]
    val mb = fb.apply_method
    mb.emit(emit(EmitRegion.default(mb)))
    val f = fb.result(None)()
    f(_)
  }

  def assertEvalsTo[T: TypeInfo](expected: T)(emit: EmitRegion => Code[T]) {
    val f = compile(emit)
    val result = Region.scoped(f)
    assertEquals(result, expected)
  }

  def assertEvalsToRV(expected: Any, typ: PType)(emit: EmitRegion => Code[Long]) {
    assert(typ.virtualType.typeCheck(expected))
    val f: Region => Long = compile(emit)
    val result = Region.scoped { region =>
      val off = f(region)
      SafeRow.read(typ, region, off)
    }
    assertEquals(result, expected)
  }

  def numBlocks(er: EmitRegion, bll: StagedBlockLinkedList): Code[Int] = {
    val count = er.mb.newField[Int]
    Code(
      count := 0,
      bll.foreachNode { b => count := count + 1 },
      count)
  }

  def sumElements(er: EmitRegion, bll: StagedBlockLinkedList): Code[Int] = {
    assert(bll.elemType.isOfType(PInt32()))
    val acc = er.mb.newField[Int]
    Code(
      acc := 0,
      bll.foreachElemAddress { e =>
        e.m.mux(Code._empty,
          acc := acc + er.region.loadInt(e.value))
      },
      acc)
  }

  def some(v: Code[_]): EmitTriplet =
    EmitTriplet(Code._empty, false, v)
  def none: EmitTriplet =
    EmitTriplet(Code._empty, true, Code._fatal("unreachable"))

  @Test def testInitialBlock() {
    assertEvalsTo(1) { er =>
      val bll = new StagedBlockLinkedList(PInt32(), er)
      Code(bll.init, numBlocks(er, bll))
    }
    assertEvalsTo(StagedBlockLinkedList.defaultBlockCap) { er =>
      val bll = new StagedBlockLinkedList(PInt32(), er)
      Code(bll.init, bll.capacity(bll.firstNode))
    }
  }

  @Test def testPushNewBlocks() {
    assertEvalsTo(3) { er =>
      val bll = new StagedBlockLinkedList(PInt32(), er)
      Code(
        bll.init,
        bll.pushNewBlockNode(20),
        bll.pushNewBlockNode(30),
        numBlocks(er, bll))
    }
    assertEvalsTo(55) { er =>
      val bll = new StagedBlockLinkedList(PInt32(), er)
      Code(
        bll.init,
        bll.pushNewBlockNode(55),
        bll.capacity(bll.lastNode))
    }
  }

  @Test def testPushInts() {
    def pushThen[T](er: EmitRegion, n: Int)(k: StagedBlockLinkedList => Code[T]): Code[T] = {
      val bll = new StagedBlockLinkedList(PInt32(), er)
      var setup: Code[Unit] = bll.initWithCapacity(8)
      for (i <- 1 to n) {
        setup = Code(setup, bll.push(
          if (i == 6)
            none
          else
            some(i)))
      }
      Code(setup, k(bll))
    }

    assertEvalsTo(5) { er => pushThen(er, n = 5)(_.totalCount) }
    assertEvalsTo(1) { er => pushThen(er, n = 5)(numBlocks(er, _)) }
    assertEvalsTo(2) { er => pushThen(er, n = 9)(numBlocks(er, _)) }
    assertEvalsTo(45 - 6) { er => pushThen(er, n = 9)(sumElements(er, _)) }
    assertEvalsToRV(IndexedSeq(1, 2, 3, 4, 5, null, 7), PArray(PInt32())) { er =>
      pushThen(er, n = 7)(_.toArray)
    }
  }

  @Test def testPushLotsOfLongs() {
    assertEvalsToRV((0L until 1000).toIndexedSeq, PArray(PInt64Optional)) { er =>
      val bll = new StagedBlockLinkedList(PInt64(), er)
      val i = er.mb.newField[Long]
      Code(bll.init,
        i := 0,
        Code.whileLoop(i < 1000,
          bll.push(some(i)),
          i := i + 1),
        bll.toArray)
    }
  }

  def allocString(er: EmitRegion, str: String): Code[Long] = {
    val off = er.mb.newField[Long]
    val setup = Code(
      off := PBinary.allocate(er.region, str.length()),
      er.region.storeInt(off, str.length()),
      er.region.storeBytes(PBinary.bytesOffset(off), const(str).invoke[Array[Byte]]("getBytes")))
    Code(setup, off)
  }

  @Test def testPushStrings() {
    assertEvalsToRV(IndexedSeq("hello", null, "world!"), PArray(PStringOptional)) { er =>
      val bll = new StagedBlockLinkedList(PString(), er)
      Code(bll.init,
        bll.push(some(allocString(er, "hello"))),
        bll.push(none),
        bll.push(some(allocString(er, "world!"))),
        bll.toArray)
    }
  }

  @Test def testAppend() {
    assertEvalsToRV(IndexedSeq(1, 2, null, 3, null, 4), PArray(PInt32Optional)) { er =>
      val bll1 = new StagedBlockLinkedList(PInt32(), er)
      val bll2 = new StagedBlockLinkedList(PInt32(), er)
      Code(
        bll1.init,
        bll2.init,
        Code(bll1.push(some(1)), bll1.push(some(2)), bll1.push(none)),
        Code(bll2.push(some(3)), bll2.push(none), bll2.push(some(4))),
        bll1.append(bll2),
        bll1.toArray)
    }
  }
}
