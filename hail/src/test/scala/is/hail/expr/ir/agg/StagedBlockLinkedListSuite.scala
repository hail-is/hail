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

  def assertBLLContents(elemType: PType, expected: IndexedSeq[Any])(
    emit: (StagedBlockLinkedList, EmitRegion) => Code[Unit]
  ) =
    assertEvalsToRV(expected, PArray(elemType)) { er =>
      val bll = new StagedBlockLinkedList(elemType, er.mb.fb)
      Code(bll.init(er.region), emit(bll, er), bll.toArray)
    }

  def numBlocks(bll: StagedBlockLinkedList): Code[Int] = {
    val count = bll.fb.newField[Int]
    Code(count := 0, bll.foreachNode { b => count := count + 1 }, count)
  }

  def sumElements(bll: StagedBlockLinkedList): Code[Int] = {
    assert(bll.elemType.isOfType(PInt32()))
    val acc = bll.fb.newField[Int]
    Code(
      acc := 0,
      bll.foreach { e =>
        e.m.mux(Code._empty, acc := acc + e.value)
      },
      acc)
  }

  def some(v: Code[_]): EmitTriplet =
    EmitTriplet(Code._empty, false, v)
  def none: EmitTriplet =
    EmitTriplet(Code._empty, true, Code._fatal("unreachable"))

  @Test def testInitialBlock() {
    assertEvalsTo(1) { case EmitRegion(mb, r) =>
      val bll = new StagedBlockLinkedList(PInt32(), mb.fb)
      Code(bll.init(r), numBlocks(bll))
    }
    assertEvalsTo(StagedBlockLinkedList.defaultBlockCap) { case EmitRegion(mb, r) =>
      val bll = new StagedBlockLinkedList(PInt32(), mb.fb)
      Code(bll.init(r), bll.capacity(bll.firstNode))
    }
  }

  @Test def testPushNewBlocks() {
    assertEvalsTo(3) { case EmitRegion(mb, r) =>
      val bll = new StagedBlockLinkedList(PInt32(), mb.fb)
      Code(
        bll.init(r),
        bll.pushNewBlockNode(r, 20),
        bll.pushNewBlockNode(r, 30),
        numBlocks(bll))
    }
    assertEvalsTo(55) { case EmitRegion(mb, r) =>
      val bll = new StagedBlockLinkedList(PInt32(), mb.fb)
      Code(
        bll.init(r),
        bll.pushNewBlockNode(r, 55),
        bll.capacity(bll.lastNode))
    }
  }

  @Test def testPushInts() {
    def pushThen[T](er: EmitRegion, n: Int)(k: StagedBlockLinkedList => Code[T]): Code[T] = {
      val bll = new StagedBlockLinkedList(PInt32(), er.mb.fb)
      var setup: Code[Unit] = bll.initWithCapacity(er.region, 8)
      for (i <- 1 to n) {
        setup = Code(setup, bll.push(er.region,
          if (i == 6)
            none
          else
            some(i)))
      }
      Code(setup, k(bll))
    }

    assertEvalsTo(5) { er => pushThen(er, n = 5)(_.totalCount) }
    assertEvalsTo(1) { er => pushThen(er, n = 5)(numBlocks) }
    assertEvalsTo(2) { er => pushThen(er, n = 9)(numBlocks) }
    assertEvalsTo(45 - 6) { er => pushThen(er, n = 9)(sumElements) }
    assertEvalsToRV(IndexedSeq(1, 2, 3, 4, 5, null, 7), PArray(PInt32())) { er =>
      pushThen(er, n = 7)(_.toArray)
    }
  }

  @Test def testPushLotsOfLongs() {
    assertBLLContents(PInt64(), (0L until 1000).toIndexedSeq) { case (bll, EmitRegion(mb, r)) =>
      val i = mb.newField[Long]
      Code(
        i := 0,
        Code.whileLoop(i < 1000,
          bll.push(r, some(i)),
          i := i + 1))
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
    assertBLLContents(PString(), IndexedSeq("hello", null, "world!")) { case (bll, er@EmitRegion(_, r)) =>
      Code(
        bll.push(r, some(allocString(er, "hello"))),
        bll.push(r, none),
        bll.push(r, some(allocString(er, "world!"))))
    }
  }

  @Test def testAppend() {
    assertEvalsToRV(IndexedSeq(1, 2, null, 3, null, 4), PArray(PInt32Optional)) { case er@EmitRegion(mb, r) =>
      val bll1 = new StagedBlockLinkedList(PInt32(), mb.fb)
      val bll2 = new StagedBlockLinkedList(PInt32(), mb.fb)
      Code(
        bll1.init(r),
        bll2.init(r),
        Code(bll1.push(r, some(1)), bll1.push(r, some(2)), bll1.push(r, none)),
        Code(bll2.push(r, some(3)), bll2.push(r, none), bll2.push(r, some(4))),
        bll1.append(r, bll2),
        bll1.toArray)
    }
  }

  // [1, null, 2]
  def buildTestArrayWithMissing(er: EmitRegion): Code[Long] = {
    val srvb = new StagedRegionValueBuilder(er, PArray(PInt32Optional))
    Code(
      srvb.start(length = 3, init = true),
      srvb.addInt(1), srvb.advance(),
      srvb.setMissing(), srvb.advance(),
      srvb.addInt(2), srvb.advance(),
      srvb.end())
  }

  @Test def testAppendShallow() {
    assertEvalsToRV(IndexedSeq(1, null, 2, 1, null, 2), PArray(PInt32Optional)) { case er@EmitRegion(mb, r) =>
      val bll = new StagedBlockLinkedList(PInt32(), mb.fb)
      val aoff = mb.newField[Long]
      Code(
        bll.init(r),
        aoff := buildTestArrayWithMissing(er),
        bll.appendShallow(r, PArray(PInt32Optional), aoff),
        bll.appendShallow(r, PArray(PInt32Optional), aoff),
        bll.toArray)
    }
  }
}
