package is.hail.expr.ir.agg

import scala.collection.generic.Growable
import is.hail.annotations.{Region, SafeRow, ScalaToRegionValue, StagedRegionValueBuilder}
import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitFunctionBuilder, EmitRegion, EmitCode, PCode}
import is.hail.expr.types.physical._
import is.hail.utils._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import org.testng.Assert._

class StagedBlockLinkedListSuite extends TestNGSuite {

  class BlockLinkedList[E](region: Region, val elemPType: PType, initImmediately: Boolean = true)
      extends Growable[E] {
    val arrayPType = PArray(elemPType)

    private val initF: Region => Long = {
      val fb = EmitFunctionBuilder[Region, Long]("init")
      val cb = fb.ecb
      val sbll = new StagedBlockLinkedList(elemPType, cb)

      val ptr = fb.genFieldThisRef[Long]()
      val r = fb.getArg[Region](1)
      fb.emit(Code(
        ptr := r.allocate(sbll.storageType.alignment, sbll.storageType.byteSize),
        sbll.init(r),
        sbll.store(ptr),
        ptr))

      fb.result()()(_)
    }

    private val pushF: (Region, Long, E) => Unit = {
      val fb = EmitFunctionBuilder[Region, Long, Long, Unit]("push")
      val cb = fb.ecb
      val sbll = new StagedBlockLinkedList(elemPType, cb)

      val r = fb.getArg[Region](1)
      val ptr = fb.getArg[Long](2)
      val eltOff = fb.getArg[Long](3)
      fb.emit(Code(
        sbll.load(ptr),
        sbll.push(r, EmitCode(Code._empty,
          eltOff.get.ceq(0),
          PCode(elemPType, Region.getIRIntermediate(elemPType)(eltOff)))),
        sbll.store(ptr)))

      val f = fb.result()()
      ({ (r, ptr, elt) =>
        f(r, ptr, if(elt == null) 0L else ScalaToRegionValue(r, elemPType, elt))
      })
    }

    private val appendF: (Region, Long, BlockLinkedList[E]) => Unit = {
      val fb = EmitFunctionBuilder[Region, Long, Long, Unit]("append")
      val cb = fb.ecb
      val sbll1 = new StagedBlockLinkedList(elemPType, cb)
      val sbll2 = new StagedBlockLinkedList(elemPType, cb)

      val r = fb.getArg[Region](1)
      val ptr1 = fb.getArg[Long](2)
      val ptr2 = fb.getArg[Long](3)
      fb.emit(Code(
        sbll1.load(ptr1),
        sbll2.load(ptr2),
        sbll1.append(r, sbll2),
        sbll1.store(ptr1)))

      val f = fb.result()()
      ({ (r, ptr, other) =>
        assert(other.elemPType.required == elemPType.required)
        f(r, ptr, other.ptr)
      })
    }

    private val materializeF: (Region, Long) => IndexedSeq[E] = {
      val fb = EmitFunctionBuilder[Region, Long, Long]("materialize")
      val cb = fb.ecb
      val sbll = new StagedBlockLinkedList(elemPType, cb)

      val rArg = fb.getArg[Region](1).load
      val ptr = fb.getArg[Long](2).load
      val rField = fb.genFieldThisRef[Region]()
      val srvb = new StagedRegionValueBuilder(EmitRegion(fb.apply_method, rField), arrayPType)
      fb.emit(Code(
        rField := rArg,
        sbll.load(ptr),
        sbll.writeToSRVB(srvb),
        srvb.end()))

      val f = fb.result()()
      ({ (r, ptr) =>
        SafeRow.read(arrayPType, f(r, ptr))
          .asInstanceOf[IndexedSeq[E]]
      })
    }

    private val initWithDeepCopyF: (Region, BlockLinkedList[E]) => Long = {
      val fb = EmitFunctionBuilder[Region, Long, Long]("init_with_copy")
      val cb = fb.ecb
      val sbll2 = new StagedBlockLinkedList(elemPType, cb)
      val sbll1 = new StagedBlockLinkedList(elemPType, cb)
      val dstPtr = fb.genFieldThisRef[Long]()
      val r = fb.getArg[Region](1)
      val srcPtr = fb.getArg[Long](2)
      fb.emit(Code(
        dstPtr := r.allocate(sbll1.storageType.alignment, sbll1.storageType.byteSize),
        sbll2.load(srcPtr),
        sbll1.initWithDeepCopy(r, sbll2),
        sbll1.store(dstPtr),
        dstPtr))

      val f = fb.result()()
      ({ (r, other) => f(r, other.ptr) })
     }

    private var ptr = 0L

    def clear(): Unit = { ptr = initF(region) }
    def +=(e: E): this.type = { pushF(region, ptr, e) ; this }
    def ++=(other: BlockLinkedList[E]): this.type = { appendF(region, ptr, other) ; this }
    def toIndexedSeq: IndexedSeq[E] = materializeF(region, ptr)

    if (initImmediately) clear()

    def copy(): BlockLinkedList[E] = {
      val b = new BlockLinkedList[E](region, elemPType, initImmediately = false)
      b.ptr = b.initWithDeepCopyF(region, this)
      b
    }
  }

  @Test def testPushIntsRequired() {
    Region.scoped { region =>
      val b = new BlockLinkedList[Int](region, PInt32Required)
      for (i <- 1 to 100) b += i
      assertEquals(b.toIndexedSeq, IndexedSeq.tabulate(100)(_ + 1))
    }
  }

  @Test def testPushStrsMissing() {
    Region.scoped { region =>
      val a = new ArrayBuilder[String]()
      val b = new BlockLinkedList[String](region, PString())
      for (i <- 1 to 100) {
        val elt = if(i%3 == 0) null else i.toString()
        a += elt
        b += elt
      }
      assertEquals(b.toIndexedSeq, a.result().toIndexedSeq)
    }
  }

  @Test def testAppendAnother() {
    Region.scoped { region =>
      val b1 = new BlockLinkedList[String](region, PString())
      val b2 = new BlockLinkedList[String](region, PString())
      b1 += "{"
      b2 ++= Seq("foo", "bar")
      b1 ++= b2
      b1 ++= b2
      b1 += "}"
      assertEquals(b1.toIndexedSeq, "{ foo bar foo bar }".split(" ").toIndexedSeq)
    }
  }

  @Test def testDeepCopy() {
    Region.scoped { region =>
      val b1 = new BlockLinkedList[Double](region, PFloat64())
      b1 ++= Seq(1.0, 2.0, 3.0)
      val b2 = b1.copy()
      b1 += 4.0
      b2 += 5.0
      assertEquals(b1.toIndexedSeq, IndexedSeq(1.0, 2.0, 3.0, 4.0))
      assertEquals(b2.toIndexedSeq, IndexedSeq(1.0, 2.0, 3.0, 5.0))
    }
  }
}
