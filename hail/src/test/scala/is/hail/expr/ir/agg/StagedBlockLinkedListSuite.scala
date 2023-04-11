package is.hail.expr.ir.agg

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeRow, ScalaToRegionValue}
import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitCode, EmitFunctionBuilder}
import is.hail.types.physical._
import is.hail.utils._
import org.testng.Assert._
import org.testng.annotations.Test

import scala.collection.generic.Growable

class StagedBlockLinkedListSuite extends HailSuite {

  class BlockLinkedList[E](region: Region, val elemPType: PType, initImmediately: Boolean = true)
      extends Growable[E] {
    val arrayPType = PCanonicalArray(elemPType)

    private val initF: Region => Long = {
      val fb = EmitFunctionBuilder[Region, Long](ctx, "init")
      val cb = fb.ecb
      val sbll = new StagedBlockLinkedList(elemPType, cb)

      val ptr = fb.genFieldThisRef[Long]()
      val r = fb.getCodeParam[Region](1)
      fb.emitWithBuilder[Long] { cb =>
        cb.assign(ptr, r.allocate(sbll.storageType.alignment, sbll.storageType.byteSize))
        sbll.init(cb, r)
        sbll.store(cb, ptr)
        ptr
      }

      fb.result(ctx)(theHailClassLoader)(_)
    }

    private val pushF: (Region, Long, E) => Unit = {
      val fb = EmitFunctionBuilder[Region, Long, Long, Unit](ctx, "push")
      val cb = fb.ecb
      val sbll = new StagedBlockLinkedList(elemPType, cb)

      val r = fb.getCodeParam[Region](1)
      val ptr = fb.getCodeParam[Long](2)
      val eltOff = fb.getCodeParam[Long](3)
      fb.emitWithBuilder[Unit] { cb =>

        sbll.load(cb, ptr)
        sbll.push(cb, r, EmitCode(Code._empty,
          eltOff.get.ceq(0L),
          elemPType.loadCheapSCode(cb, eltOff)))
        sbll.store(cb, ptr)
        Code._empty
      }

      val f = fb.result(ctx)(theHailClassLoader)
      ({ (r, ptr, elt) =>
        f(r, ptr, if(elt == null) 0L else ScalaToRegionValue(ctx.stateManager, r, elemPType, elt))
      })
    }

    private val appendF: (Region, Long, BlockLinkedList[E]) => Unit = {
      val fb = EmitFunctionBuilder[Region, Long, Long, Unit](ctx, "append")
      val cb = fb.ecb
      val sbll1 = new StagedBlockLinkedList(elemPType, cb)
      val sbll2 = new StagedBlockLinkedList(elemPType, cb)

      val r = fb.getCodeParam[Region](1)
      val ptr1 = fb.getCodeParam[Long](2)
      val ptr2 = fb.getCodeParam[Long](3)
      fb.emitWithBuilder { cb =>
        sbll1.load(cb, ptr1)
        sbll2.load(cb, ptr2)
        sbll1.append(cb, r, sbll2)
        sbll1.store(cb, ptr1)
        Code._empty
      }

      val f = fb.result(ctx)(theHailClassLoader)
      ({ (r, ptr, other) =>
        assert(other.elemPType.required == elemPType.required)
        f(r, ptr, other.ptr)
      })
    }

    private val materializeF: (Region, Long) => IndexedSeq[E] = {
      val fb = EmitFunctionBuilder[Region, Long, Long](ctx, "materialize")
      val cb = fb.ecb
      val sbll = new StagedBlockLinkedList(elemPType, cb)

      val rArg = fb.getCodeParam[Region](1)
      val ptr = fb.getCodeParam[Long](2)
      val rField = fb.genFieldThisRef[Region]()
      fb.emitWithBuilder { cb =>
        cb.assign(rField, rArg)
        sbll.load(cb, ptr)
        sbll.resultArray(cb, rArg, arrayPType).a
      }

      val f = fb.result(ctx)(theHailClassLoader)
      ({ (r, ptr) =>
        SafeRow.read(arrayPType, f(r, ptr))
          .asInstanceOf[IndexedSeq[E]]
      })
    }

    private val initWithDeepCopyF: (Region, BlockLinkedList[E]) => Long = {
      val fb = EmitFunctionBuilder[Region, Long, Long](ctx, "init_with_copy")
      val cb = fb.ecb
      val sbll2 = new StagedBlockLinkedList(elemPType, cb)
      val sbll1 = new StagedBlockLinkedList(elemPType, cb)
      val dstPtr = fb.genFieldThisRef[Long]()
      val r = fb.getCodeParam[Region](1)
      val srcPtr = fb.getCodeParam[Long](2)
      fb.emitWithBuilder { cb =>
        cb.assign(dstPtr, r.allocate(sbll1.storageType.alignment, sbll1.storageType.byteSize))
        sbll2.load(cb, srcPtr)
        sbll1.initWithDeepCopy(cb, r, sbll2)
        sbll1.store(cb, dstPtr)
        dstPtr
      }

      val f = fb.result(ctx)(theHailClassLoader)
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
    pool.scopedRegion { region =>
      val b = new BlockLinkedList[Int](region, PInt32Required)
      for (i <- 1 to 100) b += i
      assertEquals(b.toIndexedSeq, IndexedSeq.tabulate(100)(_ + 1))
    }
  }

  @Test def testPushStrsMissing() {
    pool.scopedRegion { region =>
      val a = new BoxedArrayBuilder[String]()
      val b = new BlockLinkedList[String](region, PCanonicalString())
      for (i <- 1 to 100) {
        val elt = if(i%3 == 0) null else i.toString()
        a += elt
        b += elt
      }
      assertEquals(b.toIndexedSeq, a.result().toIndexedSeq)
    }
  }

  @Test def testAppendAnother() {
    pool.scopedRegion { region =>
      val b1 = new BlockLinkedList[String](region, PCanonicalString())
      val b2 = new BlockLinkedList[String](region, PCanonicalString())
      b1 += "{"
      b2 ++= Seq("foo", "bar")
      b1 ++= b2
      b1 ++= b2
      b1 += "}"
      assertEquals(b1.toIndexedSeq, "{ foo bar foo bar }".split(" ").toIndexedSeq)
    }
  }

  @Test def testDeepCopy() {
    pool.scopedRegion { region =>
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
