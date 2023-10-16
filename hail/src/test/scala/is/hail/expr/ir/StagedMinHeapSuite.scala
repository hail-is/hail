package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeIndexedSeq}
import is.hail.asm4s._
import is.hail.check.Prop.forAll
import is.hail.expr.ir.streams.EmitMinHeap
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.SIndexablePointerValue
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Value}
import is.hail.types.physical.{PCanonicalArray, PInt32}
import is.hail.utils.{FastSeq, using}
import org.testng.annotations.Test

class StagedMinHeapSuite extends HailSuite {
  @Test def testSorting(): Unit =
    forAll { (xs: IndexedSeq[Int]) => sort(xs) == xs.sorted }.check()

  @Test def testHeapProperty(): Unit =
    forAll { (xs: IndexedSeq[Int]) =>
      val heap = heapify(xs)
      (0 until heap.size / 2).forall { i =>
        ((2 * i + 1) >= heap.size || heap(i) <= heap(2 * i + 1)) &&
          ((2 * i + 2) >= heap.size || heap(i) <= heap(2 * i + 2))
      }
    }.check()


  def sort(xs: IndexedSeq[Int]): IndexedSeq[Int] =
    emitHeap("Sort", xs) { heap =>
      heap.init()
      IndexedSeq.fill(xs.size)(heap.pop())
    }

  def heapify(xs: IndexedSeq[Int]): IndexedSeq[Int] =
    emitHeap("Heapify", xs) { heap =>
      pool.scopedRegion { r =>
        heap.init()
        val ptr = heap.toArray(r)
        SafeIndexedSeq(PCanonicalArray(PInt32()), ptr).asInstanceOf[IndexedSeq[Int]]
      }
    }

  def emitHeap[A](name: String, xs: IndexedSeq[Int])(f: Heap => A): A = {
    val modb = new EmitModuleBuilder(ctx, new ModuleBuilder())
    val Main = modb.genEmitClass[Heap with AutoCloseable](name)
    Main.cb.addInterface(implicitly[TypeInfo[AutoCloseable]].iname)

    val MinHeap = EmitMinHeap(modb, SInt32) { _ =>
      (cb: EmitCodeBuilder, a: SValue, b: SValue) =>
        cb.memoize({
          val x = a.asPrimitive.primitiveValue[Int]
          val y = b.asPrimitive.primitiveValue[Int]
          Code.invokeStatic[Int](classOf[Integer], "compare", Array.fill(2)(classOf[Int]), Array(x, y))
        })
    }(Main)

    Main.defineEmitMethod("init", FastSeq(), UnitInfo) { mb =>
      mb.voidWithBuilder { cb =>
        MinHeap.init(cb, Main.pool())
        for (x <- xs) MinHeap.push(cb, new SInt32Value(x))
      }
    }

    Main.defineEmitMethod("close", FastSeq(), UnitInfo) { mb =>
      mb.voidWithBuilder { MinHeap.close }
    }

    Main.defineEmitMethod("pop", FastSeq(), IntInfo) { mb =>
      mb.emitWithBuilder[Int] { cb =>
        val res = MinHeap.peek(cb)
        MinHeap.pop(cb)
        MinHeap.realloc(cb)
        res.asPrimitive.primitiveValue[Int]
      }
    }

    Main.defineEmitMethod("toArray", FastSeq(typeInfo[Region]), LongInfo) { mb =>
      mb.emitWithBuilder { cb =>
        val region = mb.getCodeParam[Region](1)
        val arr = MinHeap.toArray(cb, region)
        arr.asInstanceOf[SIndexablePointerValue].a
      }
    }

    pool.scopedRegion { r =>
      using(Main.resultWithIndex(ctx.shouldWriteIRFiles())(theHailClassLoader, ctx.fs, ctx.taskContext, r)) { f }
    }
  }

  trait Heap {
    def init(): Unit
    def pop(): Int
    def toArray(r: Region): Long
  }
}
