package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.asm4s._
import is.hail.expr.ir.streams.EmitMinHeap
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Value}
import is.hail.utils.{FastSeq, using}
import org.testng.annotations.Test

class StagedMinHeapSuite extends HailSuite {

  trait F extends java.lang.AutoCloseable {
    def init(): Unit
    def apply(): Int
  }

  @Test def testAlternatingInts(): Unit = {
    val Main = EmitFunctionBuilder[F](ctx, "Main", FastSeq(), IntInfo)
    val MinHeap = EmitMinHeap(Main.ecb.emodb, SInt32) { _ =>
      (cb: EmitCodeBuilder, a: SValue, b: SValue) =>
        cb.memoize({
          val x = a.asPrimitive.primitiveValue[Int]
          val y = b.asPrimitive.primitiveValue[Int]
          Code.invokeStatic[Int](classOf[Integer], "compare", Array.fill(2)(classOf[Int]), Array(x, y))
        })
    }(Main.ecb)

    Main.ecb.defineEmitMethod("init", FastSeq(), UnitInfo) { mb =>
      mb.voidWithBuilder { cb =>
        MinHeap.initialize(cb, Main.ecb.pool())
        val i = cb.newLocal[Int]("i", 0)
        cb.forLoop({}, i < 10, cb.assign(i, i + 1), {
          val x = cb.memoize((i % 2 ceq 0).mux(-i + 10, i - 10))
          MinHeap.add(cb, new SInt32Value(x))
        })
      }
    }

    Main.ecb.defineEmitMethod("close", FastSeq(), UnitInfo) { mb =>
      mb.voidWithBuilder { MinHeap.close }
    }

    Main.emitWithBuilder[Int] { cb =>
      val res = MinHeap.peek(cb)
      MinHeap.poll(cb)
      MinHeap.realloc(cb)
      res.asPrimitive.primitiveValue[Int]
    }

    val actual =
      pool.scopedRegion { r =>
        using(Main.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)) { test =>
          test.init()
          IndexedSeq.fill(10)(test())
        }
      }

    assert(actual == (0 until 10).map(i => if (i % 2 == 0) 10 - i else i - 10).sorted)
  }

  @Test def testClockArithmetic(): Unit = {
    val Main = EmitFunctionBuilder[F](ctx, "Main", FastSeq(), IntInfo)
    val MinHeap = EmitMinHeap(Main.ecb.emodb, SInt32) { _ =>
      (cb: EmitCodeBuilder, a: SValue, b: SValue) =>
        cb.memoize({
          val x = a.asPrimitive.primitiveValue[Int]
          val y = b.asPrimitive.primitiveValue[Int]
          Code.invokeStatic[Int](classOf[Integer], "compare", Array.fill(2)(classOf[Int]), Array(x, y))
        })
    }(Main.ecb)

    Main.ecb.defineEmitMethod("init", FastSeq(), UnitInfo) { mb =>
      mb.voidWithBuilder { cb =>
        MinHeap.initialize(cb, Main.ecb.pool())
        val i = cb.newLocal[Int]("i", 0)
        cb.forLoop({}, i < 10, cb.assign(i, i + 1), {
          val x = cb.memoize(i % 3)
          MinHeap.add(cb, new SInt32Value(x))
        })
      }
    }

    Main.ecb.defineEmitMethod("close", FastSeq(), UnitInfo) { mb =>
      mb.voidWithBuilder {
        MinHeap.close
      }
    }

    Main.emitWithBuilder[Int] { cb =>
      val res = MinHeap.peek(cb)
      MinHeap.poll(cb)
      MinHeap.realloc(cb)
      res.asPrimitive.primitiveValue[Int]
    }

    val actual =
      pool.scopedRegion { r =>
        using(Main.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)) { test =>
          test.init()
          IndexedSeq.fill(10)(test())
        }
      }

    assert(actual == (0 until 10).map(i => i % 3).sorted)
  }
}
