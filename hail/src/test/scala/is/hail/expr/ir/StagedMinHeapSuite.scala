package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.asm4s._
import is.hail.expr.ir.streams.EmitMinHeap
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Value}
import is.hail.utils.FastSeq
import org.testng.annotations.Test

class StagedMinHeapSuite extends HailSuite {

  trait Init { def init(): Unit }

  @Test def testPureComparator(): Unit = {
    val Main = EmitFunctionBuilder[AsmFunction0[Int] with Init](ctx, "Main", FastSeq(), IntInfo)
    Main.cb.addInterface(typeInfo[Init].iname)
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

    Main.emitWithBuilder[Int] { cb =>
      val res = MinHeap.peek(cb)
      MinHeap.poll(cb)
      MinHeap.realloc(cb)
      res.asPrimitive.primitiveValue[Int]
    }

    pool.scopedRegion { r =>
      val test = Main.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)
      test.init()
      assert(IndexedSeq.fill(10)(test().abs) == (0 until 10))
    }
  }
}
