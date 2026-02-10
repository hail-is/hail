package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeRow}
import is.hail.asm4s._
import is.hail.asm4s.implicits.valueToRichCodeRegion
import is.hail.collection.FastSeq
import is.hail.collection.implicits._
import is.hail.expr.ir.agg.TakeByRVAS
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.primitives.SInt32Value

import org.scalatest.Inspectors.forAll
import org.scalatest.enablers.InspectorAsserting.assertingNatureOfAssertion
import org.testng.annotations.Test

class TakeByAggregatorSuite extends HailSuite {
  @Test def testPointers(): Unit = {
    forAll(Array((1000, 100), (1, 10), (100, 10000), (1000, 10000))) { case (size, n) =>
      val fb = EmitFunctionBuilder[Region, Long](ctx, "test_pointers")
      val cb = fb.ecb
      val stringPT = PCanonicalString(true)
      val tba = new TakeByRVAS(
        VirtualTypeWithReq(PCanonicalString(true)),
        VirtualTypeWithReq(PInt64Optional),
        cb,
      )
      pool.scopedRegion { r =>
        val argR = fb.getCodeParam[Region](1)
        val i = fb.genFieldThisRef[Long]()
        val off = fb.genFieldThisRef[Long]()
        val rt = PCanonicalArray(tba.valueType)

        fb.emitWithBuilder { cb =>
          tba.createState(cb)
          tba.newState(cb, 0L)
          tba.initialize(cb, size)
          cb += (i := 0L)
          cb.while_(
            i < n.toLong, {
              cb += argR.invoke[Unit]("clear")
              cb.assign(off, stringPT.allocateAndStoreString(cb, argR, const("str").concat(i.toS)))
              tba.seqOp(cb, false, off, false, cb.memoize(-i))
              cb += (i := i + 1L)
            },
          )
          tba.result(cb, argR, rt).a
        }

        val o = fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)(r)
        val result = SafeRow.read(rt, o)
        assert(
          result == ((n - 1) to 0 by -1)
            .iterator
            .map(i => s"str$i")
            .take(size)
            .toFastSeq,
          s"size=$size, n=$n",
        )
      }
    }
  }

  @Test def testMissing(): Unit = {
    val fb = EmitFunctionBuilder[Region, Long](ctx, "take_by_test_missing")
    val cb = fb.ecb
    val tba =
      new TakeByRVAS(VirtualTypeWithReq(PInt32Optional), VirtualTypeWithReq(PInt32Optional), cb)
    pool.scopedRegion { r =>
      val argR = fb.getCodeParam[Region](1)
      val rt = PCanonicalArray(tba.valueType)

      fb.emitWithBuilder { cb =>
        tba.createState(cb)
        tba.newState(cb, 0L)
        tba.initialize(cb, 7)
        tba.seqOp(cb, true, 0, true, 0)
        tba.seqOp(cb, true, 0, true, 0)
        tba.seqOp(cb, false, 0, false, 0)
        tba.seqOp(cb, false, 1, false, 1)
        tba.seqOp(cb, false, 2, false, 2)
        tba.seqOp(cb, false, 3, false, 3)
        tba.seqOp(cb, true, 0, true, 0)
        tba.seqOp(cb, true, 0, true, 0)
        tba.result(cb, argR, rt).a
      }

      val o = fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)(r)
      val result = SafeRow.read(rt, o)
      assert(result == FastSeq(0, 1, 2, 3, null, null, null))
    }
  }

  @Test def testRandom(): Unit =
    forAll(Array(1, 2, 10, 100, 1000, 10000, 100000, 1000000)) { n =>
      val nToTake = 1025
      val fb = EmitFunctionBuilder[Region, Long](ctx, "take_by_test_random")
      val kb = fb.ecb

      pool.scopedRegion { r =>
        val argR = fb.getCodeParam[Region](1)
        val i = fb.genFieldThisRef[Int]()
        val random = fb.genFieldThisRef[Int]()
        val resultOff = fb.genFieldThisRef[Long]()

        val tba =
          new TakeByRVAS(VirtualTypeWithReq(PInt32Required), VirtualTypeWithReq(PInt32Required), kb)
        val ab = new agg.StagedArrayBuilder(PInt32Required, kb, argR)
        val rt = PCanonicalArray(tba.valueType)
        val rng = fb.apply_method.threefryRandomEngine

        fb.emitWithBuilder { cb =>
          tba.createState(cb)
          tba.newState(cb, 0L)
          tba.initialize(cb, nToTake)
          ab.initialize(cb)
          cb += (i := 0)
          cb.while_(
            i < n, {
              cb += (random := rng.invoke[Double, Double, Double]("runif", -10000d, 10000d).toI)
              tba.seqOp(cb, false, random, false, random)
              ab.append(cb, new SInt32Value(random))
              cb += (i := i + 1)
            },
          )
          cb.if_(ab.size cne n, cb._fatal("bad size!"))
          cb += (resultOff := argR.allocate(8L, 16L))
          cb += Region.storeAddress(resultOff, tba.result(cb, argR, rt).a)
          cb += Region.storeAddress(resultOff + 8L, ab.data)
          resultOff
        }

        val o = fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)(r)
        val pqOffset = Region.loadAddress(o)
        val pq = SafeRow.read(rt, pqOffset)
        val collOffset = Region.loadAddress(o + 8)
        val collected = SafeRow.read(ab.eltArray, collOffset).asInstanceOf[IndexedSeq[Int]].take(n)
        val minValues = collected.sorted.take(nToTake)
        assert(pq == minValues, s"n=$n")
      }
    }
}
