package is.hail.types.physical

import is.hail.annotations.{Region, SafeNDArray, UnsafeRow}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitFunctionBuilder}
import is.hail.methods.LocalWhitening
import is.hail.types.physical.stypes.interfaces.{ColonIndex => Colon, _}
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class PNDArraySuite extends PhysicalTestUtils {
  @Test def copyTests() {
    def runTests(deepCopy: Boolean, interpret: Boolean = false) {
      copyTestExecutor(PCanonicalNDArray(PInt64(true), 1), PCanonicalNDArray(PInt64(true), 1), new SafeNDArray(IndexedSeq(3L), IndexedSeq(4L,5L,6L)),
        deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, true)
    runTests(false, true)
  }

  @Test def testWhitenBase(): Unit = {
    val fb = EmitFunctionBuilder[Region, Double](ctx, "whiten_test")
    val matType = PCanonicalNDArray(PFloat64Required, 2)
    val vecType = PCanonicalNDArray(PFloat64Required, 1)
    val m = SizeValueStatic(2000)
    val w = SizeValueStatic(50)
    val n = SizeValueStatic(200)
    val wpn = SizeValueStatic(w.v + n.v)
    val blocksize = SizeValueStatic(5)
    val btwpn = SizeValueStatic(blocksize.v * wpn.v)

    this.pool.scopedRegion { region =>
      fb.emitWithBuilder { cb =>
        val region = fb.getCodeParam[Region](1)
        val A = matType.constructUninitialized(FastSeq(m, wpn), cb, region)
        val Acopy = matType.constructUninitialized(FastSeq(m, wpn), cb, region)
        val Q = matType.constructUninitialized(FastSeq(m, wpn), cb, region)
        val R = matType.constructUninitialized(FastSeq(wpn, wpn), cb, region)
        val Qout = matType.constructUninitialized(FastSeq(m, w), cb, region)
        val W = matType.constructUninitialized(FastSeq(m, n), cb, region)
        val work1 = matType.constructUninitialized(FastSeq(wpn, wpn), cb, region)
        val work2 = matType.constructUninitialized(FastSeq(wpn, n), cb, region)
        val work3 = vecType.constructUninitialized(FastSeq(btwpn), cb, region)
        val T = vecType.constructUninitialized(FastSeq(btwpn), cb, region)

        A.coiterateMutate(cb, region) { case Seq(a) =>
          primitive(cb.memoize(cb.emb.newRNG(0L).invoke[Double]("rnorm")))
        }
        Acopy.coiterateMutate(cb, region, (A, "A")) { case Seq(acopy, a) => a }

        SNDArray.geqrt_full(cb, Acopy, Q, R, T, work3, blocksize)

        new LocalWhitening(cb, m, w, n, blocksize, region, false).whitenBlockPreOrthogonalized(cb, Q.slice(cb, Colon, (null, w)), Q.slice(cb, Colon, (w, null)), Qout, R, W, work1, work2, blocksize)

        SNDArray.trmm(cb, "R", "U", "N", "N", 1.0, R.slice(cb, (n, null), (n, null)), Qout)

        val normDiff = cb.newLocal[Double]("normDiff", 0.0)
        val normA = cb.newLocal[Double]("normA", 0.0)
        val diff = cb.newLocal[Double]("diff")

        SNDArray.coiterate(cb, (A.slice(cb, Colon, (n, null)), "A"), (Qout, "Qout")) {
          case Seq(l, r) =>
            def lCode = l.asDouble.value
            cb.assign(diff, lCode - r.asDouble.value)
            cb.assign(normDiff, normDiff + diff*diff)
            cb.assign(normA, normA + lCode*lCode)
        }
        Code.invokeStatic1[java.lang.Math, Double, Double]("sqrt", normDiff / normA)
      }

      val f = fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, region)

      assert(f(region) < 1e-14)
    }
  }

  @Test def testQrPivot(): Unit = {
    val fb = EmitFunctionBuilder[Region, Double](ctx, "whiten_test")
    val matType = PCanonicalNDArray(PFloat64Required, 2)
    val vecType = PCanonicalNDArray(PFloat64Required, 1)
    val m = SizeValueStatic(2000)
    val w = SizeValueStatic(50)
    val n = SizeValueStatic(500)
    val p = 5
    val blocksize = SizeValueStatic(5)
    val btn = SizeValueStatic(blocksize.v * n.v)
    val btm = SizeValueStatic(blocksize.v * m.v)

    this.pool.scopedRegion { region =>
      fb.emitWithBuilder { cb =>
        val region = fb.getCodeParam[Region](1)
        val A = matType.constructUninitialized(FastSeq(m, n), FastSeq(8, 8*m.v), cb, region)
        val Acopy = matType.constructUninitialized(FastSeq(m, n), FastSeq(8, 8*m.v), cb, region)
        val Q = matType.constructUninitialized(FastSeq(m, n), FastSeq(8, 8*m.v), cb, region)
        val R = matType.constructUninitialized(FastSeq(n, n), FastSeq(8, 8*n.v), cb, region)
        val work = vecType.constructUninitialized(FastSeq(btm), FastSeq(8), cb, region)
        val T = vecType.constructUninitialized(FastSeq(btn), FastSeq(8), cb, region)

        A.coiterateMutate(cb, region) { case Seq(a) =>
          primitive(cb.memoize(cb.emb.newRNG(0L).invoke[Double]("rnorm")))
        }
        Acopy.coiterateMutate(cb, region, (A, "A")) { case Seq(acopy, a) => a }
        SNDArray.geqrt_full(cb, Acopy, Q, R, T, work, blocksize)

        new LocalWhitening(cb, m, w, n, blocksize, region, false).qrPivot(cb, Q, R, 0, p)

        SNDArray.trmm(cb, "R", "U", "N", "N", 1.0, R.slice(cb, (null, p), (null, p)), Q.slice(cb, Colon, (null, p)))
        SNDArray.gemm(cb, "N", "N", 1.0, Q.slice(cb, Colon, (p, null)), R.slice(cb, (p, null), (null, p)), 1.0, Q.slice(cb, Colon, (null, p)))
        SNDArray.trmm(cb, "R", "U", "N", "N", 1.0, R.slice(cb, (p, null), (p, null)), Q.slice(cb, Colon, (p, null)))

        val normDiff = cb.newLocal[Double]("normDiff", 0.0)
        val normA = cb.newLocal[Double]("normA", 0.0)
        val diff = cb.newLocal[Double]("diff")

        SNDArray.coiterate(cb, (A, "A"), (Q, "Q")) {
          case Seq(l, r) =>
            def lCode = l.asDouble.value
            cb.assign(diff, lCode - r.asDouble.value)
            cb.assign(normDiff, normDiff + diff*diff)
            cb.assign(normA, normA + lCode*lCode)
        }
        Code.invokeStatic1[java.lang.Math, Double, Double]("sqrt", normDiff / normA)
      }

      val f = fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, region)

      assert(f(region) < 1e-14)
    }
  }

  def whitenNaive(cb: EmitCodeBuilder, X: SNDArrayValue, w: Int, blocksize: Int, region: Value[Region]): SNDArrayValue = {
    val Seq(m, n) = X.shapes
    val vecType = PCanonicalNDArray(PFloat64Required, 1)
    val matType = PCanonicalNDArray(PFloat64Required, 2)
    val Xw = matType.constructUninitialized(X.shapes, cb, region)
    Xw.coiterateMutate(cb, region, (X, "X")) { case Seq(_, v) => v }
    val curWindow = matType.constructUninitialized(FastSeq(m, SizeValueStatic(w)), cb, region)
    val btm = SizeValueDyn(cb.memoize(m * blocksize))
    val btn = SizeValueDyn(cb.memoize(n * blocksize))
    val work = vecType.constructUninitialized(FastSeq(btm), cb, region)
    val T = vecType.constructUninitialized(FastSeq(btn), cb, region)

    val j = cb.newLocal[Long]("j")
    cb.forLoop(cb.assign(j, 0L), j < n, cb.assign(j, j+1), {
      val windowStart = cb.memoize((j-w).max(0))
      val windowSize = cb.memoize(j - windowStart)
      val window = curWindow.slice(cb, Colon, (null, windowSize))
      window.coiterateMutate(cb, region, (X.slice(cb, Colon, (windowStart, j)), "X")) { case Seq(_, v) => v }
      val bs = cb.memoize(windowSize.min(blocksize).max(1))
      SNDArray.geqrt(window, T, work, bs, cb)
      val curCol = Xw.slice(cb, Colon, j)
      SNDArray.gemqrt("L", "T", window, T, curCol, work, bs, cb)
      curCol.slice(cb, (null, windowSize)).setToZero(cb)
      SNDArray.gemqrt("L", "N", window, T, curCol, work, bs, cb)
    })

    Xw
  }

  @Test def testWhitenNonrecur(): Unit = {
    val fb = EmitFunctionBuilder[Region, Unit](ctx, "whiten_test")
    val matType = PCanonicalNDArray(PFloat64Required, 2)
    val m = SizeValueStatic(2000)
    val w = SizeValueStatic(50)
    val n = SizeValueStatic(500)
    val wpn = SizeValueStatic(w.v + n.v)
    val blocksize = SizeValueStatic(5)

    this.pool.scopedRegion { region =>
      fb.emitWithBuilder { cb =>
        val region = fb.getCodeParam[Region](1)
        val Aorig = matType.constructUninitialized(FastSeq(m, wpn), cb, region)
        val A = matType.constructUninitialized(FastSeq(m, n), cb, region)
        val state = new LocalWhitening(cb, m, w, n, blocksize, region, false)

        Aorig.coiterateMutate(cb, region) { case Seq(_) =>
          primitive(cb.memoize(cb.emb.newRNG(0L).invoke[Double]("rnorm")))
        }
        A.coiterateMutate(cb, region, (Aorig.slice(cb, Colon, (w, null)), "Aorig")) { case Seq(_, a) => a }
        state.Qtemp2.coiterateMutate(cb, region, (Aorig.slice(cb, Colon, (null, w)), "Aorig")) { case Seq(_, a) => a }

        SNDArray.geqrt_full(cb, state.Qtemp2, state.Q, state.R, state.T, state.work3, blocksize)

        state.whitenBlockSmallWindow(cb, state.Q, state.R, A, state.Qtemp, state.Qtemp2, state.Rtemp, state.work1, state.work2, blocksize)

        // Q = Q*R
        SNDArray.trmm(cb, "R", "U", "N", "N", 1.0, state.R, state.Q)

        val normDiff = cb.newLocal[Double]("normDiff", 0.0)
        val normA = cb.newLocal[Double]("normA", 0.0)
        val diff = cb.newLocal[Double]("diff")

        SNDArray.coiterate(cb, (Aorig.slice(cb, Colon, (n, null)), "A"), (state.Q, "Qout")) {
          case Seq(l, r) =>
            def lCode = l.asDouble.value
            cb.assign(diff, lCode - r.asDouble.value)
            cb.assign(normDiff, normDiff + diff*diff)
            cb.assign(normA, normA + lCode*lCode)
        }

        var relError: Value[Double] = cb.memoize(Code.invokeStatic1[java.lang.Math, Double, Double]("sqrt", normDiff / normA))
        cb.ifx(relError > 1e-14, {
          cb._fatal("backwards error too large: ", relError.toS)
        })

        val W2 = whitenNaive(cb, Aorig, w.v.toInt, blocksize.v.toInt, region)
          .slice(cb, Colon, (w, null))

        cb.assign(normDiff, 0.0)
        val normW2 = cb.newLocal[Double]("normW2", 0.0)

        SNDArray.coiterate(cb, (W2, "W2"), (A, "A")) {
          case Seq(l, r) =>
            def lCode = l.asDouble.value
            cb.assign(diff, lCode - r.asDouble.value)
            cb.assign(normDiff, normDiff + diff*diff)
            cb.assign(normW2, normW2 + lCode*lCode)
        }

        relError = cb.memoize(Code.invokeStatic1[java.lang.Math, Double, Double]("sqrt", normDiff / normW2))
        cb.println(relError.toS)
        cb.ifx(!(relError < 1e-14), {
          cb._fatal("relative error vs naive too large: ", relError.toS)
        })

        Code._empty
      }

      val f = fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, region)

      f(region)
    }
  }

  @Test def testWhiten(): Unit = {
    val fb = EmitFunctionBuilder[Region, Unit](ctx, "whiten_test")
    val matType = PCanonicalNDArray(PFloat64Required, 2)
    val m = SizeValueStatic(2000)
    val w = SizeValueStatic(100)
    val n = SizeValueStatic(500)
    val b = const(25L)
    val blocksize = SizeValueStatic(5)

    this.pool.scopedRegion { region =>
      fb.emitWithBuilder { cb =>
        val region = fb.getCodeParam[Region](1)
        val Aorig = matType.constructUninitialized(FastSeq(m, n), cb, region)
        val A = matType.constructUninitialized(FastSeq(m, n), cb, region)

        Aorig.coiterateMutate(cb, region) { case Seq(_) =>
          primitive(cb.memoize(cb.emb.newRNG(0L).invoke[Double]("rnorm")))
        }
        SNDArray.copyMatrix(cb, " ", Aorig, A)

        val state = new LocalWhitening(cb, m, w, b, blocksize, region, false)
        val i = cb.newLocal[Long]("i", 0)
        cb.whileLoop(i < n, {
          state.whitenBlock(cb, A.slice(cb, Colon, (i, (i+b).min(n))))
          cb.assign(i, i+b)
        })

        val W2 = whitenNaive(cb, Aorig, w.v.toInt, blocksize.v.toInt, region)

        val normDiff = cb.newLocal[Double]("normDiff", 0.0)
        val diff = cb.newLocal[Double]("diff")
        val normW2 = cb.newLocal[Double]("normW2", 0.0)
        cb.assign(normDiff, 0.0)

        SNDArray.coiterate(cb, (W2, "W2"), (A, "A")) {
          case Seq(l, r) =>
            def lCode = l.asDouble.value
            cb.assign(diff, lCode - r.asDouble.value)
            cb.assign(normDiff, normDiff + diff*diff)
            cb.assign(normW2, normW2 + lCode*lCode)
        }

        val relError = cb.memoize(Code.invokeStatic1[java.lang.Math, Double, Double]("sqrt", normDiff / normW2))
        cb.ifx(!(relError < 1e-14), {
          cb._fatal("relative error vs naive too large: ", relError.toS)
        })

        Code._empty
      }

      val f = fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, region)

      f(region)
    }
  }

  @Test def testRefCounted(): Unit = {
    val nd = PCanonicalNDArray(PInt32Required, 1)

    val region1 = Region(pool=this.pool)
    val region2 = Region(pool=this.pool)
    val region3 = Region(pool=this.pool)
    val fb = EmitFunctionBuilder[Region, Region, Region, Long](ctx, "ref_count_test")
    val codeRegion1 = fb.getCodeParam[Region](1)
    val codeRegion2 = fb.getCodeParam[Region](2)
    val codeRegion3 = fb.getCodeParam[Region](3)

    try {
      fb.emitWithBuilder{ cb =>
        val r2PointerToNDAddress1 = cb.newLocal[Long]("r2_ptr_to_nd_addr1")

        val shapeSeq = IndexedSeq(const(3L))

        // Region 1 just gets 1 ndarray.
        val (_, snd1Finisher) = nd.constructDataFunction(shapeSeq, shapeSeq, cb, codeRegion1)

        val snd1 = snd1Finisher(cb)

        // Region 2 gets an ndarray at ndaddress2, plus a reference to the one at ndarray 1.
        val (_, snd2Finisher) = nd.constructDataFunction(shapeSeq, shapeSeq, cb, codeRegion2)
        val snd2 = snd2Finisher(cb)
        cb.assign(r2PointerToNDAddress1, nd.store(cb, codeRegion2, snd1, true))

        // Return the 1st ndarray
        snd1.a
      }
    } catch {
      case e: AssertionError =>
        region1.clear()
        region2.clear()
        region3.clear()
        throw e
    }

    val f = fb.result(ctx)(theHailClassLoader)
    val result1 = f(region1, region2, region3)
    val result1Data = nd.unstagedDataFirstElementPointer(result1)

    // Check number of ndarrays in each region:
    assert(region1.memory.listNDArrayRefs().size == 1)
    assert(region1.memory.listNDArrayRefs()(0) == result1Data)

    assert(region2.memory.listNDArrayRefs().size == 2)
    assert(region2.memory.listNDArrayRefs()(1) == result1Data)

    // Check that the reference count of ndarray1 is 2:
    val rc1A = Region.loadLong(result1Data - Region.sharedChunkHeaderBytes)
    assert(rc1A == 2)

    region1.clear()
    assert(region1.memory.listNDArrayRefs().size == 0)

    // Check that ndarray 1 wasn't actually cleared, ref count should just be 1 now:
    val rc1B = Region.loadLong(result1Data - Region.sharedChunkHeaderBytes)
    assert(rc1B == 1)


    assert(region3.memory.listNDArrayRefs().size == 0)
    // Do an unstaged copy into region3
    nd.copyFromAddress(ctx.stateManager, region3, nd, result1, true)
    assert(region3.memory.listNDArrayRefs().size == 1)

    // Check that clearing region2 removes both ndarrays
    region2.clear()
    assert(region2.memory.listNDArrayRefs().size == 0)
  }

  @Test def testUnstagedCopy(): Unit = {
    val region1 = Region(pool=this.pool)
    val region2 = Region(pool=this.pool)
    val x = SafeNDArray(IndexedSeq(3L, 2L), (0 until 6).map(_.toDouble))
    val pNd = PCanonicalNDArray(PFloat64Required, 2, true)
    val ndAddr1 = pNd.unstagedStoreJavaObject(ctx.stateManager, x, region=region1)
    val ndAddr2 = pNd.copyFromAddress(ctx.stateManager, region2, pNd, ndAddr1, true)
    val unsafe1 = UnsafeRow.read(pNd, region1, ndAddr1)
    val unsafe2 = UnsafeRow.read(pNd, region2, ndAddr2)
    // Deep copy same ptype just increments reference count, doesn't change the address.
    val dataAddr1 = Region.loadAddress(pNd.representation.loadField(ndAddr1, 2))
    val dataAddr2 = Region.loadAddress(pNd.representation.loadField(ndAddr2, 2))
    assert(dataAddr1 == dataAddr2)
    assert(Region.getSharedChunkRefCount(dataAddr1) == 2)
    assert(unsafe1 == unsafe2)
    region1.clear()
    assert(Region.getSharedChunkRefCount(dataAddr1) == 1)

    // Deep copy with elements that contain pointers, so have to actually do a full copy
    // FIXME: Currently ndarrays do not support this, reference counting needs to account for this.
//    val pNDOfArrays = PCanonicalNDArray(PCanonicalArray(PInt32Required, true), 1)
//    val annotationNDOfArrays = new SafeNDArray(IndexedSeq(3L), (0 until 3).map(idx => (0 to idx).toArray.toIndexedSeq))
//    val addr3 = pNDOfArrays.unstagedStoreJavaObject(annotationNDOfArrays, region=region1)
//    val unsafe3 = UnsafeRow.read(pNDOfArrays, region1, addr3)
//    val addr4 = pNDOfArrays.copyFromAddress(region2, pNDOfArrays, addr3, true)
//    val unsafe4 = UnsafeRow.read(pNDOfArrays, region2, addr4)
//    assert(addr3 != addr4)
//    assert(unsafe3 == unsafe4)
//    assert(PNDArray.getReferenceCount(addr3) == 1L)
//    assert(PNDArray.getReferenceCount(addr4) == 1L)

    // Deep copy with PTypes with different requirements
    val pNDOfStructs1 = PCanonicalNDArray(PCanonicalStruct(true, ("x", PInt32Required), ("y", PInt32())), 1)
    val pNDOfStructs2 = PCanonicalNDArray(PCanonicalStruct(true, ("x", PInt32()), ("y", PInt32Required)), 1)
    val annotationNDOfStructs = new SafeNDArray(IndexedSeq(5L), (0 until 5).map(idx => Row(idx, idx + 100)))

    val addr5 = pNDOfStructs1.unstagedStoreJavaObject(ctx.stateManager, annotationNDOfStructs, region=region1)
    val unsafe5 = UnsafeRow.read(pNDOfStructs1, region1, addr5)
    val addr6 = pNDOfStructs2.copyFromAddress(ctx.stateManager, region2, pNDOfStructs1, addr5, true)
    val unsafe6 = UnsafeRow.read(pNDOfStructs2, region2, addr6)
    assert(unsafe5 == unsafe6)
  }
}
