package is.hail.expr.ir.agg

import breeze.linalg.{DenseMatrix, DenseVector, diag, inv}
import is.hail.annotations.{Region, RegionValueBuilder, StagedRegionValueBuilder, UnsafeRow}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.types.physical._
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.virtual.{TArray, TFloat64, TInt32, Type}
import is.hail.utils.FastIndexedSeq

class LinearRegressionAggregatorState(val kb: EmitClassBuilder[_]) extends AbstractTypedRegionBackedAggState(LinearRegressionAggregator.stateType)

object LinearRegressionAggregator {

  val scalar = PFloat64(true)
  val vector = PCanonicalArray(scalar, true)
  val stateType: PCanonicalTuple = PCanonicalTuple(true, vector, vector, PInt32(true))

  private val optVector = vector.setRequired(false)

  def resultType: PCanonicalStruct = PCanonicalStruct(required = true, "xty" -> optVector, "beta" -> optVector, "diag_inv" -> optVector, "beta0" -> optVector)

  def computeResult(region: Region, xtyPtr: Long, xtxPtr: Long, k0: Int): Long = {
    val xty = DenseVector(UnsafeRow.readArray(vector, null, xtyPtr)
      .asInstanceOf[IndexedSeq[Double]].toArray[Double])
    val k = xty.length
    val xtx = DenseMatrix.create(k, k, UnsafeRow.readArray(vector, null, xtxPtr)
      .asInstanceOf[IndexedSeq[Double]].toArray[Double])

    val rvb = new RegionValueBuilder(region)
    rvb.start(resultType)
    rvb.startStruct()

    try {
      val b = xtx \ xty
      val diagInv = diag(inv(xtx))

      val xtx0 = xtx(0 until k0, 0 until k0)
      val xty0 = xty(0 until k0)
      val b0 = xtx0 \ xty0

      rvb.startArray(k)
      var i = 0
      while (i < k) {
        rvb.addDouble(xty(i))
        i += 1
      }
      rvb.endArray()

      rvb.startArray(k)
      i = 0
      while (i < k) {
        rvb.addDouble(b(i))
        i += 1
      }
      rvb.endArray()

      rvb.startArray(k)
      i = 0
      while (i < k) {
        rvb.addDouble(diagInv(i))
        i += 1
      }
      rvb.endArray()

      rvb.startArray(k0)
      i = 0
      while (i < k0) {
        rvb.addDouble(b0(i))
        i += 1
      }
      rvb.endArray()
    } catch {
      case _: breeze.linalg.MatrixSingularException |
           _: breeze.linalg.NotConvergedException =>
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
    }

    rvb.endStruct()
    rvb.end()
  }
}

class LinearRegressionAggregator() extends StagedAggregator {

  import LinearRegressionAggregator._

  type State = AbstractTypedRegionBackedAggState

  override def resultType: PType = LinearRegressionAggregator.resultType

  val initOpTypes: Seq[Type] = Array(TInt32, TInt32)
  val seqOpTypes: Seq[Type] = Array(TFloat64, TArray(TFloat64))

  def initOpF(state: State)(cb: EmitCodeBuilder, kc: Code[Int], k0c: Code[Int]): Unit = {
    val k = cb.newLocal[Int]("lra_init_k", kc)
    val k0 = cb.newLocal[Int]("lra_init_k0", k0c)
    cb.ifx((k0 < 0) || (k0 > k),
      cb += Code._fatal[Unit](const("linreg: `nested_dim` must be between 0 and the number (")
        .concat(k.toS)
        .concat(") of covariates, inclusive"))
    )
    cb.assign(state.off, stateType.allocate(state.region))
    cb += Region.storeAddress(stateType.fieldOffset(state.off, 0), vector.zeroes(cb.emb, state.region, k))
    cb += Region.storeAddress(stateType.fieldOffset(state.off, 1), vector.zeroes(cb.emb, state.region, k * k))
    cb += Region.storeInt(stateType.loadField(state.off, 2), k0)
  }

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(kt, k0t) = init
    kt.toI(cb)
      .consume(cb,
        {
          cb += Code._fatal[Unit]("linreg: init args may not be missing")
        },
        { ktCode =>
          k0t.toI(cb)
            .consume(cb,
              {
                cb += Code._fatal[Unit]("linreg: init args may not be missing")
              },
              k0tCode => initOpF(state)(cb, ktCode.asInt.intCode(cb), k0tCode.asInt.intCode(cb))
            )
        })
  }

  def seqOpF(state: State)(cb: EmitCodeBuilder, y: Code[Double], xc: SCode): Unit = {
    val k = cb.newLocal[Int]("linreg_agg_seqop_k")
    val i = cb.newLocal[Int]("linreg_agg_seqop_i")
    val j = cb.newLocal[Int]("linreg_agg_seqop_j")
    val sptr = cb.newLocal[Long]("linreg_agg_seqop_sptr")
    val xty = cb.newLocal[Long]("linreg_agg_seqop_xty")
    val xtx = cb.newLocal[Long]("linreg_agg_seqop_xtx")

    val x = xc.memoize(cb, "lra_seqop_x").asInstanceOf[SIndexableValue]

    cb.ifx(!x.hasMissingValues(cb),
      {
        cb.assign(xty, stateType.loadField(state.off, 0))
        cb.assign(xtx, stateType.loadField(state.off, 1))
        cb.assign(k, vector.loadLength(xty))
        cb.assign(sptr, vector.firstElementOffset(xty, k))
        cb.assign(i, 0)
        cb.whileLoop(i < k,
          {
            cb += Region.storeDouble(sptr, Region.loadDouble(sptr) + x.loadElement(cb, i).get(cb).asDouble.doubleCode(cb) * y)
            cb.assign(i, i + 1)
            cb.assign(sptr, sptr + scalar.byteSize)
          })

        cb.assign(i, 0)
        cb.assign(sptr, vector.firstElementOffset(xtx, k))

        cb.whileLoop(i < k,
          {
            cb.assign(j, 0)
            cb.whileLoop(j < k,
              {
                // add x[i] * x[j] to the value at sptr
                cb += Region.storeDouble(sptr, Region.loadDouble(sptr) +
                  (x.loadElement(cb, i).get(cb).asDouble.doubleCode(cb) * x.loadElement(cb, j).get(cb).asDouble.doubleCode(cb)))
                cb.assign(j, j + 1)
                cb.assign(sptr, sptr + scalar.byteSize)
              })
            cb.assign(i, i + 1)
          })
      })
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(y, x) = seq
    y.toI(cb)
      .consume(cb,
        {},
        { yCode =>
          x.toI(cb)
            .consume(cb,
              {},
              xCode => seqOpF(state)(cb, yCode.asDouble.doubleCode(cb), xCode)
            )
        })
  }

  def combOpF(state: State, other: State)(cb: EmitCodeBuilder): Unit = {
    val n = cb.newLocal[Int]("n")
    val i = cb.newLocal[Int]("i")
    val sptr = cb.newLocal[Long]("sptr")
    val optr = cb.newLocal[Long]("optr")
    val xty = cb.newLocal[Long]("xty")
    val xtx = cb.newLocal[Long]("xtx")
    val oxty = cb.newLocal[Long]("oxty")
    val oxtx = cb.newLocal[Long]("oxtx")

    cb += Code(FastIndexedSeq(
      xty := stateType.loadField(state.off, 0),
      xtx := stateType.loadField(state.off, 1),
      oxty := stateType.loadField(other.off, 0),
      oxtx := stateType.loadField(other.off, 1),
      n := vector.loadLength(xty),
      i := 0,
      sptr := vector.firstElementOffset(xty, n),
      optr := vector.firstElementOffset(oxty, n),
      Code.whileLoop(i < n, Code(
        Region.storeDouble(sptr, Region.loadDouble(sptr) + Region.loadDouble(optr)),
        i := i + 1,
        sptr := sptr + scalar.byteSize,
        optr := optr + scalar.byteSize)),

      n := vector.loadLength(xtx),
      i := 0,
      sptr := vector.firstElementOffset(xtx, n),
      optr := vector.firstElementOffset(oxtx, n),
      Code.whileLoop(i < n, Code(
        Region.storeDouble(sptr, Region.loadDouble(sptr) + Region.loadDouble(optr)),
        i := i + 1,
        sptr := sptr + scalar.byteSize,
        optr := optr + scalar.byteSize))))
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    combOpF(state, other)(cb)
  }

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    val res = state.kb.genFieldThisRef[Long]()
    cb += coerce[Unit](Code(
      res := Code.invokeScalaObject4[Region, Long, Long, Int, Long](LinearRegressionAggregator.getClass, "computeResult",
        srvb.region,
        stateType.loadField(state.off, 0),
        stateType.loadField(state.off, 1),
        Region.loadInt(stateType.loadField(state.off, 2))),
      srvb.addIRIntermediate(resultType)(res)
    ))
  }
}
