package is.hail.expr.ir.agg

import breeze.linalg.{DenseMatrix, DenseVector, diag, inv}
import is.hail.annotations.{Region, RegionValueBuilder, UnsafeRow}
import is.hail.asm4s._
import is.hail.backend.{ExecuteContext, HailStateManager}
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SIndexablePointer, SIndexablePointerValue}
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.virtual.{TArray, TFloat64, TInt32, Type}
import is.hail.utils.FastSeq

class LinearRegressionAggregatorState(val kb: EmitClassBuilder[_]) extends AbstractTypedRegionBackedAggState(LinearRegressionAggregator.stateType)

object LinearRegressionAggregator {

  val scalar = PFloat64(true)
  val vector = PCanonicalArray(scalar, true)
  val stateType: PCanonicalTuple = PCanonicalTuple(true, vector, vector, PInt32(true))

  private val optVector = vector.setRequired(false)

  val resultPType: PCanonicalStruct = PCanonicalStruct(required = false, "xty" -> optVector, "beta" -> optVector, "diag_inv" -> optVector, "beta0" -> optVector)

  def computeResult(region: Region, xtyPtr: Long, xtxPtr: Long, k0: Int): Long = {
    val xty = DenseVector(UnsafeRow.readArray(vector, null, xtyPtr)
      .asInstanceOf[IndexedSeq[Double]].toArray[Double])
    val k = xty.length
    val xtx = DenseMatrix.create(k, k, UnsafeRow.readArray(vector, null, xtxPtr)
      .asInstanceOf[IndexedSeq[Double]].toArray[Double])

    val rvb = new RegionValueBuilder(HailStateManager(Map.empty), region)
    rvb.start(resultPType)
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

  override def resultEmitType: EmitType = EmitType(SBaseStructPointer(LinearRegressionAggregator.resultPType), true)

  val initOpTypes: Seq[Type] = Array(TInt32, TInt32)
  val seqOpTypes: Seq[Type] = Array(TFloat64, TArray(TFloat64))

  def initOpF(state: State)(cb: EmitCodeBuilder, kc: Code[Int], k0c: Code[Int]): Unit = {
    val k = cb.newLocal[Int]("lra_init_k", kc)
    val k0 = cb.newLocal[Int]("lra_init_k0", k0c)
    cb.if_((k0 < 0) || (k0 > k),
      cb += Code._fatal[Unit](const("linreg: `nested_dim` must be between 0 and the number (")
        .concat(k.toS)
        .concat(") of covariates, inclusive"))
    )
    cb.assign(state.off, stateType.allocate(state.region))
    cb += Region.storeAddress(stateType.fieldOffset(state.off, 0), vector.zeroes(cb, state.region, k))
    cb += Region.storeAddress(stateType.fieldOffset(state.off, 1), vector.zeroes(cb, state.region, k * k))
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
              k0tCode => initOpF(state)(cb, ktCode.asInt.value, k0tCode.asInt.value)
            )
        })
  }

  def seqOpF(state: State)(cb: EmitCodeBuilder, y: Code[Double], x: SIndexableValue): Unit = {
    val k = cb.newLocal[Int]("linreg_agg_seqop_k")
    val i = cb.newLocal[Int]("linreg_agg_seqop_i")
    val j = cb.newLocal[Int]("linreg_agg_seqop_j")
    val sptr = cb.newLocal[Long]("linreg_agg_seqop_sptr")
    val xty = cb.newLocal[Long]("linreg_agg_seqop_xty")
    val xtx = cb.newLocal[Long]("linreg_agg_seqop_xtx")

    cb.if_(!x.hasMissingValues(cb),
      {
        cb.assign(xty, stateType.loadField(state.off, 0))
        cb.assign(xtx, stateType.loadField(state.off, 1))
        cb.assign(k, vector.loadLength(xty))
        cb.assign(sptr, vector.firstElementOffset(xty, k))
        cb.assign(i, 0)
        x.st match {
          case SIndexablePointer(pt: PCanonicalArray) =>
            assert(pt.elementType.isInstanceOf[PFloat64])

            val xAddr = x.asInstanceOf[SIndexablePointerValue].a
            val xptr = cb.newLocal[Long]("linreg_agg_seqop_xptr")
            val xptr2 = cb.newLocal[Long]("linreg_agg_seqop_xptr2")
            cb.assign(xptr, pt.firstElementOffset(xAddr, k))
            cb.while_(i < k,
              {
                cb += Region.storeDouble(sptr, Region.loadDouble(sptr) + (Region.loadDouble(xptr) * y))
                cb.assign(i, i + 1)
                cb.assign(sptr, sptr + scalar.byteSize)
                cb.assign(xptr, xptr + scalar.byteSize)
              })

            cb.assign(i, 0)
            cb.assign(sptr, vector.firstElementOffset(xtx, k))
            cb.assign(xptr, pt.firstElementOffset(xAddr, k))

            cb.while_(i < k,
              {
                cb.assign(j, 0)
                cb.assign(xptr2, pt.firstElementOffset(xAddr, k))
                cb.while_(j < k,
                  {
                    // add x[i] * x[j] to the value at sptr
                    cb += Region.storeDouble(sptr, Region.loadDouble(sptr) + (Region.loadDouble(xptr) * Region.loadDouble(xptr2)))
                    cb.assign(j, j + 1)
                    cb.assign(sptr, sptr + scalar.byteSize)
                    cb.assign(xptr2, xptr2 + scalar.byteSize)
                  })
                cb.assign(i, i + 1)
                cb.assign(xptr, xptr + scalar.byteSize)
              })

          case _ =>
            cb.while_(i < k,
              {
                cb += Region.storeDouble(sptr, Region.loadDouble(sptr) + x.loadElement(cb, i).get(cb).asDouble.value * y)
                cb.assign(i, i + 1)
                cb.assign(sptr, sptr + scalar.byteSize)
              })

            cb.assign(i, 0)
            cb.assign(sptr, vector.firstElementOffset(xtx, k))

            cb.while_(i < k,
              {
                cb.assign(j, 0)
                cb.while_(j < k,
                  {
                    // add x[i] * x[j] to the value at sptr
                    cb += Region.storeDouble(sptr, Region.loadDouble(sptr) +
                      (x.loadElement(cb, i).get(cb).asDouble.value * x.loadElement(cb, j).get(cb).asDouble.value))
                    cb.assign(j, j + 1)
                    cb.assign(sptr, sptr + scalar.byteSize)
                  })
                cb.assign(i, i + 1)
              })
        }
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
              xCode => seqOpF(state)(cb, yCode.asDouble.value, xCode.asIndexable)
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

    cb += Code(
      xty := stateType.loadField(state.off, 0),
      xtx := stateType.loadField(state.off, 1),
      oxty := stateType.loadField(other.off, 0),
      oxtx := stateType.loadField(other.off, 1),
      n := vector.loadLength(xty),
      i := 0,
      sptr := vector.firstElementOffset(xty, n),
      optr := vector.firstElementOffset(oxty, n)
    )

    cb.while_(i < n, {
      cb += Code(
        Region.storeDouble(sptr, Region.loadDouble(sptr) + Region.loadDouble(optr)),
        i := i + 1,
        sptr := sptr + scalar.byteSize,
        optr := optr + scalar.byteSize
      )
    })

    cb += Code(
      n := vector.loadLength(xtx),
      i := 0,
      sptr := vector.firstElementOffset(xtx, n),
      optr := vector.firstElementOffset(oxtx, n)
    )

    cb.while_(i < n, {
      cb += Code(
        Region.storeDouble(sptr, Region.loadDouble(sptr) + Region.loadDouble(optr)),
        i := i + 1,
        sptr := sptr + scalar.byteSize,
        optr := optr + scalar.byteSize)
    })
  }

  protected def _combOp(ctx: ExecuteContext, cb: EmitCodeBuilder, region: Value[Region], state: AbstractTypedRegionBackedAggState, other: AbstractTypedRegionBackedAggState): Unit = {
    combOpF(state, other)(cb)
  }

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode = {
    val resAddr = cb.newLocal[Long]("linear_regression_agg_res", Code.invokeScalaObject4[Region, Long, Long, Int, Long](
      LinearRegressionAggregator.getClass, "computeResult",
      region,
      stateType.loadField(state.off, 0),
      stateType.loadField(state.off, 1),
      Region.loadInt(stateType.loadField(state.off, 2))))
    IEmitCode.present(cb, LinearRegressionAggregator.resultPType.loadCheapSCode(cb, resAddr))
  }
}
