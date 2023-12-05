package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s.{Code, _}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitMethodBuilder, IEmitCode}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}
import is.hail.types.virtual.{TInt32, Type}
import is.hail.utils._

class ReservoirSampleRVAS(val eltType: VirtualTypeWithReq, val kb: EmitClassBuilder[_]) extends AggregatorState {
  val eltPType = eltType.canonicalPType

  private val r: ThisFieldRef[Region] = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r
  private val rand = kb.genFieldThisRef[java.util.Random]()

  val builder = new StagedArrayBuilder(eltPType, kb, region)
  val storageType: PCanonicalTuple = PCanonicalTuple(true, PInt32Required, PInt64Required, PInt64Required, builder.stateType)
  val maxSize = kb.genFieldThisRef[Int]()
  val seenSoFar = kb.genFieldThisRef[Long]()
  private val garbage = kb.genFieldThisRef[Long]()
  private val maxSizeOffset: Code[Long] => Code[Long] = storageType.loadField(_, 0)
  private val elementsSeenOffset: Code[Long] => Code[Long] = storageType.loadField(_, 1)
  private val garbageOffset: Code[Long] => Code[Long] = storageType.loadField(_, 2)
  private val builderStateOffset: Code[Long] => Code[Long] = storageType.loadField(_, 3)

  def newState(cb: EmitCodeBuilder, off: Value[Long]): Unit = {
    cb += region.getNewRegion(regionSize)
  }

  def createState(cb: EmitCodeBuilder): Unit = {
    cb.assign(rand, Code.newInstance[java.util.Random])
    cb.if_(region.isNull, {
      cb.assign(r, Region.stagedCreate(regionSize, kb.pool()))
    })
  }

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Value[Long]): Unit = {
    regionLoader(cb, r)
    cb.assign(maxSize, Region.loadInt(maxSizeOffset(src)))
    cb.assign(seenSoFar, Region.loadLong(elementsSeenOffset(src)))
    cb.assign(garbage, Region.loadLong(garbageOffset(src)))
    builder.loadFrom(cb, builderStateOffset(src))
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Value[Long]): Unit = {
    cb.if_(region.isValid,
      {
        regionStorer(cb, region)
        cb += region.invalidate()
        cb += Region.storeInt(maxSizeOffset(dest), maxSize)
        cb += Region.storeLong(elementsSeenOffset(dest), seenSoFar)
        cb += Region.storeLong(garbageOffset(dest), garbage)
        builder.storeTo(cb, builderStateOffset(dest))
      })
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      cb += ob.writeInt(maxSize)
      cb += ob.writeLong(seenSoFar)
      builder.serialize(codec)(cb, ob)
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      cb.assign(maxSize, ib.readInt())
      cb.assign(seenSoFar, ib.readLong())
      cb.assign(garbage, 0L)
      builder.deserialize(codec)(cb, ib)
  }

  def init(cb: EmitCodeBuilder, _maxSize: Code[Int]): Unit = {
    cb.assign(maxSize, _maxSize)
    cb.assign(seenSoFar, 0L)
    cb.assign(garbage, 0L)
    builder.initialize(cb)
  }

  private[this] val gc: EmitMethodBuilder[_] =
    kb.defineEmitMethod(genName("m", "reservoir_sample_gc"), FastSeq(), UnitInfo) { mb =>
      mb.voidWithBuilder { cb =>
        cb.if_(garbage > (maxSize.toL * 2L + 1024L), {
          val oldRegion = mb.newLocal[Region]("old_region")
          cb.assign(oldRegion, region)
          cb.assign(r, Region.stagedCreate(regionSize, kb.pool()))
          builder.reallocateData(cb)
          cb.assign(garbage, 0L)
          cb += oldRegion.invoke[Unit]("invalidate")
        })
      }
    }

  def seqOp(cb: EmitCodeBuilder, elt: EmitCode): Unit = {
    val eltVal = cb.memoize(elt)
    cb.assign(seenSoFar, seenSoFar + 1)
    cb.if_(builder.size < maxSize,
      eltVal.toI(cb)
        .consume(cb,
          builder.setMissing(cb),
          sc => builder.append(cb, sc)),
    {
      // swaps the next element into the reservoir with probability (k / n), where
      // k is the reservoir size and n is the number of elements seen so far (including current)
      cb.if_(rand.invoke[Double]("nextDouble") * seenSoFar.toD <= maxSize.toD, {
        val idxToSwap = cb.memoize(rand.invoke[Int, Int]("nextInt", maxSize))
        builder.overwrite(cb, eltVal, idxToSwap)
        cb.assign(garbage, garbage + 1L)
        cb.invokeVoid(gc, cb.this_)
      })
    })
  }

  def dump(cb: EmitCodeBuilder, prefix: String): Unit = {
    cb.println(s"> dumping reservoir: $prefix with size=", maxSize.toS,", seen=", seenSoFar.toS)
    val j = cb.newLocal[Int]("j", 0)
    cb.while_(j < builder.size, {
      cb.println("    j=", j.toS, ", elt=", cb.strValue(builder.loadElement(cb, j)))
      cb.assign(j, j + 1)
    })

  }

  def combine(cb: EmitCodeBuilder, other: ReservoirSampleRVAS): Unit = {
    val j = cb.newLocal[Int]("j")
    cb.if_(other.builder.size < maxSize, {

      cb.assign(j, 0)
      cb.while_(j < other.builder.size, {
        seqOp(cb, cb.memoize(other.builder.loadElement(cb, j)))
        cb.assign(j, j + 1)
      })
    }, {
      cb.if_(builder.size < maxSize, {
        cb.assign(j, 0)
        cb.while_(j < builder.size, {
          other.seqOp(cb, cb.memoize(builder.loadElement(cb, j)))
          cb.assign(j, j + 1)
        })

        cb.assign(seenSoFar, other.seenSoFar)
        cb.assign(garbage, other.garbage)
        val tmpRegion = cb.newLocal[Region]("tmpRegion", region)
        cb.assign(r, other.region)
        cb.assign(other.r, tmpRegion)
        cb += tmpRegion.invoke[Unit]("invalidate")
        builder.cloneFrom(cb, other.builder)

      }, {
        val newBuilder = new StagedArrayBuilder(eltPType, kb, region)
        newBuilder.initializeWithCapacity(cb, maxSize)

        val totalWeightLeft = cb.newLocal("totalWeightLeft", seenSoFar.toD)
        val totalWeightRight = cb.newLocal("totalWeightRight", other.seenSoFar.toD)

        val leftSize = cb.newLocal[Int]("leftSize", builder.size)
        val rightSize = cb.newLocal[Int]("rightSize", other.builder.size)

        cb.assign(j, 0)
        cb.while_(j < maxSize, {
          val x = cb.memoize(rand.invoke[Double]("nextDouble"))
          cb.if_(x * (totalWeightLeft + totalWeightRight) <= totalWeightLeft, {

            val idxToSample = cb.memoize(rand.invoke[Int, Int]("nextInt", leftSize))
            builder.loadElement(cb, idxToSample).toI(cb).consume(cb,
              newBuilder.setMissing(cb),
              newBuilder.append(cb, _, false))
            cb.assign(leftSize, leftSize - 1)
            cb.assign(totalWeightLeft, totalWeightLeft - 1)
            cb.if_(idxToSample < leftSize, {
              builder.overwrite(cb, cb.memoize(builder.loadElement(cb, leftSize)), idxToSample, false)
            })
          }, {
            val idxToSample = cb.memoize(rand.invoke[Int, Int]("nextInt", rightSize))
            other.builder.loadElement(cb, idxToSample).toI(cb).consume(cb,
              newBuilder.setMissing(cb),
              newBuilder.append(cb, _, true))
            cb.assign(rightSize, rightSize - 1)
            cb.assign(totalWeightRight, totalWeightRight - 1)
            cb.if_(idxToSample < rightSize, {
              other.builder.overwrite(cb, cb.memoize(other.builder.loadElement(cb, rightSize)), idxToSample, false)
            })
          })
          cb.assign(j, j + 1)
        })
        builder.cloneFrom(cb, newBuilder)
        cb.assign(seenSoFar, seenSoFar + other.seenSoFar)
        cb.assign(garbage, garbage + leftSize.toL)
        cb.invokeVoid(gc, cb.this_)
      })
    })
  }

  def resultArray(cb: EmitCodeBuilder, region: Value[Region], resType: PCanonicalArray): SIndexablePointerValue = {
    resType.constructFromElements(cb, region, builder.size, deepCopy = true) { (cb, idx) =>
      builder.loadElement(cb, idx).toI(cb)
    }
  }

  def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
    cb.assign(maxSize, Region.loadInt(maxSizeOffset(src)))
    cb.assign(seenSoFar, Region.loadLong(elementsSeenOffset(src)))
    cb.assign(garbage, Region.loadLong(garbageOffset(src)))
    builder.copyFrom(cb, builderStateOffset(src))
  }
}

class ReservoirSampleAggregator(typ: VirtualTypeWithReq) extends StagedAggregator {
  type State = ReservoirSampleRVAS

  private val pt = typ.canonicalPType
  val resultPType: PCanonicalArray = PCanonicalArray(pt)
  val resultEmitType: EmitType = EmitType(SIndexablePointer(resultPType), true)
  val initOpTypes: Seq[Type] = Array(TInt32)
  val seqOpTypes: Seq[Type] = Array(typ.t)

  protected def _initOp(cb: EmitCodeBuilder, state: ReservoirSampleRVAS, init: Array[EmitCode]): Unit = {
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    sizeTriplet.toI(cb)
      .consume(cb,
        cb += Code._fatal[Unit](s"argument 'n' for 'hl.agg.reservoir_sample' may not be missing"),
        sc => state.init(cb, sc.asInt.value)
      )
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: ReservoirSampleRVAS, seq: Array[EmitCode]): Unit = {
    val Array(elt: EmitCode) = seq
    state.seqOp(cb, elt)
  }

  protected def _combOp(ctx: ExecuteContext, cb: EmitCodeBuilder, region: Value[Region], state: ReservoirSampleRVAS, other: ReservoirSampleRVAS): Unit = state.combine(cb, other)

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode = {
    // deepCopy is handled by state.resultArray
    IEmitCode.present(cb, state.resultArray(cb, region, resultPType))
  }
}
