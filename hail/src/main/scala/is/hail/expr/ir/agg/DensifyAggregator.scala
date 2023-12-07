package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s.{Code, _}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitContext, IEmitCode}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.virtual.{TInt32, Type}
import is.hail.utils._

object DensifyAggregator {
  val END_SERIALIZATION: Int = 0xf81ea4
}

class DensifyState(val arrayVType: VirtualTypeWithReq, val kb: EmitClassBuilder[_])
    extends AggregatorState {
  val eltType: PType =
    // FIXME: VirtualTypeWithReq needs better ergonomics
    arrayVType.canonicalPType.asInstanceOf[PCanonicalArray].elementType.setRequired(false)

  private val r: ThisFieldRef[Region] = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r
  override val regionSize: Int = Region.REGULAR

  val maxRegionSize: Long = 512 * 1024 * 1024

  val arrayStorageType: PCanonicalArray = PCanonicalArray(eltType, required = true)
  val storageType: PType = PInt64Required // FIXME: there are incorrect loads in TupleAggState
  private val length = kb.genFieldThisRef[Int]("densify_len")
  private val arrayAddr = kb.genFieldThisRef[Long]("densify_addr")

  def newState(cb: EmitCodeBuilder, off: Value[Long]): Unit =
    cb += region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit =
    cb.if_(region.isNull, cb.assign(r, Region.stagedCreate(regionSize, kb.pool())))

  override def load(
    cb: EmitCodeBuilder,
    regionLoader: (EmitCodeBuilder, Value[Region]) => Unit,
    src: Value[Long],
  ): Unit = {
    regionLoader(cb, r)
    cb.assign(arrayAddr, Region.loadAddress(src))
    cb.assign(length, arrayStorageType.loadLength(arrayAddr))
  }

  override def store(
    cb: EmitCodeBuilder,
    regionStorer: (EmitCodeBuilder, Value[Region]) => Unit,
    dest: Value[Long],
  ): Unit = {
    regionStorer(cb, region)
    cb += region.invalidate()
    cb += Region.storeAddress(dest, arrayAddr)
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val codecSpec = TypedCodecSpec(arrayStorageType, codec)
    (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) => {

      val arrayCode = arrayStorageType.loadCheapSCode(cb, arrayAddr)
      codecSpec.encodedType.buildEncoder(arrayCode.st, kb)
        .apply(cb, arrayCode, ob)
      cb += ob.writeInt(const(DensifyAggregator.END_SERIALIZATION))
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val codecSpec = TypedCodecSpec(arrayStorageType, codec)

    (cb: EmitCodeBuilder, ib: Value[InputBuffer]) => {

      val decValue = codecSpec.encodedType.buildDecoder(arrayStorageType.virtualType, kb)
        .apply(cb, region, ib)

      cb.assign(arrayAddr, arrayStorageType.store(cb, region, decValue, deepCopy = false))
      cb.assign(length, arrayStorageType.loadLength(arrayAddr))
      cb.if_(
        ib.readInt().cne(const(DensifyAggregator.END_SERIALIZATION)),
        cb += Code._fatal[Unit](s"densify serialization failed"),
      )
    }
  }

  def init(cb: EmitCodeBuilder, _maxSize: Code[Int]): Unit = {
    cb.assign(length, _maxSize)
    cb.assign(arrayAddr, arrayStorageType.allocate(region, length))
    arrayStorageType.stagedInitialize(cb, arrayAddr, length, setMissing = true)
  }

  private def gc(cb: EmitCodeBuilder): Unit = {
    cb.if_(
      region.totalManagedBytes() > maxRegionSize, {
        val newRegion =
          cb.newLocal[Region]("densify_gc", Region.stagedCreate(regionSize, kb.pool()))
        cb.assign(
          arrayAddr,
          arrayStorageType.store(
            cb,
            newRegion,
            arrayStorageType.loadCheapSCode(cb, arrayAddr),
            deepCopy = true,
          ),
        )
        cb += region.invalidate()
        cb.assign(r, newRegion)

      },
    )
  }

  def seqOp(cb: EmitCodeBuilder, a: EmitCode): Unit = {
    a.toI(cb)
      .consume(
        cb, {
          /* do nothing if missing */
        },
        arr =>
          arr.asIndexable.forEachDefined(cb) { case (cb, idx, element) =>
            arrayStorageType.setElementPresent(cb, arrayAddr, idx)
            eltType.storeAtAddress(
              cb,
              arrayStorageType.elementOffset(arrayAddr, length, idx),
              region,
              element,
              deepCopy = true,
            )
          },
      )
    gc(cb)
  }

  def combine(cb: EmitCodeBuilder, other: DensifyState): Unit = {
    assert(other.arrayStorageType == this.arrayStorageType)
    val arr = arrayStorageType.loadCheapSCode(cb, other.arrayAddr)
    arr.asInstanceOf[SIndexableValue].forEachDefined(cb) { case (cb, idx, element) =>
      arrayStorageType.setElementPresent(cb, arrayAddr, idx)
      eltType.storeAtAddress(
        cb,
        arrayStorageType.elementOffset(arrayAddr, length, idx),
        region,
        element,
        deepCopy = true,
      )
    }
    gc(cb)
  }

  def result(cb: EmitCodeBuilder, region: Value[Region]): SIndexablePointerValue =
    arrayStorageType.loadCheapSCode(cb, arrayAddr)

  def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
    cb.assign(
      arrayAddr,
      arrayStorageType.store(
        cb,
        region,
        arrayStorageType.loadCheapSCode(cb, arrayStorageType.loadFromNested(src)),
        deepCopy = true,
      ),
    )
    cb.assign(length, arrayStorageType.loadLength(arrayAddr))
  }
}

class DensifyAggregator(val arrayVType: VirtualTypeWithReq) extends StagedAggregator {
  type State = DensifyState

  private val pt = {
    // FIXME: VirtualTypeWithReq needs better ergonomics
    val eltType =
      arrayVType.canonicalPType.asInstanceOf[PCanonicalArray].elementType.setRequired(false)
    PCanonicalArray(eltType)
  }

  val resultEmitType: EmitType = EmitType(SIndexablePointer(pt), true)
  val initOpTypes: Seq[Type] = Array(TInt32)
  val seqOpTypes: Seq[Type] = Array(resultEmitType.virtualType)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    sizeTriplet.toI(cb)
      .consume(
        cb,
        cb += Code._fatal[Unit](s"argument 'n' for 'hl.agg.densify' may not be missing"),
        sc => state.init(cb, sc.asInt.value),
      )
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt: EmitCode) = seq
    state.seqOp(cb, elt)
  }

  protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: DensifyState,
    other: DensifyState,
  ): Unit = state.combine(cb, other)

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode = {
    val resultInWrongRegion = state.result(cb, region)
    // deepCopy needs to be done here
    val ptrInRightRegion = pt.store(cb, region, resultInWrongRegion, true)
    IEmitCode.present(cb, pt.loadCheapSCode(cb, ptrInRightRegion))
  }
}
