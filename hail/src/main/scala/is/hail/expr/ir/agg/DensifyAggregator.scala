package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.virtual.{TInt32, Type}
import is.hail.utils._

object DensifyAggregator {
  val END_SERIALIZATION: Int = 0xf81ea4
}

class DensifyState(val arrayVType: VirtualTypeWithReq, val kb: EmitClassBuilder[_]) extends AggregatorState {
  val eltType: PType = {
    // FIXME: VirtualTypeWithReq needs better ergonomics
    arrayVType.canonicalPType.asInstanceOf[PCanonicalArray].elementType.setRequired(false)
  }

  private val r: ThisFieldRef[Region] = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r
  override val regionSize: Int = Region.REGULAR

  val maxRegionSize: Long = 512 * 1024 * 1024

  val arrayStorageType: PCanonicalArray = PCanonicalArray(eltType, required = true)
  val storageType: PType = PInt64Required // FIXME: there are incorrect loads in TupleAggState
  private val length = kb.genFieldThisRef[Int]("densify_len")
  private val arrayAddr = kb.genFieldThisRef[Long]("densify_addr")

  def newState(cb: EmitCodeBuilder, off: Code[Long]): Unit = {
    cb += region.getNewRegion(regionSize)
  }

  def createState(cb: EmitCodeBuilder): Unit =
    cb.ifx(region.isNull, {
      cb.assign(r, Region.stagedCreate(regionSize, kb.pool()))
    })

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, srcc: Code[Long]): Unit = {
    regionLoader(cb, r)
    cb.assign(arrayAddr, Region.loadAddress(srcc))
    cb.assign(length, arrayStorageType.loadLength(arrayAddr))
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, destc: Code[Long]): Unit = {
    regionStorer(cb, region)
    cb += region.invalidate()
    cb += Region.storeAddress(destc, arrayAddr)
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val enc = TypedCodecSpec(arrayStorageType, codec).buildTypedEmitEncoderF[Long](arrayStorageType, kb)
    (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) => {

      cb += enc(region, arrayAddr, ob)
      cb += ob.writeInt(const(DensifyAggregator.END_SERIALIZATION))
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val (decType, dec) = TypedCodecSpec(arrayStorageType, codec).buildEmitDecoderF[Long](kb)
    assert(decType == arrayStorageType)

    (cb: EmitCodeBuilder, ib: Value[InputBuffer]) => {
      cb.assign(arrayAddr, dec(region, ib))
      cb.assign(length, arrayStorageType.loadLength(arrayAddr))
      cb.ifx(ib.readInt().cne(const(DensifyAggregator.END_SERIALIZATION)),
        cb += Code._fatal[Unit](s"densify serialization failed"))
    }
  }

  def init(cb: EmitCodeBuilder, _maxSize: Code[Int]): Unit = {
    cb.assign(length, _maxSize)
    cb.assign(arrayAddr, arrayStorageType.allocate(region, length))
    cb += arrayStorageType.stagedInitialize(arrayAddr, length, setMissing = true)
  }

  private def gc(cb: EmitCodeBuilder): Unit = {
    cb.ifx(region.totalManagedBytes() > maxRegionSize, {
      val newRegion = cb.newLocal[Region]("densify_gc", Region.stagedCreate(regionSize, kb.pool()))
      cb.assign(arrayAddr, arrayStorageType.store(cb, newRegion, arrayStorageType.loadCheapPCode(cb, arrayAddr), deepCopy = true))
      cb += region.invalidate()
      cb.assign(r, newRegion)

    })
  }

  def seqOp(cb: EmitCodeBuilder, a: EmitCode): Unit = {
    a.toI(cb)
      .consume(cb,
        {
          /* do nothing if missing */
        },
        { sc =>
          val arr = sc.memoize(cb, "densify_seq_arr")
          arr.asInstanceOf[SIndexableValue].forEachDefined(cb) { case (cb, idx, element) =>
            cb += arrayStorageType.setElementPresent(arrayAddr, idx)
            eltType.storeAtAddress(cb, arrayStorageType.elementOffset(arrayAddr, length, idx), region, element, deepCopy = true)
          }
        })
    gc(cb)
  }

  def combine(cb: EmitCodeBuilder, other: DensifyState): Unit = {
    assert(other.arrayStorageType == this.arrayStorageType)
    val arr = arrayStorageType.loadCheapPCode(cb, other.arrayAddr).memoize(cb, "densify_comb_other")
    arr.asInstanceOf[SIndexableValue].forEachDefined(cb) { case (cb, idx, element) =>
      cb += arrayStorageType.setElementPresent(arrayAddr, idx)
      eltType.storeAtAddress(cb, arrayStorageType.elementOffset(arrayAddr, length, idx), region, element, deepCopy = true)
    }
    gc(cb)
  }

  def result(cb: EmitCodeBuilder, resultType: PArray, srvb: StagedRegionValueBuilder): Unit = {
    cb += srvb.addIRIntermediate(arrayStorageType.loadCheapPCode(cb, arrayAddr), deepCopy = true)
  }

  def copyFrom(cb: EmitCodeBuilder, srcCode: Code[Long]): Unit = {
    cb.assign(arrayAddr,
      arrayStorageType.store(cb,
        region,
        arrayStorageType.loadCheapPCode(cb, arrayStorageType.loadFromNested(cb, srcCode)),
        deepCopy = true))
    cb.assign(length, arrayStorageType.loadLength(arrayAddr))
  }
}


class DensifyAggregator(val arrayVType: VirtualTypeWithReq) extends StagedAggregator {
  type State = DensifyState

  private val pt = {
    // FIXME: VirtualTypeWithReq needs better ergonomics
    val eltType = arrayVType.canonicalPType.asInstanceOf[PCanonicalArray].elementType.setRequired(false)
    PCanonicalArray(eltType, required = true)
  }
  val resultType: PCanonicalArray = pt
  val initOpTypes: Seq[Type] = Array(TInt32)
  val seqOpTypes: Seq[Type] = Array(pt.virtualType)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    sizeTriplet.toI(cb)
      .consume(cb,
        cb += Code._fatal[Unit](s"argument 'n' for 'hl.agg.densify' may not be missing"),
        sc => state.init(cb, sc.asInt.intCode(cb))
      )
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt: EmitCode) = seq
    state.seqOp(cb, elt)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = state.combine(cb, other)

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    assert(srvb.currentPType().fundamentalType == state.arrayStorageType.fundamentalType,
      s"mismatch:\n  current=${ srvb.currentPType() }\n  state=  ${ state.arrayStorageType }\n  result= ${ resultType }")
    state.result(cb, resultType, srvb)
  }
}
