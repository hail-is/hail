package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.SStackStruct
import is.hail.types.physical.stypes.interfaces.SBinaryValue
import is.hail.utils._

trait AggregatorState {
  def kb: EmitClassBuilder[_]

  def storageType: PType

  def regionSize: Int = Region.TINY

  def createState(cb: EmitCodeBuilder): Unit

  def newState(cb: EmitCodeBuilder, off: Value[Long]): Unit

  // null to safeguard against users of off
  def newState(cb: EmitCodeBuilder): Unit = newState(cb, null)

  def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Value[Long]): Unit

  def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Value[Long]): Unit

  def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit

  def deserializeFromBytes(cb: EmitCodeBuilder, bytes: SBinaryValue): Unit = {
    val lazyBuffer = kb.getOrDefineLazyField[MemoryBufferWrapper](Code.newInstance[MemoryBufferWrapper](), ("AggregatorStateBufferWrapper"))
    cb += lazyBuffer.invoke[Array[Byte], Unit]("set", bytes.loadBytes(cb))
    val ib = cb.memoize(lazyBuffer.invoke[InputBuffer]("buffer"))
    deserialize(BufferSpec.blockedUncompressed)(cb, ib)
    cb += lazyBuffer.invoke[Unit]("invalidate")
  }

  def serializeToRegion(cb: EmitCodeBuilder, t: PBinary, r: Value[Region]): SValue = {
    val lazyBuffer = kb.getOrDefineLazyField[MemoryWriterWrapper](Code.newInstance[MemoryWriterWrapper](), ("AggregatorStateWriterWrapper"))
    val addr = kb.genFieldThisRef[Long]("addr")
    cb += lazyBuffer.invoke[Unit]("clear")
    val ob = cb.memoize(lazyBuffer.invoke[OutputBuffer]("buffer"))
    serialize(BufferSpec.blockedUncompressed)(cb, ob)
    cb.assign(addr, t.allocate(r, lazyBuffer.invoke[Int]("length")))
    t.storeLength(cb, addr, lazyBuffer.invoke[Int]("length"))
    cb += lazyBuffer.invoke[Long, Unit]("copyToAddress", t.bytesAddress(addr))

    t.loadCheapSCode(cb, addr)
  }
}

trait RegionBackedAggState extends AggregatorState {
  protected val r: Settable[Region] = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r

  def newState(cb: EmitCodeBuilder, off: Value[Long]): Unit = cb += region.getNewRegion(const(regionSize))

  def createState(cb: EmitCodeBuilder): Unit = {
    cb.if_(region.isNull, cb.assign(r, Region.stagedCreate(regionSize, kb.pool())))
  }

  def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Value[Long]): Unit = regionLoader(cb, r)

  def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Value[Long]): Unit =
    cb.if_(region.isValid,
      {
        regionStorer(cb, region)
        cb += region.invalidate()
      })
}

trait PointerBasedRVAState extends RegionBackedAggState {
  val off: Settable[Long] = kb.genFieldThisRef[Long]()
  val storageType: PType = PInt64(true)

  override val regionSize: Int = Region.TINIER

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Value[Long]): Unit = {
    super.load(cb, regionLoader, src)
    cb.assign(off, Region.loadAddress(src))
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Value[Long]): Unit = {
    cb.if_(region.isValid,
      {
        cb += Region.storeAddress(dest, off)
        super.store(cb, regionStorer, dest)
      })
  }

  def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit =
    copyFromAddress(cb, cb.memoize(Region.loadAddress(src)))

  def copyFromAddress(cb: EmitCodeBuilder, src: Value[Long]): Unit
}

class TypedRegionBackedAggState(val typ: VirtualTypeWithReq, val kb: EmitClassBuilder[_]) extends AbstractTypedRegionBackedAggState(typ.canonicalPType)

abstract class AbstractTypedRegionBackedAggState(val ptype: PType) extends RegionBackedAggState {

  override val regionSize: Int = Region.TINIER
  val storageType: PTuple = PCanonicalTuple(required = true, ptype)
  val off: Settable[Long] = kb.genFieldThisRef[Long]()

  override def newState(cb: EmitCodeBuilder): Unit = {
    cb += region.getNewRegion(const(regionSize))
    cb.assign(off, storageType.allocate(region))
  }

  override def newState(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
    cb.assign(off, src)
    super.newState(cb, off)
  }

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Value[Long]): Unit = {
    super.load(cb, { (cb: EmitCodeBuilder, r: Value[Region]) => cb += r.invalidate() }, src)
    cb.assign(off, src)
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Value[Long]): Unit = {
    cb.if_(region.isValid,
      cb.if_(dest.cne(off),
        cb += Region.copyFrom(off, dest, const(storageType.byteSize))))
    super.store(cb, regionStorer, dest)
  }

  def storeMissing(cb: EmitCodeBuilder): Unit = {
    storageType.setFieldMissing(cb, off, 0)
  }

  def storeNonmissing(cb: EmitCodeBuilder, sc: SValue): Unit = {
    cb += region.getNewRegion(const(regionSize))
    storageType.setFieldPresent(cb, off, 0)
    ptype.storeAtAddress(cb, storageType.fieldOffset(off, 0), region, sc, deepCopy = true)
  }

  def get(cb: EmitCodeBuilder): IEmitCode = {
    IEmitCode(cb, storageType.isFieldMissing(cb, off, 0), ptype.loadCheapSCode(cb, storageType.loadField(off, 0)))
  }

  def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
    newState(cb, off)
    storageType.storeAtAddress(cb, off, region, storageType.loadCheapSCode(cb, src), deepCopy = true)
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val codecSpec = TypedCodecSpec(storageType, codec)
    val enc = codecSpec.encodedType.buildEncoder(storageType.sType, kb)
    (cb, ob: Value[OutputBuffer]) => enc(cb, storageType.loadCheapSCode(cb, off), ob)
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val codecSpec = TypedCodecSpec(storageType, codec)

    val dec = codecSpec.encodedType.buildDecoder(storageType.virtualType, kb)
    (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      storageType.storeAtAddress(cb, off, region, dec(cb, region, ib), deepCopy = false)
  }
}

class PrimitiveRVAState(val vtypes: Array[VirtualTypeWithReq], val kb: EmitClassBuilder[_]) extends AggregatorState {
  private[this] val emitTypes = vtypes.map(_.canonicalEmitType)
  assert(emitTypes.forall(_.st.isPrimitive))

  val nFields: Int = emitTypes.length
  val fields: Array[EmitSettable] = Array.tabulate(nFields) { i => kb.newEmitField(s"primitiveRVA_${ i }_v", emitTypes(i)) }
  val storageType = PCanonicalTuple(true, emitTypes.map(_.typeWithRequiredness.canonicalPType): _*)
  val sStorageType = storageType.sType

  def foreachField(f: (Int, EmitSettable) => Unit): Unit = {
    (0 until nFields).foreach { i =>
      f(i, fields(i))
    }
  }

  def newState(cb: EmitCodeBuilder, off: Value[Long]): Unit = {}

  def createState(cb: EmitCodeBuilder): Unit = {}

  private[this] def loadVarsFromRegion(cb: EmitCodeBuilder, srcc: Code[Long]): Unit = {
    val pv = storageType.loadCheapSCode(cb, srcc)
    foreachField { (i, es) =>
      cb.assign(es, pv.loadField(cb, i))
    }
  }

  def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Value[Long]): Unit = {
    loadVarsFromRegion(cb, src)
  }

  def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Value[Long]): Unit = {
    storageType.storeAtAddress(cb,
      dest,
      null,
      SStackStruct.constructFromArgs(cb, null, storageType.virtualType, fields.map(_.load): _*),
      false)
  }

  def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit = loadVarsFromRegion(cb, src)

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    (cb, ob: Value[OutputBuffer]) =>
      foreachField { case (_, es) =>
        if (es.emitType.required) {
          ob.writePrimitive(cb, es.get(cb))
        } else {
          es.toI(cb).consume(cb,
            cb += ob.writeBoolean(true),
            { sc =>
              cb += ob.writeBoolean(false)
              ob.writePrimitive(cb, sc)
            })
        }
      }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    (cb, ib: Value[InputBuffer]) =>
      foreachField { case (_, es) =>
        if (es.emitType.required) {
          cb.assign(es, EmitCode.present(cb.emb, ib.readPrimitive(cb, es.st.virtualType)))
        } else {
          cb.if_(ib.readBoolean(),
            cb.assign(es, EmitCode.missing(cb.emb, es.st)),
            cb.assign(es, EmitCode.present(cb.emb, ib.readPrimitive(cb, es.st.virtualType))))
        }
      }
  }
}

case class StateTuple(states: Array[AggregatorState]) {
  val nStates: Int = states.length
  val storageType: PTuple = PCanonicalTuple(true, states.map { s => s.storageType }: _*)

  def apply(i: Int): AggregatorState = {
    if (i >= states.length)
      throw new RuntimeException(s"tried to access state $i, but there are only ${ states.length } states")
    states(i)
  }

  def toCode(f: (Int, AggregatorState) => Unit): Unit = {
    (0 until nStates).foreach { i =>
      f(i, states(i))
    }
  }

  def toCodeWithArgs(
    cb: EmitCodeBuilder, f: (EmitCodeBuilder, Int, AggregatorState) => Unit
  ): Unit = {
    (0 until nStates).foreach { i =>
      f(cb, i, states(i))
    }
  }

  def createStates(cb: EmitCodeBuilder): Unit =
    toCode((i, s) => s.createState(cb))
}

class TupleAggregatorState(val kb: EmitClassBuilder[_], val states: StateTuple, val topRegion: Value[Region], val off: Value[Long], val rOff: Value[Int] = const(0)) {
  val storageType: PTuple = states.storageType

  private def getRegion(i: Int): (EmitCodeBuilder, Value[Region]) => Unit = { (cb: EmitCodeBuilder, r: Value[Region]) =>
    cb += r.setFromParentReference(topRegion, rOff + const(i), states(i).regionSize)
  }

  private def setRegion(i: Int): (EmitCodeBuilder, Value[Region]) => Unit = { (cb: EmitCodeBuilder, r: Value[Region]) =>
    cb += topRegion.setParentReference(r, rOff + const(i))
  }

  private def getStateOffset(cb: EmitCodeBuilder, i: Int): Value[Long] = cb.memoize(storageType.loadField(off, i))

  def toCode(f: (Int, AggregatorState) => Unit): Unit =
    (0 until states.nStates).foreach(i => f(i, states(i)))

  def newState(cb: EmitCodeBuilder, i: Int): Unit = states(i).newState(cb, getStateOffset(cb, i))

  def newState(cb: EmitCodeBuilder): Unit = states.toCode((i, s) => s.newState(cb, getStateOffset(cb, i)))

  def load(cb: EmitCodeBuilder): Unit =
    states.toCode((i, s) => s.load(cb, getRegion(i), getStateOffset(cb, i)))

  def store(cb: EmitCodeBuilder): Unit = {
    states.toCode((i, s) => s.store(cb, setRegion(i), getStateOffset(cb, i)))
  }

  def copyFrom(cb: EmitCodeBuilder, statesOffset: Value[Long]): Unit = {
    states.toCodeWithArgs(cb,
      { case (cb, i, s) => s.copyFrom(cb, cb.memoize(storageType.loadField(statesOffset, i))) })
  }
}
