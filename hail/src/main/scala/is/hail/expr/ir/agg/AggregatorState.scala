package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SBaseStructPointerCode}
import is.hail.utils._

trait AggregatorState {
  def kb: EmitClassBuilder[_]

  def storageType: PType

  def regionSize: Int = Region.TINY

  def createState(cb: EmitCodeBuilder): Unit

  def newState(cb: EmitCodeBuilder, off: Code[Long]): Unit

  // null to safeguard against users of off
  def newState(cb: EmitCodeBuilder): Unit = newState(cb, null)

  def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Code[Long]): Unit

  def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Code[Long]): Unit

  def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit

  def deserializeFromBytes(cb: EmitCodeBuilder, t: PBinary, address: Code[Long]): Unit = {
    val lazyBuffer = kb.getOrDefineLazyField[MemoryBufferWrapper](Code.newInstance[MemoryBufferWrapper](), (this, "bufferWrapper"))
    val addr = cb.newField[Long]("addr", address)
    cb += lazyBuffer.invoke[Long, Int, Unit]("clearAndSetFrom", t.bytesAddress(addr), t.loadLength(addr))
    val ib = cb.newLocal("aggstate_deser_from_bytes_ib", lazyBuffer.invoke[InputBuffer]("buffer"))
    deserialize(BufferSpec.defaultUncompressed)(cb, ib)
  }

  def serializeToRegion(cb: EmitCodeBuilder, t: PBinary, r: Code[Region]): Code[Long] = {
    val lazyBuffer = kb.getOrDefineLazyField[MemoryWriterWrapper](Code.newInstance[MemoryWriterWrapper](), (this, "writerWrapper"))
    val addr = kb.genFieldThisRef[Long]("addr")
    cb += lazyBuffer.invoke[Unit]("clear")
    val ob = cb.newLocal("aggstate_ser_to_region_ob", lazyBuffer.invoke[OutputBuffer]("buffer"))
    serialize(BufferSpec.defaultUncompressed)(cb, ob)
    cb.assign(addr, t.allocate(r, lazyBuffer.invoke[Int]("length")))
    cb += t.storeLength(addr, lazyBuffer.invoke[Int]("length"))
    cb += lazyBuffer.invoke[Long, Unit]("copyToAddress", t.bytesAddress(addr))

    addr
  }
}

trait RegionBackedAggState extends AggregatorState {
  protected val r: Settable[Region] = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r

  def newState(cb: EmitCodeBuilder, off: Code[Long]): Unit = cb += region.getNewRegion(const(regionSize))

  def createState(cb: EmitCodeBuilder): Unit = {
    cb.ifx(region.isNull, cb.assign(r, Region.stagedCreate(regionSize, kb.pool())))
  }

  def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Code[Long]): Unit = regionLoader(cb, r)

  def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Code[Long]): Unit =
    cb.ifx(region.isValid,
      {
        regionStorer(cb, region)
        cb += region.invalidate()
      })
}

trait PointerBasedRVAState extends RegionBackedAggState {
  val off: Settable[Long] = kb.genFieldThisRef[Long]()
  val storageType: PType = PInt64(true)

  override val regionSize: Int = Region.TINIER

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Code[Long]): Unit = {
    super.load(cb, regionLoader, src)
    cb.assign(off, Region.loadAddress(src))
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Code[Long]): Unit = {
    cb.ifx(region.isValid,
      {
        cb += Region.storeAddress(dest, off)
        super.store(cb, regionStorer, dest)
      })
  }

  def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = copyFromAddress(cb, Region.loadAddress(src))

  def copyFromAddress(cb: EmitCodeBuilder, src: Code[Long]): Unit
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

  override def newState(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    cb.assign(off, src)
    super.newState(cb, off)
  }

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Code[Long]): Unit = {
    super.load(cb, { (cb: EmitCodeBuilder, r: Value[Region]) => cb += r.invalidate() }, src)
    cb.assign(off, src)
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, destc: Code[Long]): Unit = {
    val dest = cb.newLocal[Long]("trbas_dest", destc)
    cb.ifx(region.isValid,
      cb.ifx(dest.cne(off),
        cb += Region.copyFrom(off, dest, const(storageType.byteSize))))
    super.store(cb, regionStorer, dest)
  }

  def storeMissing(cb: EmitCodeBuilder): Unit = {
    cb += storageType.setFieldMissing(off, 0)
  }

  def storeNonmissing(cb: EmitCodeBuilder, sc: SCode): Unit = {
    cb += region.getNewRegion(const(regionSize))
    cb += storageType.setFieldPresent(off, 0)
    ptype.storeAtAddress(cb, storageType.fieldOffset(off, 0), region, sc, deepCopy = true)
  }

  def get(cb: EmitCodeBuilder): IEmitCode = {
    IEmitCode(cb, storageType.isFieldMissing(off, 0), ptype.loadCheapPCode(cb, storageType.loadField(off, 0)))
  }

  def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    newState(cb, off)
    storageType.storeAtAddress(cb, off, region, storageType.loadCheapPCode(cb, src), deepCopy = true)
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val codecSpec = TypedCodecSpec(storageType, codec)
    val enc = codecSpec.encodedType.buildEncoder(storageType.sType, kb)
    (cb, ob: Value[OutputBuffer]) => enc(cb, storageType.loadCheapPCode(cb, off), ob)
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val codecSpec = TypedCodecSpec(storageType, codec)

    val dec = codecSpec.encodedType.buildDecoder(storageType.virtualType, kb)
    ((cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      storageType.storeAtAddress(cb, off, region, dec(cb, region, ib), deepCopy = false))
  }
}

class PrimitiveRVAState(val vtypes: Array[VirtualTypeWithReq], val kb: EmitClassBuilder[_]) extends AggregatorState {
  private[this] val ptypes = vtypes.map(_.canonicalPType)
  assert(ptypes.forall(_.isPrimitive))

  val nFields: Int = ptypes.length
  val fields: Array[EmitSettable] = Array.tabulate(nFields) { i => kb.newEmitField(s"primitiveRVA_${ i }_v", ptypes(i)) }
  val storageType = PCanonicalTuple(true, ptypes: _*)
  val sStorageType = SBaseStructPointer(storageType)

  def foreachField(f: (Int, EmitSettable) => Unit): Unit = {
    (0 until nFields).foreach { i =>
      f(i, fields(i))
    }
  }

  def newState(cb: EmitCodeBuilder, off: Code[Long]): Unit = {}

  def createState(cb: EmitCodeBuilder): Unit = {}

  private[this] def loadVarsFromRegion(cb: EmitCodeBuilder, srcc: Code[Long]): Unit = {
    val pv = new SBaseStructPointerCode(sStorageType, srcc).memoize(cb, "prim_rvastate_load_vars")
    foreachField { (i, es) =>
      cb.assign(es, pv.loadField(cb, i).map(cb)(_.asPCode))
    }
  }

  def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Code[Long]): Unit = {
    loadVarsFromRegion(cb, src)
  }

  def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, destc: Code[Long]): Unit = {
    val dest = cb.newLocal("prim_rvastate_store_dest", destc)
    storageType.storeAtAddressFromFields(cb, dest, null, fields.map(_.load), false)
  }

  def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = loadVarsFromRegion(cb, src)

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    (cb, ob: Value[OutputBuffer]) =>
      foreachField { case (_, es) =>
        if (es.pt.required) {
          cb += ob.writePrimitive(es.pt)(es.v)
        } else {
          cb += ob.writeBoolean(es.m)
          cb.ifx(!es.m, cb += ob.writePrimitive(es.pt)(es.v))
        }
      }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    (cb, ib: Value[InputBuffer]) =>
      foreachField { case (_, es) =>
        if (es.pt.required) {
          cb.assign(es, EmitCode.present(cb.emb, PCode(es.pt, ib.readPrimitive(es.pt))))
        } else {
          cb.ifx(ib.readBoolean(),
            cb.assign(es, EmitCode.missing(cb.emb, es.pt)),
            cb.assign(es, EmitCode.present(cb.emb, PCode(es.pt, ib.readPrimitive(es.pt)))))
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
    cb: EmitCodeBuilder, args: IndexedSeq[Code[_]], f: (EmitCodeBuilder, Int, AggregatorState, Seq[Code[_]]) => Unit
  ): Unit = {
    val targs = args.zipWithIndex.map { case (arg, i) =>
      cb.newLocalAny(s"astcwa_arg$i", arg)(arg.ti)
    }
    (0 until nStates).foreach { i =>
      f(cb, i, states(i), targs.map(_.get))
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

  private def getStateOffset(i: Int): Code[Long] = storageType.loadField(off, i)

  def toCode(f: (Int, AggregatorState) => Unit): Unit =
    (0 until states.nStates).foreach(i => f(i, states(i)))

  def newState(cb: EmitCodeBuilder, i: Int): Unit = states(i).newState(cb, getStateOffset(i))

  def newState(cb: EmitCodeBuilder): Unit = states.toCode((i, s) => s.newState(cb, getStateOffset(i)))

  def load(cb: EmitCodeBuilder): Unit =
    states.toCode((i, s) => s.load(cb, getRegion(i), getStateOffset(i)))

  def store(cb: EmitCodeBuilder): Unit = {
    states.toCode((i, s) => s.store(cb, setRegion(i), getStateOffset(i)))
  }

  def copyFrom(cb: EmitCodeBuilder, statesOffset: Code[Long]): Unit = {
    states.toCodeWithArgs(cb,
      Array(statesOffset),
      { case (cb, i, s, Seq(o: Code[Long@unchecked])) => s.copyFrom(cb, storageType.loadField(o, i)) })
  }
}
