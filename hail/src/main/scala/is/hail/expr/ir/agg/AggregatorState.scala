package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{coerce, _}
import is.hail.expr.ir._
import is.hail.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.utils._

trait AggregatorState {
  def kb: EmitClassBuilder[_]

  def storageType: PType

  def regionSize: Int = Region.TINY

  def createState(cb: EmitCodeBuilder): Unit
  def newState(off: Code[Long]): Code[Unit]

  // null to safeguard against users of off
  def newState(): Code[Unit] = newState(null)

  def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit]
  def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit]

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

  def newState(off: Code[Long]): Code[Unit] = region.getNewRegion(const(regionSize))

  def createState(cb: EmitCodeBuilder): Unit =
    cb.ifx(region.isNull, cb.assign(r, Region.stagedCreate(regionSize)))

  def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] = regionLoader(r)

  def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    region.isValid.orEmpty(Code(regionStorer(region), region.invalidate()))
}

trait PointerBasedRVAState extends RegionBackedAggState {
  val off: Settable[Long] = kb.genFieldThisRef[Long]()
  val storageType: PType = PInt64(true)

  override val regionSize: Int = Region.TINIER

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    Code(super.load(regionLoader, src), off := Region.loadAddress(src))

  override def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    Code(region.isValid.orEmpty(Region.storeAddress(dest, off)), super.store(regionStorer, dest))

  def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = copyFromAddress(cb, Region.loadAddress(src))

  def copyFromAddress(cb: EmitCodeBuilder, src: Code[Long]): Unit
}

class TypedRegionBackedAggState(val typ: PType, val kb: EmitClassBuilder[_]) extends RegionBackedAggState {
  override val regionSize: Int = Region.TINIER
  val storageType: PTuple = PCanonicalTuple(required = true, typ)
  val off: Settable[Long] = kb.genFieldThisRef[Long]()

  override def newState(): Code[Unit] = Code(region.getNewRegion(const(regionSize)), off := storageType.allocate(region))
  override def newState(src: Code[Long]): Code[Unit] = Code(off := src, super.newState(off))
  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    Code(super.load(r => r.invalidate(), src), off := src)
  override def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    Code.memoize(dest, "trbas_dest") { dest =>
      Code(region.isValid.orEmpty(dest.cne(off).orEmpty(
        Region.copyFrom(off, dest, const(storageType.byteSize)))),
        super.store(regionStorer, dest))
    }

  def storeMissing(): Code[Unit] = storageType.setFieldMissing(off, 0)
  def storeNonmissing(v: Code[_]): Code[Unit] = Code(
    region.getNewRegion(const(regionSize)),
    storageType.setFieldPresent(off, 0),
    StagedRegionValueBuilder.deepCopy(kb, region, typ, v, storageType.fieldOffset(off, 0)))

  def get(): EmitCode = EmitCode(Code._empty,
    storageType.isFieldMissing(off, 0),
    PCode(typ, Region.loadIRIntermediate(typ)(storageType.fieldOffset(off, 0))))

  def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit =
    cb += Code(newState(off), StagedRegionValueBuilder.deepCopy(kb, region, storageType, src, off))

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val enc = TypedCodecSpec(storageType, codec).buildTypedEmitEncoderF[Long](storageType, kb)
    (cb, ob: Value[OutputBuffer]) => cb += enc(region, off, ob)
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val (t, dec) = TypedCodecSpec(storageType, codec).buildTypedEmitDecoderF[Long](storageType.virtualType, kb)
    val off2: Settable[Long] = kb.genFieldThisRef[Long]()
    (cb, ib: Value[InputBuffer]) => cb += Code(off2 := dec(region, ib), Region.copyFrom(off2, off, const(storageType.byteSize)))
  }
}

class PrimitiveRVAState(val types: Array[PType], val kb: EmitClassBuilder[_]) extends AggregatorState {
  type ValueField = (Option[Settable[Boolean]], Settable[_], PType)
  assert(types.forall(_.isPrimitive))

  val nFields: Int = types.length
  val fields: Array[ValueField] = Array.tabulate(nFields) { i =>
    val m = if (types(i).required) None else Some(kb.genFieldThisRef[Boolean](s"primitiveRVA_${i}_m"))
    val v = kb.genFieldThisRef(s"primitiveRVA_${i}_v")(typeToTypeInfo(types(i)))
    (m, v, types(i))
  }
  val storageType: PTuple = PCanonicalTuple(false, types: _*)

  def foreachField(f: (Int, ValueField) => Code[Unit]): Code[Unit] =
    Code(Array.tabulate(nFields)(i => f(i, fields(i))))

  def newState(off: Code[Long]): Code[Unit] = Code._empty
  def createState(cb: EmitCodeBuilder): Unit = {}

  private[this] def loadVarsFromRegion(src: Code[Long]): Code[Unit] =
    foreachField {
      case (i, (None, v, t)) =>
        v.storeAny(Region.loadPrimitive(t)(storageType.fieldOffset(src, i)))
      case (i, (Some(m), v, t)) =>
        Code.memoize(src, "prim_rvastate_load_vars_src") { src =>
          Code(
            m := storageType.isFieldMissing(src, i),
            m.mux(Code._empty,
              v.storeAny(Region.loadPrimitive(t)(storageType.fieldOffset(src, i)))))
        }
    }

  def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    loadVarsFromRegion(src)

  def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
      foreachField {
        case (i, (None, v, t)) =>
          Region.storePrimitive(t, storageType.fieldOffset(dest, i))(v)
        case (i, (Some(m), v, t)) =>
          Code.memoize(dest, "prim_rvastate_store_dest") { dest =>
            m.mux(storageType.setFieldMissing(dest, i),
              Code(storageType.setFieldPresent(dest, i),
                Region.storePrimitive(t, storageType.fieldOffset(dest, i))(v)))
          }
      }

  def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = cb += loadVarsFromRegion(src)

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    (cb, ob: Value[OutputBuffer]) =>
      cb += (foreachField {
        case (_, (None, v, t)) => ob.writePrimitive(t)(v)
        case (_, (Some(m), v, t)) => Code(
          ob.writeBoolean(m),
          m.mux(Code._empty, ob.writePrimitive(t)(v)))
      })
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    (cb, ib: Value[InputBuffer]) =>
      cb += (foreachField {
        case (_, (None, v, t)) =>
          v.storeAny(ib.readPrimitive(t))
        case (_, (Some(m), v, t)) => Code(
          m := ib.readBoolean(),
          m.mux(Code._empty, v.storeAny(ib.readPrimitive(t))))
      })
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

  def toCode(cb: EmitCodeBuilder, prefix: String, f: (EmitCodeBuilder, Int, AggregatorState) => Unit): Unit =
    cb.emb.wrapVoids(cb,
      Array.tabulate(nStates) { (i: Int) =>
        (cb: EmitCodeBuilder) => f(cb, i, states(i))
      },
      prefix
    )

  def toCodeWithArgs(cb: EmitCodeBuilder, prefix: String, argTypes: IndexedSeq[TypeInfo[_]],
    args: IndexedSeq[Code[_]], f: (EmitCodeBuilder, Int, AggregatorState, Seq[Code[_]]) => Unit
  ): Unit =
    cb.emb.ecb.wrapVoidsWithArgs(cb,
      Array.tabulate(nStates) { (i: Int) => { (cb: EmitCodeBuilder, args: Seq[Code[_]]) =>
        f(cb, i, states(i), args)
      }},
      prefix, argTypes, args)

  def createStates(cb: EmitCodeBuilder): Unit =
    toCode(cb, "create_states", (cb, i, s) => s.createState(cb))
}

class TupleAggregatorState(val kb: EmitClassBuilder[_], val states: StateTuple, val topRegion: Value[Region], val off: Value[Long], val rOff: Value[Int] = const(0)) {
  val storageType: PTuple = states.storageType
  private def getRegion(i: Int): Value[Region] => Code[Unit] = { r: Value[Region] =>
    r.setFromParentReference(topRegion, rOff + const(i), states(i).regionSize) }
  private def setRegion(i: Int): Value[Region] => Code[Unit] = { r: Value[Region] =>
    topRegion.setParentReference(r, rOff + const(i))
  }
  private def getStateOffset(i: Int): Code[Long] = storageType.loadField(off, i)

  def toCode(f: (Int, AggregatorState) => Code[Unit]): Code[Unit] =
    Code(Array.tabulate(states.nStates)(i => f(i, states(i))))

  def newState(i: Int): Code[Unit] = states(i).newState(getStateOffset(i))

  def newState(cb: EmitCodeBuilder): Unit = states.toCode(cb, "new_state", (cb, i, s) => cb += s.newState(getStateOffset(i)))

  def load(cb: EmitCodeBuilder): Unit =
    states.toCode(cb, "load", (cb, i, s) => cb += s.load(getRegion(i), getStateOffset(i)))

  def store(cb: EmitCodeBuilder): Unit =
    states.toCode(cb, "store", (cb, i, s) => cb += s.store(setRegion(i), getStateOffset(i)))

  def copyFrom(cb: EmitCodeBuilder, statesOffset: Code[Long]): Unit = {
    states.toCodeWithArgs(cb, "copy_from", Array[TypeInfo[_]](LongInfo),
      Array(statesOffset),
      { case (cb, i, s, Seq(o: Code[Long@unchecked])) => s.copyFrom(cb, storageType.loadField(o, i)) })
  }
}
