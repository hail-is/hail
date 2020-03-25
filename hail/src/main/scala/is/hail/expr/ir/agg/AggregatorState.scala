package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{coerce, _}
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.utils._

trait AggregatorState {
  def cb: EmitClassBuilder[_]

  def storageType: PType

  def regionSize: Int = Region.TINY

  def createState: Code[Unit]
  def newState(off: Code[Long]): Code[Unit]

  def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit]
  def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit]

  def copyFrom(src: Code[Long]): Code[Unit]

  def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit]

  def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit]

  def deserializeFromBytes(t: PBinary, address: Code[Long]): Code[Unit] = {
    val lazyBuffer = cb.getOrDefineLazyField[MemoryBufferWrapper](Code.newInstance[MemoryBufferWrapper](), (this, "buffer"))
    val addr = cb.genFieldThisRef[Long]("addr")
    Code(addr := address,
      lazyBuffer.invoke[Long, Int, Unit]("clearAndSetFrom", t.bytesAddress(addr), t.loadLength(addr)),
      Code.memoize(lazyBuffer.invoke[InputBuffer]("buffer"), "aggstate_deser_from_bytes_ib") { ib =>
        deserialize(BufferSpec.defaultUncompressed)(ib)
      })
  }

  def serializeToRegion(t: PBinary, r: Code[Region]): Code[Long] = {
    val lazyBuffer = cb.getOrDefineLazyField[MemoryWriterWrapper](Code.newInstance[MemoryWriterWrapper](), (this, "buffer"))
    val addr = cb.genFieldThisRef[Long]("addr")
    Code(
      lazyBuffer.invoke[Unit]("clear"),
      Code.memoize(lazyBuffer.invoke[OutputBuffer]("buffer"), "aggstate_ser_to_region_ob") { ob =>
        serialize(BufferSpec.defaultUncompressed)(ob)
      },
      addr := t.allocate(r, lazyBuffer.invoke[Int]("length")),
      t.storeLength(addr, lazyBuffer.invoke[Int]("length")),
      lazyBuffer.invoke[Long, Unit]("copyToAddress", t.bytesAddress(addr)),
      addr)
  }
}

trait RegionBackedAggState extends AggregatorState {
  protected val r: Settable[Region] = cb.genFieldThisRef[Region]()
  val region: Value[Region] = r

  def newState(off: Code[Long]): Code[Unit] = region.getNewRegion(const(regionSize))

  def createState: Code[Unit] = region.isNull.mux(r := Region.stagedCreate(regionSize), Code._empty)

  def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] = regionLoader(r)

  def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    region.isValid.orEmpty(Code(regionStorer(region), region.invalidate()))
}

trait PointerBasedRVAState extends RegionBackedAggState {
  val off: Settable[Long] = cb.genFieldThisRef[Long]()
  val storageType: PType = PInt64(true)

  override val regionSize: Int = Region.TINIER

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    Code(super.load(regionLoader, src), off := Region.loadAddress(src))

  override def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    Code(region.isValid.orEmpty(Region.storeAddress(dest, off)), super.store(regionStorer, dest))

  def copyFrom(src: Code[Long]): Code[Unit] = copyFromAddress(Region.loadAddress(src))

  def copyFromAddress(src: Code[Long]): Code[Unit]
}

class TypedRegionBackedAggState(val typ: PType, val cb: EmitClassBuilder[_]) extends RegionBackedAggState {
  override val regionSize: Int = Region.TINIER
  val storageType: PTuple = PTuple(required = true, typ)
  val off: Settable[Long] = cb.genFieldThisRef[Long]()

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
    StagedRegionValueBuilder.deepCopy(cb, region, typ, v, storageType.fieldOffset(off, 0)))

  def get(): EmitCode = EmitCode(Code._empty,
    storageType.isFieldMissing(off, 0),
    PCode(typ, Region.loadIRIntermediate(typ)(storageType.fieldOffset(off, 0))))

  def copyFrom(src: Code[Long]): Code[Unit] =
    Code(newState(off), StagedRegionValueBuilder.deepCopy(cb, region, storageType, src, off))

  def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit] = {
    val enc = TypedCodecSpec(storageType, codec).buildEmitEncoderF[Long](storageType, cb)
    ob: Value[OutputBuffer] => enc(region, off, ob)
  }

  def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit] = {
    val (t, dec) = TypedCodecSpec(storageType, codec).buildEmitDecoderF[Long](storageType.virtualType, cb)
    val off2: Settable[Long] = cb.genFieldThisRef[Long]()
    ib: Value[InputBuffer] => Code(off2 := dec(region, ib), Region.copyFrom(off2, off, const(storageType.byteSize)))
  }
}

class PrimitiveRVAState(val types: Array[PType], val cb: EmitClassBuilder[_]) extends AggregatorState {
  type ValueField = (Option[Settable[Boolean]], Settable[_], PType)
  assert(types.forall(_.isPrimitive))

  val nFields: Int = types.length
  val fields: Array[ValueField] = Array.tabulate(nFields) { i =>
    val m = if (types(i).required) None else Some(cb.genFieldThisRef[Boolean](s"primitiveRVA_${i}_m"))
    val v = cb.genFieldThisRef(s"primitiveRVA_${i}_v")(typeToTypeInfo(types(i)))
    (m, v, types(i))
  }
  val storageType: PTuple = PTuple(types: _*)

  def foreachField(f: (Int, ValueField) => Code[Unit]): Code[Unit] =
    Code(Array.tabulate(nFields)(i => f(i, fields(i))))

  def newState(off: Code[Long]): Code[Unit] = Code._empty
  def createState: Code[Unit] = Code._empty

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

  def copyFrom(src: Code[Long]): Code[Unit] = loadVarsFromRegion(src)

  def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit] = {
    ob: Value[OutputBuffer] =>
      foreachField {
        case (_, (None, v, t)) => ob.writePrimitive(t)(v)
        case (_, (Some(m), v, t)) => Code(
          ob.writeBoolean(m),
          m.mux(Code._empty, ob.writePrimitive(t)(v)))
      }
  }

  def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit] = {
    ib: Value[InputBuffer] =>
      foreachField {
        case (_, (None, v, t)) =>
          v.storeAny(ib.readPrimitive(t))
        case (_, (Some(m), v, t)) => Code(
          m := ib.readBoolean(),
          m.mux(Code._empty, v.storeAny(ib.readPrimitive(t))))
      }
  }
}

case class StateTuple(states: Array[AggregatorState]) {
  val nStates: Int = states.length
  val storageType: PTuple = PTuple(true, states.map { s => s.storageType }: _*)

  def apply(i: Int): AggregatorState = {
    if (i >= states.length)
      throw new RuntimeException(s"tried to access state $i, but there are only ${ states.length } states")
    states(i)
  }

  def toCode(cb: EmitClassBuilder[_], prefix: String, f: (Int, AggregatorState) => Code[Unit]): Code[Unit] =
    cb.wrapVoids(Array.tabulate(nStates)((i: Int) => f(i, states(i))), prefix)

  def toCodeWithArgs(cb: EmitClassBuilder[_], prefix: String, argTypes: IndexedSeq[TypeInfo[_]], args: IndexedSeq[Code[_]],
    f: (Int, AggregatorState, Seq[Code[_]]) => Code[Unit]): Code[Unit] =
    cb.wrapVoidsWithArgs(Array.tabulate(nStates)((i: Int) => { args: Seq[Code[_]] => f(i, states(i), args) }), prefix, argTypes, args)

  def createStates(cb: EmitClassBuilder[_]): Code[Unit] =
    toCode(cb, "create_states", (i, s) => s.createState)
}

class TupleAggregatorState(val cb: EmitClassBuilder[_], val states: StateTuple, val topRegion: Value[Region], val off: Value[Long], val rOff: Value[Int] = const(0)) {
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
  def newState: Code[Unit] = states.toCode(cb, "new_state", (i, s) => s.newState(getStateOffset(i)))
  def load: Code[Unit] = states.toCode(cb, "load", (i, s) => s.load(getRegion(i), getStateOffset(i)))
  def store: Code[Unit] = states.toCode(cb, "store", (i, s) => s.store(setRegion(i), getStateOffset(i)))
  def copyFrom(statesOffset: Code[Long]): Code[Unit] = {
    states.toCodeWithArgs(cb, "copy_from", Array[TypeInfo[_]](LongInfo),
      Array(statesOffset),
      { case (i, s, Seq(o: Code[Long@unchecked])) => s.copyFrom(storageType.loadField(o, i)) })
  }
}
