package is.hail.expr.ir.agg

import is.hail.annotations.{Region, RegionUtils, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, CodecSpec, CodecSpec2, InputBuffer, OutputBuffer, PackCodecSpec2}
import is.hail.utils._
import is.hail.asm4s.coerce

trait AggregatorState {
  def fb: EmitFunctionBuilder[_]

  def storageType: PType

  def regionSize: Int = Region.TINY

  def createState: Code[Unit]
  def newState(off: Code[Long]): Code[Unit]

  def load(regionLoader: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit]
  def store(regionStorer: Code[Region] => Code[Unit], dest: Code[Long]): Code[Unit]

  def copyFrom(src: Code[Long]): Code[Unit]

  def serialize(codec: BufferSpec): Code[OutputBuffer] => Code[Unit]

  def deserialize(codec: BufferSpec): Code[InputBuffer] => Code[Unit]
}

trait RegionBackedAggState extends AggregatorState {
  protected val r: ClassFieldRef[Region] = fb.newField[Region]
  val region: Code[Region] = r.load()

  def newState(off: Code[Long]): Code[Unit] = region.getNewRegion(regionSize)

  def createState: Code[Unit] = region.isNull.mux(r := Region.stagedCreate(regionSize), Code._empty)

  def load(regionLoader: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit] = regionLoader(r)

  def store(regionStorer: Code[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    region.isValid.orEmpty(Code(regionStorer(region), region.invalidate()))
}

trait PointerBasedRVAState extends RegionBackedAggState {
  val off: ClassFieldRef[Long] = fb.newField[Long]
  val storageType: PType = PInt64(true)

  override val regionSize: Int = Region.TINIER

  override def load(regionLoader: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    Code(super.load(regionLoader, src), off := Region.loadAddress(src))

  override def store(regionStorer: Code[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    Code(region.isValid.orEmpty(Region.storeAddress(dest, off)), super.store(regionStorer, dest))

  def copyFrom(src: Code[Long]): Code[Unit] = copyFromAddress(Region.loadAddress(src))

  def copyFromAddress(src: Code[Long]): Code[Unit]
}

class TypedRegionBackedAggState(val typ: PType, val fb: EmitFunctionBuilder[_]) extends RegionBackedAggState {
  override val regionSize: Int = Region.TINIER
  val storageType: PTuple = PTuple(required = true, typ)
  val off: ClassFieldRef[Long] = fb.newField[Long]

  override def newState(src: Code[Long]): Code[Unit] = Code(off := src, super.newState(off))
  override def load(regionLoader: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    Code(super.load(r => r.invalidate(), src), off := src)
  override def store(regionStorer: Code[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    Code(region.isValid.orEmpty(dest.cne(off).orEmpty(
      Region.copyFrom(off, dest, storageType.byteSize))),
      super.store(regionStorer, dest))

  def storeMissing(): Code[Unit] = Code(Code._println(RegionUtils.printAddr(off, "off")), storageType.setFieldMissing(off, 0))
  def storeNonmissing(v: Code[_]): Code[Unit] = Code(
    region.getNewRegion(regionSize),
    storageType.setFieldPresent(off, 0),
    StagedRegionValueBuilder.deepCopy(fb, region, typ, v, storageType.fieldOffset(off, 0)))

  def get(): EmitTriplet = EmitTriplet(Code._empty, storageType.isFieldMissing(off, 0), Region.loadIRIntermediate(typ)(storageType.fieldOffset(off, 0)))

  def copyFrom(src: Code[Long]): Code[Unit] =
    Code(newState(off), StagedRegionValueBuilder.deepCopy(fb, region, storageType, src, off))

  def serialize(codec: BufferSpec): Code[OutputBuffer] => Code[Unit] = {
    val enc = PackCodecSpec2(storageType, codec).buildEmitEncoderF[Long](storageType, fb)
    ob: Code[OutputBuffer] => enc(region, off, ob)
  }

  def deserialize(codec: BufferSpec): Code[InputBuffer] => Code[Unit] = {
    val (t, dec) = PackCodecSpec2(storageType, codec).buildEmitDecoderF[Long](storageType.virtualType, fb)
    val off2: ClassFieldRef[Long] = fb.newField[Long]
    ib: Code[InputBuffer] => Code(off2 := dec(region, ib), Region.copyFrom(off2, off, storageType.byteSize))
  }
}

class PrimitiveRVAState(val types: Array[PType], val fb: EmitFunctionBuilder[_]) extends AggregatorState {
  type ValueField = (Option[ClassFieldRef[Boolean]], ClassFieldRef[_], PType)
  assert(types.forall(_.isPrimitive))

  val nFields: Int = types.length
  val fields: Array[ValueField] = Array.tabulate(nFields) { i =>
    val m = if (types(i).required) None else Some(fb.newField[Boolean](s"primitiveRVA_${i}_m"))
    val v = fb.newField(s"primitiveRVA_${i}_v")(typeToTypeInfo(types(i)))
    (m, v, types(i))
  }
  val storageType: PTuple = PTuple(types: _*)

  def foreachField(f: (Int, ValueField) => Code[Unit]): Code[Unit] =
    coerce[Unit](Code(Array.tabulate(nFields)(i => f(i, fields(i))) :_*))

  def newState(off: Code[Long]): Code[Unit] = Code._empty
  def createState: Code[Unit] = Code._empty

  private[this] def loadVarsFromRegion(src: Code[Long]): Code[Unit] =
    foreachField {
      case (i, (None, v, t)) =>
        v.storeAny(Region.loadPrimitive(t)(storageType.fieldOffset(src, i)))
      case (i, (Some(m), v, t)) => Code(
        m := storageType.isFieldMissing(src, i),
        m.mux(Code._empty,
          v.storeAny(Region.loadPrimitive(t)(storageType.fieldOffset(src, i)))))
    }

  def load(regionLoader: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    loadVarsFromRegion(src)

  def store(regionStorer: Code[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
      foreachField {
        case (i, (None, v, t)) =>
          Region.storePrimitive(t, storageType.fieldOffset(dest, i))(v)
        case (i, (Some(m), v, t)) =>
          m.mux(storageType.setFieldMissing(dest, i),
            Code(storageType.setFieldPresent(dest, i),
              Region.storePrimitive(t, storageType.fieldOffset(dest, i))(v)))
      }

  def copyFrom(src: Code[Long]): Code[Unit] = loadVarsFromRegion(src)

  def serialize(codec: BufferSpec): Code[OutputBuffer] => Code[Unit] = {
    ob: Code[OutputBuffer] =>
      foreachField {
        case (_, (None, v, t)) => ob.writePrimitive(t)(v)
        case (_, (Some(m), v, t)) => Code(
          ob.writeBoolean(m),
          m.mux(Code._empty, ob.writePrimitive(t)(v)))
      }
  }

  def deserialize(codec: BufferSpec): Code[InputBuffer] => Code[Unit] = {
    ib: Code[InputBuffer] =>
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

  def apply(i: Int): AggregatorState = states(i)
  private def getRegion(region: Code[Region], rOffset: Code[Int], i: Int): Code[Region] => Code[Unit] = { r: Code[Region] =>
    r.setFromParentReference(region, rOffset + i, states(i).regionSize) }
  private def getStateOffset(off: Code[Long], i: Int): Code[Long] = storageType.loadField(off, i)

  def toCode(f: (Int, AggregatorState) => Code[Unit]): Code[Unit] =
    coerce[Unit](Code(Array.tabulate(nStates)(i => f(i, states(i))): _*))

  def createStates: Code[Unit] =
    toCode((i, s) => s.createState)

  def newStates(stateOffset: Code[Long]): Code[Unit] =
    toCode((i, s) => s.newState(getStateOffset(stateOffset, i)))

  def load(region: Code[Region], rOffset: Code[Int], stateOffset: Code[Long]): Code[Unit] =
    toCode((i, s) => s.load(getRegion(region, rOffset, i), getStateOffset(stateOffset, i)))

  def store(region: Code[Region], rOffset: Code[Int], statesOffset: Code[Long]): Code[Unit] =
    toCode((i, s) => s.store(r => region.setParentReference(r, rOffset + i), getStateOffset(statesOffset, i)))

  def copyFrom(statesOffset: Code[Long]): Code[Unit] =
    toCode((i, s) => s.copyFrom(getStateOffset(statesOffset, i)))
}

case class TupleAggregatorState(states: StateTuple, topRegion: Code[Region], off: Code[Long], rOff: Code[Int] = 0) {
  val storageType: PTuple = states.storageType
  private def getRegion(i: Int): Code[Region] => Code[Unit] = { r: Code[Region] =>
    r.setFromParentReference(topRegion, rOff + i, states(i).regionSize) }
  private def getStateOffset(i: Int): Code[Long] = storageType.loadField(off, i)

  def newState(i: Int): Code[Unit] = states(i).newState(getStateOffset(i))
  def newState: Code[Unit] = states.newStates(off)
  def load: Code[Unit] = states.toCode((i, s) => s.load(getRegion(i), getStateOffset(i)))
  def store: Code[Unit] = states.store(topRegion, rOff, off)
}