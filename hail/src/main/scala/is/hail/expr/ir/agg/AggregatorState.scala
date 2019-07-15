package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.io.{CodecSpec, InputBuffer, OutputBuffer}
import is.hail.utils._
import is.hail.asm4s.coerce

object StorageStrategy extends Enumeration {
  type StorageStrategy = Value
  val MutableStoredInline, MutableRegionBacked = Value
}

trait AggregatorState {
  def mb: EmitMethodBuilder

  def storageType: PType

  def regionSize: Int = Region.TINY

  def isLoaded: Code[Boolean]

  def createState: Code[Unit]
  def newState: Code[Unit]

  def loadRegion(f: Code[Region] => Code[Unit]): Code[Unit]
  def loadState(src: Code[Long]): Code[Unit]
  def load(f: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    Code(loadRegion(f), loadState(src))

  def storeRegion(topRegion: Code[Region], rIdx: Code[Int]): Code[Unit]
  def storeState(off: Code[Long]): Code[Unit]
  def store(topRegion: Code[Region], rIdx: Code[Int], off: Code[Long]): Code[Unit] =
    Code(storeState(off), storeRegion(topRegion, rIdx))

  def addState(srvb: StagedRegionValueBuilder): Code[Unit]

  def copyFrom(src: Code[Long]): Code[Unit]

  def serialize(codec: CodecSpec): Code[OutputBuffer] => Code[Unit]

  def unserialize(codec: CodecSpec): Code[InputBuffer] => Code[Unit]
}

trait PointerBasedRVAState extends AggregatorState {
  private val r: ClassFieldRef[Region] = mb.newField[Region]
  val off: ClassFieldRef[Long] = mb.newField[Long]
  val storageType: PType = PInt64(true)
  val region: Code[Region] = r.load()
  val er: EmitRegion = EmitRegion(mb, region)

  override val regionSize: Int = Region.TINY

  def isLoaded: Code[Boolean] = region.isValid()

  def loadRegion(f: Code[Region] => Code[Unit]): Code[Unit] = Code(region.isValid().mux(Code._empty, region.closeButKeepContainer()), f(r))

  def newState: Code[Unit] = Code(region.isValid().mux(Code._empty, region.closeButKeepContainer()), region.getNewRegion(regionSize))

  def createState: Code[Unit] = region.isNull.mux(Code(r := Code.newInstance[Region, Int](regionSize), region.closeButKeepContainer()), Code._empty)

  def loadState(src: Code[Long]): Code[Unit] = off := Region.loadAddress(src)

  def storeRegion(topRegion: Code[Region], rIdx: Code[Int]): Code[Unit] =
    region.isValid().mux(Code(topRegion.setParentReference(region, rIdx), region.closeButKeepContainer()), Code._empty)

  def storeState(dest: Code[Long]): Code[Unit] = Region.storeAddress(dest, off)

  def addState(srvb: StagedRegionValueBuilder): Code[Unit] = srvb.addAddress(off)

  def copyFrom(src: Code[Long]): Code[Unit] = copyFromAddress(Region.loadAddress(src))

  def copyFromAddress(src: Code[Long]): Code[Unit]
}

case class TypedRVAState(valueType: PType, mb: EmitMethodBuilder) extends PointerBasedRVAState {
  def copyFromAddress(src: Code[Long]): Code[Unit] = off := StagedRegionValueBuilder.deepCopy(er, valueType, src)

  def serialize(codec: CodecSpec): Code[OutputBuffer] => Code[Unit] = {
    val enc = codec.buildEmitEncoderF[Long](valueType, valueType, mb.fb)
    ob: Code[OutputBuffer] => enc(region, off, ob)
  }

  def unserialize(codec: CodecSpec): Code[InputBuffer] => Code[Unit] = {
    val dec = codec.buildEmitDecoderF[Long](valueType, valueType, mb.fb)
    ib: Code[InputBuffer] => off := dec(region, ib)
  }
}

case class PrimitiveRVAState(types: Array[PType], mb: EmitMethodBuilder) extends AggregatorState {
  type ValueField = (Option[ClassFieldRef[Boolean]], ClassFieldRef[_], PType)
  assert(types.forall(_.isPrimitive))

  val nFields: Int = types.length
  val fields: Array[ValueField] = Array.tabulate(nFields) { i =>
    val m = if (types(i).required) None else Some(mb.newField[Boolean](s"primitiveRVA_${i}_m"))
    val v = mb.newField(s"primitiveRVA_${i}_v")(typeToTypeInfo(types(i)))
    (m, v, types(i))
  }
  val storageType: PTuple = PTuple(types.toFastIndexedSeq)

  def foreachField(f: (Int, ValueField) => Code[Unit]): Code[Unit] =
    coerce[Unit](Code(Array.tabulate(nFields)(i => f(i, fields(i))) :_*))

  val _loaded: ClassFieldRef[Boolean] = mb.newField[Boolean]

  def isLoaded: Code[Boolean] = _loaded

  def loadRegion(f: Code[Region] => Code[Unit]): Code[Unit] = Code._empty
  def newState: Code[Unit] = Code._empty
  def createState: Code[Unit] = Code._empty

  def loadState(src: Code[Long]): Code[Unit] = Code(
    foreachField {
      case (i, (None, v, t)) =>
        v.storeAny(Region.loadPrimitive(t)(storageType.fieldOffset(src, i)))
      case (i, (Some(m), v, t)) => Code(
        m := storageType.isFieldMissing(src, i),
        m.mux(Code._empty,
          v.storeAny(Region.loadPrimitive(t)(storageType.fieldOffset(src, i)))))
    },
    _loaded := true)

  def storeRegion(topRegion: Code[Region], rIdx: Code[Int]): Code[Unit] = Code._empty

  def storeState(dest: Code[Long]): Code[Unit] = Code(
    foreachField {
      case (i, (None, v, t)) =>
        Region.storePrimitive(t, storageType.fieldOffset(dest, i))(v)
      case (i, (Some(m), v, t)) =>
        m.mux(storageType.setFieldMissing(dest, i),
          Code(storageType.setFieldPresent(dest, i),
            Region.storePrimitive(t, storageType.fieldOffset(dest, i))(v)))
    },
    _loaded := false)

  def addState(srvb: StagedRegionValueBuilder): Code[Unit] = srvb.addBaseStruct(storageType, ssb =>
    Code(
      ssb.start(),
      foreachField {
        case (_, (None, v, t)) => Code(ssb.addIRIntermediate(t)(v.load()), ssb.advance())
        case (_, (Some(m), v, t)) => Code(m.mux(ssb.setMissing(), ssb.addIRIntermediate(t)(v.load())), ssb.advance())
      })
  )

  def copyFrom(src: Code[Long]): Code[Unit] = loadState(src)

  def serialize(codec: CodecSpec): Code[OutputBuffer] => Code[Unit] = {
    ob: Code[OutputBuffer] => Code(
      foreachField {
        case (_, (None, v, t)) => ob.writePrimitive(t)(v)
        case (_, (Some(m), v, t)) => Code(
          ob.writeBoolean(m),
          m.mux(Code._empty, ob.writePrimitive(t)(v)))
      }, _loaded := false)
  }

  def unserialize(codec: CodecSpec): Code[InputBuffer] => Code[Unit] = {
    ib: Code[InputBuffer] => Code(
      foreachField {
        case (_, (None, v, t)) =>
          v.storeAny(ib.readPrimitive(t))
        case (_, (Some(m), v, t)) => Code(
          m := ib.readBoolean(),
          m.mux(Code._empty, v.storeAny(ib.readPrimitive(t))))
      },
      _loaded := true)
  }
}

case class StateContainer(states: Array[AggregatorState], topRegion: Code[Region]) {
  val nStates: Int = states.length
  val typ: PTuple = PTuple(states.map { s => s.storageType }, required = true)

  def apply(i: Int): AggregatorState = states(i)
  def getRegion(rOffset: Code[Int], i: Int): Code[Region] => Code[Unit] = { r: Code[Region] =>
    r.setFromParentReference(topRegion, rOffset + i, states(i).regionSize) }
  def getStateOffset(off: Code[Long], i: Int): Code[Long] = typ.loadField(topRegion, off, i)

  def setAllMissing(off: Code[Long]): Code[Unit] = toCode((i, _) =>
    topRegion.storeAddress(typ.fieldOffset(off, i), 0L))

  def toCode(f: (Int, AggregatorState) => Code[Unit]): Code[Unit] =
    coerce[Unit](Code(Array.tabulate(nStates)(i => f(i, states(i))): _*))

  def loadOneIfMissing(stateOffset: Code[Long], idx: Int): Code[Unit] =
    states(idx).isLoaded.mux(Code._empty,
      states(idx).load(getRegion(0, idx), getStateOffset(stateOffset, idx)))

  def createStates: Code[Unit] =
    toCode((i, s) => s.createState)

  def loadRegions(rOffset: Code[Int]): Code[Unit] =
    toCode((i, s) => s.loadRegion(getRegion(rOffset, i)))

  def load(rOffset: Code[Int], stateOffset: Code[Long]): Code[Unit] =
    toCode((i, s) => s.load(getRegion(rOffset, i), getStateOffset(stateOffset, i)))

  def store(rOffset: Code[Int], statesOffset: Code[Long]): Code[Unit] =
    toCode((i, s) => s.store(topRegion, rOffset + i, getStateOffset(statesOffset, i)))

  def addState(srvb: StagedRegionValueBuilder): Code[Unit] = {
    srvb.addBaseStruct(typ, ssb =>
      Code(ssb.start(),
        toCode((_, s) => Code(s.addState(ssb), ssb.advance()))))
  }
}