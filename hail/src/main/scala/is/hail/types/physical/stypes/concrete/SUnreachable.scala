package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitValue, IEmitCode}
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.SInt64Value
import is.hail.types.physical.{PCanonicalNDArray, PNDArray, PType}
import is.hail.types.virtual._
import is.hail.utils.FastSeq

object SUnreachable {
  def fromVirtualType(t: Type): SType = {
    require(t.isRealizable)
    t match {
      case t if t.isPrimitive => SType.canonical(t)
      case TRNGState => SRNGState(Some(SRNGStateStaticInfo(0, false, 0)))
      case ts: TBaseStruct => SUnreachableStruct(ts)
      case tc: TContainer => SUnreachableContainer(tc)
      case tnd: TNDArray => SUnreachableNDArray(tnd)
      case tl: TLocus => SUnreachableLocus(tl)
      case ti: TInterval => SUnreachableInterval(ti)
      case TCall => SUnreachableCall
      case TBinary => SUnreachableBinary
      case TString => SUnreachableString
      case TVoid => SVoid
    }
  }
}

abstract class SUnreachable extends SType {
  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq()

  override def storageType(): PType = PType.canonical(virtualType, required = false, innerRequired = true)

  override def asIdent: String = s"s_unreachable"

  override def castRename(t: Type): SType = SUnreachable.fromVirtualType(t)

  val sv: SUnreachableValue

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = sv

  override def fromValues(values: IndexedSeq[Value[_]]): SUnreachableValue = sv

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = sv

  override def copiedType: SType = this

  override def containsPointers: Boolean = false
}

abstract class SUnreachableValue extends SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq()

  override def valueTuple: IndexedSeq[Value[_]] = FastSeq()

  override def store(cb: EmitCodeBuilder, v: SValue): Unit = {}

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = new SInt64Value(-1L)
}

case class SUnreachableStruct(virtualType: TBaseStruct) extends SUnreachable with SBaseStruct {
  override def size: Int = virtualType.size

  override val fieldTypes: IndexedSeq[SType] = virtualType.types.map(SUnreachable.fromVirtualType)
  override val fieldEmitTypes: IndexedSeq[EmitType] = fieldTypes.map(f => EmitType(f, true))

  override def fieldIdx(fieldName: String): Int = virtualType.fieldIdx(fieldName)

  override val sv = new SUnreachableStructValue(this)
}

class SUnreachableStructValue(override val st: SUnreachableStruct) extends SUnreachableValue with SBaseStructValue {
  override def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode =
    IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.types(fieldIdx)).defaultValue)

  override def isFieldMissing(cb: EmitCodeBuilder, fieldIdx: Int): Value[Boolean] = false

  override def subset(fieldNames: String*): SBaseStructValue = {
    val oldType = st.virtualType.asInstanceOf[TStruct]
    val newType = TStruct(fieldNames.map(f => (f, oldType.fieldType(f))): _*)
    new SUnreachableStructValue(SUnreachableStruct(newType))
  }

  override def insert(cb: EmitCodeBuilder, region: Value[Region], newType: TStruct, fields: (String, EmitValue)*): SBaseStructValue =
    new SUnreachableStructValue(SUnreachableStruct(newType))

  override def _insert(newType: TStruct, fields: (String, EmitValue)*): SBaseStructValue =
    new SUnreachableStructValue(SUnreachableStruct(newType))
}

case object SUnreachableBinary extends SUnreachable with SBinary {
  override def virtualType: Type = TBinary

  override val sv = new SUnreachableBinaryValue
}

class SUnreachableBinaryValue extends SUnreachableValue with SBinaryValue {
  override def loadByte(cb: EmitCodeBuilder, i: Code[Int]): Value[Byte] = const(0.toByte)

  override def loadBytes(cb: EmitCodeBuilder): Value[Array[Byte]] = Code._null[Array[Byte]]

  override def loadLength(cb: EmitCodeBuilder): Value[Int] = const(0)

  override def st: SUnreachableBinary.type = SUnreachableBinary
}

case object SUnreachableString extends SUnreachable with SString {
  override def virtualType: Type = TString

  override val sv = new SUnreachableStringValue

  override def constructFromString(cb: EmitCodeBuilder, r: Value[Region], s: Code[String]): SStringValue = sv
}

class SUnreachableStringValue extends SUnreachableValue with SStringValue {
  override def st: SUnreachableString.type = SUnreachableString

  override def loadLength(cb: EmitCodeBuilder): Value[Int] = const(0)

  override def loadString(cb: EmitCodeBuilder): Value[String] = Code._null[String]

  override def toBytes(cb: EmitCodeBuilder): SBinaryValue = SUnreachableBinary.sv
}

case class SUnreachableLocus(virtualType: TLocus) extends SUnreachable with SLocus {
  override val sv = new SUnreachableLocusValue(this)

  override def contigType: SString = SUnreachableString

  override def rg: String = virtualType.rg
}

class SUnreachableLocusValue(override val st: SUnreachableLocus) extends SUnreachableValue with SLocusValue {
  override def position(cb: EmitCodeBuilder): Value[Int] = const(0)

  override def contig(cb: EmitCodeBuilder): SStringValue = SUnreachableString.sv

  override def contigIdx(cb: EmitCodeBuilder): Value[Int] = const(0)

  override def structRepr(cb: EmitCodeBuilder): SBaseStructValue = SUnreachableStruct(TStruct("contig" -> TString, "position" -> TInt32)).defaultValue.asInstanceOf[SUnreachableStructValue]
}

case object SUnreachableCall extends SUnreachable with SCall {
  override def virtualType: Type = TCall

  override val sv = new SUnreachableCallValue
}

class SUnreachableCallValue extends SUnreachableValue with SCallValue {
  override def unphase(cb: EmitCodeBuilder): SCallValue = this

  def containsAllele(cb: EmitCodeBuilder, allele: Value[Int]): Value[Boolean] = const(false)

  override def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit = {}

  override def isPhased(cb: EmitCodeBuilder): Value[Boolean] = const(false)

  override def ploidy(cb: EmitCodeBuilder): Value[Int] = const(0)

  override def canonicalCall(cb: EmitCodeBuilder): Value[Int] = const(0)

  override def st: SUnreachableCall.type = SUnreachableCall

  override def lgtToGT(cb: EmitCodeBuilder, localAlleles: SIndexableValue, errorID: Value[Int]): SCallValue = st.sv
}


case class SUnreachableInterval(virtualType: TInterval) extends SUnreachable with SInterval {
  override val sv = new SUnreachableIntervalValue(this)

  override def pointType: SType = SUnreachable.fromVirtualType(virtualType.pointType)

  override def pointEmitType: EmitType = EmitType(pointType, true)
}

class SUnreachableIntervalValue(override val st: SUnreachableInterval) extends SUnreachableValue with SIntervalValue {
  override def includesStart: Value[Boolean] = const(false)

  override def includesEnd: Value[Boolean] = const(false)

  override def loadStart(cb: EmitCodeBuilder): IEmitCode = IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.pointType).defaultValue)

  override def startDefined(cb: EmitCodeBuilder): Value[Boolean] = const(false)

  override def loadEnd(cb: EmitCodeBuilder): IEmitCode = IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.pointType).defaultValue)

  override def endDefined(cb: EmitCodeBuilder): Value[Boolean] = const(false)

  override def isEmpty(cb: EmitCodeBuilder): Value[Boolean] = const(false)
}


case class SUnreachableNDArray(virtualType: TNDArray) extends SUnreachable with SNDArray {
  override val sv = new SUnreachableNDArrayValue(this)

  override def nDims: Int = virtualType.nDims

  lazy val elementType: SType = SUnreachable.fromVirtualType(virtualType.elementType)

  override def elementPType: PType = PType.canonical(elementType.storageType())

  override def pType: PNDArray = PCanonicalNDArray(elementPType.setRequired(true), nDims, false)

  override def elementByteSize: Long = 0L
}

class SUnreachableNDArrayValue(override val st: SUnreachableNDArray) extends SUnreachableValue with SNDArraySettable {
  val pt = st.pType

  override def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SValue = SUnreachable.fromVirtualType(st.virtualType.elementType).defaultValue

  override def loadElementAddress(indices: IndexedSeq[is.hail.asm4s.Value[Long]],cb: is.hail.expr.ir.EmitCodeBuilder): is.hail.asm4s.Code[Long] = const(0L)

  override def shapes: IndexedSeq[SizeValue] = (0 until st.nDims).map(_ => SizeValueStatic(0L))

  override def shapeStruct(cb: EmitCodeBuilder): SBaseStructValue = SUnreachableStruct(TTuple((0 until st.nDims).map(_ => TInt64): _*)).sv

  override def strides: IndexedSeq[Value[Long]] = (0 until st.nDims).map(_ => const(0L))

  override def outOfBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Boolean] = const(false)

  override def assertInBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder, errorId: Int = -1): Unit = {}

  override def sameShape(cb: EmitCodeBuilder, other: SNDArrayValue): Code[Boolean] = const(false)

  override def coerceToShape(cb: EmitCodeBuilder, otherShape: IndexedSeq[SizeValue]): SNDArrayValue = this

  override def firstDataAddress: Value[Long] = const(0L)

  override def coiterateMutate(cb: EmitCodeBuilder, region: Value[Region], deepCopy: Boolean, indexVars: IndexedSeq[String],
    destIndices: IndexedSeq[Int], arrays: (SNDArrayValue, IndexedSeq[Int], String)*)(body: IndexedSeq[SValue] => SValue): Unit = ()
}

case class SUnreachableContainer(virtualType: TContainer) extends SUnreachable with SContainer {
  override val sv = new SUnreachableContainerValue(this)

  lazy val elementType: SType = SUnreachable.fromVirtualType(virtualType.elementType)

  lazy val elementEmitType: EmitType = EmitType(elementType, true)
}

class SUnreachableContainerValue(override val st: SUnreachableContainer) extends SUnreachableValue with SIndexableValue {
  override def loadLength(): Value[Int] = const(0)

  override def isElementMissing(cb: EmitCodeBuilder, i: Code[Int]): Value[Boolean] = const(false)

  override def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.elementType).defaultValue)

  override def hasMissingValues(cb: EmitCodeBuilder): Value[Boolean] = const(false)

  override def castToArray(cb: EmitCodeBuilder): SIndexableValue =
    SUnreachable.fromVirtualType(st.virtualType.arrayElementsRepr).defaultValue.asIndexable
}
