package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitValue, IEmitCode}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes._
import is.hail.types.physical.{PCanonicalNDArray, PNDArray, PType}
import is.hail.types.virtual._
import is.hail.utils.FastIndexedSeq
import is.hail.variant.{Locus, ReferenceGenome}

object SUnreachable {
  def fromVirtualType(t: Type): SType = {
    require(t.isRealizable)
    t match {
      case t if t.isPrimitive => SType.canonical(t)
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
  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq()

  override def storageType(): PType = PType.canonical(virtualType, required = false, innerRequired = true)

  override def asIdent: String = s"s_unreachable"

  override def castRename(t: Type): SType = SUnreachable.fromVirtualType(t)

  val sv: SUnreachableValue

  val sc: SUnreachableCode

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = sv

  override def fromValues(values: IndexedSeq[Value[_]]): SUnreachableValue = sv

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = sv

  override def copiedType: SType = this

  override def containsPointers: Boolean = false
}

abstract class SUnreachableCode extends SCode

abstract class SUnreachableValue extends SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq()

  override def valueTuple: IndexedSeq[Value[_]] = FastIndexedSeq()

  override def store(cb: EmitCodeBuilder, v: SCode): Unit = {}
}

case class SUnreachableStruct(virtualType: TBaseStruct) extends SUnreachable with SBaseStruct {
  override def size: Int = virtualType.size

  override val fieldTypes: IndexedSeq[SType] = virtualType.types.map(SUnreachable.fromVirtualType)
  override val fieldEmitTypes: IndexedSeq[EmitType] = fieldTypes.map(f => EmitType(f, true))

  override def fieldIdx(fieldName: String): Int = virtualType.fieldIdx(fieldName)

  override val sv = new SUnreachableStructValue(this)

  override val sc = new SUnreachableStructCode(this)
}

class SUnreachableStructCode(override val st: SUnreachableStruct) extends SUnreachableCode with SBaseStructCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SBaseStructValue = st.sv

  override def memoize(cb: EmitCodeBuilder, name: String): SBaseStructValue = st.sv

  override def loadSingleField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode =
    IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.types(fieldIdx)).defaultValue)

  override def insert(cb: EmitCodeBuilder, region: Value[Region], newType: TStruct, fields: (String, EmitCode)*): SBaseStructCode =
    new SUnreachableStructCode(SUnreachableStruct(newType))

  override def _insert(newType: TStruct, fields: (String, EmitCode)*): SBaseStructCode =
    new SUnreachableStructCode(SUnreachableStruct(newType))
}

class SUnreachableStructValue(override val st: SUnreachableStruct) extends SUnreachableValue with SBaseStructValue {
  override def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode =
    IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.types(fieldIdx)).defaultValue)

  override def isFieldMissing(fieldIdx: Int): Code[Boolean] = false

  override def subset(fieldNames: String*): SBaseStructValue = {
    val oldType = st.virtualType.asInstanceOf[TStruct]
    val newType = TStruct(fieldNames.map(f => (f, oldType.fieldType(f))): _*)
    new SUnreachableStructValue(SUnreachableStruct(newType))
  }

  override def insert(cb: EmitCodeBuilder, region: Value[Region], newType: TStruct, fields: (String, EmitValue)*): SBaseStructValue =
    new SUnreachableStructValue(SUnreachableStruct(newType))

  override def _insert(newType: TStruct, fields: (String, EmitValue)*): SBaseStructValue =
    new SUnreachableStructValue(SUnreachableStruct(newType))

  override def get: SBaseStructCode = st.sc
}

case object SUnreachableBinary extends SUnreachable with SBinary {
  override def virtualType: Type = TBinary

  override val sv = new SUnreachableBinaryValue

  override val sc = new SUnreachableBinaryCode
}

class SUnreachableBinaryCode extends SUnreachableCode with SBinaryCode {
  override def st: SUnreachableBinary.type = SUnreachableBinary

  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableBinaryValue = st.sv

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableBinaryValue = st.sv

  override def loadBytes(): Code[Array[Byte]] = Code._null[Array[Byte]]

  override def loadLength(): Code[Int] = const(0)
}

class SUnreachableBinaryValue extends SUnreachableValue with SBinaryValue {
  override def loadByte(i: Code[Int]): Code[Byte] = const(0.toByte)

  override def loadBytes(): Code[Array[Byte]] = Code._null[Array[Byte]]

  override def loadLength(): Code[Int] = const(0)

  override def st: SUnreachableBinary.type = SUnreachableBinary

  override def get: SUnreachableBinaryCode = st.sc
}

case object SUnreachableString extends SUnreachable with SString {
  override def virtualType: Type = TString

  override val sv = new SUnreachableStringValue

  override val sc = new SUnreachableStringCode

  override def constructFromString(cb: EmitCodeBuilder, r: Value[Region], s: Code[String]): SStringValue = sv
}

class SUnreachableStringCode extends SUnreachableCode with SStringCode {
  override def st: SUnreachableString.type = SUnreachableString

  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableStringValue = st.sv

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableStringValue = st.sv

  override def toBytes(): SBinaryCode = SUnreachableBinary.sc

  override def loadLength(): Code[Int] = const(0)

  override def loadString(): Code[String] = Code._null[String]
}

class SUnreachableStringValue extends SUnreachableValue with SStringValue {
  override def st: SUnreachableString.type = SUnreachableString

  override def loadLength(cb: EmitCodeBuilder): Value[Int] = const(0)

  override def loadString(cb: EmitCodeBuilder): Value[String] = Code._null[String]

  override def toBytes(cb: EmitCodeBuilder): SBinaryValue = SUnreachableBinary.sv

  override def get: SUnreachableStringCode = st.sc
}

case class SUnreachableLocus(virtualType: TLocus) extends SUnreachable with SLocus {
  override val sv = new SUnreachableLocusValue(this)

  override val sc = new SUnreachableLocusCode(this)

  override def contigType: SString = SUnreachableString

  override def rg: ReferenceGenome = virtualType.rg
}

class SUnreachableLocusValue(override val st: SUnreachableLocus) extends SUnreachableValue with SLocusValue {
  override def position(cb: EmitCodeBuilder): Value[Int] = const(0)

  override def contig(cb: EmitCodeBuilder): SStringValue = SUnreachableString.sv

  override def contigLong(cb: EmitCodeBuilder): Value[Long] = const(0)

  override def structRepr(cb: EmitCodeBuilder): SBaseStructValue = SUnreachableStruct(TStruct("contig" -> TString, "position" -> TInt32)).defaultValue.asInstanceOf[SUnreachableStructValue]

  override def get: SUnreachableLocusCode = st.sc
}

class SUnreachableLocusCode(override val st: SUnreachableLocus) extends SUnreachableCode with SLocusCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableLocusValue = st.sv

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableLocusValue = st.sv

  override def position(cb: EmitCodeBuilder): Code[Int] = const(0)

  override def contig(cb: EmitCodeBuilder): SStringCode = SUnreachableString.sc

  override def getLocusObj(cb: EmitCodeBuilder): Code[Locus] = Code._null[Locus]
}

case object SUnreachableCall extends SUnreachable with SCall {
  override def virtualType: Type = TCall

  override val sv = new SUnreachableCallValue

  override val sc = new SUnreachableCallCode
}

class SUnreachableCallCode extends SUnreachableCode with SCallCode {
  override def st: SUnreachableCall.type = SUnreachableCall

  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableCallValue = st.sv

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableCallValue = st.sv

  override def loadCanonicalRepresentation(cb: EmitCodeBuilder): Code[Int] = const(0)

  override def isPhased(): Code[Boolean] = const(false)

  override def ploidy(): Code[Int] = const(0)
}

class SUnreachableCallValue extends SUnreachableValue with SCallValue {
  override def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit = {}

  override def isPhased(): Code[Boolean] = const(false)

  override def ploidy(): Code[Int] = const(0)

  override def canonicalCall(cb: EmitCodeBuilder): Value[Int] = const(0)

  override def st: SUnreachableCall.type = SUnreachableCall

  override def get: SUnreachableCallCode = st.sc

  override def lgtToGT(cb: EmitCodeBuilder, localAlleles: SIndexableValue, errorID: Value[Int]): SCallCode = st.sc
}


case class SUnreachableInterval(virtualType: TInterval) extends SUnreachable with SInterval {
  override val sv = new SUnreachableIntervalValue(this)

  override val sc = new SUnreachableIntervalCode(this)

  override def pointType: SType = SUnreachable.fromVirtualType(virtualType.pointType)

  override def pointEmitType: EmitType = EmitType(pointType, true)
}

class SUnreachableIntervalCode(override val st: SUnreachableInterval) extends SUnreachableCode with SIntervalCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableIntervalValue = st.sv

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableIntervalValue = st.sv

  override def codeIncludesStart(): Code[Boolean] = const(false)

  override def codeIncludesEnd(): Code[Boolean] = const(false)
}

class SUnreachableIntervalValue(override val st: SUnreachableInterval) extends SUnreachableValue with SIntervalValue {
  override def includesStart(): Value[Boolean] = const(false)

  override def includesEnd(): Value[Boolean] = const(false)

  override def loadStart(cb: EmitCodeBuilder): IEmitCode = IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.pointType).defaultValue)

  override def startDefined(cb: EmitCodeBuilder): Code[Boolean] = const(false)

  override def loadEnd(cb: EmitCodeBuilder): IEmitCode = IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.pointType).defaultValue)

  override def endDefined(cb: EmitCodeBuilder): Code[Boolean] = const(false)

  override def isEmpty(cb: EmitCodeBuilder): Code[Boolean] = const(false)

  override def get: SUnreachableIntervalCode = st.sc
}


case class SUnreachableNDArray(virtualType: TNDArray) extends SUnreachable with SNDArray {
  override val sv = new SUnreachableNDArrayValue(this)

  override val sc = new SUnreachableNDArrayCode(this)

  override def nDims: Int = virtualType.nDims

  lazy val elementType: SType = SUnreachable.fromVirtualType(virtualType.elementType)

  override def elementPType: PType = PType.canonical(elementType.storageType())

  override def pType: PNDArray = PCanonicalNDArray(elementPType.setRequired(true), nDims, false)

  override def elementByteSize: Long = 0L
}

class SUnreachableNDArrayCode(override val st: SUnreachableNDArray) extends SUnreachableCode with SNDArrayCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableNDArrayValue = st.sv

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableNDArrayValue = st.sv

  override def shape(cb: EmitCodeBuilder): SBaseStructCode = SUnreachableStruct(TTuple((0 until st.nDims).map(_ => TInt64): _*)).sc
}

class SUnreachableNDArrayValue(override val st: SUnreachableNDArray) extends SUnreachableValue with SNDArraySettable {
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

  override def get: SUnreachableNDArrayCode = st.sc

  override def coiterateMutate(cb: EmitCodeBuilder, region: Value[Region], deepCopy: Boolean, indexVars: IndexedSeq[String],
    destIndices: IndexedSeq[Int], arrays: (SNDArrayValue, IndexedSeq[Int], String)*)(body: IndexedSeq[SValue] => SValue): Unit = ()
}

case class SUnreachableContainer(virtualType: TContainer) extends SUnreachable with SContainer {
  override val sv = new SUnreachableContainerValue(this)

  override val sc = new SUnreachableContainerCode(this)

  lazy val elementType: SType = SUnreachable.fromVirtualType(virtualType.elementType)

  lazy val elementEmitType: EmitType = EmitType(elementType, true)
}

class SUnreachableContainerValue(override val st: SUnreachableContainer) extends SUnreachableValue with SIndexableValue {
  override def loadLength(): Value[Int] = const(0)

  override def isElementMissing(i: Code[Int]): Code[Boolean] = const(false)

  override def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.elementType).defaultValue)

  override def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] = const(false)

  override def castToArray(cb: EmitCodeBuilder): SIndexableValue =
    SUnreachable.fromVirtualType(st.virtualType.arrayElementsRepr).defaultValue.asIndexable

  override def get: SUnreachableContainerCode = st.sc
}

class SUnreachableContainerCode(override val st: SUnreachableContainer) extends SUnreachableCode with SIndexableCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableContainerValue = st.sv

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableContainerValue = st.sv

  override def codeLoadLength(): Code[Int] = const(0)

  override def castToArray(cb: EmitCodeBuilder): SIndexableCode =
    SUnreachable.fromVirtualType(st.virtualType.arrayElementsRepr).defaultValue.get.asIndexable
}
