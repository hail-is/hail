package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.{PCanonicalNDArray, PNDArray, PType}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType}
import is.hail.types.virtual._
import is.hail.utils.FastIndexedSeq
import is.hail.variant.ReferenceGenome

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
      case ts: TShuffle => SUnreachableShuffle(ts)
      case TCall => SUnreachableCall
      case TBinary => SUnreachableBinary
      case TString => SUnreachableString
      case TVoid => SVoid
    }
  }
}

abstract class SUnreachable extends SType {
  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq()

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq()

  def canonicalPType(): PType = PType.canonical(virtualType, required = false, innerRequired = true)

  override def asIdent: String = s"s_unreachable"

  def castRename(t: Type): SType = SUnreachable.fromVirtualType(t)

  val sv: SUnreachableValue

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = sv

  override def fromCodes(codes: IndexedSeq[Code[_]]): SUnreachableValue = sv

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = sv
}

abstract class SUnreachableValue extends SCode with SSettable {
  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq()

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq()

  def store(cb: EmitCodeBuilder, v: SCode): Unit = {}

  override def get: SCode = this
}

case class SUnreachableStruct(virtualType: TBaseStruct) extends SUnreachable with SBaseStruct {
  override def size: Int = virtualType.size

  val fieldTypes: IndexedSeq[SType] = virtualType.types.map(SUnreachable.fromVirtualType)
  val fieldEmitTypes: IndexedSeq[EmitType] = fieldTypes.map(f => EmitType(f, true))

  def fieldIdx(fieldName: String): Int = virtualType.fieldIdx(fieldName)

  val sv = new SUnreachableStructValue(this)

  override def fromCodes(codes: IndexedSeq[Code[_]]): SUnreachableStructValue = sv
}

class SUnreachableStructValue(val st: SUnreachableStruct) extends SUnreachableValue with SBaseStructValue with SBaseStructCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SBaseStructValue = this

  override def memoize(cb: EmitCodeBuilder, name: String): SBaseStructValue = this

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.types(fieldIdx)).defaultValue)

  override def isFieldMissing(fieldIdx: Int): Code[Boolean] = false

  override def loadSingleField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = loadField(cb, fieldIdx)

  override def subset(fieldNames: String*): SBaseStructCode = {
    val oldType = st.virtualType.asInstanceOf[TStruct]
    val newType = TStruct(fieldNames.map(f => (f, oldType.fieldType(f))): _*)
    new SUnreachableStructValue(SUnreachableStruct(newType))
  }

  override def insert(cb: EmitCodeBuilder, region: Value[Region], newType: TStruct, fields: (String, EmitCode)*): SBaseStructCode =
    new SUnreachableStructValue(SUnreachableStruct(newType))

  override def _insert(newType: TStruct, fields: (String, EmitCode)*): SBaseStructCode =
    new SUnreachableStructValue(SUnreachableStruct(newType))
}

case object SUnreachableBinary extends SUnreachable with SBinary {
  override def virtualType: Type = TBinary

  val sv = new SUnreachableBinaryValue
}

class SUnreachableBinaryValue extends SUnreachableValue with SBinaryValue with SBinaryCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableBinaryValue = this

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableBinaryValue = this

  override def loadByte(i: Code[Int]): Code[Byte] = const(0.toByte)

  override def loadBytes(): Code[Array[Byte]] = Code._null

  override def loadLength(): Code[Int] = const(0)

  def st: SUnreachableBinary.type = SUnreachableBinary

  override def get: SUnreachableBinaryValue = this
}

case object SUnreachableString extends SUnreachable with SString {
  override def virtualType: Type = TString

  val sv = new SUnreachableStringValue

  override def constructFromString(cb: EmitCodeBuilder, r: Value[Region], s: Code[String]): SStringCode = sv
}

class SUnreachableStringValue extends SUnreachableValue with SStringValue with SStringCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableStringValue = this

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableStringValue = this

  override def loadLength(): Code[Int] = const(0)

  def st: SUnreachableString.type = SUnreachableString

  override def loadString(): Code[String] = Code._null

  override def toBytes(): SBinaryCode = new SUnreachableBinaryValue

  override def get: SUnreachableStringValue = this
}

case class SUnreachableShuffle(virtualType: TShuffle) extends SUnreachable with SShuffle {
  val sv = new SUnreachableShuffleValue(this)
}

class SUnreachableShuffleValue(val st: SUnreachableShuffle) extends SUnreachableValue with SShuffleValue with SShuffleCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableShuffleValue = this

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableShuffleValue = this

  override def loadBytes(): Code[Array[Byte]] = Code._null

  override def loadLength(): Code[Int] = const(0)

  override def get: SUnreachableShuffleValue = this
}

case class SUnreachableLocus(virtualType: TLocus) extends SUnreachable with SLocus {
  val sv = new SUnreachableLocusValue(this)

  override def contigType: SString = SUnreachableString

  override def rg: ReferenceGenome = virtualType.rg
}

class SUnreachableLocusValue(val st: SUnreachableLocus) extends SUnreachableValue with SLocusValue with SLocusCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableLocusValue = this

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableLocusValue = this

  override def position(cb: EmitCodeBuilder): Code[Int] = const(0)

  override def contig(cb: EmitCodeBuilder): SStringCode = new SUnreachableStringValue

  override def structRepr(cb: EmitCodeBuilder): SBaseStructValue = SUnreachableStruct(TStruct("contig" -> TString, "position" -> TInt32)).defaultValue.asInstanceOf[SUnreachableStructValue]
}


case object SUnreachableCall extends SUnreachable with SCall {
  override def virtualType: Type = TCall

  val sv = new SUnreachableCallValue
}

class SUnreachableCallValue extends SUnreachableValue with SCallValue with SCallCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableCallValue = this

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableCallValue = this

  override def loadCanonicalRepresentation(cb: EmitCodeBuilder): Code[Int] = const(0)

  override def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit = {}

  override def isPhased(): Code[Boolean] = const(false)

  override def ploidy(): Code[Int] = const(0)

  override def canonicalCall(cb: EmitCodeBuilder): Code[Int] = const(0)

  def st: SUnreachableCall.type = SUnreachableCall

  override def get: SUnreachableCallValue = this

  override def lgtToGT(cb: EmitCodeBuilder, localAlleles: SIndexableValue, errorID: Value[Int]): SCallCode = this
}


case class SUnreachableInterval(virtualType: TInterval) extends SUnreachable with SInterval {
  val sv = new SUnreachableIntervalValue(this)

  override def pointType: SType = SUnreachable.fromVirtualType(virtualType.pointType)

  override def pointEmitType: EmitType = EmitType(pointType, true)
}

class SUnreachableIntervalValue(val st: SUnreachableInterval) extends SUnreachableValue with SIntervalValue with SIntervalCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableIntervalValue = this

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableIntervalValue = this

  def includesStart(): Value[Boolean] = const(false)

  def includesEnd(): Value[Boolean] = const(false)

  def codeIncludesStart(): Code[Boolean] = const(false)

  def codeIncludesEnd(): Code[Boolean] = const(false)

  def loadStart(cb: EmitCodeBuilder): IEmitCode = IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.pointType).defaultValue)

  def startDefined(cb: EmitCodeBuilder): Code[Boolean] = const(false)

  def loadEnd(cb: EmitCodeBuilder): IEmitCode = IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.pointType).defaultValue)

  def endDefined(cb: EmitCodeBuilder): Code[Boolean] = const(false)

  def isEmpty(cb: EmitCodeBuilder): Code[Boolean] = const(false)
}


case class SUnreachableNDArray(virtualType: TNDArray) extends SUnreachable with SNDArray {
  val sv = new SUnreachableNDArrayValue(this)

  override def nDims: Int = virtualType.nDims

  lazy val elementType: SType = SUnreachable.fromVirtualType(virtualType.elementType)

  override def elementPType: PType = elementType.canonicalPType()

  override def pType: PNDArray = PCanonicalNDArray(elementPType.setRequired(true), nDims, false)

  override def elementByteSize: Long = 0L
}

class SUnreachableNDArrayValue(val st: SUnreachableNDArray) extends SUnreachableValue with SNDArraySettable with SNDArrayCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableNDArrayValue = this

  def shape(cb: EmitCodeBuilder): SBaseStructCode = SUnreachableStruct(TTuple((0 until st.nDims).map(_ => TInt64): _*)).defaultValue.asBaseStruct

  def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SCode = SUnreachable.fromVirtualType(st.virtualType.elementType).defaultValue

  def shapes(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = (0 until st.nDims).map(_ => const(0L))

  def strides(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = (0 until st.nDims).map(_ => const(0L))

  def outOfBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Boolean] = const(false)

  def assertInBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder, errorId: Int = -1): Code[Unit] = Code._empty

  def sameShape(other: SNDArrayValue, cb: EmitCodeBuilder): Code[Boolean] = const(false)

  def firstDataAddress(cb: EmitCodeBuilder): Value[Long] = const(0L)

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableNDArrayValue = this

  override def get: SUnreachableNDArrayValue = this

  override def coiterateMutate(cb: EmitCodeBuilder, region: Value[Region], deepCopy: Boolean, indexVars: IndexedSeq[String],
    destIndices: IndexedSeq[Int], arrays: (SNDArrayCode, IndexedSeq[Int], String)*)(body: IndexedSeq[SCode] => SCode): Unit = ()
}

case class SUnreachableContainer(virtualType: TContainer) extends SUnreachable with SContainer {
  val sv = new SUnreachableContainerValue(this)

  lazy val elementType: SType = SUnreachable.fromVirtualType(virtualType.elementType)

  lazy val elementEmitType: EmitType = EmitType(elementType, true)
}

class SUnreachableContainerValue(val st: SUnreachableContainer) extends SUnreachableValue with SIndexableValue with SIndexableCode {
  override def memoizeField(cb: EmitCodeBuilder, name: String): SUnreachableContainerValue = this

  override def memoize(cb: EmitCodeBuilder, name: String): SUnreachableContainerValue = this

  def loadLength(): Value[Int] = const(0)

  override def codeLoadLength(): Code[Int] = const(0)

  def isElementMissing(i: Code[Int]): Code[Boolean] = const(false)

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = IEmitCode.present(cb, SUnreachable.fromVirtualType(st.virtualType.elementType).defaultValue)

  def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] = const(false)

  def castToArray(cb: EmitCodeBuilder): SIndexableCode = SUnreachable.fromVirtualType(st.virtualType.arrayElementsRepr).defaultValue.asIndexable
}
