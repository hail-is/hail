package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.{PNewArray, PType}
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType}
import is.hail.types.physical.stypes.interfaces.{SContainer, SIndexableCode, SIndexableValue}
import is.hail.types.virtual.Type
import is.hail.utils.FastIndexedSeq

case class SStackArray(pType: PNewArray) extends SContainer {
  lazy val elementType: SType = pType.elementType.sType

  lazy val elementEmitType: EmitType = EmitType(elementType, pType.elementRequired)

  lazy val virtualType: Type = pType.virtualType

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = ???

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = if (elementEmitType.required) {
    FastIndexedSeq(IntInfo, LongInfo)
  } else {
    FastIndexedSeq(IntInfo, LongInfo, LongInfo)
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = settables match {
    case IndexedSeq(length, elements, missing) => new SStackArraySettable(this, coerce(length), coerce(elements), Some(coerce(missing)))
    case IndexedSeq(length, elements) => new SStackArraySettable(this, coerce(length), coerce(elements), None)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SCode = codes match {
    case IndexedSeq(length, elements, missing) => new SStackArrayCode(this, coerce(length), coerce(elements), Some(coerce(missing)))
    case IndexedSeq(length, elements) => new SStackArrayCode(this, coerce(length), coerce(elements), None)
  }

  def canonicalPType(): PType = pType

  def castRename(t: Type): SType = SStackArray(pType.deepRename(t).asInstanceOf)
}

class SStackArrayCode(val st: SStackArray,
  length: Code[Int],
  elements: Code[Long],
  missing: Option[Code[Long]]) extends SIndexableCode {
  require(missing.isEmpty == st.elementEmitType.required)

  def loadLength(): Code[Int] = length

  def memoize(cb: EmitCodeBuilder, name: String): SIndexableValue = ???

  def memoizeField(cb: EmitCodeBuilder, name: String): SIndexableValue = ???

  def castToArray(cb: EmitCodeBuilder): SIndexableCode = ???

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(length, elements) ++ missing
}

class SStackArraySettable(val st: SStackArray,
  length: Settable[Int],
  elements: Settable[Long],
  missing: Option[Settable[Long]]
) extends SIndexableValue with SSettable {
  require(missing.isEmpty == st.elementEmitType.required)
  val pt = st.pType

  def loadLength(): Value[Int] = length

  def isElementMissing(i: Code[Int]): Code[Boolean] = missing.map(m => Region.loadBit(m, i.toL)).getOrElse(true)

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    val iv = cb.newLocal("stack_array_load_element_i", i)
    IEmitCode(cb, isElementMissing(iv), pt.elementType.loadCheapPCode(cb, pt.elementOffsetFromBase(elements, iv)))
  }

  def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] = missing.map(m => Region.containsNonZeroBits(m, length.toL)).getOrElse(false)

  def store(cb: EmitCodeBuilder, v: SCode): Unit = ???

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(length, elements) ++ missing

  def get: SCode = new SStackArrayCode(st, length, elements, missing.map(_.get))
}