package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.stypes.interfaces.{SContainer, SIndexableCode, SIndexableValue}
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType}
import is.hail.types.physical.{PArray, PCanonicalArray, PCanonicalDict, PCanonicalSet, PContainer, PType}
import is.hail.types.virtual.Type
import is.hail.utils.FastIndexedSeq

case class SIndexablePointer(pType: PContainer) extends SContainer {
  require(!pType.required)

  lazy val virtualType: Type = pType.virtualType

  override def castRename(t: Type): SType = SIndexablePointer(pType.deepRename(t).asInstanceOf[PContainer])

  override def elementType: SType = pType.elementType.sType

  def elementEmitType: EmitType = EmitType(elementType, pType.elementType.required)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value match {
      case value: SIndexablePointerCode if !deepCopy => new SIndexablePointerCode(this, value.length, value.elements, value.missing)
      case _ =>
        val addr = pType.store(cb, region, value, deepCopy)
        pType.loadCheapPCode(cb, addr)
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo, IntInfo, LongInfo)

  def fromSettables(settables: IndexedSeq[Settable[_]]): SIndexablePointerSettable = {
    val (length, elements, missing) = settables match {
      case IndexedSeq(length: Settable[Int@unchecked], elements: Settable[Long@unchecked], missing: Settable[Long@unchecked]) =>
        (length, elements, Some(missing))
      case IndexedSeq(length: Settable[Int@unchecked], elements: Settable[Long@unchecked]) =>
        (length, elements, None)
    }
    assert(length.ti == IntInfo)
    assert(elements.ti == LongInfo)
    missing.foreach(m => assert(m.ti == LongInfo))
    new SIndexablePointerSettable(this, length, elements, missing)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SIndexablePointerCode = {
    val (length, elements, missing) = codes match {
      case IndexedSeq(length: Code[Int@unchecked], elements: Code[Long@unchecked], missing: Code[Long@unchecked]) =>
        (length, elements, Some(missing))
      case IndexedSeq(length: Code[Int@unchecked], elements: Code[Long@unchecked]) =>
        (length, elements, None)
    }
    assert(length.ti == IntInfo)
    assert(elements.ti == LongInfo)
    missing.foreach(m => assert(m.ti == LongInfo))
    new SIndexablePointerCode(this, length, elements, missing)
  }

  def canonicalPType(): PType = pType
}


class SIndexablePointerCode(val st: SIndexablePointer,
  val length: Code[Int],
  val elements: Code[Long],
  val missing: Option[Code[Long]]
) extends SIndexableCode {
  val pt: PContainer = st.pType
  require(missing.isEmpty == pt.elementType.required)

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(length, elements) ++ missing

  def loadLength(): Code[Int] = length

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SIndexableValue = {
    val s = SIndexablePointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SIndexableValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SIndexableValue = memoize(cb, name, cb.fieldBuilder)

  def castToArray(cb: EmitCodeBuilder): SIndexableCode = {
    pt match {
      case t: PCanonicalDict => new SIndexablePointerCode(SIndexablePointer(t.arrayRep), length, elements, missing)
      case t: PCanonicalSet => new SIndexablePointerCode(SIndexablePointer(t.arrayRep), length, elements, missing)
      case _: PArray => this
    }
  }
}

object SIndexablePointerSettable {
  def apply(sb: SettableBuilder, st: SIndexablePointer, name: String): SIndexablePointerSettable = {
    new SIndexablePointerSettable(st,
      sb.newSettable[Int](s"${ name }_length"),
      sb.newSettable[Long](s"${ name }_elements"),
      if (st.pType.elementType.required) None else Some(sb.newSettable[Long](s"${ name }_missing")))
  }
}

class SIndexablePointerSettable(
  val st: SIndexablePointer,
  val length: Settable[Int],
  val elements: Settable[Long],
  val missing: Option[Settable[Long]]
) extends SIndexableValue with SSettable {
  val pt: PContainer = st.pType
  require(missing.isEmpty == pt.elementType.required)

  def get: SIndexableCode = new SIndexablePointerCode(st, length, elements, missing.map(_.get))

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(length, elements) ++ missing

  def loadLength(): Value[Int] = length

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    val iv = cb.newLocal("pcindval_i", i)
    IEmitCode(cb,
      isElementMissing(iv),
      pt.elementType.loadCheapPCode(cb, pt.elementOffsetFromBase(elements, iv))) // FIXME loadElement should take elementsAddress
  }

  def isElementMissing(i: Code[Int]): Code[Boolean] =
    missing.map(m => Region.loadBit(m, i.toL)).getOrElse(false)

  def store(cb: EmitCodeBuilder, sc: SCode): Unit = sc match {
    case sc: SIndexablePointerCode =>
      cb.assign(length, sc.length)
      cb.assign(elements, sc.elements)
      missing.foreach(missing => cb.assign(missing, sc.missing.get))
  }

  override def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] =
    missing.map(m => Region.containsNonZeroBits(m, length.toL)).getOrElse(false)

  override def forEachDefined(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, Value[Int], SCode) => Unit) {
    st.pType match {
      case pca: PCanonicalArray =>
        val idx = cb.newLocal[Int]("foreach_pca_idx", 0)
        val elementPtr = cb.newLocal[Long]("foreach_pca_elt_ptr", elements)
        val et = pca.elementType
        cb.whileLoop(idx < length, {
          cb.ifx(isElementMissing(idx),
            {}, // do nothing,
            {
              val elt = et.loadCheapPCode(cb, et.loadFromNested(elementPtr))
              f(cb, idx, elt)
            })
          cb.assign(idx, idx + 1)
          cb.assign(elementPtr, elementPtr + pca.elementByteSize)
        })
      case _ => super.forEachDefined(cb)(f)
    }
  }

  /**
    * If this SVector was created from a canonical [[PContainer]]
    * (one of [[PCanonicalArray]], [[PCanonicalDict]], [[PCanonicalSet]])
    * then get the base address that corresponds to the pointer that would
    * be passed to [[PType.loadCheapPCode]]
    * @return [[Code]]
    */
  def baseAddress(): Code[Long] = pt match {
    case _: PCanonicalSet | _: PCanonicalDict | _: PCanonicalArray =>
      elements - pt.elementsOffset(length)
  }
}
