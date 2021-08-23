package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.stypes.interfaces.{SContainer, SIndexableCode, SIndexableValue}
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType}
import is.hail.types.physical.{PArray, PCanonicalArray, PCanonicalDict, PCanonicalSet, PContainer, PType}
import is.hail.types.virtual.Type
import is.hail.utils.FastIndexedSeq


final case class SIndexablePointer(pType: PContainer) extends SContainer {
  require(!pType.required)

  override lazy val virtualType: Type = pType.virtualType

  override def castRename(t: Type): SType = SIndexablePointer(pType.deepRename(t).asInstanceOf[PContainer])

  override def elementType: SType = pType.elementType.sType

  override def elementEmitType: EmitType = EmitType(elementType, pType.elementType.required)

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SIndexablePointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  override def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo, IntInfo, LongInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SIndexablePointerSettable = {
    val IndexedSeq(a: Settable[Long@unchecked], length: Settable[Int@unchecked], elementsAddress: Settable[Long@unchecked]) = settables
    assert(a.ti == LongInfo)
    assert(length.ti == IntInfo)
    assert(elementsAddress.ti == LongInfo)
    new SIndexablePointerSettable(this, a, length, elementsAddress)
  }

  override def fromCodes(codes: IndexedSeq[Code[_]]): SIndexablePointerCode = {
    val IndexedSeq(a: Code[Long@unchecked]) = codes
    assert(a.ti == LongInfo)
    new SIndexablePointerCode(this, a)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SIndexablePointerValue = {
    val IndexedSeq(a: Value[Long@unchecked], length: Value[Int@unchecked], elementsAddress: Value[Long@unchecked]) = values
    assert(a.ti == LongInfo)
    assert(length.ti == IntInfo)
    assert(elementsAddress.ti == LongInfo)
    new SIndexablePointerValue(this, a, length, elementsAddress)
  }

  override def storageType(): PType = pType

  override def copiedType: SType = SIndexablePointer(pType.copiedType.asInstanceOf[PContainer])

  override def containsPointers: Boolean = pType.containsPointers
}


class SIndexablePointerCode(val st: SIndexablePointer, val a: Code[Long]) extends SIndexableCode {
  val pt: PContainer = st.pType

  def code: Code[_] = a

  override def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  override def codeLoadLength(): Code[Int] = pt.loadLength(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SIndexableValue = {
    val s = SIndexablePointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  override def memoize(cb: EmitCodeBuilder, name: String): SIndexableValue = memoize(cb, name, cb.localBuilder)

  override def memoizeField(cb: EmitCodeBuilder, name: String): SIndexableValue = memoize(cb, name, cb.fieldBuilder)

  override def castToArray(cb: EmitCodeBuilder): SIndexableCode = {
    pt match {
      case t: PArray => this
      case t: PCanonicalDict => new SIndexablePointerCode(SIndexablePointer(t.arrayRep), a)
      case t: PCanonicalSet => new SIndexablePointerCode(SIndexablePointer(t.arrayRep), a)
    }
  }
}

class SIndexablePointerValue(
  val st: SIndexablePointer,
  val a: Value[Long],
  val length: Value[Int],
  val elementsAddress: Value[Long]
) extends SIndexableValue {
  val pt: PContainer = st.pType

  def get: SIndexableCode = new SIndexablePointerCode(st, a)

  def loadLength(): Value[Int] = length

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    val iv = cb.newLocal("pcindval_i", i)
    IEmitCode(cb,
      isElementMissing(iv),
      pt.elementType.loadCheapSCode(cb, pt.loadElement(a, length, iv))) // FIXME loadElement should take elementsAddress
  }

  def isElementMissing(i: Code[Int]): Code[Boolean] = pt.isElementMissing(a, i)

  override def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] = pt.hasMissingValues(a)

  override def forEachDefined(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, Value[Int], SCode) => Unit) {
    st.pType match {
      case pca: PCanonicalArray =>
        val idx = cb.newLocal[Int]("foreach_pca_idx", 0)
        val elementPtr = cb.newLocal[Long]("foreach_pca_elt_ptr", elementsAddress)
        val et = pca.elementType
        cb.whileLoop(idx < length, {
          cb.ifx(isElementMissing(idx),
            {}, // do nothing,
            {
              val elt = et.loadCheapSCode(cb, et.loadFromNested(elementPtr))
              f(cb, idx, elt)
            })
          cb.assign(idx, idx + 1)
          cb.assign(elementPtr, elementPtr + pca.elementByteSize)
        })
      case _ => super.forEachDefined(cb)(f)
    }
  }
}

object SIndexablePointerSettable {
  def apply(sb: SettableBuilder, st: SIndexablePointer, name: String): SIndexablePointerSettable = {
    new SIndexablePointerSettable(st,
      sb.newSettable[Long](s"${ name }_a"),
      sb.newSettable[Int](s"${ name }_length"),
      sb.newSettable[Long](s"${ name }_elems_addr"))
  }
}

final class SIndexablePointerSettable(
  st: SIndexablePointer,
  override val a: Settable[Long],
  override val length: Settable[Int],
  override val elementsAddress: Settable[Long]
) extends SIndexablePointerValue(st, a, length, elementsAddress) with SSettable {
  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a, length, elementsAddress)

  def store(cb: EmitCodeBuilder, pc: SCode): Unit = {
    cb.assign(a, pc.asInstanceOf[SIndexablePointerCode].a)
    cb.assign(length, pt.loadLength(a))
    cb.assign(elementsAddress, pt.firstElementOffset(a, length))
  }
}
