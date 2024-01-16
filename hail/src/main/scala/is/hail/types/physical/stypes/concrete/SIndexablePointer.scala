package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical._
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.interfaces.{SContainer, SIndexableValue}
import is.hail.types.virtual.Type
import is.hail.utils.FastSeq

final case class SIndexablePointer(pType: PContainer) extends SContainer {
  require(!pType.required)

  override lazy val virtualType: Type = pType.virtualType

  override def castRename(t: Type): SType =
    SIndexablePointer(pType.deepRename(t).asInstanceOf[PContainer])

  override def elementType: SType = pType.elementType.sType

  override def elementEmitType: EmitType = EmitType(elementType, pType.elementType.required)

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue =
    value match {
      case value: SIndexableValue =>
        val a = pType.store(cb, region, value, deepCopy)
        new SIndexablePointerValue(
          this,
          a,
          value.loadLength(),
          cb.memoize(pType.firstElementOffset(a)),
        )
    }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(LongInfo, IntInfo, LongInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SIndexablePointerSettable = {
    val IndexedSeq(
      a: Settable[Long @unchecked],
      length: Settable[Int @unchecked],
      elementsAddress: Settable[Long @unchecked],
    ) = settables
    assert(a.ti == LongInfo)
    assert(length.ti == IntInfo)
    assert(elementsAddress.ti == LongInfo)
    new SIndexablePointerSettable(this, a, length, elementsAddress)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SIndexablePointerValue = {
    val IndexedSeq(
      a: Value[Long @unchecked],
      length: Value[Int @unchecked],
      elementsAddress: Value[Long @unchecked],
    ) = values
    assert(a.ti == LongInfo)
    assert(length.ti == IntInfo)
    assert(elementsAddress.ti == LongInfo)
    new SIndexablePointerValue(this, a, length, elementsAddress)
  }

  override def storageType(): PType = pType

  override def copiedType: SType = SIndexablePointer(pType.copiedType.asInstanceOf[PContainer])

  override def containsPointers: Boolean = pType.containsPointers
}

class SIndexablePointerValue(
  override val st: SIndexablePointer,
  val a: Value[Long],
  val length: Value[Int],
  val elementsAddress: Value[Long],
) extends SIndexableValue {
  val pt: PContainer = st.pType

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(a, length, elementsAddress)

  override def loadLength(): Value[Int] = length

  override def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    val iv = cb.memoize(i)
    IEmitCode(
      cb,
      isElementMissing(cb, iv),
      pt.elementType.loadCheapSCode(cb, pt.loadElement(a, length, iv)),
    ) // FIXME loadElement should take elementsAddress
  }

  override def isElementMissing(cb: EmitCodeBuilder, i: Code[Int]): Value[Boolean] =
    cb.memoize(pt.isElementMissing(a, i))

  override def hasMissingValues(cb: EmitCodeBuilder): Value[Boolean] =
    cb.memoize(pt.hasMissingValues(a))

  override def castToArray(cb: EmitCodeBuilder): SIndexableValue =
    pt match {
      case _: PArray => this
      case t: PCanonicalDict =>
        new SIndexablePointerValue(SIndexablePointer(t.arrayRep), a, length, elementsAddress)
      case t: PCanonicalSet =>
        new SIndexablePointerValue(SIndexablePointer(t.arrayRep), a, length, elementsAddress)
    }

  override def forEachDefined(
    cb: EmitCodeBuilder
  )(
    f: (EmitCodeBuilder, Value[Int], SValue) => Unit
  ): Unit = {
    st.pType match {
      case pca: PCanonicalArray => pca.forEachDefined(cb, a)(f)
      case _ => super.forEachDefined(cb)(f)
    }
  }
}

object SIndexablePointerSettable {
  def apply(sb: SettableBuilder, st: SIndexablePointer, name: String): SIndexablePointerSettable =
    new SIndexablePointerSettable(
      st,
      sb.newSettable[Long](s"${name}_a"),
      sb.newSettable[Int](s"${name}_length"),
      sb.newSettable[Long](s"${name}_elems_addr"),
    )
}

final class SIndexablePointerSettable(
  st: SIndexablePointer,
  override val a: Settable[Long],
  override val length: Settable[Int],
  override val elementsAddress: Settable[Long],
) extends SIndexablePointerValue(st, a, length, elementsAddress) with SSettable {
  def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(a, length, elementsAddress)

  def store(cb: EmitCodeBuilder, v: SValue): Unit = v match {
    case v: SIndexablePointerValue =>
      cb.assign(a, v.a)
      cb.assign(length, v.length)
      cb.assign(elementsAddress, v.elementsAddress)
  }
}
