package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.stypes.interfaces.{SContainer, SIndexableCode, SIndexableValue}
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType}
import is.hail.types.physical.{PArray, PArrayBackedContainer, PCanonicalArray, PCanonicalDict, PCanonicalSet, PContainer, PType}
import is.hail.types.virtual.{TContainer, Type}
import is.hail.utils.FastIndexedSeq

case class SIndexablePointer(pType: PContainer) extends SContainer {
  require(!pType.required)

  lazy val virtualType: TContainer = pType.virtualType

  override def castRename(t: Type): SType = SIndexablePointer(pType.deepRename(t).asInstanceOf[PContainer])

  override def elementType: SType = pType.elementType.sType

  def elementEmitType: EmitType = EmitType(elementType, pType.elementType.required)

  def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SIndexablePointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo, IntInfo, LongInfo)

  def fromSettables(settables: IndexedSeq[Settable[_]]): SIndexablePointerSettable = {
    val IndexedSeq(a: Settable[Long@unchecked], length: Settable[Int@unchecked], elementsAddress: Settable[Long@unchecked]) = settables
    assert(a.ti == LongInfo)
    assert(length.ti == IntInfo)
    assert(elementsAddress.ti == LongInfo)
    new SIndexablePointerSettable(this, a, length, elementsAddress)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SIndexablePointerCode = {
    val IndexedSeq(a: Code[Long@unchecked]) = codes
    assert(a.ti == LongInfo)
    new SIndexablePointerCode(this, a)
  }

  def storageType(): PType = pType

  def copiedType: SType = SIndexablePointer(pType.copiedType.asInstanceOf[PContainer])

  def containsPointers: Boolean = pType.containsPointers
  
  def constructFromFunctionsKnownLength(cb: EmitCodeBuilder, region: Value[Region], length: Value[Int], deepCopy: Boolean):
  ((EmitCodeBuilder, IEmitCode) => Unit, EmitCodeBuilder => SIndexablePointerCode) = {
    val arrayPType: PCanonicalArray = pType match {
      case t: PCanonicalArray => t
      case t: PCanonicalSet => t.arrayRep
      case t: PCanonicalDict => t.arrayRep
    }

    val addr = cb.newLocal[Long]("pcarray_construct2_addr", arrayPType.allocate(region, length))
    cb += arrayPType.stagedInitialize(addr, length, setMissing = false)
    val currentElementIndex = cb.newLocal[Int]("pcarray_construct2_current_idx", 0)
    val currentElementAddress = cb.newLocal[Long]("pcarray_construct2_current_addr", arrayPType.firstElementOffset(addr, length))

    val push: (EmitCodeBuilder, IEmitCode) => Unit = { (cb, iec) =>
      iec.consume(cb,
        cb += arrayPType.setElementMissing(addr, currentElementIndex),
        { sc =>
          arrayPType.elementType.storeAtAddress(cb, currentElementAddress, region, sc, deepCopy = deepCopy)
        })
      cb.assign(currentElementIndex, currentElementIndex + 1)
      cb.assign(currentElementAddress, currentElementAddress + arrayPType.elementByteSize)
    }
    val finish: EmitCodeBuilder => SIndexablePointerCode = { (cb: EmitCodeBuilder) =>
      cb.ifx(currentElementIndex.cne(length), cb._fatal("SIndexablePointer.constructFromFunctions push was called the wrong number of times: len=",
        length.toS, ", calls=", currentElementIndex.toS))
      new SIndexablePointerCode(this, addr)
    }
    (push, finish)
  }

  def constructFromFunctionsUnknownLength(cb: EmitCodeBuilder, region: Value[Region], deepCopy: Boolean):
  ((EmitCodeBuilder, IEmitCode) => Unit, EmitCodeBuilder => SIndexableCode) = {
    val arrayPType: PCanonicalArray = pType match {
      case t: PCanonicalArray => t
      case t: PCanonicalSet => t.arrayRep
      case t: PCanonicalDict => t.arrayRep
    }

    val capacity = cb.newLocal("spointer_construct_length", 8)
    val addr = cb.newLocal[Long]("spointer_construct_addr", arrayPType.allocate(region, capacity))
    cb += arrayPType.stagedInitialize(addr, capacity, setMissing = false)
    val currentElementIndex = cb.newLocal[Int]("spointer_construct_current_idx", 0)
    val currentElementAddress = cb.newLocal[Long]("spointer_construct_current_addr", arrayPType.firstElementOffset(addr, capacity))

    val push: (EmitCodeBuilder, IEmitCode) => Unit = { (cb, iec) =>
      cb.ifx(currentElementIndex.ceq(capacity), {
        cb.assign(capacity, capacity * 2)
        val newAddr = cb.newLocal("newaddr", arrayPType.allocate(region, capacity))
        cb += arrayPType.storeLength(newAddr, capacity)
        if (!arrayPType.elementType.required) {
          cb += Region.copyFrom(addr + arrayPType.lengthHeaderBytes, newAddr + arrayPType.lengthHeaderBytes, arrayPType.nMissingBytes(currentElementIndex).toL)
        }
        cb += Region.copyFrom(arrayPType.firstElementOffset(addr, currentElementIndex), arrayPType.firstElementOffset(newAddr, capacity), const(arrayPType.elementByteSize) * currentElementIndex.toL)
        cb.assign(addr, newAddr)
        cb.assign(currentElementAddress, arrayPType.elementOffset(addr, capacity, currentElementIndex))
      })
      iec.consume(cb,
        cb += arrayPType.setElementMissing(addr, currentElementIndex),
        { sc =>
          arrayPType.elementType.storeAtAddress(cb, currentElementAddress, region, sc, deepCopy = deepCopy)
        })
      cb.assign(currentElementIndex, currentElementIndex + 1)
      cb.assign(currentElementAddress, currentElementAddress + arrayPType.elementByteSize)
    }
    val finish: (EmitCodeBuilder) => SIndexablePointerCode = { cb =>
      cb.ifx(currentElementIndex.cne(capacity), {
        val len = currentElementIndex
        val newAddr = cb.newLocal("newaddr", arrayPType.allocate(region, len))
        cb += arrayPType.stagedInitialize(newAddr, len, setMissing = false)
        if (!arrayPType.elementType.required) {
          cb += Region.copyFrom(addr + arrayPType.lengthHeaderBytes, newAddr + arrayPType.lengthHeaderBytes, arrayPType.nMissingBytes(len).toL)
        }
        cb += Region.copyFrom(arrayPType.firstElementOffset(addr, capacity), arrayPType.firstElementOffset(newAddr, len), const(arrayPType.elementByteSize) * len.toL)
        cb.assign(addr, newAddr)
      })
      new SIndexablePointerCode(this, addr)
    }

    push -> finish
  }
}


class SIndexablePointerCode(val st: SIndexablePointer, val a: Code[Long]) extends SIndexableCode {
  val pt: PContainer = st.pType

  def code: Code[_] = a

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def codeLoadLength(): Code[Int] = pt.loadLength(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SIndexableValue = {
    val s = SIndexablePointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SIndexableValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SIndexableValue = memoize(cb, name, cb.fieldBuilder)

  def castToArray(cb: EmitCodeBuilder): SIndexableCode = {
    pt match {
      case t: PArray => this
      case t: PCanonicalDict => new SIndexablePointerCode(SIndexablePointer(t.arrayRep), a)
      case t: PCanonicalSet => new SIndexablePointerCode(SIndexablePointer(t.arrayRep), a)
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

class SIndexablePointerSettable(
  val st: SIndexablePointer,
  val a: Settable[Long],
  val length: Settable[Int],
  val elementsAddress: Settable[Long]
) extends SIndexableValue with SSettable {
  val pt: PContainer = st.pType

  def get: SIndexablePointerCode = new SIndexablePointerCode(st, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a, length, elementsAddress)

  def loadLength(): Value[Int] = length

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    val iv = cb.newLocal("pcindval_i", i)
    IEmitCode(cb,
      isElementMissing(cb, iv),
      pt.elementType.loadCheapSCode(cb, pt.loadElement(a, length, iv))) // FIXME loadElement should take elementsAddress
  }

  def isElementMissing(cb: EmitCodeBuilder, i: Code[Int]): Code[Boolean] = pt.isElementMissing(a, i)

  def store(cb: EmitCodeBuilder, pc: SCode): Unit = {
    cb.assign(a, pc.asInstanceOf[SIndexablePointerCode].a)
    cb.assign(length, pt.loadLength(a))
    cb.assign(elementsAddress, pt.firstElementOffset(a, length))
  }

  override def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] = pt.hasMissingValues(a)

  override def forEachDefined(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, Value[Int], SCode) => Unit) {
    st.pType match {
      case pca: PCanonicalArray =>
        val idx = cb.newLocal[Int]("foreach_pca_idx", 0)
        val elementPtr = cb.newLocal[Long]("foreach_pca_elt_ptr", elementsAddress)
        val et = pca.elementType
        cb.whileLoop(idx < length, {
          cb.ifx(isElementMissing(cb, idx),
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

  def reallocate(cb: EmitCodeBuilder, region: Value[Region], newLength: Code[Int], deepCopy: Boolean): Unit = {
    val len = cb.newLocal("new_len", newLength)
    val arr = cb.newLocal("new_address", st.pType.allocate(region, len))
    val elems = cb.newLocal("new_elements", st.pType.firstElementOffset(arr, len))
    cb += st.pType.stagedInitialize(arr, length)

    val toCopy = cb.newLocal("to_copy", len.min(length))
    if (!st.pType.elementType.required) {
      cb += Region.copyFrom(a + st.pType.lengthHeaderBytes, arr + st.pType.lengthHeaderBytes, st.pType.nMissingBytes(toCopy).toL)
    }

    if (deepCopy && st.pType.elementType.containsPointers) {
      val out = CodeLabel()
      forEachDefined(cb) { (cb, i, sc) =>
        cb.ifx(i >= toCopy, cb.goto(out))
        st.pType.elementType.storeAtAddress(cb, elems + st.pType.elementByteOffset(i), region, sc, deepCopy)
      }
      cb.define(out)
    } else {
      cb += Region.copyFrom(elementsAddress, elems, const(st.pType.elementByteSize) * toCopy.toL)
    }

    cb.assign(a, arr)
    cb.assign(length, len)
    cb.assign(elementsAddress, elems)
  }
}
