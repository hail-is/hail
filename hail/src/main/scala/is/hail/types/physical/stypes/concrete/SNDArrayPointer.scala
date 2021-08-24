package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.{SBaseStructCode, SNDArray, SNDArrayCode, SNDArraySettable, SNDArrayValue}
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.physical.{PCanonicalNDArray, PPrimitive, PType}
import is.hail.types.virtual.Type
import is.hail.utils.{FastIndexedSeq, toRichIterable}

final case class SNDArrayPointer(pType: PCanonicalNDArray) extends SNDArray {
  require(!pType.required)

  override def nDims: Int = pType.nDims

  override def elementByteSize: Long = pType.elementType.byteSize

  override def elementType: SType = pType.elementType.sType

  override def elementPType: PType = pType.elementType

  override lazy val virtualType: Type = pType.virtualType

  override def castRename(t: Type): SType = SNDArrayPointer(pType.deepRename(t))

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SNDArrayPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = Array.fill(2 + nDims * 2)(LongInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SNDArrayPointerSettable = {
    val a = settables(0).asInstanceOf[Settable[Long@unchecked]]
    val shape = settables.slice(1, 1 + pType.nDims).asInstanceOf[IndexedSeq[Settable[Long@unchecked]]]
    val strides = settables.slice(1 + pType.nDims, 1 + 2 * pType.nDims).asInstanceOf[IndexedSeq[Settable[Long@unchecked]]]
    val dataFirstElementPointer = settables.last.asInstanceOf[Settable[Long]]
    assert(a.ti == LongInfo)
    new SNDArrayPointerSettable(this, a, shape, strides, dataFirstElementPointer)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SNDArrayPointerValue = {
    val a = values(0).asInstanceOf[Value[Long@unchecked]]
    val shape = values.slice(1, 1 + pType.nDims).asInstanceOf[IndexedSeq[Value[Long@unchecked]]]
    val strides = values.slice(1 + pType.nDims, 1 + 2 * pType.nDims).asInstanceOf[IndexedSeq[Value[Long@unchecked]]]
    val dataFirstElementPointer = values.last.asInstanceOf[Value[Long]]
    assert(a.ti == LongInfo)
    new SNDArrayPointerValue(this, a, shape, strides, dataFirstElementPointer)
  }

  override def storageType(): PType = pType

  override def copiedType: SType = SNDArrayPointer(pType.copiedType.asInstanceOf[PCanonicalNDArray])

  override def containsPointers: Boolean = pType.containsPointers
}

class SNDArrayPointerValue(
  val st: SNDArrayPointer,
  val a: Value[Long],
  val shape: IndexedSeq[Value[Long]],
  val strides: IndexedSeq[Value[Long]],
  val dataFirstElement: Value[Long]
) extends SNDArrayValue {
  val pt: PCanonicalNDArray = st.pType

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastIndexedSeq(a) ++ shape ++ strides ++ FastIndexedSeq(dataFirstElement)

  override def loadElementAddress(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Long] = {
    assert(indices.size == pt.nDims)
    pt.loadElementFromDataAndStrides(cb, indices, dataFirstElement, strides)
  }

  override def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SCode = {
    assert(indices.size == pt.nDims)
    pt.elementType.loadCheapSCode(cb, loadElementAddress(indices, cb))
  }

  override def get: SNDArrayPointerCode = new SNDArrayPointerCode(st, a)

  override def shapes(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = shape

  override def strides(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = strides

  override def firstDataAddress(cb: EmitCodeBuilder): Value[Long] = dataFirstElement

  override def coiterateMutate(
    cb: EmitCodeBuilder,
    region: Value[Region],
    deepCopy: Boolean,
    indexVars: IndexedSeq[String],
    destIndices: IndexedSeq[Int],
    arrays: (SNDArrayCode, IndexedSeq[Int], String)*
  )(body: IndexedSeq[SCode] => SCode
  ): Unit = {
    SNDArray._coiterate(cb, indexVars, (this.get, destIndices, "dest") +: arrays: _*) { ptrs =>
      val codes = (this.get +: arrays.map(_._1)).zip(ptrs).toFastIndexedSeq.map { case (array, ptr) =>
        val pt: PType = array.st.pType.elementType
        pt.loadCheapSCode(cb, pt.loadFromNested(ptr)).get
      }
      pt.elementType.storeAtAddress(cb, ptrs.head, region, body(codes), deepCopy)
    }
  }
}

object SNDArrayPointerSettable {
  def apply(sb: SettableBuilder, st: SNDArrayPointer, name: String): SNDArrayPointerSettable = {
    new SNDArrayPointerSettable(st, sb.newSettable[Long](name),
      Array.tabulate(st.pType.nDims)(i => sb.newSettable[Long](s"${name}_nd_shape_$i")),
      Array.tabulate(st.pType.nDims)(i => sb.newSettable[Long](s"${name}_nd_strides_$i")),
      sb.newSettable[Long](s"${name}_nd_first_element")
    )
  }
}

final class SNDArrayPointerSettable(
  st: SNDArrayPointer,
  a: Settable[Long],
  shape: IndexedSeq[Settable[Long]],
  strides: IndexedSeq[Settable[Long]],
  dataFirstElement: Settable[Long]
 ) extends SNDArrayPointerValue(st, a, shape, strides, dataFirstElement) with SNDArraySettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a) ++ shape ++ strides ++ FastIndexedSeq(dataFirstElement)

  override def store(cb: EmitCodeBuilder, v: SCode): Unit = {
    cb.assign(a, v.asInstanceOf[SNDArrayPointerCode].a)
    pt.loadShapes(cb, a, shape)
    pt.loadStrides(cb, a, strides)
    cb.assign(dataFirstElement, pt.dataFirstElementPointer(a))
  }
}

class SNDArrayPointerCode(val st: SNDArrayPointer, val a: Code[Long]) extends SNDArrayCode {
  val pt: PCanonicalNDArray = st.pType

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SNDArrayPointerValue = {
    val s = SNDArrayPointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  override def memoize(cb: EmitCodeBuilder, name: String): SNDArrayPointerValue =
    memoize(cb, name, cb.localBuilder)

  override def memoizeField(cb: EmitCodeBuilder, name: String): SNDArrayPointerValue =
    memoize(cb, name, cb.fieldBuilder)

  override def shape(cb: EmitCodeBuilder): SBaseStructCode =
    pt.shapeType.loadCheapSCode(cb, pt.representation.loadField(a, "shape")).get
}
