package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.{PCanonicalNDArray, PType}
import is.hail.types.virtual.Type
import is.hail.utils.{FastSeq, toRichIterable}

final case class SNDArrayPointer(pType: PCanonicalNDArray) extends SNDArray {
  require(!pType.required)

  override def nDims: Int = pType.nDims

  override def elementByteSize: Long = pType.elementType.byteSize

  override def elementType: SType = pType.elementType.sType

  override def elementPType: PType = pType.elementType

  override lazy val virtualType: Type = pType.virtualType

  override def castRename(t: Type): SType = SNDArrayPointer(pType.deepRename(t))

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue =
    value match {
      case value: SNDArrayValue =>
        val a = pType.store(cb, region, value, deepCopy)
        new SNDArrayPointerValue(this, a, value.shapes, value.strides, cb.memoize(pType.dataFirstElementPointer(a)))
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
    new SNDArrayPointerValue(this, a, shape.map(SizeValueDyn.apply), strides, dataFirstElementPointer)
  }

  override def storageType(): PType = pType

  override def copiedType: SType = SNDArrayPointer(pType.copiedType.asInstanceOf[PCanonicalNDArray])

  override def containsPointers: Boolean = pType.containsPointers
}

class SNDArrayPointerValue(
  val st: SNDArrayPointer,
  val a: Value[Long],
  val shapes: IndexedSeq[SizeValue],
  val strides: IndexedSeq[Value[Long]],
  val firstDataAddress: Value[Long]
) extends SNDArrayValue {
  val pt: PCanonicalNDArray = st.pType

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(a) ++ shapes ++ strides ++ FastSeq(firstDataAddress)

  override def shapeStruct(cb: EmitCodeBuilder): SBaseStructValue =
    pt.shapeType.loadCheapSCode(cb, pt.representation.loadField(a, "shape"))

  override def loadElementAddress(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Long] = {
    assert(indices.size == pt.nDims)
    pt.loadElementFromDataAndStrides(cb, indices, firstDataAddress, strides)
  }

  override def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SValue = {
    assert(indices.size == pt.nDims)
    pt.elementType.loadCheapSCode(cb, loadElementAddress(indices, cb))
  }

  override def coerceToShape(cb: EmitCodeBuilder, otherShape: IndexedSeq[SizeValue]): SNDArrayValue = {
    cb.ifx(!hasShape(cb, otherShape), cb._fatal("incompatible shapes"))
    new SNDArrayPointerValue(st, a, otherShape, strides, firstDataAddress)
  }

  override def coiterateMutate(
    cb: EmitCodeBuilder,
    region: Value[Region],
    deepCopy: Boolean,
    indexVars: IndexedSeq[String],
    destIndices: IndexedSeq[Int],
    arrays: (SNDArrayValue, IndexedSeq[Int], String)*
  )(body: IndexedSeq[SValue] => SValue
  ): Unit = {
    SNDArray._coiterate(cb, indexVars, (this, destIndices, "dest") +: arrays: _*) { ptrs =>
      val codes = (this +: arrays.map(_._1)).zip(ptrs).toFastSeq.map { case (array, ptr) =>
        val pt: PType = array.st.pType.elementType
        pt.loadCheapSCode(cb, pt.loadFromNested(ptr))
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
  override val a: Settable[Long],
  val shape: IndexedSeq[Settable[Long]],
  override val strides: IndexedSeq[Settable[Long]],
  override val firstDataAddress: Settable[Long]
) extends SNDArrayPointerValue(st, a, shape.map(SizeValueDyn.apply), strides, firstDataAddress) with SNDArraySettable {
  def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(a) ++ shape ++ strides ++ FastSeq(firstDataAddress)

  def store(cb: EmitCodeBuilder, v: SValue): Unit = v match {
    case v: SNDArrayPointerValue =>
      cb.assign(a, v.a)
      (shape, v.shapes).zipped.foreach(cb.assign(_, _))
      (strides, v.strides).zipped.foreach(cb.assign(_, _))
      cb.assign(firstDataAddress, v.firstDataAddress)
  }
}