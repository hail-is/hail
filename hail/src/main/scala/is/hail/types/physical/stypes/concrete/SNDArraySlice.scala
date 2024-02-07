package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitValue}
import is.hail.types.physical.{PCanonicalNDArray, PType}
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.SInt64
import is.hail.types.virtual.{TNDArray, Type}
import is.hail.utils.toRichIterable

final case class SNDArraySlice(pType: PCanonicalNDArray) extends SNDArray {
  override def nDims: Int = pType.nDims

  override def elementByteSize: Long = pType.elementType.byteSize

  override def elementType: SType = pType.elementType.sType

  override def elementPType: PType = pType.elementType

  override lazy val virtualType: TNDArray = pType.virtualType

  override def copiedType: SType = SNDArrayPointer(pType)

  override def storageType(): PType = pType

  override def castRename(t: Type): SType = SNDArrayPointer(pType.deepRename(t))

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue =
    value.st match {
      case SNDArraySlice(`pType`) if !deepCopy => value
    }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = Array.fill(2 * nDims + 1)(LongInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SNDArraySliceSettable = {
    assert(settables.length == 2 * nDims + 1)
    val shape = settables.slice(0, nDims).asInstanceOf[IndexedSeq[Settable[Long @unchecked]]]
    val strides =
      settables.slice(nDims, 2 * nDims).asInstanceOf[IndexedSeq[Settable[Long @unchecked]]]
    val dataFirstElementPointer = settables.last.asInstanceOf[Settable[Long]]
    new SNDArraySliceSettable(this, shape, strides, dataFirstElementPointer)
  }

  override def fromValues(settables: IndexedSeq[Value[_]]): SNDArraySliceValue = {
    assert(settables.length == 2 * nDims + 1)
    val shape = settables.slice(0, nDims).asInstanceOf[IndexedSeq[Value[Long @unchecked]]]
    val strides = settables.slice(nDims, 2 * nDims).asInstanceOf[IndexedSeq[Value[Long @unchecked]]]
    val dataFirstElementPointer = settables.last.asInstanceOf[Value[Long]]
    new SNDArraySliceValue(this, shape.map(SizeValueDyn.apply), strides, dataFirstElementPointer)
  }

  override def containsPointers: Boolean = true

  override def isIsomorphicTo(st: SType): Boolean =
    st match {
      case a: SNDArraySlice => pType.sType isIsomorphicTo a.pType.sType
      case _ => false
    }
}

class SNDArraySliceValue(
  override val st: SNDArraySlice,
  override val shapes: IndexedSeq[SizeValue],
  override val strides: IndexedSeq[Value[Long]],
  override val firstDataAddress: Value[Long],
) extends SNDArrayValue {
  val pt: PCanonicalNDArray = st.pType

  override lazy val valueTuple: IndexedSeq[Value[_]] = shapes ++ strides :+ firstDataAddress

  override def shapeStruct(cb: EmitCodeBuilder): SStackStructValue = {
    val shapeType =
      SStackStruct(st.virtualType.shapeType, Array.fill(st.nDims)(EmitType(SInt64, true)))
    new SStackStructValue(shapeType, shapes.map(x => EmitValue.present(primitive(x))))
  }

  override def loadElementAddress(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder)
    : Code[Long] = {
    assert(indices.size == pt.nDims)
    pt.loadElementFromDataAndStrides(cb, indices, firstDataAddress, strides)
  }

  override def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SValue =
    pt.elementType.loadCheapSCode(cb, loadElementAddress(indices, cb))

  def coerceToShape(cb: EmitCodeBuilder, otherShape: IndexedSeq[SizeValue]): SNDArrayValue = {
    cb.if_(!hasShape(cb, otherShape), cb._fatal("incompatible shapes"))
    new SNDArraySliceValue(st, otherShape, strides, firstDataAddress)
  }

  override def coiterateMutate(
    cb: EmitCodeBuilder,
    region: Value[Region],
    deepCopy: Boolean,
    indexVars: IndexedSeq[String],
    destIndices: IndexedSeq[Int],
    arrays: (SNDArrayValue, IndexedSeq[Int], String)*
  )(
    body: IndexedSeq[SValue] => SValue
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

object SNDArraySliceSettable {
  def apply(sb: SettableBuilder, st: SNDArraySlice, name: String): SNDArraySliceSettable =
    new SNDArraySliceSettable(
      st,
      Array.tabulate(st.pType.nDims)(i => sb.newSettable[Long](s"${name}_nd_shape_$i")),
      Array.tabulate(st.pType.nDims)(i => sb.newSettable[Long](s"${name}_nd_strides_$i")),
      sb.newSettable[Long](s"${name}_nd_first_element"),
    )
}

final class SNDArraySliceSettable(
  st: SNDArraySlice,
  shape: IndexedSeq[Settable[Long]],
  override val strides: IndexedSeq[Settable[Long]],
  override val firstDataAddress: Settable[Long],
) extends SNDArraySliceValue(st, shape.map(SizeValueDyn.apply), strides, firstDataAddress)
    with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = shape ++ strides :+ firstDataAddress

  override def store(cb: EmitCodeBuilder, v: SValue): Unit = v match {
    case v: SNDArraySliceValue =>
      (shape, v.shapes).zipped.foreach((x, s) => cb.assign(x, s))
      (strides, v.strides).zipped.foreach((x, s) => cb.assign(x, s))
      cb.assign(firstDataAddress, v.firstDataAddress)
  }
}
