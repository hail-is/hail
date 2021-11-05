package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.SInt64
import is.hail.types.physical.stypes._
import is.hail.types.physical.{PCanonicalNDArray, PNumeric, PPrimitive, PType}
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

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue =
    value.st match {
      case SNDArraySlice(`pType`) if !deepCopy => value
    }


  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = Array.fill(2*nDims + 1)(LongInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SNDArraySliceSettable = {
    assert(settables.length == 2*nDims + 1)
    val shape = settables.slice(0, nDims).asInstanceOf[IndexedSeq[Settable[Long@unchecked]]]
    val strides = settables.slice(nDims, 2 * nDims).asInstanceOf[IndexedSeq[Settable[Long@unchecked]]]
    val dataFirstElementPointer = settables.last.asInstanceOf[Settable[Long]]
    new SNDArraySliceSettable(this, shape, strides, dataFirstElementPointer)
  }

  override def fromValues(settables: IndexedSeq[Value[_]]): SNDArraySliceValue = {
    assert(settables.length == 2*nDims + 1)
    val shape = settables.slice(0, nDims).asInstanceOf[IndexedSeq[Value[Long@unchecked]]]
    val strides = settables.slice(nDims, 2 * nDims).asInstanceOf[IndexedSeq[Value[Long@unchecked]]]
    val dataFirstElementPointer = settables.last.asInstanceOf[Value[Long]]
    new SNDArraySliceValue(this, shape.map(SizeValueDyn.apply), strides, dataFirstElementPointer)
  }

  override def containsPointers: Boolean = true
}

class SNDArraySliceValue(
  override val st: SNDArraySlice,
  override val shapes: IndexedSeq[SizeValue],
  override val strides: IndexedSeq[Value[Long]],
  override val firstDataAddress: Value[Long]
) extends SNDArrayValue {
  val pt: PCanonicalNDArray = st.pType

  override lazy val valueTuple: IndexedSeq[Value[_]] = shapes ++ strides :+ firstDataAddress

  override def shapeStruct(cb: EmitCodeBuilder): SStackStructValue = {
    val shapeType = SStackStruct(st.virtualType.shapeType, Array.fill(st.nDims)(EmitType(SInt64, true)))
    new SStackStructValue(shapeType, shapes.map(x => EmitValue.present(primitive(x))))
  }

  override def loadElementAddress(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Long] = {
    assert(indices.size == pt.nDims)
    pt.loadElementFromDataAndStrides(cb, indices, firstDataAddress, strides)
  }

  override def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SValue =
    pt.elementType.loadCheapSCode(cb, loadElementAddress(indices, cb))

  override def get: SNDArraySliceCode = new SNDArraySliceCode(st, shapes, strides, firstDataAddress)

  def coerceToShape(cb: EmitCodeBuilder, otherShape: IndexedSeq[SizeValue]): SNDArrayValue = {
    cb.ifx(!hasShape(cb, otherShape), cb._fatal("incompatible shapes"))
    new SNDArraySliceValue(st, otherShape, strides, firstDataAddress)
  }

  // FIXME: only optimized for column major
  override def setToZero(cb: EmitCodeBuilder): Unit = {
    val eltType = pt.elementType.asInstanceOf[PNumeric with PPrimitive]

    val contiguousDims = contiguousDimensions(cb)

    def recur(startPtr: Value[Long], dim: Int): Unit =
      if (dim > 0) {
        cb.ifx(contiguousDims.ceq(dim), {
          cb += Region.setMemory(startPtr, shapes(dim-1) * strides(dim-1), 0: Byte)
        }, {
          val ptr = cb.mb.newLocal[Long](s"NDArray_setToZero_ptr_$dim")
          val end = cb.mb.newLocal[Long](s"NDArray_setToZero_end_$dim")
          cb.assign(ptr, startPtr)
          cb.assign(end, ptr + strides(dim-1) * shapes(dim-1))
          cb.forLoop({}, ptr < end, cb.assign(ptr, ptr + strides(dim-1)), recur(ptr, dim - 1))
        })
      } else {
        eltType.storePrimitiveAtAddress(cb, startPtr, primitive(eltType.virtualType, eltType.zero))
      }

    recur(firstDataAddress, st.nDims)
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
      val codes = (this +: arrays.map(_._1)).zip(ptrs).toFastIndexedSeq.map { case (array, ptr) =>
        val pt: PType = array.st.pType.elementType
        pt.loadCheapSCode(cb, pt.loadFromNested(ptr))
      }
      pt.elementType.storeAtAddress(cb, ptrs.head, region, body(codes), deepCopy)
    }
  }
}

object SNDArraySliceSettable {
  def apply(sb: SettableBuilder, st: SNDArraySlice, name: String): SNDArraySliceSettable = {
    new SNDArraySliceSettable(st,
      Array.tabulate(st.pType.nDims)(i => sb.newSettable[Long](s"${name}_nd_shape_$i")),
      Array.tabulate(st.pType.nDims)(i => sb.newSettable[Long](s"${name}_nd_strides_$i")),
      sb.newSettable[Long](s"${name}_nd_first_element")
    )
  }
}

final class SNDArraySliceSettable(
  st: SNDArraySlice,
  shape: IndexedSeq[Settable[Long]],
  override val strides: IndexedSeq[Settable[Long]],
  override val firstDataAddress: Settable[Long]
) extends SNDArraySliceValue(st, shape.map(SizeValueDyn.apply), strides, firstDataAddress) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = shape ++ strides :+ firstDataAddress

  override def store(cb: EmitCodeBuilder, v: SCode): Unit = {
    val vSlice = v.asInstanceOf[SNDArraySliceCode]
    shape.zip(vSlice.shape).foreach { case (x, s) => cb.assign(x, s) }
    strides.zip(vSlice.strides).foreach { case (x, s) => cb.assign(x, s) }
    cb.assign(firstDataAddress, vSlice.dataFirstElement)
  }
}

class SNDArraySliceCode(val st: SNDArraySlice, val shape: IndexedSeq[Code[Long]], val strides: IndexedSeq[Code[Long]], val dataFirstElement: Code[Long]) extends SNDArrayCode {
  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SNDArrayValue = {
    val s = SNDArraySliceSettable(sb, st, name)
    s.store(cb, this)
    s
  }

  override def memoize(cb: EmitCodeBuilder, name: String): SNDArrayValue =
    memoize(cb, name, cb.localBuilder)

  override def memoizeField(cb: EmitCodeBuilder, name: String): SValue =
    memoize(cb, name, cb.fieldBuilder)
}
