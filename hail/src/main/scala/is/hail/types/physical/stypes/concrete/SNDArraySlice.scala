package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder}
import is.hail.types.physical.{PCanonicalNDArray, PNDArray, PNumeric, PPrimitive, PType}
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType, SValue}
import is.hail.types.physical.stypes.interfaces.{SBaseStructCode, SNDArray, SNDArrayCode, SNDArrayValue, primitive}
import is.hail.types.physical.stypes.primitives.SInt64
import is.hail.types.virtual.{TInt64, TNDArray, TTuple, Type}
import is.hail.utils.{FastIndexedSeq, toRichIterable}

final case class SNDArraySlice(pType: PCanonicalNDArray) extends SNDArray {
  override def nDims: Int = pType.nDims

  override def elementByteSize: Long = pType.elementType.byteSize

  override def elementType: SType = pType.elementType.sType

  override def elementPType: PType = pType.elementType

  override lazy val virtualType: TNDArray = pType.virtualType

  override def copiedType: SType = SNDArrayPointer(pType)

  override def storageType(): PType = pType

  override def castRename(t: Type): SType = SNDArrayPointer(pType.deepRename(t))

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode =
    value.st match {
      case SNDArraySlice(`pType`) if !deepCopy => value
    }


  override def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = Array.fill(2*nDims + 1)(LongInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SNDArraySliceSettable = {
    assert(settables.length == 2*nDims + 1)
    val shape = settables.slice(0, nDims).asInstanceOf[IndexedSeq[Settable[Long@unchecked]]]
    val strides = settables.slice(nDims, 2 * nDims).asInstanceOf[IndexedSeq[Settable[Long@unchecked]]]
    val dataFirstElementPointer = settables.last.asInstanceOf[Settable[Long]]
    new SNDArraySliceSettable(this, shape, strides, dataFirstElementPointer)
  }

  override def fromCodes(codes: IndexedSeq[Code[_]]): SNDArraySliceCode = {
    val codesT = codes.asInstanceOf[IndexedSeq[Code[Long@unchecked]]]
    assert(codesT.length == 2*nDims + 1)
    val shape = codesT.slice(0, nDims)
    val strides = codesT.slice(nDims, 2*nDims)
    val dataFirstElement = codesT.last
    new SNDArraySliceCode(this, shape, strides, dataFirstElement)
  }

  override def fromValues(settables: IndexedSeq[Value[_]]): SNDArraySliceValue = {
    assert(settables.length == 2*nDims + 1)
    val shape = settables.slice(0, nDims).asInstanceOf[IndexedSeq[Value[Long@unchecked]]]
    val strides = settables.slice(nDims, 2 * nDims).asInstanceOf[IndexedSeq[Value[Long@unchecked]]]
    val dataFirstElementPointer = settables.last.asInstanceOf[Value[Long]]
    new SNDArraySliceValue(this, shape, strides, dataFirstElementPointer)
  }

  override def containsPointers: Boolean = true
}

class SNDArraySliceValue(
  val st: SNDArraySlice,
  val shape: IndexedSeq[Value[Long]],
  val strides: IndexedSeq[Value[Long]],
  val dataFirstElement: Value[Long]
) extends SNDArrayValue {
  val pt: PCanonicalNDArray = st.pType

  override def loadElementAddress(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Long] = {
    assert(indices.size == pt.nDims)
    pt.loadElementFromDataAndStrides(cb, indices, dataFirstElement, strides)
  }

  override def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SCode =
    pt.elementType.loadCheapSCode(cb, loadElementAddress(indices, cb))

  override def get: SNDArraySliceCode = new SNDArraySliceCode(st, shape, strides, dataFirstElement)

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
  override val shape: IndexedSeq[Settable[Long]],
  override val strides: IndexedSeq[Settable[Long]],
  override val dataFirstElement: Settable[Long]
) extends SNDArraySliceValue(st, shape, strides, dataFirstElement) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = shape ++ strides :+ dataFirstElement

  override def store(cb: EmitCodeBuilder, v: SCode): Unit = {
    val vSlice = v.asInstanceOf[SNDArraySliceCode]
    shape.zip(vSlice.shape).foreach { case (x, s) => cb.assign(x, s) }
    strides.zip(vSlice.strides).foreach { case (x, s) => cb.assign(x, s) }
    cb.assign(dataFirstElement, vSlice.dataFirstElement)
  }
}

class SNDArraySliceCode(val st: SNDArraySlice, val shape: IndexedSeq[Code[Long]], val strides: IndexedSeq[Code[Long]], val dataFirstElement: Code[Long]) extends SNDArrayCode {
  override def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = shape ++ strides :+ dataFirstElement

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SNDArrayValue = {
    val s = SNDArraySliceSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  override def memoize(cb: EmitCodeBuilder, name: String): SNDArrayValue =
    memoize(cb, name, cb.localBuilder)

  override def memoizeField(cb: EmitCodeBuilder, name: String): SValue =
    memoize(cb, name, cb.fieldBuilder)

  override def shape(cb: EmitCodeBuilder): SStackStructCode = {
    val shapeType = SStackStruct(st.virtualType.shapeType, Array.fill(st.nDims)(EmitType(SInt64, true)))
    new SStackStructCode(shapeType, shape.map(x => EmitCode.present(cb.emb, primitive(x))))
  }
}
