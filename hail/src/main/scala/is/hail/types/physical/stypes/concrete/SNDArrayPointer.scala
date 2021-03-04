package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Code, IntInfo, LongInfo, Settable, SettableBuilder, TypeInfo, Value, const}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.interfaces.{SNDArray, SNDArrayValue}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.{PBaseStructCode, PCanonicalNDArray, PCode, PNDArray, PNDArrayCode, PNDArrayValue, PSettable, PType, PValue}
import is.hail.utils.FastIndexedSeq

case class SNDArrayPointer(pType: PCanonicalNDArray) extends SNDArray {
  def nDims: Int = pType.nDims

  override def elementType: SType = pType.elementType.sType

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SNDArrayPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    if (pt == this.pType)
      new SNDArrayPointerCode(this, addr)
    else
      coerceOrCopy(cb, region, pt.loadCheapPCode(cb, addr), deepCopy = false)
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SNDArrayPointerSettable = {
    val a = settables(0).asInstanceOf[Settable[Long@unchecked]]
    val shape = settables.slice(1, 1 + pType.nDims).asInstanceOf[IndexedSeq[Settable[Long@unchecked]]]
    val strides = settables.slice(1 + pType.nDims, 1 + 2 * pType.nDims).asInstanceOf[IndexedSeq[Settable[Long@unchecked]]]
    val dataFirstElementPointer = settables.last.asInstanceOf[Settable[Long]]
    assert(a.ti == LongInfo)
    new SNDArrayPointerSettable(this, a, shape, strides, dataFirstElementPointer)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SNDArrayPointerCode = {
    val IndexedSeq(a: Code[Long@unchecked]) = codes
    assert(a.ti == LongInfo)
    new SNDArrayPointerCode(this, a)
  }

  def canonicalPType(): PType = pType
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

class SNDArrayPointerSettable(
   val st: SNDArrayPointer,
   val a: Settable[Long],
   val shape: IndexedSeq[Settable[Long]],
   val strides: IndexedSeq[Settable[Long]],
   val dataFirstElement: Settable[Long]
 ) extends PNDArrayValue with PSettable {
  val pt: PCanonicalNDArray = st.pType

  def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): PCode = {
    assert(indices.size == pt.nDims)
    pt.elementType.loadCheapPCode(cb, pt.loadElementFromDataAndStrides(cb, indices, dataFirstElement, strides))
  }

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a) ++ shape ++ strides ++ FastIndexedSeq(dataFirstElement)

  def store(cb: EmitCodeBuilder, v: PCode): Unit = {
    cb.assign(a, v.asInstanceOf[SNDArrayPointerCode].a)
    pt.loadShapes(cb, a, shape)
    pt.loadStrides(cb, a, strides)
    cb.assign(dataFirstElement, pt.dataFirstElementPointer(a))
  }

  override def get: PCode = new SNDArrayPointerCode(st, a)

  override def outOfBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Boolean] = {
    val shape = this.shapes(cb)
    val outOfBounds = cb.newLocal[Boolean]("sndarray_out_of_bounds", false)

    (0 until pt.nDims).foreach { dimIndex =>
      cb.assign(outOfBounds, outOfBounds || (indices(dimIndex) >= shape(dimIndex)))
    }
    outOfBounds
  }

  override def assertInBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder, errorId: Int): Code[Unit] = {
    val shape = this.shapes(cb)
    Code.foreach(0 until pt.nDims) { dimIndex =>
      val eMsg = const("Index ").concat(indices(dimIndex).toS)
        .concat(s" is out of bounds for axis $dimIndex with size ")
        .concat(shape(dimIndex).toS)
      (indices(dimIndex) >= shape(dimIndex)).orEmpty(Code._fatalWithID[Unit](eMsg, errorId))
    }
  }

  override def shapes(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = shape

  override def strides(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = strides

  override def sameShape(other: SNDArrayValue, cb: EmitCodeBuilder): Code[Boolean] = {
    val otherShapes = other.shapes(cb)
    val b = cb.newLocal[Boolean]("sameShape_b", true)
    assert(shape.length == otherShapes.length)
    shape.zip(otherShapes).foreach { case (s1, s2) =>
      cb.assign(b, b && s1.ceq(s2))
    }
    b
  }

  def firstDataAddress(cb: EmitCodeBuilder): Value[Long] = dataFirstElement
}

class SNDArrayPointerCode(val st: SNDArrayPointer, val a: Code[Long]) extends PNDArrayCode {
  val pt: PCanonicalNDArray = st.pType

  override def code: Code[_] = a

  override def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PNDArrayValue = {
    val s = SNDArrayPointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  override def memoize(cb: EmitCodeBuilder, name: String): PNDArrayValue = memoize(cb, name, cb.localBuilder)

  override def memoizeField(cb: EmitCodeBuilder, name: String): PValue = memoize(cb, name, cb.fieldBuilder)

  override def shape(cb: EmitCodeBuilder): PBaseStructCode = pt.shapeType.loadCheapPCode(cb, pt.representation.loadField(a, "shape"))
}
