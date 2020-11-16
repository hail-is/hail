package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{???, Code, IntInfo, LongInfo, Settable, SettableBuilder, TypeInfo, Value, const}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.{PBaseStructCode, PCanonicalNDArray, PCode, PNDArray, PNDArrayCode, PNDArrayValue, PSettable, PType, PValue}
import is.hail.utils.FastIndexedSeq

trait SNDArray extends SType

case class SNDArrayPointer(pType: PNDArray) extends SNDArray {
  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = pType.codeOrdering(mb)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: PCode, deepCopy: Boolean): PCode = {
    new SNDArrayPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): PCode = {
    if (pt == this.pType)
      new SNDArrayPointerCode(this, addr)
    else
      coerceOrCopy(cb, region, pt.getPointerTo(cb, addr), deepCopy = false)
  }

}

object SNDArrayPointerSettable {
  def apply(sb: SettableBuilder, st: SNDArrayPointer, name: String): SNDArrayPointerSettable = {
    new SNDArrayPointerSettable(st, sb.newSettable[Long](name))
  }
}

class SNDArrayPointerSettable(val st: SNDArrayPointer, val a: Settable[Long]) extends PNDArrayValue with PSettable {
  val pt: PNDArray = st.pType

  def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): PCode = {
    assert(indices.size == pt.nDims)
    pt.elementType.getPointerTo(cb, pt.loadElement(cb, indices, a))
  }

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  def store(cb: EmitCodeBuilder, v: PCode): Unit = {
    cb.assign(a, v.asInstanceOf[SNDArrayPointerCode].a)
  }

  override def get: PCode = new SNDArrayPointerCode(st, a)

  override def outOfBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Boolean] = {
    val shape = this.shapes(cb)
    val outOfBounds = cb.newLocal[Boolean]("sndarray_out_of_bounds", false)

    val idx = cb.newLocal[Int]("i")

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

  override def shapes(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = Array.tabulate(pt.nDims) { i =>
    new Value[Long] {
      def get: Code[Long] = pt.loadShape(cb, a, i)
    }
  }

  override def strides(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = Array.tabulate(pt.nDims) { i =>
    new Value[Long] {
      def get: Code[Long] = pt.loadStride(cb, a, i)
    }
  }

  override def sameShape(other: PNDArrayValue, cb: EmitCodeBuilder): Code[Boolean] = {
    val comparator = this.pt.shape.pType.codeOrdering(cb.emb, other.pt.shape.pType)
    val thisShape = this.pt.shape.load(this.a).asInstanceOf[Code[comparator.T]]
    val otherShape = other.pt.shape.load(other.value.asInstanceOf[Value[Long]]).asInstanceOf[Code[comparator.T]]
    comparator.equivNonnull(thisShape, otherShape)
  }
}

class SNDArrayPointerCode(val st: SNDArrayPointer, val a: Code[Long]) extends PNDArrayCode {
  val pt: PNDArray = st.pType

  override def code: Code[_] = a

  override def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PNDArrayValue = {
    val s = SNDArrayPointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  override def memoize(cb: EmitCodeBuilder, name: String): PNDArrayValue = memoize(cb, name, cb.localBuilder)

  override def memoizeField(cb: EmitCodeBuilder, name: String): PValue = memoize(cb, name, cb.fieldBuilder)

  override def shape: PBaseStructCode = PCode(pt.shape.pType, pt.shape.load(a)).asBaseStruct
}
