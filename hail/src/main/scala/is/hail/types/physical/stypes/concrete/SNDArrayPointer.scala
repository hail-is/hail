package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.{SBaseStructCode, SNDArray, SNDArrayCode, SNDArrayValue}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PCanonicalNDArray, PType}
import is.hail.types.virtual.Type
import is.hail.utils.{FastIndexedSeq, toRichIterable}

case class SNDArrayPointer(pType: PCanonicalNDArray) extends SNDArray {
  require(!pType.required)

  def nDims: Int = pType.nDims

  override def elementType: SType = pType.elementType.sType

  override def elementPType: PType = pType.elementType

  lazy val virtualType: Type = pType.virtualType

  override def castRename(t: Type): SType = SNDArrayPointer(pType.deepRename(t))

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    new SNDArrayPointerCode(this, pType.store(cb, region, value, deepCopy))
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(LongInfo)

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = Array.fill(2 + nDims * 2)(LongInfo)

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
 ) extends SNDArrayValue with SSettable {
  val pt: PCanonicalNDArray = st.pType

  def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SCode = {
    assert(indices.size == pt.nDims)
    pt.elementType.loadCheapSCode(cb, pt.loadElementFromDataAndStrides(cb, indices, dataFirstElement, strides))
  }

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a) ++ shape ++ strides ++ FastIndexedSeq(dataFirstElement)

  def store(cb: EmitCodeBuilder, v: SCode): Unit = {
    cb.assign(a, v.asInstanceOf[SNDArrayPointerCode].a)
    pt.loadShapes(cb, a, shape)
    pt.loadStrides(cb, a, strides)
    cb.assign(dataFirstElement, pt.dataFirstElementPointer(a))
  }

  override def get: SNDArrayPointerCode = new SNDArrayPointerCode(st, a)

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

  // Note: to iterate through an array in column major order, make sure the indices are in ascending order. E.g.
  // A.coiterate(cb, region, IndexedSeq("i", "j"), IndexedSeq((A, IndexedSeq(0, 1), "A"), (B, IndexedSeq(0, 1), "B")), {
  //   SCode.add(cb, a, b)
  // })
  // computes A += B.
  def coiterateMutate(
    cb: EmitCodeBuilder,
    region: Value[Region],
    deepCopy: Boolean,
    indexVars: IndexedSeq[String],
    destIndices: IndexedSeq[Int],
    arrays: (SNDArrayCode, IndexedSeq[Int], String)*
  )(body: IndexedSeq[SCode] => SCode
  ): Unit = {

    val indexSizes = new Array[Settable[Int]](indexVars.length)
    val indexCoords = Array.tabulate(indexVars.length) { i => cb.newLocal[Int](indexVars(i)) }

    case class ArrayInfo(
      array: SNDArrayValue,
      strides: IndexedSeq[Value[Long]],
      pos: IndexedSeq[Settable[Long]],
      elt: SCode,
      indexToDim: Map[Int, Int],
      name: String)

    val arrayValues = arrays.map { case (_array, indices, name) =>
      val array = _array.memoize(cb, s"${name}_copy")
      (array, indices, name)
    }

    val info = ((this, destIndices, "dest") +: arrayValues).toIndexedSeq.map { case (array, indices, name) =>
      for (idx <- indices) assert(idx < indexVars.length && idx >= 0)
      // FIXME: relax this assumption to handle transposing, non-column major
      for (i <- 0 until indices.length - 1) assert(indices(i) < indices(i+1))
      assert(indices.length == array.st.nDims)

      val shape = array.shapes(cb)
      for (i <- indices.indices) {
        val idx = indices(i)
        if (indexSizes(idx) == null) {
          indexSizes(idx) = cb.newLocal[Int](s"${indexVars(idx)}_max")
          cb.assign(indexSizes(idx), shape(i).toI)
        } else {
          cb.ifx(indexSizes(idx).cne(shape(i).toI), s"${indexVars(idx)} indexes incompatible dimensions")
        }
      }
      val strides = array.strides(cb)
      val pos = Array.tabulate(array.st.nDims + 1) { i => cb.newLocal[Long](s"$name$i") }
      // FIXME: need to use `pos` of smallest index var to support non-column major
      val pt: PType = array.st.pType.elementType
      val elt = pt.loadCheapSCode(cb, pt.loadFromNested(pos(0)))
      val indexToDim = indices.zipWithIndex.toMap
      ArrayInfo(array, strides, pos, elt, indexToDim, name)
    }

    def recurLoopBuilder(idx: Int): Unit = {
      if (idx < 0) {
        pt.storeAtAddress(cb, info.head.pos(0), region, body(info.map(_.elt)), deepCopy)
      } else {
        val coord = indexCoords(idx)
        def init(): Unit = {
          cb.assign(coord, 0)
          for (n <- arrays.indices) {
            if (info(n).indexToDim.contains(idx)) {
              val i = info(n).indexToDim(idx)
              // FIXME: assumes array's indices in ascending order
              cb.assign(info(n).pos(i), info(n).pos(i+1))
            }
          }
        }
        def increment(): Unit = {
          cb.assign(coord, coord + 1)
          for (n <- arrays.indices) {
            if (info(n).indexToDim.contains(idx)) {
              val i = info(n).indexToDim(idx)
              cb.assign(info(n).pos(i), info(n).pos(i) + info(n).strides(i))
            }
          }
        }

        cb.forLoop(init(), coord < indexSizes(idx), increment(), recurLoopBuilder(idx - 1))
      }
    }

    for (n <- arrays.indices) {
      cb.assign(info(n).pos(info(n).array.st.nDims), info(n).array.firstDataAddress(cb))
    }
    recurLoopBuilder(indexVars.length - 1)
  }
}

class SNDArrayPointerCode(val st: SNDArrayPointer, val a: Code[Long]) extends SNDArrayCode {
  val pt: PCanonicalNDArray = st.pType

  override def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SNDArrayValue = {
    val s = SNDArrayPointerSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  override def memoize(cb: EmitCodeBuilder, name: String): SNDArrayValue = memoize(cb, name, cb.localBuilder)

  override def memoizeField(cb: EmitCodeBuilder, name: String): SValue = memoize(cb, name, cb.fieldBuilder)

  override def shape(cb: EmitCodeBuilder): SBaseStructCode = pt.shapeType.loadCheapSCode(cb, pt.representation.loadField(a, "shape"))
}
