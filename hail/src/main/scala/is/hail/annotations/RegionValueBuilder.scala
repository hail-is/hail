package is.hail.annotations

import is.hail.backend.HailStateManager
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._

class RegionValueBuilder(sm: HailStateManager, var region: Region) {
  def this(sm: HailStateManager) = this(sm, null)

  var start: Long = _
  var root: PType = _

  val typestk = new ObjectArrayStack[PType]()
  val indexstk = new IntArrayStack()
  val offsetstk = new LongArrayStack()
  val elementsOffsetstk = new LongArrayStack()

  def inactive: Boolean =
    root == null && typestk.isEmpty && offsetstk.isEmpty && elementsOffsetstk.isEmpty && indexstk.isEmpty

  def clear(): Unit = {
    root = null
    typestk.clear()
    offsetstk.clear()
    elementsOffsetstk.clear()
    indexstk.clear()
  }

  def set(newRegion: Region): Unit = {
    assert(inactive)
    region = newRegion
  }

  def currentOffset(): Long = {
    if (typestk.isEmpty)
      start
    else {
      val i = indexstk.top
      typestk.top match {
        case t: PCanonicalBaseStruct =>
          offsetstk.top + t.byteOffsets(i)
        case t: PArray =>
          t.incrementElementOffset(elementsOffsetstk.top, i)
      }
    }
  }

  def currentType(): PType = {
    if (typestk.isEmpty)
      root
    else {
      typestk.top match {
        case t: PCanonicalBaseStruct =>
          val i = indexstk.top
          t.types(i)
        case t: PArray =>
          t.elementType
      }
    }
  }

  def start(newRoot: PType): Unit = {
    assert(inactive)
    root = newRoot
  }

  def allocateRoot(): Unit = {
    assert(typestk.isEmpty)
    root match {
      case _: PArray =>
      case _: PBinary =>
      case _ =>
        start = region.allocate(root.alignment, root.byteSize)
    }
  }

  def end(): Long = {
    assert(root != null)
    root = null
    assert(inactive)
    start
  }

  def advance(): Unit = {
    if (indexstk.nonEmpty)
      indexstk(0) = indexstk(0) + 1
  }

  def startBaseStruct(init: Boolean = true, setMissing: Boolean = false): Unit = {
    val t = currentType().asInstanceOf[PBaseStruct]
    if (typestk.isEmpty)
      allocateRoot()

    val off = currentOffset()
    typestk.push(t)
    offsetstk.push(off)
    indexstk.push(0)

    if (init)
      t.initialize(off, setMissing)
  }

  def endBaseStruct(): Unit = {
    val t = typestk.top.asInstanceOf[PBaseStruct]
    typestk.pop()
    offsetstk.pop()
    val last = indexstk.pop()
    assert(last == t.size)

    advance()
  }

  def startStruct(init: Boolean = true, setMissing: Boolean = false): Unit = {
    assert(currentType().isInstanceOf[PStruct])
    startBaseStruct(init, setMissing)
  }

  def endStruct(): Unit = {
    assert(typestk.top.isInstanceOf[PStruct])
    endBaseStruct()
  }

  def startTuple(init: Boolean = true): Unit = {
    assert(currentType().isInstanceOf[PTuple])
    startBaseStruct(init)
  }

  def endTuple(): Unit = {
    assert(typestk.top.isInstanceOf[PTuple])
    endBaseStruct()
  }

  def startArray(length: Int, init: Boolean = true): Unit = {
    startArrayInternal(length, init, false)
  }

  // using this function, rather than startArray will set all elements of the array to missing by
  // default, you will need to use setPresent to add a value to this array.
  def startMissingArray(length: Int, init: Boolean = true): Unit = {
    val t = currentType().asInstanceOf[PArray]
    if (t.elementType.required)
      fatal(s"cannot use random array pattern for required type ${t.elementType}")
    startArrayInternal(length, init, true)
  }

  private def startArrayInternal(length: Int, init: Boolean, setMissing: Boolean): Unit = {
    val t = currentType() match {
      case abc: PArrayBackedContainer => abc.arrayRep
      case arr: PArray => arr
    }
    val aoff = t.allocate(region, length)

    if (typestk.nonEmpty) {
      val off = currentOffset()
      Region.storeAddress(off, aoff)
    } else
      start = aoff

    typestk.push(t)
    elementsOffsetstk.push(aoff + t.elementsOffset(length))
    indexstk.push(0)
    offsetstk.push(aoff)

    if (init)
      t.initialize(aoff, length, setMissing)
  }

  def endArray(): Unit = {
    val t = typestk.top.asInstanceOf[PArray]
    val aoff = offsetstk.top
    val length = t.loadLength(aoff)
    assert(length == indexstk.top)

    endArrayUnchecked()
  }

  def endArrayUnchecked(): Unit = {
    typestk.pop()
    offsetstk.pop()
    elementsOffsetstk.pop()
    indexstk.pop()

    advance()
  }

  def setArrayIndex(newI: Int): Unit = {
    assert(typestk.top.isInstanceOf[PArray])
    indexstk(0) = newI
  }

  def setFieldIndex(newI: Int): Unit = {
    assert(typestk.top.isInstanceOf[PBaseStruct])
    indexstk(0) = newI
  }

  def setMissing(): Unit = {
    val i = indexstk.top
    typestk.top match {
      case t: PBaseStruct =>
        if (t.fieldRequired(i))
          fatal(s"cannot set missing field for required type ${t.types(i)}")
        t.setFieldMissing(offsetstk.top, i)
      case t: PArray =>
        if (t.elementType.required)
          fatal(s"cannot set missing field for required type ${t.elementType}")
        t.setElementMissing(offsetstk.top, i)
    }
    advance()
  }

  def setPresent(): Unit = {
    val i = indexstk.top
    typestk.top match {
      case t: PBaseStruct =>
        t.setFieldPresent(offsetstk.top, i)
      case t: PArray =>
        t.setElementPresent(offsetstk.top, i)
    }
  }

  def addBoolean(b: Boolean): Unit = {
    assert(currentType().isInstanceOf[PBoolean])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    Region.storeByte(off, b.toByte)
    advance()
  }

  def addInt(i: Int): Unit = {
    assert(currentType().isInstanceOf[PInt32])
    addIntInternal(i)
  }

  def addCall(c: Int): Unit = {
    assert(currentType().isInstanceOf[PCall])
    addIntInternal(c)
  }

  def addIntInternal(i: Int): Unit = {
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    Region.storeInt(off, i)
    advance()
  }

  def addLong(l: Long): Unit = {
    assert(currentType().isInstanceOf[PInt64])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    Region.storeLong(off, l)
    advance()
  }

  def addFloat(f: Float): Unit = {
    assert(currentType().isInstanceOf[PFloat32])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    Region.storeFloat(off, f)
    advance()
  }

  def addDouble(d: Double): Unit = {
    assert(currentType().isInstanceOf[PFloat64])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    Region.storeDouble(off, d)
    advance()
  }

  def addString(s: String): Unit = {
    assert(currentType().isInstanceOf[PString])
    currentType().asInstanceOf[PString].unstagedStoreJavaObjectAtAddress(
      sm,
      currentOffset(),
      s,
      region,
    )
    advance()
  }

  def addLocus(contig: String, pos: Int): Unit = {
    assert(currentType().isInstanceOf[PLocus])
    currentType().asInstanceOf[PLocus].unstagedStoreLocus(sm, currentOffset(), contig, pos, region)
    advance()
  }

  def addField(t: PBaseStruct, fromRegion: Region, fromOff: Long, i: Int): Unit = {
    addField(t, fromOff, i, region.ne(fromRegion))
  }

  def addField(t: PBaseStruct, fromOff: Long, i: Int, deepCopy: Boolean): Unit = {
    if (t.isFieldDefined(fromOff, i))
      addRegionValue(t.types(i), t.loadField(fromOff, i), deepCopy)
    else
      setMissing()
  }

  def skipFields(n: Int): Unit = {
    var i = 0
    while (i < n) {
      setMissing()
      i += 1
    }
  }

  def addAllFields(t: PBaseStruct, fromRegion: Region, fromOff: Long): Unit = {
    var i = 0
    while (i < t.size) {
      addField(t, fromRegion, fromOff, i)
      i += 1
    }
  }

  def addAllFields(t: PBaseStruct, fromRV: RegionValue): Unit = {
    addAllFields(t, fromRV.region, fromRV.offset)
  }

  def addFields(t: PBaseStruct, fromRegion: Region, fromOff: Long, fieldIdx: Array[Int]): Unit = {
    var i = 0
    while (i < fieldIdx.length) {
      addField(t, fromRegion, fromOff, fieldIdx(i))
      i += 1
    }
  }

  def addFields(t: PBaseStruct, fromRV: RegionValue, fieldIdx: Array[Int]): Unit = {
    addFields(t, fromRV.region, fromRV.offset, fieldIdx)
  }

  def selectRegionValue(fromT: PStruct, fromFieldIdx: Array[Int], fromRV: RegionValue): Unit = {
    selectRegionValue(fromT, fromFieldIdx, fromRV.region, fromRV.offset)
  }

  def selectRegionValue(fromT: PStruct, fromFieldIdx: Array[Int], region: Region, offset: Long): Unit = {
    // too expensive!
    // val t = fromT.typeAfterSelect(fromFieldIdx)
    // assert(currentType().setRequired(true) == t.setRequired(true), s"${currentType()} != ${t}")
    // assert(t.size == fromFieldIdx.length)
    startStruct()
    addFields(fromT, region, offset, fromFieldIdx)
    endStruct()
  }

  def addRegionValue(t: PType, rv: RegionValue): Unit = {
    addRegionValue(t, rv.region, rv.offset)
  }

  def addRegionValue(t: PType, fromRegion: Region, fromOff: Long): Unit = {
    addRegionValue(t, fromOff, region.ne(fromRegion))
  }

  def addRegionValue(t: PType, fromOff: Long, deepCopy: Boolean): Unit = {
    val toT = currentType()

    if (typestk.isEmpty) {
      val r = toT.copyFromAddress(sm, region, t, fromOff, deepCopy)
      start = r
      return
    }

    val toOff = currentOffset()
    assert(typestk.nonEmpty || toOff == start)

    toT.unstagedStoreAtAddress(sm, toOff, region, t, fromOff, deepCopy)

    advance()
  }

  def addAnnotation(t: Type, a: Annotation): Unit = {
    assert(typestk.nonEmpty)
    if (a == null) {
      setMissing()
    } else {
      currentType().unstagedStoreJavaObjectAtAddress(sm, currentOffset(), a, region)
      advance()
    }
  }

  def result(): RegionValue = RegionValue(region, start)
}
