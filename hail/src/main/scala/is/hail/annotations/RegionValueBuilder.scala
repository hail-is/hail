package is.hail.annotations

import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils._
import is.hail.variant.Locus
import org.apache.spark.sql.Row

class RegionValueBuilder(var region: Region) {
  def this() = this(null)

  var start: Long = _
  var root: PType = _

  val typestk = new ArrayStack[PType]()
  val indexstk = new ArrayStack[Int]()
  val offsetstk = new ArrayStack[Long]()
  val elementsOffsetstk = new ArrayStack[Long]()

  def inactive: Boolean = root == null && typestk.isEmpty && offsetstk.isEmpty && elementsOffsetstk.isEmpty && indexstk.isEmpty

  def clear(): Unit = {
    root = null
    typestk.clear()
    offsetstk.clear()
    elementsOffsetstk.clear()
    indexstk.clear()
  }

  def set(newRegion: Region) {
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
          elementsOffsetstk.top + i * t.elementByteSize
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

  def start(newRoot: PType) {
    assert(inactive)
    root = newRoot.fundamentalType
  }

  def allocateRoot() {
    assert(typestk.isEmpty)
    root match {
      case t: PArray =>
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

  def advance() {
    if (indexstk.nonEmpty)
      indexstk(0) = indexstk(0) + 1
  }

  /**
    * Unsafe unless the bytesize of every type being "advanced past" is size
    * 0. The primary use-case is when adding an array of hl.PStruct()
    * (i.e. empty structs).
    *
    **/
  def unsafeAdvance(i: Int) {
    if (indexstk.nonEmpty)
      indexstk(0) = indexstk(0) + i
  }

  def startBaseStruct(init: Boolean = true) {
    val t = currentType().asInstanceOf[PBaseStruct]
    if (typestk.isEmpty)
      allocateRoot()

    val off = currentOffset()
    typestk.push(t)
    offsetstk.push(off)
    indexstk.push(0)

    if (init)
      t.initialize(off)
  }

  def endBaseStruct() {
    val t = typestk.top.asInstanceOf[PBaseStruct]
    typestk.pop()
    offsetstk.pop()
    val last = indexstk.pop()
    assert(last == t.size)

    advance()
  }

  def startStruct(init: Boolean = true) {
    assert(currentType().isInstanceOf[PStruct])
    startBaseStruct(init)
  }

  def endStruct() {
    assert(typestk.top.isInstanceOf[PStruct])
    endBaseStruct()
  }

  def startTuple(init: Boolean = true) {
    assert(currentType().isInstanceOf[PTuple])
    startBaseStruct(init)
  }

  def endTuple() {
    assert(typestk.top.isInstanceOf[PTuple])
    endBaseStruct()
  }

  def startArray(length: Int, init: Boolean = true) {
    startArrayInternal(length, init, false)
  }

  // using this function, rather than startArray will set all elements of the array to missing by
  // default, you will need to use setPresent to add a value to this array.
  def startMissingArray(length: Int, init: Boolean = true) {
    val t = currentType().asInstanceOf[PArray]
    if (t.elementType.required)
      fatal(s"cannot use random array pattern for required type ${ t.elementType }")
    startArrayInternal(length, init, true)
  }

  private def startArrayInternal(length: Int, init: Boolean, setMissing: Boolean) {
    val t = currentType().asInstanceOf[PArray]
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

  def endArray() {
    val t = typestk.top.asInstanceOf[PArray]
    val aoff = offsetstk.top
    val length = t.loadLength(aoff)
    assert(length == indexstk.top)

    endArrayUnchecked()
  }

  def endArrayUnchecked() {
    typestk.pop()
    offsetstk.pop()
    elementsOffsetstk.pop()
    indexstk.pop()

    advance()
  }

  def setArrayIndex(newI: Int) {
    assert(typestk.top.isInstanceOf[PArray])
    indexstk(0) = newI
  }

  def setFieldIndex(newI: Int) {
    assert(typestk.top.isInstanceOf[PBaseStruct])
    indexstk(0) = newI
  }

  def setMissing() {
    val i = indexstk.top
    typestk.top match {
      case t: PBaseStruct =>
        if (t.types(i).required)
          fatal(s"cannot set missing field for required type ${ t.types(i) }")
        t.setFieldMissing(offsetstk.top, i)
      case t: PArray =>
        if (t.elementType.required)
          fatal(s"cannot set missing field for required type ${ t.elementType }")
        t.setElementMissing(offsetstk.top, i)
    }
    advance()
  }

  def setPresent() {
    val i = indexstk.top
    typestk.top match {
      case t: PBaseStruct =>
        t.setFieldPresent(offsetstk.top, i)
      case t: PArray =>
        t.setElementPresent(offsetstk.top, i)
    }
  }

  def addBoolean(b: Boolean) {
    assert(currentType().isInstanceOf[PBoolean])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    Region.storeByte(off, b.toByte)
    advance()
  }

  def addInt(i: Int) {
    assert(currentType().isInstanceOf[PInt32])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    Region.storeInt(off, i)
    advance()
  }

  def addLong(l: Long) {
    assert(currentType().isInstanceOf[PInt64])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    Region.storeLong(off, l)
    advance()
  }

  def addFloat(f: Float) {
    assert(currentType().isInstanceOf[PFloat32])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    Region.storeFloat(off, f)
    advance()
  }

  def addDouble(d: Double) {
    assert(currentType().isInstanceOf[PFloat64])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    Region.storeDouble(off, d)
    advance()
  }

  def addBinary(bytes: Array[Byte]) {
    val pbt = currentType().asInstanceOf[PBinary]
    val valueAddress = pbt.allocate(region, bytes.length)
    pbt.store(valueAddress, bytes)

    if (typestk.nonEmpty)
      Region.storeAddress(currentOffset(), valueAddress)
    else
      start = valueAddress

    advance()
  }

  def addString(s: String) {
    addBinary(s.getBytes)
  }

  def addRow(t: TBaseStruct, r: Row) {
    assert(r != null)
    startBaseStruct()
    var i = 0
    while (i < t.size) {
      addAnnotation(t.types(i), r.get(i))
      i += 1
    }
    endBaseStruct()
  }

  def addField(t: PBaseStruct, fromRegion: Region, fromOff: Long, i: Int) {
    if (t.isFieldDefined(fromOff, i))
      addRegionValue(t.types(i), fromRegion, t.loadField(fromOff, i))
    else
      setMissing()
  }

  def addField(t: PBaseStruct, rv: RegionValue, i: Int) {
    addField(t, rv.region, rv.offset, i)
  }

  def skipFields(n: Int) {
    var i = 0
    while (i < n) {
      setMissing()
      i += 1
    }
  }

  def addAllFields(t: PBaseStruct, fromRegion: Region, fromOff: Long) {
    var i = 0
    while (i < t.size) {
      addField(t, fromRegion, fromOff, i)
      i += 1
    }
  }

  def addAllFields(t: PBaseStruct, fromRV: RegionValue) {
    addAllFields(t, fromRV.region, fromRV.offset)
  }

  def addFields(t: PBaseStruct, fromRegion: Region, fromOff: Long, fieldIdx: Array[Int]) {
    var i = 0
    while (i < fieldIdx.length) {
      addField(t, fromRegion, fromOff, fieldIdx(i))
      i += 1
    }
  }

  def addFields(t: PBaseStruct, fromRV: RegionValue, fieldIdx: Array[Int]) {
    addFields(t, fromRV.region, fromRV.offset, fieldIdx)
  }

  def addElement(t: PArray, fromRegion: Region, fromAOff: Long, i: Int) {
    if (t.isElementDefined(fromAOff, i))
      addRegionValue(t.elementType, fromRegion,
        t.elementOffset(fromAOff, i))
    else
      setMissing()
  }

  def addElement(t: PArray, rv: RegionValue, i: Int) {
    addElement(t, rv.region, rv.offset, i)
  }

  def selectRegionValue(fromT: PStruct, fromFieldIdx: Array[Int], fromRV: RegionValue) {
    selectRegionValue(fromT, fromFieldIdx, fromRV.region, fromRV.offset)
  }

  def selectRegionValue(fromT: PStruct, fromFieldIdx: Array[Int], region: Region, offset: Long) {
    val t = fromT.typeAfterSelect(fromFieldIdx).fundamentalType
    assert(currentType() == t)
    assert(t.size == fromFieldIdx.length)
    startStruct()
    addFields(fromT, region, offset, fromFieldIdx)
    endStruct()
  }

  def addRegionValue(t: PType, rv: RegionValue) {
    addRegionValue(t, rv.region, rv.offset)
  }

  def addRegionValue(t: PType, fromRegion: Region, fromOff: Long) {
    val toT = currentType()

    if (typestk.isEmpty) {
      val r = toT.copyFromType(region, t.fundamentalType, fromOff, region.ne(fromRegion))
      start = r
      return
    }

    val toOff = currentOffset()
    assert(typestk.nonEmpty || toOff == start)

    toT.constructAtAddress(toOff, region, t.fundamentalType, fromOff, region.ne(fromRegion))

    advance()
  }

  def addUnsafeRow(t: PBaseStruct, ur: UnsafeRow) {
    assert(t == ur.t)
    addRegionValue(t, ur.region, ur.offset)
  }

  def addUnsafeArray(t: PArray, uis: UnsafeIndexedSeq) {
    assert(t == uis.t)
    addRegionValue(t, uis.region, uis.aoff)
  }

  def addAnnotation(t: Type, a: Annotation) {
    if (a == null)
      setMissing()
    else
      t match {
        case TBoolean => addBoolean(a.asInstanceOf[Boolean])
        case TInt32 => addInt(a.asInstanceOf[Int])
        case TInt64 => addLong(a.asInstanceOf[Long])
        case TFloat32 => addFloat(a.asInstanceOf[Float])
        case TFloat64 => addDouble(a.asInstanceOf[Double])
        case TString => addString(a.asInstanceOf[String])
        case TBinary => addBinary(a.asInstanceOf[Array[Byte]])

        case t: TArray =>
          a match {
            case uis: UnsafeIndexedSeq if currentType() == uis.t =>
              addUnsafeArray(uis.t.asInstanceOf[PArray], uis)

            case is: IndexedSeq[Annotation] =>
              startArray(is.length)
              var i = 0
              while (i < is.length) {
                addAnnotation(t.elementType, is(i))
                i += 1
              }
              endArray()
          }

        case t: TBaseStruct =>
          a match {
            case ur: UnsafeRow if currentType() == ur.t =>
              addUnsafeRow(ur.t, ur)
            case r: Row =>
              addRow(t, r)
          }

        case TSet(elementType) =>
          val s = a.asInstanceOf[Set[Annotation]]
            .toArray
            .sorted(elementType.ordering.toOrdering)
          startArray(s.length)
          s.foreach { x => addAnnotation(elementType, x) }
          endArray()

        case td: TDict =>
          val m = a.asInstanceOf[Map[Annotation, Annotation]]
            .map { case (k, v) => Row(k, v) }
            .toArray
            .sorted(td.elementType.ordering.toOrdering)
          startArray(m.length)
          m.foreach { case Row(k, v) =>
            startStruct()
            addAnnotation(td.keyType, k)
            addAnnotation(td.valueType, v)
            endStruct()
          }
          endArray()

        case TCall =>
          addInt(a.asInstanceOf[Int])

        case t: TLocus =>
          val l = a.asInstanceOf[Locus]
          startStruct()
          addString(l.contig)
          addInt(l.position)
          endStruct()

        case t: TInterval =>
          val i = a.asInstanceOf[Interval]
          startStruct()
          addAnnotation(t.pointType, i.start)
          addAnnotation(t.pointType, i.end)
          addBoolean(i.includesStart)
          addBoolean(i.includesEnd)
          endStruct()
        case t: TNDArray =>
          addAnnotation(t.representation, a)
      }
  }

  def addInlineRow(t: PBaseStruct, a: Row) {
    var i = 0
    if (a == null) {
      while (i < t.size) {
        setMissing()
        i += 1
      }
    } else {
      while(i < t.size) {
        addAnnotation(t.types(i).virtualType, a(i))
        i += 1
      }
    }
  }

  def result(): RegionValue = RegionValue(region, start)
}