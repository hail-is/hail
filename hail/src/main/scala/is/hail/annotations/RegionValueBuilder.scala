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
        case t: PBaseStruct =>
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
        case t: PBaseStruct =>
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
      t.clearMissingBits(region, off)
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
    val aoff = region.allocate(t.contentsAlignment, t.contentsByteSize(length))

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
      t.initialize(region, aoff, length, setMissing)
  }

  def endArray() {
    val t = typestk.top.asInstanceOf[PArray]
    val aoff = offsetstk.top
    val length = t.loadLength(region, aoff)
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
        t.setFieldMissing(region, offsetstk.top, i)
      case t: PArray =>
        if (t.elementType.required)
          fatal(s"cannot set missing field for required type ${ t.elementType }")
        t.setElementMissing(region, offsetstk.top, i)
    }
    advance()
  }

  def setPresent() {
    val i = indexstk.top
    typestk.top match {
      case t: PBaseStruct =>
        t.setFieldPresent(region, offsetstk.top, i)
      case t: PArray =>
        t.setElementPresent(region, offsetstk.top, i)
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
    assert(currentType().isInstanceOf[PBinary])

    val boff = region.appendBinary(bytes)

    if (typestk.nonEmpty) {
      val off = currentOffset()
      Region.storeAddress(off, boff)
    } else
      start = boff

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

  def fixupBinary(fromRegion: Region, fromBOff: Long): Long = {
    val length = PBinary.loadLength(fromRegion, fromBOff)
    val toBOff = PBinary.allocate(region, length)
    Region.copyFrom(fromBOff, toBOff, PBinary.contentByteSize(length))
    toBOff
  }

  def requiresFixup(t: PType): Boolean = {
    t match {
      case t: PBaseStruct => t.types.exists(requiresFixup)
      case _: PArray | _: PBinary => true
      case _ => false
    }
  }

  def fixupArray(t: PArray, fromRegion: Region, fromAOff: Long): Long = {
    val length = t.loadLength(fromRegion, fromAOff)
    val toAOff = t.allocate(region, length)

    Region.copyFrom(fromAOff, toAOff, t.contentsByteSize(length))

    if (region.ne(fromRegion) && requiresFixup(t.elementType)) {
      var i = 0
      while (i < length) {
        if (t.isElementDefined(fromRegion, fromAOff, i)) {
          t.elementType match {
            case t2: PBaseStruct =>
              fixupStruct(t2, t.elementOffset(toAOff, length, i), fromRegion, t.elementOffset(fromAOff, length, i))

            case t2: PArray =>
              val toAOff2 = fixupArray(t2, fromRegion, t.loadElement(fromRegion, fromAOff, length, i))
              Region.storeAddress(t.elementOffset(toAOff, length, i), toAOff2)

            case _: PBinary =>
              val toBOff = fixupBinary(fromRegion, t.loadElement(fromRegion, fromAOff, length, i))
              Region.storeAddress(t.elementOffset(toAOff, length, i), toBOff)

            case _ =>
          }
        }
        i += 1
      }
    }

    toAOff
  }

  def fixupStruct(t: PBaseStruct, toOff: Long, fromRegion: Region, fromOff: Long) {
    assert(region.ne(fromRegion))

    var i = 0
    while (i < t.size) {
      if (t.isFieldDefined(fromRegion, fromOff, i)) {
        t.types(i) match {
          case t2: PBaseStruct =>
            fixupStruct(t2, t.fieldOffset(toOff, i), fromRegion, t.fieldOffset(fromOff, i))

          case _: PBinary =>
            val toBOff = fixupBinary(fromRegion, t.loadField(fromRegion, fromOff, i))
            Region.storeAddress(t.fieldOffset(toOff, i), toBOff)

          case t2: PArray =>
            val toAOff = fixupArray(t2, fromRegion, t.loadField(fromRegion, fromOff, i))
            Region.storeAddress(t.fieldOffset(toOff, i), toAOff)

          case _ =>
        }
      }
      i += 1
    }
  }

  def addField(t: PBaseStruct, fromRegion: Region, fromOff: Long, i: Int) {
    if (t.isFieldDefined(fromRegion, fromOff, i))
      addRegionValue(t.types(i), fromRegion, t.loadField(fromRegion, fromOff, i))
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
    if (t.isElementDefined(fromRegion, fromAOff, i))
      addRegionValue(t.elementType, fromRegion,
        t.elementOffsetInRegion(fromRegion, fromAOff, i))
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
    assert(toT == t.fundamentalType)

    if (typestk.isEmpty) {
      if (region.eq(fromRegion)) {
        start = fromOff
        advance()
        return
      }

      allocateRoot()
    }

    val toOff = currentOffset()
    assert(typestk.nonEmpty || toOff == start)

    t.fundamentalType match {
      case t: PBaseStruct =>
        Region.copyFrom(fromOff, toOff, t.byteSize)
        if (region.ne(fromRegion))
          fixupStruct(t, toOff, fromRegion, fromOff)
      case t: PArray =>
        if (region.eq(fromRegion)) {
          assert(!typestk.isEmpty)
          Region.storeAddress(toOff, fromOff)
        } else {
          val toAOff = fixupArray(t, fromRegion, fromOff)
          if (typestk.nonEmpty)
            Region.storeAddress(toOff, toAOff)
          else
            start = toAOff
        }
      case _: PBinary =>
        if (region.eq(fromRegion)) {
          assert(!typestk.isEmpty)
          Region.storeAddress(toOff, fromOff)
        } else {
          val toBOff = fixupBinary(fromRegion, fromOff)
          if (typestk.nonEmpty)
            Region.storeAddress(toOff, toBOff)
          else
            start = toBOff
        }
      case _ =>
        Region.copyFrom(fromOff, toOff, t.byteSize)
    }
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
        case _: TBoolean => addBoolean(a.asInstanceOf[Boolean])
        case _: TInt32 => addInt(a.asInstanceOf[Int])
        case _: TInt64 => addLong(a.asInstanceOf[Long])
        case _: TFloat32 => addFloat(a.asInstanceOf[Float])
        case _: TFloat64 => addDouble(a.asInstanceOf[Double])
        case _: TString => addString(a.asInstanceOf[String])
        case _: TBinary => addBinary(a.asInstanceOf[Array[Byte]])

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

        case TSet(elementType, _) =>
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

        case _: TCall =>
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
