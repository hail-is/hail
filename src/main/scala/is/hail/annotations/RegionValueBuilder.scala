package is.hail.annotations

import is.hail.expr.{TAltAllele, TArray, TBinary, TBoolean, TCall, TDict, TFloat32, TFloat64, TInt32, TInt64, TInterval, TLocus, TSet, TString, TStruct, TVariant, Type}
import is.hail.utils._
import is.hail.variant.{AltAllele, Locus, Variant}
import org.apache.spark.sql.Row

class RegionValueBuilder(var region: Region) {
  def this() = this(null)

  var start: Long = _
  var root: Type = _

  val typestk = new ArrayStack[Type]()
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
        case t: TStruct =>
          offsetstk.top + t.byteOffsets(i)
        case t: TArray =>
          elementsOffsetstk.top + i * t.elementByteSize
      }
    }
  }

  def currentType(): Type = {
    if (typestk.isEmpty)
      root
    else {
      typestk.top match {
        case t: TStruct =>
          val i = indexstk.top
          t.fields(i).typ
        case t: TArray =>
          t.elementType
      }
    }
  }

  def start(newRoot: Type) {
    assert(inactive)
    root = newRoot.fundamentalType
  }

  def allocateRoot() {
    assert(typestk.isEmpty)
    root match {
      case t: TArray =>
      case _: TBinary =>
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

  def startStruct(init: Boolean = true) {
    if (typestk.isEmpty)
      allocateRoot()

    val t = currentType().asInstanceOf[TStruct]
    val off = currentOffset()
    typestk.push(t)
    offsetstk.push(off)
    indexstk.push(0)

    if (init)
      t.clearMissingBits(region, off)
  }

  def endStruct() {
    typestk.top match {
      case t: TStruct =>
        typestk.pop()
        offsetstk.pop()
        val last = indexstk.pop()
        assert(last == t.size)

        advance()
    }
  }

  def startArray(length: Int, init: Boolean = true) {
    val t = currentType().asInstanceOf[TArray]
    val aoff = region.allocate(t.contentsAlignment, t.contentsByteSize(length))

    if (typestk.nonEmpty) {
      val off = currentOffset()
      region.storeAddress(off, aoff)
    } else
      start = aoff

    typestk.push(t)
    elementsOffsetstk.push(aoff + t.elementsOffset(length))
    indexstk.push(0)
    offsetstk.push(aoff)

    if (init)
      t.initialize(region, aoff, length)
  }

  def endArray() {
    val t = typestk.top.asInstanceOf[TArray]
    val aoff = offsetstk.top
    val length = t.loadLength(region, aoff)
    assert(length == indexstk.top)

    typestk.pop()
    offsetstk.pop()
    elementsOffsetstk.pop()
    indexstk.pop()

    advance()
  }

  def setFieldIndex(newI: Int) {
    assert(typestk.top.isInstanceOf[TStruct])
    indexstk(0) = newI
  }

  def setMissing() {
    val i = indexstk.top
    typestk.top match {
      case t: TStruct =>
        if (t.fieldType(i).required)
          fatal(s"cannot set missing field for required type ${ t.fieldType(i) }")
        t.setFieldMissing(region, offsetstk.top, i)
      case t: TArray =>
        if (t.elementType.required)
          fatal(s"cannot set missing field for required type ${ t.elementType }")
        t.setElementMissing(region, offsetstk.top, i)
    }
    advance()
  }

  def addBoolean(b: Boolean) {
    assert(currentType().isInstanceOf[TBoolean])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    region.storeByte(off, b.toByte)
    advance()
  }

  def addInt(i: Int) {
    assert(currentType().isInstanceOf[TInt32])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    region.storeInt(off, i)
    advance()
  }

  def addLong(l: Long) {
    assert(currentType().isInstanceOf[TInt64])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    region.storeLong(off, l)
    advance()
  }

  def addFloat(f: Float) {
    assert(currentType().isInstanceOf[TFloat32])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    region.storeFloat(off, f)
    advance()
  }

  def addDouble(d: Double) {
    assert(currentType().isInstanceOf[TFloat64])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    region.storeDouble(off, d)
    advance()
  }

  def addBinary(bytes: Array[Byte]) {
    assert(currentType().isInstanceOf[TBinary])

    val boff = region.appendInt(bytes.length)
    region.appendBytes(bytes)

    if (typestk.nonEmpty) {
      val off = currentOffset()
      region.storeAddress(off, boff)
    } else
      start = boff

    advance()
  }

  def addString(s: String) {
    addBinary(s.getBytes)
  }

  def addRow(t: TStruct, r: Row) {
    assert(r != null)

    startStruct()
    var i = 0
    while (i < t.size) {
      addAnnotation(t.fields(i).typ, r.get(i))
      i += 1
    }
    endStruct()
  }

  def fixupBinary(fromRegion: Region, fromBOff: Long): Long = {
    val length = TBinary.loadLength(fromRegion, fromBOff)
    val toBOff = TBinary.allocate(region, length)
    region.copyFrom(fromRegion, fromBOff, toBOff, TBinary.contentByteSize(length))
    toBOff
  }

  def requiresFixup(t: Type): Boolean = {
    t match {
      case t: TStruct => t.fields.exists(f => requiresFixup(f.typ))
      case _: TArray | _: TBinary => true
      case _ => false
    }
  }

  def fixupArray(t: TArray, fromRegion: Region, fromAOff: Long): Long = {
    val length = t.loadLength(fromRegion, fromAOff)
    val toAOff = t.allocate(region, length)

    region.copyFrom(fromRegion, fromAOff, toAOff, t.contentsByteSize(length))

    if (region.ne(fromRegion) && requiresFixup(t.elementType)) {
      var i = 0
      while (i < length) {
        if (t.isElementDefined(fromRegion, fromAOff, i)) {
          t.elementType match {
            case t2: TStruct =>
              fixupStruct(t2, t.elementOffset(toAOff, length, i), fromRegion, t.elementOffset(fromAOff, length, i))

            case t2: TArray =>
              val toAOff2 = fixupArray(t2, fromRegion, t.loadElement(fromRegion, fromAOff, length, i))
              region.storeAddress(t.elementOffset(toAOff, length, i), toAOff2)

            case _: TBinary =>
              val toBOff = fixupBinary(fromRegion, t.loadElement(fromRegion, fromAOff, length, i))
              region.storeAddress(t.elementOffset(toAOff, length, i), toBOff)

            case _ =>
          }
        }
        i += 1
      }
    }

    toAOff
  }

  def fixupStruct(t: TStruct, toOff: Long, fromRegion: Region, fromOff: Long) {
    assert(region.ne(fromRegion))

    var i = 0
    while (i < t.size) {
      if (t.isFieldDefined(fromRegion, fromOff, i)) {
        t.fields(i).typ match {
          case t2: TStruct =>
            fixupStruct(t2, t.fieldOffset(toOff, i), fromRegion, t.fieldOffset(fromOff, i))

          case _: TBinary =>
            val toBOff = fixupBinary(fromRegion, t.loadField(fromRegion, fromOff, i))
            region.storeAddress(t.fieldOffset(toOff, i), toBOff)

          case t2: TArray =>
            val toAOff = fixupArray(t2, fromRegion, t.loadField(fromRegion, fromOff, i))
            region.storeAddress(t.fieldOffset(toOff, i), toAOff)

          case _ =>
        }
      }
      i += 1
    }
  }

  def addField(t: TStruct, fromRegion: Region, fromOff: Long, i: Int) {
    if (t.isFieldDefined(fromRegion, fromOff, i))
      addRegionValue(t.fieldType(i), fromRegion, t.loadField(fromRegion, fromOff, i))
    else
      setMissing()
  }

  def addField(t: TStruct, rv: RegionValue, i: Int) {
    addField(t, rv.region, rv.offset, i)
  }

  def addElement(t: TArray, fromRegion: Region, fromAOff: Long, i: Int) {
    if (t.isElementDefined(fromRegion, fromAOff, i))
      addRegionValue(t.elementType, fromRegion,
        t.elementOffsetInRegion(fromRegion, fromAOff, i))
    else
      setMissing()
  }

  def addElement(t: TArray, rv: RegionValue, i: Int) {
    addElement(t, rv.region, rv.offset, i)
  }

  def addRegionValue(t: Type, rv: RegionValue) {
    addRegionValue(t, rv.region, rv.offset)
  }

  def addRegionValue(t: Type, fromRegion: Region, fromOff: Long) {
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
      case t: TStruct =>
        region.copyFrom(fromRegion, fromOff, toOff, t.byteSize)
        if (region.ne(fromRegion))
          fixupStruct(t, toOff, fromRegion, fromOff)
      case t: TArray =>
        if (region.eq(fromRegion)) {
          assert(!typestk.isEmpty)
          region.storeAddress(toOff, fromOff)
        } else {
          val toAOff = fixupArray(t, fromRegion, fromOff)
          if (typestk.nonEmpty)
            region.storeAddress(toOff, toAOff)
          else
            start = toAOff
        }
      case _: TBinary =>
        if (region.eq(fromRegion)) {
          assert(!typestk.isEmpty)
          region.storeAddress(toOff, fromOff)
        } else {
          val toBOff = fixupBinary(fromRegion, fromOff)
          if (typestk.nonEmpty)
            region.storeAddress(toOff, toBOff)
          else
            start = toBOff
        }
      case _ =>
        region.copyFrom(fromRegion, fromOff, toOff, t.byteSize)
    }
    advance()
  }

  def addUnsafeRow(t: TStruct, ur: UnsafeRow) {
    assert(t == ur.t)
    addRegionValue(t, ur.region, ur.offset)
  }

  def addUnsafeArray(t: TArray, uis: UnsafeIndexedSeq) {
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
            case uis: UnsafeIndexedSeq if t == uis.t =>
              addUnsafeArray(t, uis)

            case is: IndexedSeq[Annotation] =>
              startArray(is.length)
              var i = 0
              while (i < is.length) {
                addAnnotation(t.elementType, is(i))
                i += 1
              }
              endArray()
          }

        case t: TStruct =>
          a match {
            case ur: UnsafeRow if t == ur.t =>
              addUnsafeRow(t, ur)
            case r: Row =>
              addRow(t, r)
          }

        case TSet(elementType, _) =>
          val s = a.asInstanceOf[Set[Annotation]]
            .toArray
            .sorted(elementType.ordering(true))
          startArray(s.length)
          s.foreach { x => addAnnotation(elementType, x) }
          endArray()

        case td: TDict =>
          val m = a.asInstanceOf[Map[Annotation, Annotation]]
            .map { case (k, v) => Row(k, v) }
            .toArray
            .sorted(td.elementType.ordering(true))
          startArray(m.length)
          m.foreach { case Row(k, v) =>
            startStruct()
            addAnnotation(td.keyType, k)
            addAnnotation(td.valueType, v)
            endStruct()
          }
          endArray()

        case t: TVariant =>
          val v = a.asInstanceOf[Variant]
          startStruct()
          addString(v.contig)
          addInt(v.start)
          addString(v.ref)
          startArray(v.altAlleles.length)
          var i = 0
          while (i < v.altAlleles.length) {
            addAnnotation(TAltAllele(), v.altAlleles(i))
            i += 1
          }
          endArray()
          endStruct()

        case _: TAltAllele =>
          val aa = a.asInstanceOf[AltAllele]
          startStruct()
          addString(aa.ref)
          addString(aa.alt)
          endStruct()

        case _: TCall =>
          addInt(a.asInstanceOf[Int])

        case t: TLocus =>
          val l = a.asInstanceOf[Locus]
          startStruct()
          addString(l.contig)
          addInt(l.position)
          endStruct()

        case t: TInterval =>
          val i = a.asInstanceOf[Interval[Locus]]
          startStruct()
          addAnnotation(TLocus(t.gr), i.start)
          addAnnotation(TLocus(t.gr), i.end)
          endStruct()
      }

  }

  def result(): RegionValue = RegionValue(region, start)
}
