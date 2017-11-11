package is.hail.variant

import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class StructView(target: TStruct, source: TStruct) {
  private val mapping: Array[Int] = target.fields
    .map(_.name)
    .map(source.index(_).getOrElse(-1))
    .toArray
  private var m: MemoryBuffer = _
  private var offset: Long = _

  def setRegion(rv: RegionValue) {
    setRegion(rv.region, rv.offset)
  }

  def setRegion(m: MemoryBuffer, offset: Long) {
    this.m = m
    this.offset = offset
  }

  def hasField(name: String): Boolean =
    hasField(target.fieldIdx(name))

  def hasField(targetIdx: Int): Boolean = {
    val sourceIdx = mapping(targetIdx)
    (sourceIdx != -1
      && source.fieldType(sourceIdx) == target.fieldType(targetIdx)
      && source.isFieldDefined(m, offset, sourceIdx))
  }

  def getBooleanField(name: String): Boolean =
    getBooleanField(target.fieldIdx(name))

  def getBooleanField(idx: Int): Boolean = {
    assert(target.fields(idx).typ.isInstanceOf[TBoolean])
    val sourceIdx = mapping(idx)
    assert(sourceIdx != -1)
    m.loadBoolean(source.loadField(m, offset, sourceIdx))
  }

  def getIntField(name: String): Int =
    getIntField(target.fieldIdx(name))

  def getIntField(idx: Int): Int = {
    assert(target.fields(idx).typ.isInstanceOf[TInt32])
    val sourceIdx = mapping(idx)
    assert(sourceIdx != -1)
    m.loadInt(source.loadField(m, offset, sourceIdx))
  }

  def getLongField(name: String): Long =
    getLongField(target.fieldIdx(name))

  def getLongField(idx: Int): Long = {
    assert(target.fields(idx).typ.isInstanceOf[TInt64])
    val sourceIdx = mapping(idx)
    assert(sourceIdx != -1)
    m.loadLong(source.loadField(m, offset, sourceIdx))
  }

  def getFloatField(name: String): Float =
    getFloatField(target.fieldIdx(name))

  def getFloatField(idx: Int): Float = {
    assert(target.fields(idx).typ.isInstanceOf[TFloat32])
    val sourceIdx = mapping(idx)
    assert(sourceIdx != -1)
    m.loadFloat(source.loadField(m, offset, sourceIdx))
  }

  def getDoubleField(name: String): Double =
    getDoubleField(target.fieldIdx(name))

  def getDoubleField(idx: Int): Double = {
    assert(target.fields(idx).typ.isInstanceOf[TFloat64])
    val sourceIdx = mapping(idx)
    assert(sourceIdx != -1)
    m.loadDouble(source.loadField(m, offset, sourceIdx))
  }

  def getStringField(name: String): String =
    getStringField(target.fieldIdx(name))

  def getStringField(idx: Int): String = {
    assert(target.fields(idx).typ.isInstanceOf[TFloat64])
    val sourceIdx = mapping(idx)
    assert(sourceIdx != -1)
    TString.loadString(m, source.loadField(m, offset, sourceIdx))
  }

  def getArrayField(name: String): Long =
    getArrayField(target.fieldIdx(name))

  def getArrayField(idx: Int): Long = {
    val t = target.fields(idx).typ.fundamentalType
    assert(t.isInstanceOf[TArray])
    val sourceIdx = mapping(idx)
    assert(sourceIdx != -1)
    m.loadAddress(source.loadField(m, offset, sourceIdx))
  }

  def getStructField(name: String): Long =
    getStructField(target.fieldIdx(name))

  def getStructField(idx: Int): Long = {
    val t = target.fields(idx).typ.fundamentalType
    assert(t.isInstanceOf[TStruct])
    val sourceIdx = mapping(idx)
    assert(sourceIdx != -1)
    source.loadField(m, offset, sourceIdx)
  }
}
