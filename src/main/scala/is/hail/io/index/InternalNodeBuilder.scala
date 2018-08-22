package is.hail.io.index

import is.hail.annotations.RegionValueBuilder
import is.hail.expr.types.{TArray, TInt64, TStruct, Type}
import is.hail.utils.ArrayBuilder

object InternalNodeBuilder {
  def typ(keyType: Type, annotationType: Type) = TStruct(
    "first_idx" -> +TInt64(),
    "children" -> +TArray(TStruct(
      "index_file_offset" -> +TInt64(),
      "first_key" -> keyType,
      "first_record_offset" -> +TInt64(),
      "first_annotation" -> annotationType,
      "last_key" -> keyType
    ), required = true)
  )
}

class InternalNodeBuilder(keyType: Type, annotationType: Type) {
  val indexFileOffsets = new ArrayBuilder[Long]()
  val firstKeys = new ArrayBuilder[Any]()
  val firstRecordOffsets = new ArrayBuilder[Long]()
  val firstAnnotations = new ArrayBuilder[Any]()
  val lastKeys = new ArrayBuilder[Any]()
  var size = 0
  var firstIdx = 0L
  val typ = InternalNodeBuilder.typ(keyType, annotationType)

  def +=(info: IndexNodeInfo) {
    if (size == 0) {
      firstIdx = info.firstIndex
    }
    indexFileOffsets += info.indexFileOffset
    firstKeys += info.firstKey
    firstRecordOffsets += info.firstRecordOffset
    firstAnnotations += info.firstAnnotation
    lastKeys += info.lastKey
    size += 1
  }

  def write(rvb: RegionValueBuilder): Long = {
    rvb.start(typ)
    rvb.startStruct()
    rvb.addLong(firstIdx)

    rvb.startArray(size)
    var i = 0
    while (i < size) {
      rvb.startStruct()
      rvb.addLong(indexFileOffsets(i))
      rvb.addAnnotation(keyType, firstKeys(i))
      rvb.addLong(firstRecordOffsets(i))
      rvb.addAnnotation(annotationType, firstAnnotations(i))
      rvb.addAnnotation(keyType, lastKeys(i))
      rvb.endStruct()
      i += 1
    }
    rvb.endArray()
    rvb.endStruct()
    rvb.end()
  }

  def clear() {
    indexFileOffsets.clear()
    firstKeys.clear()
    firstRecordOffsets.clear()
    firstAnnotations.clear()
    lastKeys.clear()
    size = 0
  }

  def getChild(idx: Int): InternalChild = {
    assert(idx >= 0 && idx < size)
    InternalChild(indexFileOffsets(idx), firstKeys(idx), firstRecordOffsets(idx), firstAnnotations(idx), lastKeys(idx))
  }

  override def toString: String = s"InternalNodeBuilder $firstIdx $size [${ (0 until size).map(i => (indexFileOffsets(i), firstKeys(i), firstRecordOffsets(i), firstAnnotations(i), lastKeys(i))).mkString(",") }]"
}
