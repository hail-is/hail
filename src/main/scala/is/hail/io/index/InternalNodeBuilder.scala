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
      "first_annotation" -> annotationType
    ), required = true)
  )
}

class InternalNodeBuilder(keyType: Type, annotationType: Type, var firstIdx: Long) {
  val indexFileOffsets = new ArrayBuilder[Long]()
  val firstKeys = new ArrayBuilder[Any]()
  val firstRecordOffsets = new ArrayBuilder[Long]()
  val firstAnnotations = new ArrayBuilder[Any]()
  var size = 0
  val typ = InternalNodeBuilder.typ(keyType, annotationType)

  def +=(info: IndexNodeInfo) {
    indexFileOffsets += info.indexFileOffset
    firstKeys += info.firstKey
    firstRecordOffsets += info.firstRecordOffset
    firstAnnotations += info.firstAnnotation
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
      rvb.endStruct()
      i += 1
    }
    rvb.endArray()
    rvb.endStruct()
    rvb.end()
  }

  def clear(idx: Long) {
    indexFileOffsets.clear()
    firstKeys.clear()
    firstRecordOffsets.clear()
    firstAnnotations.clear()
    size = 0
    firstIdx = idx
  }

  def getChild(idx: Int): InternalChild = {
    assert(size > idx)
    InternalChild(indexFileOffsets(idx), firstKeys(idx), firstRecordOffsets(idx), firstAnnotations(idx))
  }

  override def toString: String = s"InternalNodeBuilder $firstIdx $size [${ (0 until size).map(i => (indexFileOffsets(i), firstKeys(i), firstRecordOffsets(i), firstAnnotations(i))).mkString(",") }]"
}
