package is.hail.io.index

import is.hail.annotations.RegionValueBuilder
import is.hail.expr.types.{TArray, TInt64, TStruct, Type}
import is.hail.utils.ArrayBuilder

object InternalNodeBuilder {
  def typ(keyType: Type, annotationType: Type) = TStruct(
    "children" -> +TArray(+TStruct(
      "index_file_offset" -> +TInt64(),
      "first_idx" -> +TInt64(),
      "first_key" -> keyType,
      "first_record_offset" -> +TInt64(),
      "first_annotation" -> annotationType
    ), required = true)
  )
}

class InternalNodeBuilder(keyType: Type, annotationType: Type) {
  val indexFileOffsets = new ArrayBuilder[Long]()
  val firstIdxs = new ArrayBuilder[Long]()
  val firstKeys = new ArrayBuilder[Any]()
  val firstRecordOffsets = new ArrayBuilder[Long]()
  val firstAnnotations = new ArrayBuilder[Any]()

  var size = 0
  val typ = InternalNodeBuilder.typ(keyType, annotationType)

  def +=(info: IndexNodeInfo) {
    indexFileOffsets += info.indexFileOffset
    firstIdxs += info.firstIndex
    firstKeys += info.firstKey
    firstRecordOffsets += info.firstRecordOffset
    firstAnnotations += info.firstAnnotation
    size += 1
  }

  def write(rvb: RegionValueBuilder): Long = {
    rvb.start(typ.physicalType)
    rvb.startStruct()
    rvb.startArray(size)
    var i = 0
    while (i < size) {
      rvb.startStruct()
      rvb.addLong(indexFileOffsets(i))
      rvb.addLong(firstIdxs(i))
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

  def clear() {
    indexFileOffsets.clear()
    firstIdxs.clear()
    firstKeys.clear()
    firstRecordOffsets.clear()
    firstAnnotations.clear()
    size = 0
  }

  def getChild(idx: Int): InternalChild = {
    assert(idx >= 0 && idx < size)
    InternalChild(indexFileOffsets(idx), firstIdxs(idx), firstKeys(idx), firstRecordOffsets(idx), firstAnnotations(idx))
  }

  override def toString: String = s"InternalNodeBuilder $size [${ (0 until size).map { i =>
    (indexFileOffsets(i), firstIdxs(i), firstKeys(i), firstRecordOffsets(i), firstAnnotations(i))
  }.mkString(",") }]"
}
