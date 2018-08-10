package is.hail.io.index

import is.hail.annotations.RegionValueBuilder
import is.hail.expr.types.{TArray, TInt64, TStruct, Type}
import is.hail.utils.ArrayBuilder

object InternalNodeBuilder {
  def typ(keyType: Type) = TStruct(
    "block_first_key_idx" -> +TInt64(),
    "children" -> +TArray(TStruct(
      "child_offset" -> +TInt64(),
      "first_key" -> keyType,
      "first_key_offset" -> +TInt64(),
      "last_key" -> keyType
    ), required = true)
  )
}

class InternalNodeBuilder(keyType: Type) {
  val childOffsets = new ArrayBuilder[Long]()
  val firstKeys = new ArrayBuilder[Any]()
  val firstKeyOffsets = new ArrayBuilder[Long]()
  val lastKeys = new ArrayBuilder[Any]()

  var size = 0
  val typ = InternalNodeBuilder.typ(keyType)
  var firstIndex = 0L

  def setFirstIndex(idx: Long) {
    firstIndex = idx
  }

  def +=(info: IndexNodeInfo) {
    childOffsets += info.fileOffset
    firstKeys += info.firstKey
    firstKeyOffsets += info.firstKeyOffset
    lastKeys += info.lastKey
    size += 1
  }

  def write(rvb: RegionValueBuilder): Long = {
    rvb.start(typ)
    rvb.startStruct()
    rvb.addLong(firstIndex) // block_first_key_index

    rvb.startArray(size)
    var i = 0
    while (i < size) {
      rvb.startStruct()
      rvb.addLong(childOffsets(i))
      rvb.addAnnotation(keyType, firstKeys(i))
      rvb.addLong(firstKeyOffsets(i))
      rvb.addAnnotation(keyType, lastKeys(i))
      rvb.endStruct()
      i += 1
    }
    rvb.endArray()
    rvb.endStruct()
    rvb.end()
  }

  def clear() {
    childOffsets.clear()
    firstKeys.clear()
    firstKeyOffsets.clear()
    lastKeys.clear()
    size = 0
    firstIndex = 0L
  }

  def getChild(idx: Int): InternalChild = {
    assert(size > idx)
    InternalChild(childOffsets(idx), firstKeys(idx), firstKeyOffsets(idx), lastKeys(idx))
  }

  override def toString: String = s"InternalNodeBuilder $size $firstIndex [${ (0 until size).map(i => (childOffsets(i), firstKeys(i), firstKeyOffsets(i), lastKeys(i))).mkString(",") }]"
}
