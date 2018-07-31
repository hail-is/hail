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
      "first_key_offset" -> +TInt64()
    ), required = true)
  )
}

class InternalNodeBuilder(keyType: Type) {
  val childOffsets = new ArrayBuilder[Long]()
  val firstKeys = new ArrayBuilder[Any]()
  val firstKeyOffsets = new ArrayBuilder[Long]()
  var size = 0
  val typ = InternalNodeBuilder.typ(keyType)
  var firstIndex = 0L

  def firstKey: Any = {
    assert(size > 0)
    firstKeys(0)
  }

  def firstOffset: Long = {
    assert(size > 0)
    firstKeyOffsets(0)
  }

  def setFirstIndex(idx: Long) {
    firstIndex = idx
  }

  def +=(childOffset: Long, firstKey: Any, firstKeyOffset: Long) {
    childOffsets += childOffset
    firstKeys += firstKey
    firstKeyOffsets += firstKeyOffset
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
      rvb.addLong(childOffsets(i)) // child_offset
      rvb.addAnnotation(keyType, firstKeys(i)) // first_key
      rvb.addLong(firstKeyOffsets(i)) // first_key_offset
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
    size = 0
    firstIndex = 0L
  }

  override def toString: String = s"InternalNodeBuilder $size $firstIndex [${ (0 until size).map(i => (childOffsets(i), firstKeys(i), firstKeyOffsets(i))).mkString(",") }]"
}
