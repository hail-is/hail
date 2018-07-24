package is.hail.io.index

import is.hail.annotations.RegionValueBuilder
import is.hail.expr.types.{TArray, TInt64, TStruct, Type}
import is.hail.utils.ArrayBuilder

object LeafNodeBuilder {
  def typ(keyType: Type) = TStruct(
    "first_key_idx" -> TInt64(required = true),
    "keys" -> TArray(TStruct(
      "key" -> keyType,
      "offset" -> TInt64(required = true)
    ), required = true)
  )
}

class LeafNodeBuilder(keyType: Type) {
  val keys = new ArrayBuilder[Any]()
  val offsets = new ArrayBuilder[Long]()
  var size = 0
  val typ = LeafNodeBuilder.typ(keyType)

  def write(rvb: RegionValueBuilder, idx: Long): Long = {
    rvb.start(typ)
    rvb.startStruct()

    rvb.addLong(idx) // first_key_idx

    rvb.startArray(size) // keys
    var i = 0
    while (i < size) {
      rvb.startStruct()
      rvb.addAnnotation(keyType, keys(i)) // key
      rvb.addLong(offsets(i)) // offset
      rvb.endStruct()
      i += 1
    }
    rvb.endArray()
    rvb.endStruct()
    rvb.end()
  }

  def firstKey: Any = {
    assert(size > 0)
    keys(0)
  }

  def firstOffset: Long = {
    assert(size > 0)
    offsets(0)
  }

  def +=(key: Any, offset: Long) {
    keys += key
    offsets += offset
    size += 1
  }

  def clear() {
    keys.clear()
    offsets.clear()
    size = 0
  }

  override def toString: String = s"LeafNodeBuilder $size [${ (0 until size).map(i => (keys(i), offsets(i))).mkString(",") }]"
}