package is.hail.io.index

import is.hail.annotations.{Annotation, RegionValueBuilder}
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.{TArray, TInt64, TStruct, Type}
import is.hail.utils.ArrayBuilder

object LeafNodeBuilder {
  def virtualType(keyType: Type, annotationType: Type): TStruct = typ(PType.canonical(keyType), PType.canonical(annotationType)).virtualType

  def legacyTyp(keyType: PType, annotationType: PType) = PStruct(
    "first_idx" -> +PInt64(),
    "keys" -> +PArray(+PStruct(
      "key" -> keyType,
      "offset" -> +PInt64(),
      "annotation" -> annotationType
    ), required = true))

  def typ(keyType: PType, annotationType: PType) = PStruct(
    "first_idx" -> +PInt64(),
    "keys" -> +PArray(+PStruct(
      "key" -> keyType,
      "offset" -> +PInt64(),
      "annotation" -> annotationType
    ), required = true)
  )
}

class LeafNodeBuilder(keyType: PType, annotationType: PType, var firstIdx: Long) {
  val keys = new ArrayBuilder[Any]()
  val recordOffsets = new ArrayBuilder[Long]()
  val annotations = new ArrayBuilder[Any]()
  var size = 0
  val pType: PStruct = LeafNodeBuilder.typ(keyType, annotationType)

  def write(rvb: RegionValueBuilder): Long = {
    rvb.start(pType)
    rvb.startStruct()

    rvb.addLong(firstIdx)

    rvb.startArray(size)
    var i = 0
    while (i < size) {
      rvb.startStruct()
      rvb.addAnnotation(keyType.virtualType, keys(i))
      rvb.addLong(recordOffsets(i))
      rvb.addAnnotation(annotationType.virtualType, annotations(i))
      rvb.endStruct()
      i += 1
    }
    rvb.endArray()
    rvb.endStruct()
    rvb.end()
  }

  def +=(key: Annotation, offset: Long, annotation: Annotation) {
    keys += key
    recordOffsets += offset
    annotations += annotation
    size += 1
  }

  def getChild(idx: Int): LeafChild = {
    assert(idx >= 0 && idx < size)
    LeafChild(keys(idx), recordOffsets(idx), annotations(idx))
  }

  def clear(newIdx: Long) {
    keys.clear()
    recordOffsets.clear()
    annotations.clear()
    size = 0
    firstIdx = newIdx
  }

  override def toString: String = s"LeafNodeBuilder $firstIdx $size [${ (0 until size).map(i => (keys(i), recordOffsets(i), annotations(i))).mkString(",") }]"
}