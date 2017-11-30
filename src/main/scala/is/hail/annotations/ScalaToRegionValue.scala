package is.hail.annotations

import is.hail.expr._

object ScalaToRegionValue {

  def addStruct[T: HailRep](region: MemoryBuffer,
    n1: String, v1: T): Long = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(TStruct(
      n1 -> hailType[T]))
    rvb.startStruct()
    rvb.addAnnotation(hailType[T], v1)
    rvb.endStruct()
    rvb.end()
  }

  def addStruct(region: MemoryBuffer,
    n1: String, t1: Type, offset1: Long): Long = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(TStruct(n1 -> t1))
    rvb.startStruct()
    rvb.addRegionValue(t1, region, offset1)
    rvb.endStruct()
    rvb.end()
  }

  def addStruct[T: HailRep](region: MemoryBuffer,
    n1: String, v1: T,
    n2: String, t2: Type, offset2: Long): Long = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(TStruct(n1 -> hailType[T], n2 -> t2))
    rvb.startStruct()
    rvb.addAnnotation(hailType[T], v1)
    rvb.addRegionValue(t2, region, offset2)
    rvb.endStruct()
    rvb.end()
  }

  def addStruct[T: HailRep, U: HailRep](region: MemoryBuffer,
    n1: String, v1: T,
    n2: String, v2: U): Long = {

    val rvb = new RegionValueBuilder(region)
    rvb.start(TStruct(
      n1 -> hailType[T],
      n2 -> hailType[U]))
    rvb.startStruct()
    rvb.addAnnotation(hailType[T], v1)
    rvb.addAnnotation(hailType[U], v2)
    rvb.endStruct()
    rvb.end()
  }

  def addStruct[T: HailRep, U: HailRep, V: HailRep](region: MemoryBuffer,
    n1: String, v1: T,
    n2: String, v2: U,
    n3: String, v3: V): Long = {

    val rvb = new RegionValueBuilder(region)
    rvb.start(TStruct(
      n1 -> hailType[T],
      n2 -> hailType[U],
      n3 -> hailType[V]))
    rvb.startStruct()
    rvb.addAnnotation(hailType[T], v1)
    rvb.addAnnotation(hailType[U], v2)
    rvb.addAnnotation(hailType[U], v3)
    rvb.endStruct()
    rvb.end()
  }

  def addArray(region: MemoryBuffer, a: Array[Double]): Long = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(TArray(TFloat64()))
    rvb.startArray(a.length)
    a.foreach(rvb.addDouble(_))
    rvb.endArray()
    rvb.end()
  }

  def addBoxedArray(region: MemoryBuffer, a: Array[java.lang.Double]): Long = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(TArray(TFloat64()))
    rvb.startArray(a.length)
    a.foreach { e =>
      if (e == null)
        rvb.setMissing()
      else
        rvb.addDouble(e)
    }
    rvb.endArray()
    rvb.end()
  }

}
