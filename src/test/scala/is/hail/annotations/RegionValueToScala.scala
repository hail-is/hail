package is.hail.annotations

import is.hail.expr._

import scala.reflect.ClassTag
import scala.reflect.classTag

object RegionValueToScala {
  def classTagHail(t: Type): ClassTag[_] = t match {
    case TBoolean(true) => classTag[Boolean]
    case TBoolean(false) => classTag[java.lang.Boolean]
    case TInt32(true) => classTag[Int]
    case TInt32(false) => classTag[java.lang.Integer]
    case TInt64(true) => classTag[Long]
    case TInt64(false) => classTag[java.lang.Long]
    case TFloat32(true) => classTag[Float]
    case TFloat32(false) => classTag[java.lang.Float]
    case TFloat64(true) => classTag[Double]
    case TFloat64(false) => classTag[java.lang.Double]
    case t => throw new RuntimeException(s"classTagHail does not handle $t")
  }

  def load[T: HailRep](region: Region, off: Long): T =
    load(hailType[T])(region, off)

  def load[T](t: Type)(region: Region, off: Long): T = t match {
    case _: TBoolean => region.loadBoolean(off).asInstanceOf[T]
    case _: TInt32 => region.loadInt(off).asInstanceOf[T]
    case _: TInt64 => region.loadLong(off).asInstanceOf[T]
    case _: TFloat32 => region.loadFloat(off).asInstanceOf[T]
    case _: TFloat64 => region.loadDouble(off).asInstanceOf[T]
    case t: TArray => loadArray(t.elementType)(region, off)(classTagHail(t.elementType)).asInstanceOf[T]
    case t => throw new RuntimeException(s"load does not handle $t")
  }

  def loadArray[T : HailRep : ClassTag](region: Region, aOff: Long): Array[T] =
    loadArray[T](hailType[T])(region, aOff)

  def loadArray[T : ClassTag](elementType: Type)(region: Region, aOff: Long): Array[T] = {
    val arrayType = TArray(elementType)
    val len = arrayType.loadLength(region, aOff)
    println(s"rvts length $len")
    val a = new Array[T](len)
    var i = 0
    while (i < len) {
      println(s"rvts $i")
      if (arrayType.isElementDefined(region, aOff, i))
        a(i) = load[T](elementType)(region, arrayType.loadElement(region, aOff, i))
      i += 1
    }
    a
  }
}
