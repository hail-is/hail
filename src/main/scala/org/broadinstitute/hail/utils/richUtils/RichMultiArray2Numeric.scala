package org.broadinstitute.hail.utils.richUtils

import org.broadinstitute.hail.utils.MultiArray2

class RichMultiArray2Numeric[T](val ma: MultiArray2[T]) extends AnyVal {

  def addElementWise(other: MultiArray2[T])(implicit ev: T => scala.math.Numeric[T]#Ops): MultiArray2[T] = {
    require(ma.n1 == other.n1 && ma.n2 == other.n2)
    val arr1 = ma.toArray
    val arr2 = other.toArray
    var i = 0
    while (i < arr1.length) {
      arr1(i) += arr2(i)
      i += 1
    }
    ma
  }

  def multiplyElementWise(other: MultiArray2[T])(implicit ev: T => scala.math.Numeric[T]#Ops): MultiArray2[T] = {
    require(ma.n1 == other.n1 && ma.n2 == other.n2)
    val arr1 = ma.toArray
    val arr2 = other.toArray
    var i = 0
    while (i < arr1.length) {
      arr1(i) *= arr2(i)
      i += 1
    }
    ma
  }
}
