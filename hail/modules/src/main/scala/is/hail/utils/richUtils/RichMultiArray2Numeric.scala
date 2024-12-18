package is.hail.utils.richUtils

import is.hail.utils.MultiArray2

class RichMultiArray2Long(val ma: MultiArray2[Long]) extends AnyVal {

  def addElementWise(other: MultiArray2[Long]): MultiArray2[Long] = {
    require(ma.n1 == other.n1 && ma.n2 == other.n2)
    val arr1 = ma.array
    val arr2 = other.array
    var i = 0
    while (i < arr1.length) {
      arr1(i) += arr2(i)
      i += 1
    }
    ma
  }

  def multiplyElementWise(other: MultiArray2[Long]): MultiArray2[Long] = {
    require(ma.n1 == other.n1 && ma.n2 == other.n2)
    val arr1 = ma.array
    val arr2 = other.array
    var i = 0
    while (i < arr1.length) {
      arr1(i) *= arr2(i)
      i += 1
    }
    ma
  }
}

class RichMultiArray2Double(val ma: MultiArray2[Double]) extends AnyVal {

  def addElementWise(other: MultiArray2[Double]): MultiArray2[Double] = {
    require(ma.n1 == other.n1 && ma.n2 == other.n2)
    val arr1 = ma.array
    val arr2 = other.array
    var i = 0
    while (i < arr1.length) {
      arr1(i) += arr2(i)
      i += 1
    }
    ma
  }

  def multiplyElementWise(other: MultiArray2[Double]): MultiArray2[Double] = {
    require(ma.n1 == other.n1 && ma.n2 == other.n2)
    val arr1 = ma.array
    val arr2 = other.array
    var i = 0
    while (i < arr1.length) {
      arr1(i) *= arr2(i)
      i += 1
    }
    ma
  }
}

class RichMultiArray2Int(val ma: MultiArray2[Int]) extends AnyVal {

  def addElementWise(other: MultiArray2[Int]): MultiArray2[Int] = {
    require(ma.n1 == other.n1 && ma.n2 == other.n2)
    val arr1 = ma.array
    val arr2 = other.array
    var i = 0
    while (i < arr1.length) {
      arr1(i) += arr2(i)
      i += 1
    }
    ma
  }

  def multiplyElementWise(other: MultiArray2[Int]): MultiArray2[Int] = {
    require(ma.n1 == other.n1 && ma.n2 == other.n2)
    val arr1 = ma.array
    val arr2 = other.array
    var i = 0
    while (i < arr1.length) {
      arr1(i) *= arr2(i)
      i += 1
    }
    ma
  }
}
