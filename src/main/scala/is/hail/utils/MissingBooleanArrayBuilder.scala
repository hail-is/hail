package is.hail.utils

import is.hail.expr.types._
import is.hail.annotations._
import scala.collection.mutable.BitSet

class MissingBooleanArrayBuilder {
  private var len = 0
  private val elements = new BitSet()
  private val isMissing = new BitSet()

  def addMissing() {
    isMissing.add(len)
    len += 1
  }

  def add(x: Boolean) {
    elements.add(len)
    len += 1
  }

  def length(): Int = len

  def foreach(whenMissing: (Int) => Unit)(whenPresent: (Int, Boolean) => Unit) {
    var i = 0
    var j = 0
    while (i < len) {
      if (isMissing(i))
        whenMissing(i)
      else {
        whenPresent(i, elements(j))
        j += 1
      }
      i += 1
    }
  }

  val typ = TArray(TBoolean())

  def write(rvb: RegionValueBuilder) {
    rvb.startArray(len)
    var i = 0
    while (i < len) {
      if (isMissing(i))
        rvb.setMissing()
      else
        rvb.addBoolean(elements(i))
      i += 1
    }
    rvb.endArray()
  }
}
