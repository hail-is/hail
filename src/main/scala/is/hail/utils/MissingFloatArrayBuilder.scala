package is.hail.utils

import is.hail.expr.types._
import is.hail.annotations._
import scala.collection.mutable

class MissingFloatArrayBuilder extends Serializable {
  private var len = 0
  private var elements = new ArrayBuilder[Float]()
  private var isMissing = new mutable.BitSet()

  def addMissing() {
    isMissing.add(len)
    len += 1
  }

  def add(x: Float) {
    elements += x
    len += 1
  }

  def length(): Int = len

  def foreach(whenMissing: (Int) => Unit)(whenPresent: (Int, Float) => Unit) {
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

  val typ = TArray(TFloat32())

  private val rvb = new RegionValueBuilder()

  def write(rvb: RegionValueBuilder) {
    rvb.startArray(len)
    var i = 0
    var j = 0
    while (i < len) {
      if (isMissing(i))
        rvb.setMissing()
      else {
        rvb.addFloat(elements(j))
        j += 1
      }
      i += 1
    }
    rvb.endArray()
  }

  def clear() {
    len = 0
    elements.clear()
    isMissing.clear()
  }

  override def clone(): MissingFloatArrayBuilder = {
    val ab = new MissingFloatArrayBuilder()
    ab.len = len
    ab.elements = elements.clone()
    ab.isMissing = isMissing.clone()
    ab
  }
}
