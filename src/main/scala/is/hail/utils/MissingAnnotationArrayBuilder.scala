package is.hail.utils

import is.hail.annotations.{Annotation, RegionValueBuilder}
import is.hail.expr.types.Type

import scala.collection.mutable

class MissingAnnotationArrayBuilder extends Serializable {
  private var len = 0
  private var elements = new ArrayBuilder[Annotation]()
  private var isMissing = new mutable.BitSet()

  def addMissing() {
    isMissing.add(len)
    len += 1
  }

  def add(x: Annotation) {
    elements += x
    len += 1
  }

  def length(): Int = len

  def foreach(whenMissing: (Int) => Unit)(whenPresent: (Int, Annotation) => Unit) {
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

  def write(rvb: RegionValueBuilder, t: Type) {
    rvb.startArray(len)
    var i = 0
    var j = 0
    while (i < len) {
      if (isMissing(i))
        rvb.setMissing()
      else {
        rvb.addAnnotation(t, elements(j))
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

  override def clone(): MissingAnnotationArrayBuilder = {
    val ab = new MissingAnnotationArrayBuilder()
    ab.len = len
    ab.elements = elements.clone()
    ab.isMissing = isMissing.clone()
    ab
  }
}
