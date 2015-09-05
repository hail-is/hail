package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

abstract class BaseMethod extends Serializable {
  type T
  def name: String
}

case class MethodValues(methods: Map[Method, Int],
  values: Array[Any]) {
  def get[M <: Method](m: M): M#T = {
    values(methods(m)).asInstanceOf[M#T]
  }
}

abstract class Method extends BaseMethod {
  def map(g: Genotype): T = throw new UnsupportedOperationException
  def mapWithKeys(v: Variant, s: Int, g: Genotype): T = map(g)
  def foldZeroValue: T
  def fold(x: T, y: T): T
}

// FIXME CountMethod
abstract class SumMethod extends Method {
  type T = Int
  def foldZeroValue = 0
  def fold(x: Int, y: Int) = x + y
}

abstract class DerivedMethod extends BaseMethod {
  def map(values: MethodValues): T
}
