package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

abstract class BaseMethod extends Serializable {
  type T

  def name: String
}

abstract class AggregateMethod extends BaseMethod {
  def aggZeroValue: T

  def seqOp(g: Genotype, acc: T): T = throw new UnsupportedOperationException

  def seqOpWithKeys(v: Variant, s: Int, g: Genotype, acc: T): T = seqOp(g, acc)

  def combOp(x: T, y: T): T
}

case class MethodValues(methods: Map[AggregateMethod, Int],
  values: Array[Any]) {
  def get[M <: AggregateMethod](m: M): M#T = {
    values(methods(m)).asInstanceOf[M#T]
  }
}

abstract class MapRedMethod extends AggregateMethod {
  def map(g: Genotype): T = throw new UnsupportedOperationException

  def mapWithKeys(v: Variant, s: Int, g: Genotype): T = map(g)

  // FIXME single zeroValue
  def foldZeroValue: T

  def fold(x: T, y: T): T

  def aggZeroValue: T = foldZeroValue

  override def seqOpWithKeys(v: Variant, s: Int, g: Genotype, acc: T): T =
    fold(mapWithKeys(v, s, g), acc)
  def combOp(x: T, y: T): T = fold(x, y)
}

// FIXME CountMethod
abstract class SumMethod extends MapRedMethod {
  type T = Int

  def foldZeroValue = 0

  def fold(x: Int, y: Int) = x + y
}

abstract class DerivedMethod extends BaseMethod {
  def map(values: MethodValues): T
}
