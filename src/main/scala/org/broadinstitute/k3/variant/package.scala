package org.broadinstitute.k3

import org.broadinstitute.k3.variant.sparky.SparkyVSM

package object variant {
  // type VariantSampleMatrix[T, S] = managed.ManagedVSM[T, S]
  type VariantSampleMatrix[T, S <: Iterable[(Int, T)]] = sparky.SparkyVSM[T, S]
}
