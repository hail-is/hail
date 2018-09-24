package is.hail

import is.hail.annotations.Annotation
import is.hail.utils.HailIterator

import scala.language.implicitConversions

package object variant {
  type Call = Int
  type BoxedCall = java.lang.Integer
}
