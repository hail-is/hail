package is.hail

import is.hail.utils._

package object rvd {
  implicit object RVDContextIsPointed extends Pointed[RVDContext] {
    def point: RVDContext = RVDContext.default
  }
}
