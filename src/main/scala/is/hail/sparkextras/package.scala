package is.hail

import is.hail.utils._

package object sparkextras {
  type TrivialContext = TrivialContextInstance.type

  val TrivialContext = TrivialContextInstance

  implicit object TrivialContextIsPointed extends Pointed[TrivialContext] {
    def point: TrivialContext = TrivialContext
  }
}
