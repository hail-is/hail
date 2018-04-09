package is.hail

import is.hail.utils._

package object sparkextras {
  type TrivialContext = TrivialContextInstance.type

  val TrivialContext = TrivialContextInstance

  implicit object TrivialContextIsPointed
      extends Pointed[TrivialContext] with Serializable {
    def point: TrivialContext = TrivialContext
  }
}
