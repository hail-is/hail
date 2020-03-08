package is.hail.asm4s

import scala.language.implicitConversions

package object joinpoint {

  implicit def callCC[T: TypeInfo](ccc: JoinPoint.CallCC[Code[T]]): Code[T] = ccc.code

  implicit def callCC(ccc: JoinPoint.CallCC[Unit]): Code[Unit] = ccc.code
}
