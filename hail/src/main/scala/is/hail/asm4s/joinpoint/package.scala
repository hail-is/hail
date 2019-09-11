package is.hail.asm4s

import scala.language.implicitConversions

package object joinpoint {

  implicit def callCC[T: TypeInfo](f: JoinPoint.CallCC[Code[T]]): Code[T] = f.emit

  implicit def callCC(f: JoinPoint.CallCC[Unit]): Code[Unit] = f.emit
}
