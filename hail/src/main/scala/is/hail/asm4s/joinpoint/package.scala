package is.hail.asm4s

import scala.language.implicitConversions

package object joinpoint {
  implicit def codeUnitToCtrl(c: Code[Unit]): Code[Ctrl] = c.asInstanceOf[Code[Ctrl]]

  implicit def codeCtrlToUnit(c: Code[Ctrl]): Code[Unit] = c.asInstanceOf[Code[Unit]]
}
