package is.hail.asm4s.joinpoint

import is.hail.asm4s._
import org.objectweb.asm.Opcodes
import org.objectweb.asm.tree.{LabelNode, AbstractInsnNode, JumpInsnNode}
import scala.collection.mutable
import scala.collection.generic.Growable

object JoinPoint {
  abstract class CallCC[A: ParameterPack](mb: MethodBuilder) {
    def apply[X](jb: JoinPointBuilder[X], ret: JoinPoint[A, X]): Code[X]

    private[joinpoint] def emit: Code[Nothing] = {
      val jb = new JoinPointBuilder[Nothing](mb)
      val ret = JoinPoint[A, Nothing](new LabelNode)
      val body = apply(jb, ret)
      Code(body, jb.define, ret.placeLabel)
    }
  }

  // equivalent but produces better bytecode than 'cond.mux(j1(arg), j2(arg))'
  def mux[A, X](arg: A, cond: Code[Boolean], j1: JoinPoint[A, X], j2: JoinPoint[A, X])(
    implicit ap: ParameterPack[A]
  ): Code[X] =
    new Code[Nothing] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        ap.push(arg).emit(il)
        cond.toConditional.emitConditional(il, j1.label, j2.label)
      }
    }

  def mux[X](cond: Code[Boolean], j1: JoinPoint[Unit, X], j2: JoinPoint[Unit, X]): Code[X] =
    mux((), cond, j1, j2)
}


case class JoinPoint[A, X] private[joinpoint](label: LabelNode)(implicit p: ParameterPack[A])
    extends (A => Code[X]) {
  def apply(args: A): Code[X] = new Code[Nothing] {
    val push = p.push(args)
    def emit(il: Growable[AbstractInsnNode]): Unit = {
      push.emit(il)
      il += new JumpInsnNode(Opcodes.GOTO, label)
    }
  }

  private[joinpoint] def placeLabel = new Code[Nothing] {
    def emit(il: Growable[AbstractInsnNode]): Unit =
      il += label
  }
}

class DefinableJoinPoint[A: ParameterPack, X](
  jb: JoinPointBuilder[X],
  store: ParameterStore[A]
) extends JoinPoint[A, X](new LabelNode) {
  def define(f: A => Code[X]): Unit =
    jb.definitions += this -> Code(store.pop, f(store.load))
}

class JoinPointBuilder[X] private[joinpoint](val mb: MethodBuilder) {
  private[joinpoint] val definitions: mutable.ArrayBuffer[(JoinPoint[_, X], Code[X])] =
    new mutable.ArrayBuffer()

  private[joinpoint] def define: Code[Unit] =
    Code.foreach(definitions) { case (j, body) =>
      Code(j.placeLabel, body)
    }

  def joinPoint[A](implicit p: ParameterPack[A]): DefinableJoinPoint[A, X] =
    new DefinableJoinPoint[A, X](this, p.store(mb))
}
