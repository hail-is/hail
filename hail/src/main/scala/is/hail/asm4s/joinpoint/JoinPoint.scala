package is.hail.asm4s.joinpoint

import is.hail.asm4s._
import is.hail.utils.{fatal}
import org.objectweb.asm.Opcodes
import org.objectweb.asm.tree.{LabelNode, AbstractInsnNode, JumpInsnNode}
import scala.collection.mutable
import scala.collection.generic.Growable

// uninhabitable dummy type, indicating some sort of control flow rather than returning a value;
// used as the return type of JoinPoints
case class Ctrl(n: Nothing)

object JoinPoint {
  // equivalent but produces better bytecode than 'cond.mux(j1(arg), j2(arg))'
  def mux[A](arg: A, cond: Code[Boolean], j1: JoinPoint[A], j2: JoinPoint[A])(
    implicit ap: ParameterPack[A]
  ) = new Code[Ctrl] {
    def emit(il: Growable[AbstractInsnNode]): Unit = {
      ap.push(arg).emit(il)
      cond.toConditional.emitConditional(il, j1.label, j2.label)
    }
  }

  def mux(cond: Code[Boolean], j1: JoinPoint[Unit], j2: JoinPoint[Unit]): Code[Ctrl] =
    mux((), cond, j1, j2)

  case class CallCC[A: ParameterPack](
    f: (JoinPointBuilder, JoinPoint[A]) => Code[Ctrl]
  ) {
    private[joinpoint] def code: Code[Nothing] = {
      // NOTE: unsafe if you have nested CallCC's
      val jb = new JoinPointBuilder
      val ret = JoinPoint[A](new LabelNode)
      val body = f(jb, ret)
      Code(body, jb.define, ret.placeLabel)
    }
  }
}

case class JoinPoint[A] private[joinpoint](
  label: LabelNode
)(implicit p: ParameterPack[A]) extends (A => Code[Ctrl]) {
  def apply(args: A) = Code(p.push(args), gotoLabel)
  private[joinpoint] def placeLabel = Code(label)
  private[joinpoint] def gotoLabel = Code[Ctrl](new JumpInsnNode(Opcodes.GOTO, label))
}

class DefinableJoinPoint[A: ParameterPack](
  args: ParameterStore[A]
) extends JoinPoint[A](new LabelNode) {

  def define(f: A => Code[Ctrl]): Unit =
    body = Some(Code(args.store, f(args.load)))

  private[joinpoint] var body: Option[Code[Ctrl]] = None
}

class JoinPointBuilder private[joinpoint] {
  private[joinpoint] val joinPoints: mutable.ArrayBuffer[DefinableJoinPoint[_]] =
    new mutable.ArrayBuffer()

  private[joinpoint] def define: Code[Unit] =
    Code.foreach(joinPoints) { j =>
      j.body match {
        case Some(body) => Code(j.placeLabel, body)
        case None => fatal("join point never defined")
      }
    }

  private def joinPoint[A: ParameterPack](s: ParameterStore[A]): DefinableJoinPoint[A] = {
    val j = new DefinableJoinPoint[A](s)
    joinPoints += j
    j
  }

  def joinPoint[A](mb: MethodBuilder)(implicit p: ParameterPack[A]): DefinableJoinPoint[A] =
    joinPoint(p.newLocals(mb))

  def joinPoint(): DefinableJoinPoint[Unit] =
    joinPoint(ParameterStore.unit)
}
