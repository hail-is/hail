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
  ): Code[Ctrl] = {
    assert(j1.stackIndicator eq j2.stackIndicator)
    ensureStackIndicator(j1.stackIndicator)(new Code[Ctrl] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        ap.push(arg).emit(il)
        cond.toConditional.emitConditional(il, j1.label, j2.label)
      }
    })
  }

  def mux(cond: Code[Boolean], j1: JoinPoint[Unit], j2: JoinPoint[Unit]): Code[Ctrl] =
    mux((), cond, j1, j2)

  case class CallCC[A: ParameterPack](
    f: (JoinPointBuilder, JoinPoint[A]) => Code[Ctrl]
  ) {
    private[joinpoint] def code: Code[Nothing] = {
      val si = new Object()
      val jb = new JoinPointBuilder(si)
      val ret = JoinPoint[A](new LabelNode, si)
      val body = f(jb, ret)
      withStackIndicator(si) {
        Code(body, jb.define, ret.placeLabel)
      }
    }
  }

  /**
    * So-called "stack-indicators" are placed on the following `mutable.Stack` during the extent of
    * a CallCC being emitted. Whenever a join-point is called, it checks that the stack-indicator at
    * the top of this stack is the same one associated with the CallCC from which the join-point
    * originated. This ensures that you cannot escape a CallCC by returning from an outer CallCC,
    * e.g.:
    *
    *   CallCC[Code[Int]] { (jb1, ret1) =>
    *     ret1(const(1) + CallCC[Code[Int]] { (jb2, ret2) =>
    *       ret1(const(2))
    *     })
    *   }
    *
    * The above code typechecks in Scala, but at runtime would corrupt the JVM stack, leaving a "1"
    * behind without consuming it. This stack-indicator machinery catches this before bytecode
    * verification.
    */

  class EmitLongJumpError extends AssertionError("cannot jump out of nested CallCC")

  private val stack: mutable.Stack[Object] = new mutable.Stack

  private def withStackIndicator[T](si: Object)(c: Code[T]): Code[T] = new Code[T] {
    def emit(il: Growable[AbstractInsnNode]): Unit = {
      stack.push(si)
      try
        c.emit(il)
      finally
        assert(stack.pop() eq si)
    }
  }

  private[joinpoint] def ensureStackIndicator[T](si: Object)(c: Code[T]): Code[T] = new Code[T] {
    def emit(il: Growable[AbstractInsnNode]): Unit = {
      if (stack.top ne si) throw new EmitLongJumpError
      c.emit(il)
    }
  }
}

case class JoinPoint[A] private[joinpoint](
  label: LabelNode,
  stackIndicator: Object
)(implicit p: ParameterPack[A]) extends (A => Code[Ctrl]) {
  def apply(args: A): Code[Ctrl] =
    JoinPoint.ensureStackIndicator(stackIndicator) {
      Code(p.push(args), gotoLabel)
    }

  private[joinpoint] def placeLabel: Code[Nothing] = Code(label)
  private[joinpoint] def gotoLabel: Code[Ctrl] = Code(new JumpInsnNode(Opcodes.GOTO, label))
}

class DefinableJoinPoint[A: ParameterPack] private[joinpoint](
  args: ParameterStore[A],
  _stackIndicator: Object
) extends JoinPoint[A](new LabelNode, _stackIndicator) {

  def define(f: A => Code[Ctrl]): Unit =
    body = Some(Code(args.store, f(args.load)))

  private[joinpoint] var body: Option[Code[Ctrl]] = None
}

class JoinPointBuilder private[joinpoint](
  stackIndicator: Object
) {
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
    val j = new DefinableJoinPoint[A](s, stackIndicator)
    joinPoints += j
    j
  }

  def joinPoint[A](mb: MethodBuilder)(implicit p: ParameterPack[A]): DefinableJoinPoint[A] =
    joinPoint(p.newLocals(mb))

  def joinPoint(): DefinableJoinPoint[Unit] =
    joinPoint(ParameterStore.unit)
}
