package is.hail.asm4s.joinpoint

import is.hail.asm4s._
import is.hail.utils.{fatal}
import org.objectweb.asm.Opcodes
import org.objectweb.asm.tree.{LabelNode, AbstractInsnNode, JumpInsnNode, TableSwitchInsnNode}
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
    ensureStackIndicator(j1.stackIndicator)
    ensureStackIndicator(j2.stackIndicator)
    new Code[Ctrl] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        ap.push(arg).emit(il)
        cond.toConditional.emitConditional(il, j1.label, j2.label)
      }
    }
  }

  def mux(cond: Code[Boolean], j1: JoinPoint[Unit], j2: JoinPoint[Unit]): Code[Ctrl] =
    mux((), cond, j1, j2)

  def switch(
    target: Code[Int],
    dflt: JoinPoint[Unit],
    cases: Seq[JoinPoint[Unit]]
  ): Code[Ctrl] =
    if (cases.isEmpty)
      dflt(())
    else {
      ensureStackIndicator(dflt.stackIndicator)
      cases.foreach { j => ensureStackIndicator(j.stackIndicator) }
      new Code[Ctrl] {
        def emit(il: Growable[AbstractInsnNode]): Unit = {
          target.emit(il)
          il += new TableSwitchInsnNode(0, cases.length - 1, dflt.label, cases.map(_.label): _*)
        }
      }
    }

  case class CallCC[A: ParameterPack](
    f: (JoinPointBuilder, JoinPoint[A]) => Code[Ctrl]
  ) {
    private[joinpoint] def code: Code[Nothing] = withStackIndicator { si =>
      val jb = new JoinPointBuilder(si)
      val ret = new JoinPoint[A](si)
      val body = f(jb, ret)
      Code(
        assignLabels(jb.joinPoints),
        assignLabels(List(ret)),
        body,
        jb.define,
        ret.placeLabel)
    }
  }

  private def assignLabels(js: Seq[JoinPoint[_]]): Code[Unit] =
    new Code[Unit] {
      def emit(il: Growable[AbstractInsnNode]): Unit =
        for (j <- js)
          j.label = new LabelNode
    }

  /**
    * So-called "stack-indicators" (unique ints) are placed on the following `mutable.Stack` during
    * the extent of a CallCC. Whenever a join-point is called, it checks that the stack-indicator at
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

  private val stack: mutable.Stack[Int] = new mutable.Stack
  private var scopeID: Int = 0

  private def withStackIndicator[T](body: Int => T): T = {
    val si = scopeID
    scopeID += 1
    stack.push(si)
    try
      body(si)
    finally
      assert(stack.pop() == si)
  }

  private[joinpoint] def ensureStackIndicator(si: Int): Unit =
    if (stack.top != si)
      throw new EmitLongJumpError
}

class JoinPoint[A] private[joinpoint](
  val stackIndicator: Int
)(implicit p: ParameterPack[A]) extends (A => Code[Ctrl]) {

  var label: LabelNode = null

  def apply(args: A): Code[Ctrl] = {
    JoinPoint.ensureStackIndicator(stackIndicator)
    Code(p.push(args), gotoLabel)
  }

  private[joinpoint] def placeLabel: Code[Nothing] = Code(label)
  private[joinpoint] def gotoLabel: Code[Ctrl] = Code(new JumpInsnNode(Opcodes.GOTO, label))
}

class DefinableJoinPoint[A: ParameterPack] private[joinpoint](
  args: ParameterStore[A],
  stackIndicator: Int
) extends JoinPoint[A](stackIndicator) {

  def define(f: A => Code[Ctrl]): Unit =
    body = Some(Code(args.store, f(args.load)))

  private[joinpoint] var body: Option[Code[Ctrl]] = None
}

class JoinPointBuilder private[joinpoint](
  stackIndicator: Int
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
