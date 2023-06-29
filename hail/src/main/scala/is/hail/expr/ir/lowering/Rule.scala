package is.hail.expr.ir.lowering

import is.hail.expr.ir._

import scala.reflect.ClassTag

trait Rule[State] extends scala.Product with Serializable {
  def init: State
  def apply(s: State, ir: BaseIR): (State, Boolean)
}

object Rule {
  object Implicits {
    import scala.language.implicitConversions
    implicit def mkExistential[T: ClassTag](r: Rule[T]): Rule[Any] =
      ExistentialRule(r)
  }
}

case class PureRule(f: BaseIR => Boolean) extends Rule[Unit] {
  override def init: Unit =
    ()
  override def apply(unit: Unit, ir: BaseIR): (Unit, Boolean) =
    ((), f(ir))
}

case class ExistentialRule[T: ClassTag](r: Rule[T]) extends Rule[Any] {
  override def init: Any =
    r.init

  override def apply(s: Any, ir: BaseIR): (Any, Boolean) =
    r.apply(s.asInstanceOf[T], ir)
}

object NoMatrixIR extends PureRule(!_.isInstanceOf[MatrixIR])
object NoTableIR extends PureRule(!_.isInstanceOf[TableIR])
object NoBlockMatrixIR extends PureRule(!_.isInstanceOf[BlockMatrixIR])
object NoRelationalLets extends PureRule({
  case _: RelationalLet => false
  case _: RelationalLetBlockMatrix => false
  case _: RelationalLetMatrixTable => false
  case _: RelationalLetTable => false
  case _: RelationalRef => false
  case _ => true
})

object CompilableValueIRs extends PureRule({
  case x: IR => Compilable(x)
  case _ => true
})

object NoApplyIR extends PureRule(!_.isInstanceOf[ApplyIR])
object ValueIROnly extends PureRule(_.isInstanceOf[IR])
object EmittableValueIRs extends PureRule({
  case x: IR => Emittable(x)
  case _ => true
})
