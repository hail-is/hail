package is.hail.expr.ir

import is.hail.expr.ir.defs.Ref

object Env {
  type K = Name

  def empty[V]: Env[V] = new Env()

  def apply[V](bindings: (Name, V)*): Env[V] = fromSeq(bindings)

  def fromSeq[V](bindings: Iterable[(Name, V)]): Env[V] = empty[V].bindIterable(bindings)
}

object BindingEnv {
  def empty[T]: BindingEnv[T] = BindingEnv(Env.empty[T], None, None)

  def eval[T](bindings: (Name, T)*): BindingEnv[T] =
    BindingEnv(Env.fromSeq[T](bindings), None, None)

  // BindingEnv's own operations, binding each name to its value
  implicit object bindingEnvScopes extends Scopes[BindingEnv] {
    override def bind[A](e: BindingEnv[A], s: EnvScope, bs: Seq[(Name, A)]): BindingEnv[A] =
      s match {
        case Scope.EVAL => e.copy(eval = e.eval.bindIterable(bs))
        case Scope.AGG => e.copy(agg = Some(e.agg.get.bindIterable(bs)))
        case Scope.SCAN => e.copy(scan = Some(e.scan.get.bindIterable(bs)))
        case Scope.REL => e.copy(relational = e.relational.bindIterable(bs))
      }

    override def create[A](e: BindingEnv[A], s: Scope): BindingEnv[A] =
      s match {
        case Scope.AGG => e.createAgg
        case Scope.SCAN => e.createScan
        case Scope.EVAL => throw new IllegalArgumentException("can't create eval")
      }

    override def drop[A](e: BindingEnv[A], s: Scope): BindingEnv[A] =
      s match {
        case Scope.EVAL => e.noEval
        case Scope.AGG => e.noAgg
        case Scope.SCAN => e.noScan
      }

    override def promote[A](e: BindingEnv[A], s: Scope): BindingEnv[A] =
      s match {
        case Scope.EVAL => e
        case Scope.AGG =>
          if (e.agg.isEmpty) throw new RuntimeException("can't promote agg: not defined")
          e.promoteAgg
        case Scope.SCAN =>
          if (e.scan.isEmpty) throw new RuntimeException("can't promote scan: not defined")
          e.promoteScan
      }

    override def fresh[A](e: BindingEnv[A]): BindingEnv[A] =
      BindingEnv(relational = e.relational)
  }
}

case class BindingEnv[V](
  eval: Env[V] = Env.empty[V],
  agg: Option[Env[V]] = None,
  scan: Option[Env[V]] = None,
  relational: Env[V] = Env.empty[V],
) {

  def allEmpty: Boolean =
    eval.isEmpty && agg.forall(_.isEmpty) && scan.forall(_.isEmpty) && relational.isEmpty

  def promoteAgg: BindingEnv[V] = copy(eval = agg.get, agg = None)

  def promoteScan: BindingEnv[V] = copy(eval = scan.get, scan = None)

  def noAgg: BindingEnv[V] = copy(agg = None)

  def noScan: BindingEnv[V] = copy(scan = None)

  def createAgg: BindingEnv[V] = copy(agg = Some(eval))

  def createScan: BindingEnv[V] = copy(scan = Some(eval))

  def bindEval(name: Name, v: V): BindingEnv[V] =
    copy(eval = eval.bind(name, v))

  def bindEval(bindings: (Name, V)*): BindingEnv[V] =
    copy(eval = eval.bindIterable(bindings))

  def noEval: BindingEnv[V] = copy(eval = Env.empty)

  def bindAgg(name: Name, v: V): BindingEnv[V] =
    copy(agg = Some(agg.get.bind(name, v)))

  def bindAgg(bindings: (Name, V)*): BindingEnv[V] =
    copy(agg = Some(agg.get.bindIterable(bindings)))

  def bindScan(name: Name, v: V): BindingEnv[V] =
    copy(scan = Some(scan.get.bind(name, v)))

  def bindScan(bindings: (Name, V)*): BindingEnv[V] =
    copy(scan = Some(scan.get.bindIterable(bindings)))

  def pretty(valuePrinter: V => String = _.toString): String =
    s"""BindingEnv:
       |  Eval:${eval.m.map { case (k, v) => s"\n    $k -> ${valuePrinter(v)}" }.mkString("")}
       |  Agg: ${agg.map(_.m.map { case (k, v) => s"\n    $k -> ${valuePrinter(v)}" }.mkString("")).getOrElse("None")}
       |  Scan: ${scan.map(_.m.map { case (k, v) => s"\n    $k -> ${valuePrinter(v)}" }.mkString("")).getOrElse("None")}
       |  Relational: ${relational.m.map { case (k, v) =>
        s"\n    $k -> ${valuePrinter(v)}"
      }.mkString("")}""".stripMargin

  def mapValues[T](f: V => T): BindingEnv[T] =
    copy[T](
      eval = eval.mapValues(f),
      agg = agg.map(_.mapValues(f)),
      scan = scan.map(_.mapValues(f)),
      relational = relational.mapValues(f),
    )
}

final class Env[V] private (val m: Map[Env.K, V]) {
  def this() =
    this(Map())

  override def equals(other: Any): Boolean = other match {
    case env: Env[_] => this.m == env.m
    case _ => false
  }

  def contains(k: Env.K): Boolean = m.contains(k)

  def isEmpty: Boolean = m.isEmpty

  def apply(name: Name): V = m(name)

  def lookup(name: Name): V =
    m.getOrElse(name, throw new RuntimeException(s"Cannot find $name in $m"))

  def lookupOption(name: Name): Option[V] = m.get(name)

  def delete(name: Name): Env[V] = new Env(m - name)

  def delete(names: Iterable[Name]): Env[V] = new Env(m -- names)

  def lookup(r: Ref): V =
    lookup(r.name)

  def bind(name: Name, v: V): Env[V] =
    new Env(m + (name -> v))

  def bind(bindings: (Name, V)*): Env[V] = bindIterable(bindings)

  def bindIterable(bindings: Iterable[(Name, V)]): Env[V] =
    if (bindings.isEmpty) this else new Env(m ++ bindings)

  def mapValues[U](f: (V) => U): Env[U] = new Env(m.view.mapValues(f).toMap)

  override def toString: String = m.map { case (k, v) => s"$k -> $v" }.mkString("(", ",", ")")
}
