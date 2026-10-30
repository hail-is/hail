package is.hail.expr.ir

import is.hail.expr.ir.defs.Ref

object Env {
  type K = Name

  def empty[V]: Env[V] = new Env()

  def apply[V](bindings: (Name, V)*): Env[V] = fromSeq(bindings)

  def fromSeq[V](bindings: Iterable[(Name, V)]): Env[V] = empty[V].bindIterable(bindings)
}

trait GenericBindingEnv[Self, V] {
  def extend(bindings: Bindings[V]): Self

  def promoteAgg: Self

  def promoteScan: Self

  def bindEval(bindings: (Name, V)*): Self

  def noEval: Self

  def bindAgg(bindings: (Name, V)*): Self

  def bindScan(bindings: (Name, V)*): Self

  def bindInScope(name: Name, v: V, scope: Scope): Self = scope match {
    case Scope.EVAL => bindEval(name -> v)
    case Scope.AGG => bindAgg(name -> v)
    case Scope.SCAN => bindScan(name -> v)
  }

  def createAgg: Self

  def createScan: Self

  def noAgg: Self

  def noScan: Self

  def bindRelational(bindings: (Name, V)*): Self
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
) extends GenericBindingEnv[BindingEnv[V], V] {

  private def modifyWithoutNewBindings[T](bindings: Bindings[T]): BindingEnv[V] = {
    def error(): Unit =
      throw new RuntimeException(s"found inconsistent agg or scan environments:" +
        s"\n  env: agg is ${if (this.agg.isDefined) "" else "not "}defined, " +
        s"scan is ${if (this.scan.isDefined) "" else "not "}defined" +
        s"\n  bindings: agg = ${bindings.agg}, scan = ${bindings.scan}")
    val Bindings(_, _, agg, scan, _, dropEval) = bindings
    var newEnv = this
    if (dropEval) newEnv = newEnv.noEval
    agg match {
      case AggEnv.Drop => newEnv = newEnv.noAgg
      case AggEnv.Promote =>
        if (newEnv.agg.isEmpty) error()
        newEnv = newEnv.promoteAgg
      case AggEnv.Create(_) =>
        newEnv = newEnv.copy(agg = Some(newEnv.eval))
      case AggEnv.Bind(_) =>
        if (newEnv.agg.isEmpty) error()
      case _ =>
    }
    scan match {
      case AggEnv.Drop => newEnv = newEnv.noScan
      case AggEnv.Promote =>
        if (newEnv.scan.isEmpty) error()
        newEnv = newEnv.promoteScan
      case AggEnv.Create(_) =>
        newEnv = newEnv.copy(scan = Some(newEnv.eval))
      case AggEnv.Bind(_) =>
        if (newEnv.scan.isEmpty) error()
      case _ =>
    }
    newEnv
  }

  override def extend(bindings: Bindings[V]): BindingEnv[V] = {
    val Bindings(all, eval, agg, scan, relational, _) = bindings
    val newEnv = modifyWithoutNewBindings(bindings)
    if (all.isEmpty) newEnv
    else newEnv.copy(
      eval =
        if (eval.isEmpty) newEnv.eval
        else newEnv.eval.bindIterable(eval.map(all)),
      agg =
        agg match {
          case a: AggEnv.Modify => Some(newEnv.agg.get.bindIterable(a.bindings.map(all)))
          case _ => newEnv.agg
        },
      scan =
        scan match {
          case a: AggEnv.Modify => Some(newEnv.scan.get.bindIterable(a.bindings.map(all)))
          case _ => newEnv.scan
        },
      relational =
        if (relational.isEmpty) newEnv.relational
        else newEnv.relational.bindIterable(relational.map(all)),
    )
  }

  def subtract[T](bindings: Bindings[T]): BindingEnv[V] = {
    val Bindings(all, eval, agg, scan, relational, _) = bindings

    val newEnv = modifyWithoutNewBindings(bindings)
    if (all.isEmpty) newEnv
    else newEnv.copy(
      eval =
        if (eval.isEmpty) newEnv.eval
        else newEnv.eval.delete(eval.map(all(_)._1)),
      agg =
        agg match {
          case a: AggEnv.Modify => Some(newEnv.agg.get.delete(a.bindings.map(all(_)._1)))
          case _ => newEnv.agg
        },
      scan =
        scan match {
          case a: AggEnv.Modify => Some(newEnv.scan.get.delete(a.bindings.map(all(_)._1)))
          case _ => newEnv.scan
        },
      relational =
        if (relational.isEmpty) newEnv.relational
        else newEnv.relational.delete(relational.map(all(_)._1)),
    )
  }

  def allEmpty: Boolean =
    eval.isEmpty && agg.forall(_.isEmpty) && scan.forall(_.isEmpty) && relational.isEmpty

  override def promoteAgg: BindingEnv[V] = copy(eval = agg.get, agg = None)

  override def promoteScan: BindingEnv[V] = copy(eval = scan.get, scan = None)

  override def noAgg: BindingEnv[V] = copy(agg = None)

  override def noScan: BindingEnv[V] = copy(scan = None)

  override def createAgg: BindingEnv[V] = copy(agg = Some(eval))

  override def createScan: BindingEnv[V] = copy(scan = Some(eval))

  def bindEval(name: Name, v: V): BindingEnv[V] =
    copy(eval = eval.bind(name, v))

  override def bindEval(bindings: (Name, V)*): BindingEnv[V] =
    copy(eval = eval.bindIterable(bindings))

  def deleteEval(name: Name): BindingEnv[V] = copy(eval = eval.delete(name))
  def deleteEval(names: IndexedSeq[Name]): BindingEnv[V] = copy(eval = eval.delete(names))

  override def noEval: BindingEnv[V] = copy(eval = Env.empty)

  def bindAgg(name: Name, v: V): BindingEnv[V] =
    copy(agg = Some(agg.get.bind(name, v)))

  override def bindAgg(bindings: (Name, V)*): BindingEnv[V] =
    copy(agg = Some(agg.get.bindIterable(bindings)))

  def aggOrEmpty: Env[V] = agg.getOrElse(Env.empty)

  def bindScan(name: Name, v: V): BindingEnv[V] =
    copy(scan = Some(scan.get.bind(name, v)))

  override def bindScan(bindings: (Name, V)*): BindingEnv[V] =
    copy(scan = Some(scan.get.bindIterable(bindings)))

  def bindRelational(name: Name, v: V): BindingEnv[V] =
    copy(relational = relational.bind(name, v))

  override def bindRelational(bindings: (Name, V)*): BindingEnv[V] =
    copy(relational = relational.bind(bindings: _*))

  def scanOrEmpty: Env[V] = scan.getOrElse(Env.empty)

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

  def mapValuesWithKey[T](f: (Env.K, V) => T): BindingEnv[T] =
    copy[T](
      eval = eval.mapValuesWithKey(f),
      agg = agg.map(_.mapValuesWithKey(f)),
      scan = scan.map(_.mapValuesWithKey(f)),
      relational = relational.mapValuesWithKey(f),
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

  def mapValuesWithKey[U](f: (Env.K, V) => U): Env[U] =
    new Env(m.map { case (k, v) => (k, f(k, v)) })

  override def toString: String = m.map { case (k, v) => s"$k -> $v" }.mkString("(", ",", ")")
}
