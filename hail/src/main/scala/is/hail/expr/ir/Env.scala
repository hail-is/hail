package is.hail.expr.ir

object Env {
  type K = String

  def empty[V]: Env[V] = new Env()

  def apply[V](bindings: (String, V)*): Env[V] = fromSeq(bindings)

  def fromSeq[V](bindings: Iterable[(String, V)]): Env[V] = empty[V].bindIterable(bindings)
}

trait GenericBindingEnv[Self, V] {
  def extend(bindings: Bindings[V]): Self

  def promoteAgg: Self

  def promoteScan: Self

  def bindEval(bindings: (String, V)*): Self

  def noEval: Self

  def bindAgg(bindings: (String, V)*): Self

  def bindScan(bindings: (String, V)*): Self

  def bindInScope(name: String, v: V, scope: Int): Self = scope match {
    case Scope.EVAL => bindEval(name -> v)
    case Scope.AGG => bindAgg(name -> v)
    case Scope.SCAN => bindScan(name -> v)
  }

  def createAgg: Self

  def createScan: Self

  def noAgg: Self

  def noScan: Self

  def onlyRelational(keepAggCapabilities: Boolean = false): Self

  def bindRelational(bindings: (String, V)*): Self
}

object BindingEnv {
  def empty[T]: BindingEnv[T] = BindingEnv(Env.empty[T], None, None)

  def eval[T](bindings: (String, T)*): BindingEnv[T] =
    BindingEnv(Env.fromSeq[T](bindings), None, None)
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
    if (agg.isInstanceOf[AggEnv.Create] || scan.isInstanceOf[AggEnv.Create])
      newEnv =
        newEnv.copy(agg = newEnv.agg.map(_ => Env.empty), scan = newEnv.scan.map(_ => Env.empty))
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

  def extend(bindings: Bindings[V]): BindingEnv[V] = {
    val Bindings(all, eval, agg, scan, relational, _) = bindings
    var newEnv = modifyWithoutNewBindings(bindings)
    if (all.nonEmpty) {
      agg match {
        case AggEnv.Create(bindings) =>
          newEnv = newEnv.bindAgg(bindings.map(all): _*)
        case AggEnv.Bind(bindings) =>
          newEnv = newEnv.bindAgg(bindings.map(all): _*)
        case _ =>
      }
      scan match {
        case AggEnv.Create(bindings) =>
          newEnv = newEnv.bindScan(bindings.map(all): _*)
        case AggEnv.Bind(bindings) =>
          newEnv = newEnv.bindScan(bindings.map(all): _*)
        case _ =>
      }
      if (eval.nonEmpty) newEnv = newEnv.bindEval(eval.map(all): _*)
      if (relational.nonEmpty)
        newEnv = newEnv.bindRelational(relational.map(all): _*)
    }
    newEnv
  }

  def subtract[T](bindings: Bindings[T]): BindingEnv[V] = {
    val Bindings(all, eval, agg, scan, relational, _) = bindings
    var newEnv = modifyWithoutNewBindings(bindings)
    agg match {
      case AggEnv.Create(bindings) =>
        newEnv = newEnv.copy(agg = Some(newEnv.agg.get.delete(bindings.map(all(_)._1))))
      case AggEnv.Bind(bindings) =>
        newEnv = newEnv.copy(agg = Some(newEnv.agg.get.delete(bindings.map(all(_)._1))))
      case _ =>
    }
    scan match {
      case AggEnv.Create(bindings) =>
        newEnv = newEnv.copy(scan = Some(newEnv.scan.get.delete(bindings.map(all(_)._1))))
      case AggEnv.Bind(bindings) =>
        newEnv = newEnv.copy(scan = Some(newEnv.scan.get.delete(bindings.map(all(_)._1))))
      case _ =>
    }
    if (eval.nonEmpty) newEnv = newEnv.copy(eval = newEnv.eval.delete(eval.map(all(_)._1)))
    if (relational.nonEmpty)
      newEnv = newEnv.copy(relational = newEnv.relational.delete(relational.map(all(_)._1)))
    newEnv
  }

  def allEmpty: Boolean =
    eval.isEmpty && agg.forall(_.isEmpty) && scan.forall(_.isEmpty) && relational.isEmpty

  def promoteAgg: BindingEnv[V] = copy(eval = agg.get, agg = None)

  def promoteScan: BindingEnv[V] = copy(eval = scan.get, scan = None)

  def promoteScope(scope: Int): BindingEnv[V] = scope match {
    case Scope.EVAL => this
    case Scope.AGG => promoteAgg
    case Scope.SCAN => promoteScan
  }

  def noAgg: BindingEnv[V] = copy(agg = None)

  def noScan: BindingEnv[V] = copy(scan = None)

  def createAgg: BindingEnv[V] =
    copy(agg = Some(eval), scan = scan.map(_ => Env.empty))

  def createScan: BindingEnv[V] =
    copy(scan = Some(eval), agg = agg.map(_ => Env.empty))

  def onlyRelational(keepAggCapabilities: Boolean = false): BindingEnv[V] =
    BindingEnv(
      agg = if (keepAggCapabilities) agg.map(_ => Env.empty) else None,
      scan = if (keepAggCapabilities) scan.map(_ => Env.empty) else None,
      relational = relational,
    )

  def bindEval(name: String, v: V): BindingEnv[V] =
    copy(eval = eval.bind(name, v))

  def bindEval(bindings: (String, V)*): BindingEnv[V] =
    copy(eval = eval.bindIterable(bindings))

  def deleteEval(name: String): BindingEnv[V] = copy(eval = eval.delete(name))
  def deleteEval(names: IndexedSeq[String]): BindingEnv[V] = copy(eval = eval.delete(names))

  def noEval: BindingEnv[V] = copy(eval = Env.empty)

  def bindAgg(name: String, v: V): BindingEnv[V] =
    copy(agg = Some(agg.get.bind(name, v)))

  def bindAgg(bindings: (String, V)*): BindingEnv[V] =
    copy(agg = Some(agg.get.bindIterable(bindings)))

  def aggOrEmpty: Env[V] = agg.getOrElse(Env.empty)

  def bindScan(name: String, v: V): BindingEnv[V] =
    copy(scan = Some(scan.get.bind(name, v)))

  def bindScan(bindings: (String, V)*): BindingEnv[V] =
    copy(scan = Some(scan.get.bindIterable(bindings)))

  def bindRelational(name: String, v: V): BindingEnv[V] =
    copy(relational = relational.bind(name, v))

  def bindRelational(bindings: (String, V)*): BindingEnv[V] =
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

  def dropBindings[T]: BindingEnv[T] = copy(
    eval = Env.empty,
    agg = agg.map(_ => Env.empty),
    scan = scan.map(_ => Env.empty),
    relational = Env.empty,
  )
}

final class Env[V] private (val m: Map[Env.K, V]) {
  def this() =
    this(Map())

  override def equals(other: Any): Boolean = other match {
    case env: Env[V] => this.m == env.m
    case _ => false
  }

  def contains(k: Env.K): Boolean = m.contains(k)

  def isEmpty: Boolean = m.isEmpty

  def apply(name: String): V = m(name)

  def lookup(name: String): V =
    m.getOrElse(name, throw new RuntimeException(s"Cannot find $name in $m"))

  def lookupOption(name: String): Option[V] = m.get(name)

  def delete(name: String): Env[V] = new Env(m - name)

  def delete(names: Iterable[String]): Env[V] = new Env(m -- names)

  def lookup(r: Ref): V =
    lookup(r.name)

  def bind(name: String, v: V): Env[V] =
    new Env(m + (name -> v))

  def bind(bindings: (String, V)*): Env[V] = bindIterable(bindings)

  def bindIterable(bindings: Iterable[(String, V)]): Env[V] =
    if (bindings.isEmpty) this else new Env(m ++ bindings)

  def freshName(prefix: String): String = {
    var i = 0
    var name = prefix
    while (m.keySet.contains(name)) {
      name = prefix + i
      i += 1
    }
    name
  }

  def freshNames(prefixes: String*): Array[String] =
    (prefixes map freshName).toArray

  def bindFresh(prefix: String, v: V): (String, Env[V]) = {
    val name = freshName(prefix)
    (name, new Env(m + (name -> v)))
  }

  def bindFresh(bindings: (String, V)*): (Array[String], Env[V]) = {
    val names = new Array[String](bindings.length)
    var i = 0
    var e = this
    while (i < bindings.length) {
      val (prefix, v) = bindings(i)
      val (name, e2) = e.bindFresh(prefix, v)
      names(i) = name
      e = e2
      i += 1
    }
    (names, e)
  }

  def mapValues[U](f: (V) => U): Env[U] = new Env(m.mapValues(f))

  def mapValuesWithKey[U](f: (Env.K, V) => U): Env[U] =
    new Env(m.map { case (k, v) => (k, f(k, v)) })

  override def toString: String = m.map { case (k, v) => s"$k -> $v" }.mkString("(", ",", ")")
}
