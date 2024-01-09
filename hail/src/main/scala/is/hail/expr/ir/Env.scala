package is.hail.expr.ir

object Env {
  type K = String

  def empty[V]: Env[V] = new Env()

  def apply[V](bindings: (String, V)*): Env[V] = fromSeq(bindings)

  def fromSeq[V](bindings: Iterable[(String, V)]): Env[V] = empty[V].bindIterable(bindings)
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
) {
  def allEmpty: Boolean =
    eval.isEmpty && agg.forall(_.isEmpty) && scan.forall(_.isEmpty) && relational.isEmpty

  def promoteAgg: BindingEnv[V] = copy(eval = agg.get, agg = None)

  def promoteScan: BindingEnv[V] = copy(eval = scan.get, scan = None)

  def noAgg: BindingEnv[V] = copy(agg = None)

  def noScan: BindingEnv[V] = copy(scan = None)

  def createAgg: BindingEnv[V] =
    copy(agg = Some(eval), scan = scan.map(_ => Env.empty))

  def createScan: BindingEnv[V] =
    copy(scan = Some(eval), agg = agg.map(_ => Env.empty))

  def onlyRelational: BindingEnv[V] = BindingEnv(relational = relational)

  def bindEval(name: String, v: V): BindingEnv[V] =
    copy(eval = eval.bind(name, v))

  def bindEval(bindings: (String, V)*): BindingEnv[V] =
    copy(eval = eval.bindIterable(bindings))

  def deleteEval(name: String): BindingEnv[V] = copy(eval = eval.delete(name))
  def deleteEval(names: IndexedSeq[String]): BindingEnv[V] = copy(eval = eval.delete(names))

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

  def scanOrEmpty: Env[V] = scan.getOrElse(Env.empty)

  def pretty(valuePrinter: V => String = _.toString): String =
    s"""BindingEnv:
       |  Eval:${eval.m.map { case (k, v) => s"\n    $k -> ${valuePrinter(v)}" }.mkString("")}
       |  Agg: ${agg.map(_.m.map { case (k, v) => s"\n    $k -> ${valuePrinter(v)}" }.mkString("")).getOrElse("None")}
       |  Scan: ${scan.map(_.m.map { case (k, v) => s"\n    $k -> ${valuePrinter(v)}" }.mkString("")).getOrElse("None")}
       |  Relational: ${relational.m.map { case (k, v) =>
        s"\n    $k -> ${valuePrinter(v)}"
      }.mkString("")}""".stripMargin

  def merge(newBindings: BindingEnv[V]): BindingEnv[V] = {
    if (agg.isDefined != newBindings.agg.isDefined || scan.isDefined != newBindings.scan.isDefined)
      throw new RuntimeException(s"found inconsistent agg or scan environments:" +
        s"\n  left: ${agg.isDefined}, ${scan.isDefined}" +
        s"\n  right: ${newBindings.agg.isDefined}, ${newBindings.scan.isDefined}")
    if (allEmpty)
      newBindings
    else if (newBindings.allEmpty)
      this
    else {
      copy(
        eval = eval.bindIterable(newBindings.eval.m),
        agg = agg.map(a => a.bindIterable(newBindings.agg.get.m)),
        scan = scan.map(a => a.bindIterable(newBindings.scan.get.m)),
        relational = relational.bindIterable(newBindings.relational.m),
      )
    }
  }

  def subtract(newBindings: BindingEnv[_]): BindingEnv[V] = {
    if (agg.isDefined != newBindings.agg.isDefined || scan.isDefined != newBindings.scan.isDefined)
      throw new RuntimeException(s"found inconsistent agg or scan environments:" +
        s"\n  left: ${agg.isDefined}, ${scan.isDefined}" +
        s"\n  right: ${newBindings.agg.isDefined}, ${newBindings.scan.isDefined}")
    if (allEmpty || newBindings.allEmpty)
      this
    else {
      copy(
        eval = eval.delete(newBindings.eval.m.keys),
        agg = agg.map(a => a.delete(newBindings.agg.get.m.keys)),
        scan = scan.map(a => a.delete(newBindings.scan.get.m.keys)),
        relational = relational.delete(newBindings.relational.m.keys),
      )
    }
  }

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

class Env[V] private (val m: Map[Env.K, V]) {
  def this() =
    this(Map())

  def contains(k: Env.K): Boolean = m.contains(k)

  def isEmpty: Boolean = m.isEmpty

  def apply(name: String): V = m(name)

  def lookup(name: String): V =
    m.get(name).getOrElse(throw new RuntimeException(s"Cannot find $name in $m"))

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
