package is.hail.expr.ir

object Env {
  type K = (String, Boolean)
}

class Env[V] private (val m: Map[Env.K,V]) {
  def this() {
    this(Map())
  }

  def lookup(name: String, userGenerated: Boolean = true) =
    m.get(name -> userGenerated)
      .getOrElse(throw new RuntimeException(s"Cannot find $name ($userGenerated) in $m"))

  def lookup(r: Ref): V =
    lookup(r.name, r.userGenerated)

  def bind(name: String, v: V): Env[V] =
    new Env(m + ((name, true) -> v))

  def bind(bindings: (String, V)*): Env[V] =
    new Env(m ++ bindings.map { case (n, v) => ((n, true), v) })

  def bindFresh(prefix: String, v: V): (String, Env[V]) = {
    var i = 0
    var name = prefix
    while (m.keySet.contains((name, false))) {
      name = prefix + i
      i += 1
    }
    (name, new Env(m + ((name, false) -> v)))
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
}
