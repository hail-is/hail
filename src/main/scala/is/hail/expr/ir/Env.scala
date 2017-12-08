package is.hail.expr.ir

object Env {
  type K = String
  def empty[V]: Env[V] = new Env()
}

class Env[V] private (val m: Map[Env.K,V]) {
  def this() {
    this(Map())
  }

  def lookup(name: String) =
    m.get(name)
      .getOrElse(throw new RuntimeException(s"Cannot find $name in $m"))

  def lookup(r: Ref): V =
    lookup(r.name)

  def bind(name: String, v: V): Env[V] =
    new Env(m + (name -> v))

  def bind(bindings: (String, V)*): Env[V] =
    new Env(m ++ bindings)

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
}
