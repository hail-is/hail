package is.hail.expr.ir

object Env {
  type K = Sym
  def empty[V]: Env[V] = new Env()
  def apply[V](bindings: (Sym, V)*): Env[V] = empty[V].bind(bindings: _*)
}

class Env[V] private (val m: Map[Env.K,V]) {
  def this() {
    this(Map())
  }

  def lookup(name: Sym): V =
    m.get(name)
      .getOrElse(throw new RuntimeException(s"Cannot find $name in $m"))

  def lookupOption(name: Sym): Option[V] = m.get(name)

  def delete(name: Sym): Env[V] = new Env(m - name)

  def lookup(r: Ref): V =
    lookup(r.name)

  def bind(name: Sym, v: V): Env[V] =
    new Env(m + (name -> v))

  def bind(bindings: (Any, V)*): Env[V] =
    new Env(m ++ bindings.map { case (k, v) => (toSym(k), v) })

  def freshName(prefix: String): Sym = {
    var i = 0
    var name = Identifier(prefix)
    while (m.keySet.contains(name)) {
      name = Identifier(prefix + i)
      i += 1
    }
    name
  }

  def freshNames(prefixes: String*): Array[Sym] =
    (prefixes map freshName).toArray

  def bindFresh(prefix: String, v: V): (Sym, Env[V]) = {
    val name = freshName(prefix)
    (name, new Env(m + (name -> v)))
  }

  def bindFresh(bindings: (String, V)*): (Array[Sym], Env[V]) = {
    val names = new Array[Sym](bindings.length)
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

  def mapValues[U](f: (V) => U): Env[U] =
    new Env(m.mapValues(f))
}
