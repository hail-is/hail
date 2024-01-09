package is.hail.expr

abstract class NatBase {
  def clear(): Unit
  def unify(concrete: NatBase): Boolean
  def subst(): NatBase
}

case class Nat(n: Int) extends NatBase {
  override def toString: String = n.toString

  override def clear() {}

  override def unify(concrete: NatBase): Boolean =
    concrete match {
      case Nat(cN) => cN == n
      case _ => false
    }

  override def subst(): NatBase = this
}

case class NatVariable(var nat: NatBase = null) extends NatBase {
  override def toString: String = "?nat"

  override def clear() { nat = null }

  override def unify(concrete: NatBase): Boolean = {
    if (nat != null) {
      nat.unify(concrete)
    } else {
      nat = concrete
      true
    }
  }

  override def subst(): NatBase = {
    assert(nat != null)
    nat
  }
}
