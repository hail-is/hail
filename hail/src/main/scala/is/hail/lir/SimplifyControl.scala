package is.hail.lir

import is.hail.utils.UnionFind

import scala.collection.mutable

object SimplifyControl {
  def apply(m: Method): Unit = {
    new SimplifyControl(m).simplify()
  }
}

class SimplifyControl(m: Method) {
  private val uses = mutable.Map[Block, mutable.Set[(ControlX, Int)]]()

  private val q = mutable.Set[Block]()

  def removeUse(M: Block, c: ControlX, i: Int): Unit = {
    val r = uses(M).remove((c, i))
    assert(r)
  }

  def addUse(M: Block, c: ControlX, i: Int): Unit = {
    val r = uses(M).add((c, i))
    assert(r)
  }

  def finalTarget(b0: Block): Block = {
    var b = b0
    while (b.first != null &&
      b.first.isInstanceOf[GotoX])
      b = b.first.asInstanceOf[GotoX].L
    b
  }

  def simplifyBlock(L: Block): Unit = {
    val last = L.last.asInstanceOf[ControlX]

    if (uses(L).isEmpty && (L ne m.entry)) {
      var i = 0
      while (i < last.targetArity()) {
        val M = last.target(i)
        removeUse(M, last, i)
        last.setTarget(i, null)

        q += M
        if (uses(M).size == 1) {
          val (u, _) = uses(M).head
          q += u.parent
        }

        i += 1
      }

      // just popped off q
      uses -= L

      return
    }

    var i = 0
    while (i < last.targetArity()) {
      val M = last.target(i)
      val newM = finalTarget(M)
      if (M ne newM) {
        removeUse(M, last, i)
        last.setTarget(i, newM)
        addUse(newM, last, i)

        q += M
        if (uses(M).size == 1) {
          val (u, _) = uses(M).head
          q += u.parent
        }
        q += L
      }
      i += 1
    }

    last match {
      case x: IfX =>
        val M = x.Ltrue
        if (M eq x.Lfalse) {
          x.remove()
          removeUse(x.Ltrue, x, 0)
          x.Ltrue = null
          removeUse(x.Lfalse, x, 1)
          x.Lfalse = null

          val g = goto(M)
          addUse(M, g, 0)

          L.append(g)

          // if there is one parent, it is L
          q += L
        }

      case x: GotoX =>
        val M = x.L
        if ((M ne L) && uses(M).size == 1 && (m.entry ne M)) {
          x.remove()
          removeUse(x.L, x, 0)
          x.L = null

          while (M.first != null) {
            val z = M.first
            z.remove()
            L.append(z)
          }

          q -= M
          uses -= M

          q += L
        }

      case _ =>
    }
  }

  def unify(): Unit = {
    val (blocks, blockIdx) = m.findAndIndexBlocks()

    val u = new UnionFind(blocks.length)
    blocks.indices.foreach { i =>
      u.makeSet(i)
    }

    for (b <- blocks) {
      if (b.first != null &&
        b.first.isInstanceOf[GotoX]) {
        u.sameSet(
          blockIdx(b),
          blockIdx(b.first.asInstanceOf[GotoX].L))
      }
    }

    val rootFinalTarget = mutable.Map[Int, Block]()
    blocks.zipWithIndex.foreach { case (b, i) =>
      val r = u.find(i)
      val t = finalTarget(blocks(r))
      rootFinalTarget(r) = t
    }

    for (b <- blocks) {
      val last = b.last.asInstanceOf[ControlX]
      var i = 0
      while (i < last.targetArity()) {
        last.setTarget(i,
          rootFinalTarget(u.find(blockIdx(last.target(i)))))
        i += 1
      }
    }
  }

  def simplify(): Unit = {
    unify()

    val blocks = m.findBlocks()

    blocks.foreach { b =>
      uses(b) = mutable.Set()
    }

    for (b <- blocks) {
      q += b

      val last = b.last.asInstanceOf[ControlX]
      var i = 0
      while (i < last.targetArity()) {
        val t = last.target(i)
        addUse(t, last, i)
        i += 1
      }
    }

    while (q.nonEmpty) {
      val b = q.head
      q -= b
      simplifyBlock(b)
    }
  }
}
