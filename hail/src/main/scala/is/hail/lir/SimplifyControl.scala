package is.hail.lir

import is.hail.utils.UnionFind

import scala.collection.mutable

object SimplifyControl {
  def apply(m: Method): Unit =
    new SimplifyControl(m).simplify()
}

class SimplifyControl(m: Method) {
  private val q = mutable.Set[Block]()

  def finalTarget(b0: Block): Block = {
    var b = b0
    while (b.first.isInstanceOf[GotoX])
      b = b.first.asInstanceOf[GotoX].L
    b
  }

  def simplifyBlock(L: Block): Unit = {
    val last = L.last.asInstanceOf[ControlX]

    if (L.uses.isEmpty && (L ne m.entry)) {
      var i = 0
      while (i < last.targetArity()) {
        val M = last.target(i)
        last.setTarget(i, null)

        q += M
        if (M.uses.size == 1) {
          val (u, _) = M.uses.head
          q += u.parent
        }

        i += 1
      }

      return
    }

    var i = 0
    while (i < last.targetArity()) {
      val M = last.target(i)
      val newM = finalTarget(M)
      if (M ne newM) {
        last.setTarget(i, newM)

        q += M
        if (M.uses.size == 1) {
          val (u, _) = M.uses.head
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
          x.setLtrue(null)
          x.setLfalse(null)

          val g = goto(M)

          L.append(g)

          // if there is one parent, it is L
          q += L

          // if L is now a trivial jump, revisit parents to let them jump over L
          if (L.first eq L.last) {
            if (L eq m.entry)
              m.setEntry(finalTarget(L))
            else
              L.uses.foreach(q += _._1.parent)
          }
        }

      case x: GotoX =>
        val M = x.L
        if ((M ne L) && M.uses.size == 1 && (m.entry ne M)) {
          x.remove()
          x.setL(null)

          while (M.first != null) {
            val z = M.first
            z.remove()
            L.append(z)
          }

          q -= M

          q += L
        }

      case _ =>
    }
  }

  def unify(): Unit = {
    val blocks = m.findBlocks()

    val u = new UnionFind(blocks.length)
    blocks.indices.foreach(i => u.makeSet(i))

    for (b <- blocks) {
      if (
        b.first != null &&
        b.first.isInstanceOf[GotoX]
      ) {
        u.union(
          blocks.index(b),
          blocks.index(b.first.asInstanceOf[GotoX].L),
        )
      }
    }

    val rootFinalTarget = mutable.Map[Int, Block]()
    blocks.indices.foreach { i =>
      if (!blocks(i).first.isInstanceOf[GotoX]) {
        rootFinalTarget(u.find(i)) = blocks(i)
      }
    }

    for (b <- blocks) {
      val last = b.last.asInstanceOf[ControlX]
      var i = 0
      while (i < last.targetArity()) {
        last.setTarget(i, rootFinalTarget(u.find(blocks.index(last.target(i)))))
        i += 1
      }
    }

    m.setEntry(finalTarget(m.entry))
  }

  def simplify(): Unit = {
    unify()

    val blocks = m.findBlocks()

    assert(blocks.forall(!_.first.isInstanceOf[GotoX]))

    for (b <- blocks)
      q += b

    while (q.nonEmpty) {
      val b = q.head
      q -= b
      simplifyBlock(b)
    }

    assert(m.findBlocks().forall { b =>
      !b.first.isInstanceOf[GotoX] &&
      (b.last match {
        case i: IfX => i.Ltrue ne i.Lfalse
        case g: GotoX => g.L.uses.size > 1 || (g.L eq m.entry)
        case _ => true
      })
    })
  }
}
