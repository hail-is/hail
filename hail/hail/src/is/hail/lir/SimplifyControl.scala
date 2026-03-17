package is.hail.lir

import is.hail.utils.UnionFind

import scala.annotation.tailrec
import scala.collection.mutable

object SimplifyControl {
  def apply(m: Method): Unit =
    new SimplifyControl(m).simplify()
}

class SimplifyControl(m: Method) {
  private[this] val q = mutable.Set[Block]()

  @tailrec final private def finalTarget(b: Block): Block =
    b.first match {
      case g: GotoX => finalTarget(g.L)
      case _ => b
    }

  private def simplifyBlock(L: Block): Unit =
    if (L.uses.isEmpty && (L ne m.entry)) {
      val last = L.last.asInstanceOf[ControlX]
      val T = last.targetArity()
      var t = 0
      while (t < T) {
        val M = last.target(t)
        last.setTarget(t, null)
        t += 1

        q += M
        if (M.uses.size == 1) {
          q += M.uses.head._1.parent
        }
      }
    } else {
      val last = L.last.asInstanceOf[ControlX]
      val T = last.targetArity()
      var t = 0
      while (t < T) {
        val M = last.target(t)
        val newM = finalTarget(M)

        if (M ne newM) {
          last.setTarget(t, newM)

          q += M
          if (M.uses.size == 1) {
            q += M.uses.head._1.parent
          }
          q += L
        }

        t += 1
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
    val N = blocks.length

    val u = new UnionFind(N)

    var i = 0
    while (i < N) {
      u.makeSet(i)
      i += 1
    }

    i = 0
    while (i < N) {
      blocks(i).first match {
        case g: GotoX => u.union(i, blocks.index(g.L))
        case _ =>
      }
      i += 1
    }

    val rootFinalTarget = mutable.Map[Int, Block]()

    i = 0
    while (i < N) {
      val b = blocks(i)
      if (!b.first.isInstanceOf[GotoX]) {
        rootFinalTarget(u.find(i)) = b
      }
      i += 1
    }

    i = 0
    while (i < N) {
      val last = blocks(i).last.asInstanceOf[ControlX]
      val T = last.targetArity()
      var t = 0
      while (t < T) {
        last.setTarget(t, rootFinalTarget(u.find(blocks.index(last.target(t)))))
        t += 1
      }
      i += 1
    }

    m.setEntry(finalTarget(m.entry))
  }

  def simplify(): Unit = {
    unify()

    {
      val blocks = m.findBlocks()
      val N = blocks.length
      var i = 0
      while (i < N) {
        val b = blocks(i)
        assert(!b.first.isInstanceOf[GotoX])
        q += b
        i += 1
      }
    }

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
