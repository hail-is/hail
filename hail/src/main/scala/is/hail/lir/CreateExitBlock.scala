package is.hail.lir

import is.hail.asm4s.UnitInfo

import scala.collection.mutable

object CreateExitBlock {
  def apply(m: Method): Unit =  {
    val blocks = m.findBlocks()

    val exitBlocks = mutable.ArrayBuffer[Block]()
    for (b <- blocks) {
      b.last match {
        case _: ReturnX =>
          exitBlocks += b
      }
    }

    assert(exitBlocks.nonEmpty)
    if (exitBlocks.size > 1) {
      val L = new Block()
      L.method = m
      if (m.returnTypeInfo eq UnitInfo) {
        for (b <- exitBlocks) {
          b.last.replace(goto(L))
        }
        L.append(returnx())
      } else {
        val t = new Local(m, "return_value", m.returnTypeInfo)
        for (b <- exitBlocks) {
          val v = b.last.children(0)
          v.remove()
          b.last.replace(store(t, v))
        }
        L.append(returnx(load(t)))
      }
    }
  }
}
