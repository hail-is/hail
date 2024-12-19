package is.hail.lir

class Blocks(
  blocks: Array[Block]
) extends IndexedSeq[Block] {
  private val blockIdx = {
    val jm = new java.util.HashMap[Block, Int]
    blocks.iterator.zipWithIndex.foreach { case (b, i) => jm.put(b, i) }
    jm
  }

  def nBlocks: Int = blocks.length

  def length: Int = blocks.length

  def apply(i: Int): Block = blocks(i)

  def index(block: Block): Int = blockIdx.get(block)
}
