package is.hail.lir

class Blocks(
  blocks: Array[Block]
) extends IndexedSeq[Block] {
  private val blockIdx = blocks.iterator.zipWithIndex.toMap

  def nBlocks: Int = blocks.length

  def length: Int = blocks.length

  def apply(i: Int): Block = blocks(i)

  def index(block: Block): Int = blockIdx(block)
}
