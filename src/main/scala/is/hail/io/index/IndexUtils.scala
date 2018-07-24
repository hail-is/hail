package is.hail.io.index

object IndexUtils {
  def calcDepth(nElements: Long, branchingFactor: Int) =
  // max necessary for array of length 1 becomes depth = 0
    math.max(1, (math.log10(nElements) / math.log10(branchingFactor)).ceil.toInt)
}
