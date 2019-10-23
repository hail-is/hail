package is.hail.utils

object PartitionCounts {

  def getPCSubsetOffset(n: Long, pcs: Iterator[Long]): Option[(Int, Long, Long)] = {
    var i = 0
    var nLeft = n
    for (c <- pcs) {
      if (c >= nLeft)
        return Some((i, nLeft, c - nLeft))
      i = i + 1
      nLeft = nLeft - c
    }
    return None
  }

  def getHeadPCs(original: IndexedSeq[Long], n: Long): IndexedSeq[Long] =
    getPCSubsetOffset(n, original.iterator) match {
      case Some((lastIdx, nKeep, nDrop)) =>
        (0 to lastIdx).map { i =>
          if (i == lastIdx)
            nKeep
          else
            original(i)
        }
      case None => original
    }

  def getTailPCs(original: IndexedSeq[Long], n: Long): IndexedSeq[Long] =
    getPCSubsetOffset(n, original.reverseIterator) match {
      case Some((lastIdx, nKeep, nDrop)) =>
        (0 to lastIdx).reverseMap { i =>
          if (i == lastIdx)
            nKeep
          else
            original(original.length - i - 1)
        }
      case None => original
    }
}
