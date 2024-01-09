package is.hail.utils

object PartitionCounts {

  case class PCSubsetOffset(
    finalIndex: Int,
    nKeep: Long,
    nDrop: Long,
  )

  def getPCSubsetOffset(n: Long, pcs: Iterator[Long]): Option[PCSubsetOffset] = {
    var i = 0
    var nLeft = n
    for (c <- pcs) {
      if (c >= nLeft)
        return Some(PCSubsetOffset(i, nLeft, c - nLeft))
      i = i + 1
      nLeft = nLeft - c
    }
    return None
  }

  def getHeadPCs(original: IndexedSeq[Long], n: Long): IndexedSeq[Long] =
    getPCSubsetOffset(n, original.iterator) match {
      case Some(PCSubsetOffset(lastIdx, nKeep, _)) =>
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
      case Some(PCSubsetOffset(lastIdx, nKeep, _)) =>
        (0 to lastIdx).reverseMap { i =>
          if (i == lastIdx)
            nKeep
          else
            original(original.length - i - 1)
        }
      case None => original
    }

  def incrementalPCSubsetOffset(
    n: Long,
    partIndices: IndexedSeq[Int],
  )(
    computePCs: IndexedSeq[Int] => IndexedSeq[Long]
  ): PCSubsetOffset = {
    var nLeft = n
    var nPartsScanned = 0
    var lastIdx = -1
    var lastPC = 0L
    var nPartsToTry = 1

    while (nPartsScanned < partIndices.length) {
      if (nPartsScanned > 0) {
        val nSeen = n - nLeft
        // If we didn't find any rows after the previous iteration, quadruple and retry.
        // Otherwise, interpolate the number of partitions we need to try, but overestimate
        // it by 50%. We also cap the estimation in the end.
        if (nSeen == 0) {
          nPartsToTry = nPartsToTry * 4
        } else {
          // the left side of max is >=1 whenever nPartsScanned >= 2
          nPartsToTry = Math.max((1.5 * n * nPartsScanned / nSeen).toInt - nPartsScanned, 1)
          nPartsToTry = Math.min(nPartsToTry, nPartsScanned * 4)
        }
      }

      val indices = partIndices.slice(nPartsScanned, nPartsScanned + nPartsToTry)
      val pcs = computePCs(indices)

      getPCSubsetOffset(nLeft, pcs.iterator) match {
        case Some(PCSubsetOffset(i, nKeep, nDrop)) =>
          return PCSubsetOffset(indices(i), nKeep, nDrop)
        case None =>
          nLeft = nLeft - pcs.sum
          lastIdx = indices.last
          lastPC = pcs.last
          nPartsScanned = nPartsScanned + indices.length
      }
    }

    PCSubsetOffset(lastIdx, lastPC, 0L)
  }
}
