package is.hail.variant

import is.hail.annotations.Region

/** Caches the contig index of the last contig looked up, so that runs of loci on one contig, as
  * when loci are processed in key order, skip building a String and the reference genome's hash
  * lookup.
  *
  * Not thread-safe: an instance must be confined to one thread. Never store one on a
  * [[ReferenceGenome]], which executors share across threads.
  */
final class ContigIndexCache(rg: ReferenceGenome) {
  // A copy, because the region holding the last contig may be freed once we have returned.
  private[this] var cachedBytes: Array[Byte] = null
  private[this] var cachedIndex: Int = -1

  /** The contig index of the contig whose `length` bytes start at `bytesAddress`. */
  def contigIndex(bytesAddress: Long, length: Int): Int = {
    if (cachedBytes != null && cachedBytes.length == length) {
      var i = 0
      while (i < length && Region.loadByte(bytesAddress + i) == cachedBytes(i))
        i += 1
      if (i == length)
        return cachedIndex
    }

    val bytes = Region.loadBytes(bytesAddress, length)
    // Same decoding as PCanonicalString.loadString, so misses agree with the uncached lookup.
    val idx = rg.getContigIndex(new String(bytes))
    cachedBytes = bytes
    cachedIndex = idx
    idx
  }
}
