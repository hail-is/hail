package org.broadinstitute.k3.variant

object DNASeq {
  def charToNt(c:Char): Int = {
     c match {
       case 'A' => 0
       case 'C' => 1
       case 'G' => 2
         // FIXME case 'T' => return 3, case _ => fail()
       case _ =>
         assert(c == 'T')
         3
     }
  }

  def ntToChar(nt:Int):Char = {
    nt match {
      case 0 => 'A'
      case 1 => 'C'
      case 2 => 'G'
      case _ =>
        assert(nt == 3)
        'T'
    }
  }

  def apply(c: Char): DNASeq = {
    new DNASeq(IndexedSeq(charToNt(c).toByte), 1)
  }

  def apply(s:String): DNASeq = {
    val b = IndexedSeq.newBuilder[Byte]
    b.sizeHint((s.length + 3) / 4)

    var v: Int = 0
    s.zipWithIndex foreach { case (c, i) =>
      val j = i % 4
      v |= (charToNt(c) << (2*j))
      if (j == 3) {
        b += v.toByte
        v = 0
      }
    }
    if (s.length > 0 && (s.length % 4) != 3)
      b += v.toByte

    new DNASeq(b.result, s.length)
  }
}

case class DNASeq(val seq: IndexedSeq[Byte],
  val len:Int) {

  def length():Int = len

  def apply(i:Int):Char = {
    val j = i % 4
    DNASeq.ntToChar((seq(i / 4) >>> (j*2)) & 3)
  }

  override def toString():String = {
    val b = new StringBuilder(len)
    for (i <- 0 to len-1)
      b += this(i)
    b.result
  }


}
