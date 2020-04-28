package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitMethodBuilder}
import is.hail.expr.types.virtual.{TArray, TString}
import is.hail.utils._
import is.hail.variant._

object PBetterLocus {
  def apply(rg: ReferenceGenome): PLocus = PBetterLocus(rg.broadcastRG)

  def apply(rg: ReferenceGenome, required: Boolean): PLocus = PBetterLocus(rg.broadcastRG, required)
}

final case class PBetterLocus(rgBc: BroadcastRG, required: Boolean = false) extends PLocus {
  def rg: ReferenceGenome = rgBc.value

  def _asIdent = "locusplus"

  // 32 high bits are the contig, 32 low bits is the position
  val representation: PInt64 = PInt64(required)

  override def _pretty(sb: StringBuilder, indent: Call, compact: Boolean): Unit = sb.append(s"PBetterLocus($rg)")

  private def contigIndex(value: Long): Int = (value >>> 32).toInt

  def contig(value: Long): String = rg.contigs(contigIndex(value))

  def contigType: PString = PCanonicalString(required = true)

  def positionType: PInt32 = PInt32(true)

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrderingCompareConsistentWithOthers {
      type T = Long

      override def compareNonnull(lhs: Code[Long], rhsc: Code[Long]): Code[Int] = {
        val rhs = other match {
          case _: PBetterLocus => rhsc
          case t: PCanonicalLocus => t.toBetterLocus(mb, rhsc)
        }

        Code.invokeStatic2[java.lang.Long, Long, Long, Int]("compare", lhs, rhs)
      }
    }
  }

  def setRequired(required: Boolean): PType = if (this.required == required)
    this
  else
    PBetterLocus(this.rgBc, required)

  private[physical] def codeContigs(ecb: EmitClassBuilder[_]): PIndexableValue = {
    val contigs = rg.contigs.toFastIndexedSeq
    ecb.addLiteral(
      contigs,
      PType.literalPType(TArray(TString), contigs)
    ).asInstanceOf[PIndexableValue]
  }
}

object PBetterLocusSettable {
  def apply(sb: SettableBuilder, pt: PBetterLocus, name: String) =
    new PBetterLocusSettable(pt, sb.newSettable[Long](name))
}

class PBetterLocusSettable(val pt: PBetterLocus,
  val v: Settable[Long]
) extends PLocusValue with PSettable {
  def get = new PBetterLocusCode(pt, v)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(v)

  def store(pc: PCode): Code[Unit] = {
    v := pc.asInstanceOf[PBetterLocusCode].v
  }

  def contig(mb: EmitMethodBuilder[_]): PStringCode = get.contig(mb)

  def position(): Code[Int] = v.toI
}

object PBetterLocusCode {
  def fromLocusObj(pt: PBetterLocus, mb: EmitMethodBuilder[_], locus: Code[Locus]): PBetterLocusCode = {
    val value = Code.memoize(locus, "from_locus_obj_lv") { locus =>
      val rg = mb.getReferenceGenome(pt.rg)
      val contig = locus.invoke[String]("contig")
      val contigIdx = rg.invoke[String, Int]("contigIndex", contig)
      (contigIdx.toL << 32) | locus.invoke[Int]("position").toL
    }
    new PBetterLocusCode(pt, value)
  }
}

class PBetterLocusCode(val pt: PBetterLocus, val v: Code[Long]) extends PLocusCode {
  def code: Code[_] = v

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  def contig(mb: EmitMethodBuilder[_]): PStringCode = {
    val contigs = pt.codeContigs(mb.ecb)
    assert(contigs.pt.elementType.required)
    // FIXME would like a better idiom for CodeBuilder => PCode
    val c = EmitCodeBuilder.scopedCode(mb) { cb =>
      contigs.loadElement(cb, (v >>> 32).toI).handle(cb,
        cb._fatal("got missing on a required type. this is a bug")
      ).tcode[Long]
    }
    new PCanonicalStringCode(PCanonicalString(required=true), c)
  }

  def position(): Code[Int] = v.toI

  def getLocusObj(mb: EmitMethodBuilder[_]): Code[Locus] = EmitCodeBuilder.scopedCode[Locus](mb) { cb =>
    memoize(cb, "get_better_locus_obj_v").getLocusObj(mb)
  }

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PLocusValue = {
    val s = PBetterLocusSettable(sb, pt, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PLocusValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PLocusValue = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] =
    Region.storeLong(dst, v)
}
