package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.expr.types.virtual.TLocus
import is.hail.variant._

abstract class PLocus extends ComplexPType {
  def rgBc: BroadcastRG

  lazy val virtualType: TLocus = TLocus(rgBc)

  def rg: ReferenceGenome

  def contig(value: Long): String

  def contigType: PString

  def positionType: PInt32
}

abstract class PLocusValue extends PValue {
  def contig(mb: EmitMethodBuilder[_]): PStringCode

  def position(): Code[Int]

  def getLocusObj(mb: EmitMethodBuilder[_]): Code[Locus] =
    Code.invokeStatic2[Locus, String, Int, Locus]("apply", contig(mb).loadString(), position())
}

abstract class PLocusCode extends PCode {
  def pt: PLocus

  def rg: ReferenceGenome = pt.rg

  def contig(mb: EmitMethodBuilder[_]): PStringCode

  def position(): Code[Int]

  def getLocusObj(mb: EmitMethodBuilder[_]): Code[Locus]

  def memoize(cb: EmitCodeBuilder, name: String): PLocusValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PLocusValue

  // This one takes a region because it may need to allocate
  final def toCanonical(mb: EmitMethodBuilder[_], region: Value[Region]): PCanonicalLocusCode = {
    this match {
      case pc: PCanonicalLocusCode =>
        pc
      case pc =>
        val newPType = PCanonicalLocus(pt.rg, pt.required)
        val code = EmitCodeBuilder.scopedCode[Long](mb) { cb =>
          val pv = pc.memoize(cb, "locus_val")
          // PCanonicalLocus' fields are required so this naked allocation is fine
          val addr = cb.newLocal("addr", newPType.representation.allocate(region))
          cb += pv.contig(mb).store(mb, region, newPType.contigAddr(addr))
          cb += Region.storeInt(newPType.positionAddr(addr), pv.position())
          addr
        }
        new PCanonicalLocusCode(newPType, code)
    }
  }

  // This one doesn't because it doesn't need to allocate
  final def toBetter(mb: EmitMethodBuilder[_]): PBetterLocusCode = {
    this match {
      case pc: PBetterLocusCode =>
        pc
      case pc =>
        val code = EmitCodeBuilder.scopedCode[Long](mb) { cb =>
          val rg = mb.getReferenceGenome(pt.rg)
          val pv = pc.memoize(cb, "locus_val")
          val contigIdx = rg.invoke[String, Int]("contigIndex", pv.contig(mb).loadString())
          (contigIdx.toL << 32) | pv.position().toL
        }
        new PBetterLocusCode(PBetterLocus(pt.rgBc, pt.required), code)
    }
  }
}
