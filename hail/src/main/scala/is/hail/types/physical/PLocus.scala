package is.hail.types.physical

import is.hail.asm4s._
import is.hail.expr.ir.{ConsistentEmitCodeOrdering, EmitCode, EmitCodeBuilder, EmitModuleBuilder}
import is.hail.types.virtual.TLocus
import is.hail.variant._

abstract class PLocus extends ComplexPType {
  def rgBc: BroadcastRG

  lazy val virtualType: TLocus = TLocus(rgBc)

  override def codeOrdering2(modb: EmitModuleBuilder, _other: PType): ConsistentEmitCodeOrdering = {
    val self = this
    val other = _other.asInstanceOf[PLocus]
    new ConsistentEmitCodeOrdering(modb, this, other) {
      def emitCompare(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Int] = {
        val lhsv: PLocusValue = lhs.asLocus.memoize(cb, "lhs_locus_v")
        val rhsv: PLocusValue = rhs.asLocus.memoize(cb, "rhs_locus_v")
        val strord = modb.getCodeOrdering2(lhsv.contig().pt, rhsv.contig().pt)
        val posord = modb.getCodeOrdering2(self.positionType, other.positionType)

        cb.ifx(strord.equiv(cb, EmitCode.present(lhsv.contig()), EmitCode.present(rhsv.contig())), {
          cb._return(posord.compare(cb,
            EmitCode.present(self.positionType, lhsv.position()),
            EmitCode.present(other.positionType, rhsv.position())))
        })

        val codeRG = cb.emb.getReferenceGenome(rg)
        codeRG.invoke[String, String, Int]("compare",
          lhsv.contig().loadString(),
          rhsv.contig().loadString())
      }

      override def emitEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = {
        val lhsv: PLocusValue = lhs.asLocus.memoize(cb, "lhs_locus_v")
        val rhsv: PLocusValue = rhs.asLocus.memoize(cb, "rhs_locus_v")
        val strord = modb.getCodeOrdering2(lhsv.contig().pt, rhsv.contig().pt)

        lhsv.position().cne(rhsv.position()).mux(
          const(false),
          strord.equiv(cb, EmitCode.present(lhsv.contig()), EmitCode.present(rhsv.contig())))
      }
    }
  }

  def rg: ReferenceGenome

  def contig(value: Long): String

  def contigType: PString

  def position(value: Code[Long]): Code[Int]

  def positionType: PInt32
}

abstract class PLocusValue extends PValue {
  def contig(): PStringCode

  def position(): Value[Int]

  def getLocusObj(): Code[Locus] = Code.invokeStatic2[Locus, String, Int, Locus]("apply",
    contig().loadString(), position())
}

abstract class PLocusCode extends PCode {
  def pt: PLocus

  def contig(): PStringCode

  def position(): Code[Int]

  def getLocusObj(): Code[Locus]

  def memoize(cb: EmitCodeBuilder, name: String): PLocusValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PLocusValue
}
