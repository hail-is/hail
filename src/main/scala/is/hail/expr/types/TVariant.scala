package is.hail.expr.types

import is.hail.annotations._
import is.hail.check._
import is.hail.utils._
import is.hail.variant._

import scala.reflect.ClassTag
import scala.reflect.classTag

object TVariant {
  def representation(required: Boolean = false): TStruct = {
  	val rep = TStruct(
    "contig" -> +TString(),
    "start" -> +TInt32(),
    "ref" -> +TString(),
    "altAlleles" -> +TArray(+TAltAllele()))
    rep.setRequired(required).asInstanceOf[TStruct]
  }
}

case class TVariant(gr: GRBase, override val required: Boolean = false) extends ComplexType {
  def _toString = s"""Variant($gr)"""

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Variant]

  override def genNonmissingValue: Gen[Annotation] = VariantSubgen.fromGenomeRef(gr.asInstanceOf[GenomeReference]).gen

  override def desc: String =
    """
    A ``Variant(GR)`` is a Hail data type representing a variant in the dataset. It is parameterized by a genome reference (GR) such as GRCh37 or GRCh38. It is referred to as ``v`` in the expression language.

    The `pseudoautosomal region <https://en.wikipedia.org/wiki/Pseudoautosomal_region>`_ (PAR) is currently defined with respect to reference `GRCh37 <http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/>`_:

    - X: 60001 - 2699520, 154931044 - 155260560
    - Y: 10001 - 2649520, 59034050 - 59363566

    Most callers assign variants in PAR to X.
    """

  override def scalaClassTag: ClassTag[Variant] = classTag[Variant]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(gr.variantOrdering)

  override val partitionKey: Type = TLocus(gr)

  // FIXME: Remove when representation of contig/position is a naturally-ordered Long
  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = {
    val fundamentalComparators = representation.fields.map(_.typ.unsafeOrdering(missingGreatest)).toArray
    val repr = representation.fundamentalType
    new UnsafeOrdering {
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        val cOff1 = repr.loadField(r1, o1, 0)
        val cOff2 = repr.loadField(r2, o2, 0)

        val contig1 = TString.loadString(r1, cOff1)
        val contig2 = TString.loadString(r2, cOff2)

        val c = gr.compare(contig1, contig2)
        if (c != 0)
          return c

        var i = 1
        while (i < representation.size) {
          val fOff1 = repr.loadField(r1, o1, i)
          val fOff2 = repr.loadField(r2, o2, i)

          val c = fundamentalComparators(i).compare(r1, fOff1, r2, fOff2)
          if (c != 0)
            return c

          i += 1
        }
        0
      }
    }
  }

  def variantOrdering: Ordering[Variant] = gr.variantOrdering

  val representation: TStruct = TVariant.representation(required)

  override def unify(concrete: Type): Boolean = concrete match {
    case TVariant(cgr, _) => gr.unify(cgr)
    case _ => false
  }

  override def clear(): Unit = gr.clear()

  override def subst() = gr.subst().variantType
}
