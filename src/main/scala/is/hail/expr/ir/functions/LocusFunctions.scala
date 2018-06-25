package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.expr.types
import is.hail.asm4s
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.variant._
import is.hail.expr.ir._
import is.hail.utils.FastSeq

object LocusFunctions extends RegistryFunctions {

  def getLocus(mb: EmitMethodBuilder, locus: Code[Long], typeString: String): Code[Locus] = {
    val tlocus = types.coerce[TLocus](tv(typeString).t)
    Code.checkcast[Locus](wrapArg(mb, tlocus)(locus).asInstanceOf[Code[AnyRef]])
  }

  def registerLocusCode(methodName: String): Unit = {
    registerCode(methodName, tv("T", _.isInstanceOf[TLocus]), TBoolean()) {
      case (mb: EmitMethodBuilder, locus: Code[Long]) =>
        val locusObject = getLocus(mb, locus, "T")
        val tlocus = types.coerce[TLocus](tv("T").t)
        val rg = tlocus.rg.asInstanceOf[ReferenceGenome]

        val codeRG = mb.getReferenceGenome(rg)
        unwrapReturn(mb, TBoolean())(locusObject.invoke[RGBase, Boolean](methodName, codeRG))
    }
  }

  def registerAll() {
    registerCode("contig", tv("T", _.isInstanceOf[TLocus]), TString()) {
      case (mb, locus: Code[Long]) =>
        val region = getRegion(mb)
        val tlocus = types.coerce[TLocus](tv("T").t)
        tlocus.contig(region, locus)
    }

    registerCode("position", tv("T", _.isInstanceOf[TLocus]), TInt32()) {
      case (mb, locus: Code[Long]) =>
        val region = getRegion(mb)
        val tlocus = types.coerce[TLocus](tv("T").t)
        tlocus.position(region, locus)
    }

    registerLocusCode("isAutosomalOrPseudoAutosomal")
    registerLocusCode("isAutosomal")
    registerLocusCode("inYNonPar")
    registerLocusCode("inXPar")
    registerLocusCode("isMitochondrial")
    registerLocusCode("inXNonPar")
    registerLocusCode("inYPar")

    registerCode("min_rep", tv("T", _.isInstanceOf[TLocus]), TArray(TString()), TTuple(tv("T"), TArray(TString()))) { (mb, lOff, aOff) =>
      val returnTuple = mb.newLocal[(Locus, IndexedSeq[String])]
      val locus = getLocus(mb, lOff, "T")
      val alleles = Code.checkcast[IndexedSeq[String]](wrapArg(mb, TArray(TString()))(aOff).asInstanceOf[Code[AnyRef]])
      val tuple = Code.invokeScalaObject[Locus, IndexedSeq[String], (Locus, IndexedSeq[String])](VariantMethods.getClass, "minRep", locus, alleles)

      val newLocus = Code.checkcast[Locus](returnTuple.load().get[java.lang.Object]("_1"))
      val newAlleles = Code.checkcast[IndexedSeq[String]](returnTuple.load().get[java.lang.Object]("_2"))

      val srvb = new StagedRegionValueBuilder(mb, TTuple(tv("T").t, TArray(TString())))
      Code(
        returnTuple := tuple,
        srvb.start(),
        srvb.addBaseStruct(types.coerce[TBaseStruct](tv("T").t.fundamentalType), { locusBuilder =>
          Code(
            locusBuilder.start(),
            locusBuilder.addString(newLocus.invoke[String]("contig")),
            locusBuilder.advance(),
            locusBuilder.addInt(newLocus.invoke[Int]("position")))
        }),
        srvb.advance(),
        srvb.addArray(TArray(TString()), { allelesBuilder =>
          Code(
            allelesBuilder.start(newAlleles.invoke[Int]("size")),
            Code.whileLoop(allelesBuilder.arrayIdx < newAlleles.invoke[Int]("size"),
              allelesBuilder.addString(Code.checkcast[String](newAlleles.invoke[Int, java.lang.Object]("apply", allelesBuilder.arrayIdx))),
              allelesBuilder.advance()))
        }),
        srvb.offset)
    }
  }
}
