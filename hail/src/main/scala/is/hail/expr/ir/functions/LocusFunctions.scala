package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.expr.types
import is.hail.asm4s
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.variant._
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils._

object LocusFunctions extends RegistryFunctions {

  def rgCode(mb: EmitMethodBuilder[_], rg: ReferenceGenome): Code[ReferenceGenome] =
    mb.getReferenceGenome(rg)

  def tlocus(name: String): Type = tv(name, "locus")
  def tvariant(name: String): TStruct = TStruct("locus" -> tlocus(name), "alleles" -> TArray(TString))
  def tinterval(name: String): TInterval = TInterval(tlocus(name))

  def getLocus(r: EmitRegion, locus: Code[Long], locusT: PLocus): Code[Locus] = {
    val l = r.mb.newLocal[Long]()
    val lObject = r.mb.newLocal[Locus]()
    Code(
      l := locus,
      lObject := Code.invokeStatic[Locus, String, Int, Locus]("apply",
        locusT.contigType.loadString(locusT.contig(l)),
        locusT.position(l)), lObject)
  }

  def emitLocus(r: EmitRegion, locus: Code[Locus], rt: PLocus): Code[Long] = {
    val srvb = new StagedRegionValueBuilder(r, rt)
    Code(emitLocus(srvb, locus), srvb.offset)
  }

  def emitLocus(srvb: StagedRegionValueBuilder, locus: Code[Locus]): Code[Unit] = {
    val llocal = srvb.mb.newLocal[Locus]()
    Code.memoize(locus, "locus_funs_emit_locus_locus") { locus =>
      Code(
        llocal := locus,
        srvb.start(),
        srvb.addString(locus.invoke[String]("contig")),
        srvb.advance(),
        srvb.addInt(locus.invoke[Int]("position")),
        srvb.advance())
    }
  }

  def emitVariant(r: EmitRegion, variant: Code[(Locus, IndexedSeq[String])], rt: PStruct): Code[Long] = {
    val vlocal = r.mb.newLocal[(Locus, IndexedSeq[String])]()
    val alocal = r.mb.newLocal[IndexedSeq[String]]()
    val len = r.mb.newLocal[Int]()
    val srvb = new StagedRegionValueBuilder(r, rt)
    val addLocus = { srvb: StagedRegionValueBuilder =>
      emitLocus(srvb, Code.checkcast[Locus](vlocal.getField[java.lang.Object]("_1")))
    }
    val addAlleles = { srvb: StagedRegionValueBuilder =>
      Code(
        srvb.start(len),
        Code.whileLoop(srvb.arrayIdx < len,
          srvb.addString(alocal.invoke[Int, String]("apply", srvb.arrayIdx)),
          srvb.advance()))
    }

    Code(
      vlocal := variant,
      alocal := Code.checkcast[IndexedSeq[String]](vlocal.getField[java.lang.Object]("_2")),
      len := alocal.invoke[Int]("size"),
      srvb.start(),
      srvb.addBaseStruct(types.coerce[PStruct](rt.field("locus").typ.fundamentalType), addLocus),
      srvb.advance(),
      srvb.addArray(rt.field("alleles").typ.asInstanceOf[PArray], addAlleles),
      srvb.advance(),
      srvb.offset)
  }

  def emitInterval(r: EmitRegion, interval: Code[Interval], pt: PInterval): Code[Long] = {
    val srvb = new StagedRegionValueBuilder(r, pt)
    Code(emitInterval(srvb, interval, pt), srvb.offset)
  }

  def emitInterval(srvb: StagedRegionValueBuilder, interval: Code[Interval], pt: PInterval): Code[Unit] = {
    val ilocal = srvb.mb.newLocal[Interval]()
    val addLocus = { (srvb: StagedRegionValueBuilder, point: String) =>
      emitLocus(srvb, Code.checkcast[Locus](ilocal.invoke[java.lang.Object](point)))
    }

    val lt = pt.pointType.fundamentalType.asInstanceOf[PBaseStruct]
    Code(FastIndexedSeq(
      ilocal := interval,
      srvb.start(),
      srvb.addBaseStruct(types.coerce[PBaseStruct](lt), addLocus(_, "start")),
      srvb.advance(),
      srvb.addBaseStruct(types.coerce[PBaseStruct](lt), addLocus(_, "end")),
      srvb.advance(),
      srvb.addBoolean(ilocal.invoke[Boolean]("includesStart")),
      srvb.advance(),
      srvb.addBoolean(ilocal.invoke[Boolean]("includesEnd")),
      srvb.advance()))
  }

  def emitLiftoverLocus(r: EmitRegion, result: Code[(Locus, Boolean)], rt: PStruct): Code[Long] = {
    val rlocal = r.mb.newLocal[(Locus, Boolean)]()
    val blocal = r.mb.newLocal[Boolean]()
    val srvb = new StagedRegionValueBuilder(r, rt)
    val addLocus = { srvb: StagedRegionValueBuilder =>
      emitLocus(srvb, Code.checkcast[Locus](rlocal.getField[java.lang.Object]("_1")))
    }

    Code(
      rlocal := result,
      blocal := Code.checkcast[java.lang.Boolean](rlocal.getField[java.lang.Object]("_2")).invoke[Boolean]("booleanValue"),
      srvb.start(),
      srvb.addBaseStruct(types.coerce[PStruct](rt.field("result").typ.fundamentalType), addLocus),
      srvb.advance(),
      srvb.addBoolean(blocal),
      srvb.advance(),
      srvb.offset)
  }

  def emitLiftoverLocusInterval(r: EmitRegion, result: Code[(Interval, Boolean)], pt: PStruct): Code[Long] = {
    val rlocal = r.mb.newLocal[(Interval, Boolean)]()
    val ilocal = r.mb.newLocal[Interval]()
    val blocal = r.mb.newLocal[Boolean]()
    val srvb = new StagedRegionValueBuilder(r, pt)
    val pinterval = pt.field("result").typ.asInstanceOf[PInterval]
    val addInterval = { srvb: StagedRegionValueBuilder =>
      emitInterval(srvb, ilocal, pinterval)
    }

    Code(
      rlocal := result,
      ilocal := Code.checkcast[Interval](rlocal.getField[java.lang.Object]("_1")),
      blocal := Code.checkcast[java.lang.Boolean](rlocal.getField[java.lang.Object]("_2")).invoke[Boolean]("booleanValue"),
      srvb.start(),
      srvb.addBaseStruct(types.coerce[PStruct](pinterval.fundamentalType), addInterval),
      srvb.advance(),
      srvb.addBoolean(blocal),
      srvb.advance(),
      srvb.offset)
  }

  def registerLocusCode(methodName: String)(f: IR => IR): Unit =
    registerIR(methodName, tlocus("T"), TBoolean)(f)

  def inX(locus: IR): IR = {
    val xContigs = Literal(TSet(TString), locus.typ.asInstanceOf[TLocus].rg.xContigs)
    invoke("contains", TBoolean, xContigs, invoke("contig", TString, locus))
  }

  def inY(locus: IR): IR = {
    val yContigs = Literal(TSet(TString), locus.typ.asInstanceOf[TLocus].rg.yContigs)
    invoke("contains", TBoolean, yContigs, invoke("contig", TString, locus))
  }

  def inPar(locus: IR): IR = {
    val t = locus.typ.asInstanceOf[TLocus]
    val par = Literal(TArray(TInterval(t)), t.rg.par.toFastIndexedSeq)
    ArrayFunctions.exists(par, interval => invoke("contains", TBoolean, interval, locus))
  }

  def isMitochondrial(locus: IR): IR = {
    val mtContigs = Literal(TSet(TString), locus.typ.asInstanceOf[TLocus].rg.mtContigs)
    invoke("contains", TBoolean, mtContigs, invoke("contig", TString, locus))
  }
  def isAutosomal(locus: IR): IR = !(inX(locus) || inY(locus) || isMitochondrial(locus))

  def registerAll() {
    val locusClass = Locus.getClass

    registerCode("contig", tlocus("T"), TString,
      (x: PType) => -x.asInstanceOf[PLocus].contigType) {
      case (r, rt, (locusT: PLocus, locus: Code[Long])) =>
        locusT.contig(locus)
    }

    registerCode("position", tlocus("T"), TInt32, null) {
      case (r, rt, (locusT: PLocus, locus: Code[Long])) =>
        locusT.position(locus)
    }
    registerLocusCode("isAutosomalOrPseudoAutosomal") { locus =>
      isAutosomal(locus) || ((inX(locus) || inY(locus)) && inPar(locus))
    }
    registerLocusCode("isAutosomal")(isAutosomal)
    registerLocusCode("isMitochondrial")(isMitochondrial)
    registerLocusCode("inXPar") { locus => inX(locus) && inPar(locus) }
    registerLocusCode("inYPar") { locus => inY(locus) && inPar(locus) }
    registerLocusCode("inXNonPar") { locus => inX(locus) && !inPar(locus) }
    registerLocusCode("inYNonPar") { locus => inY(locus) && !inPar(locus) }

    registerCode("min_rep", tlocus("T"), TArray(TString), TStruct("locus" -> tv("T"), "alleles" -> TArray(TString)), null) {
      case (r, rt: PStruct, (locusT: PLocus, lOff), (allelesT, aOff)) =>
        val locus = getLocus(r, lOff, locusT)
        val alleles = Code.checkcast[IndexedSeq[String]](wrapArg(r, allelesT)(aOff).asInstanceOf[Code[AnyRef]])
        val tuple = Code.invokeScalaObject[Locus, IndexedSeq[String], (Locus, IndexedSeq[String])](VariantMethods.getClass, "minRep", locus, alleles)

        Code.memoize(tuple, "min_rep_tuple") { tuple =>
          Code.memoize(
            Code.checkcast[Locus](tuple.getField[java.lang.Object]("_1")), "min_rep_new_locus",
            Code.checkcast[IndexedSeq[String]](tuple.getField[java.lang.Object]("_2")), "min_rep_new_alleles"
          ) { (newLocus, newAlleles) =>
            val newLocusT = rt.field("locus").typ
            val newAllelesT = rt.field("alleles").typ.asInstanceOf[PArray]
            val srvb = new StagedRegionValueBuilder(r, rt)
            Code(
              srvb.start(),
              srvb.addBaseStruct(newLocusT.fundamentalType.asInstanceOf[PStruct], { locusBuilder =>
                Code(
                  locusBuilder.start(),
                  locusBuilder.addString(newLocus.invoke[String]("contig")),
                  locusBuilder.advance(),
                  locusBuilder.addInt(newLocus.invoke[Int]("position")))
              }),
              srvb.advance(),
              srvb.addArray(newAllelesT, { allelesBuilder =>
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

    registerCode("locus_windows_per_contig", TArray(TArray(TFloat64)), TFloat64, TTuple(TArray(TInt32), TArray(TInt32)), null) {
      case (r: EmitRegion, rt: PTuple, (groupedT: PArray, _coords: Code[Long]), (radiusT: PFloat64, _radius: Code[Double])) =>
        val coordT = types.coerce[PArray](groupedT.elementType)

        val coords = r.mb.newLocal[Long]("coords")
        val radius = r.mb.newLocal[Double]("radius")
        val ncontigs = r.mb.newLocal[Int]("ncontigs")
        val totalLen = r.mb.newLocal[Int]("l")
        val iContig = r.mb.newLocal[Int]()

        val len = r.mb.newLocal[Int]("l")
        val i = r.mb.newLocal[Int]("i")
        val idx = r.mb.newLocal[Int]("i")
        val coordsPerContig = r.mb.newLocal[Long]("coords")
        val offset = r.mb.newLocal[Int]("offset")
        val lastCoord = r.mb.newLocal[Double]("coord")

        val getCoord = { i: Code[Int] =>
          asm4s.coerce[Double](Region.loadIRIntermediate(coordT.elementType)(coordT.elementOffset(coordsPerContig, len, i)))
        }

        def forAllContigs(c: Code[Unit]): Code[Unit] = {
          Code(iContig := 0,
            Code.whileLoop(iContig < ncontigs,
              coordsPerContig := asm4s.coerce[Long](
                Region.loadIRIntermediate(coordT)(
                  groupedT.elementOffset(coords, ncontigs, iContig))),
              c,
              iContig += 1))
        }

        val startCond = getCoord(i) > (getCoord(idx) + radius)
        val endCond = getCoord(i) >= (getCoord(idx) - radius)

        val addIdxWithCondition = { (cond: Code[Boolean], sab: StagedRegionValueBuilder) =>
          Code(
            sab.start(totalLen),
            offset := 0,
            forAllContigs(
              Code(
                i := 0,
                idx := 0,
                len := coordT.loadLength(coordsPerContig),
                len.ceq(0).mux(lastCoord := 0.0, lastCoord := getCoord(0)),
                Code.whileLoop(i < len,
                  coordT.isElementMissing( coordsPerContig, i).mux(
                    Code._fatal[Unit](
                      const("locus_windows: missing value for 'coord_expr' at row ")
                        .concat((offset + i).toS)),
                    (lastCoord > getCoord(i)).mux(
                      Code._fatal[Unit]("locus_windows: 'coord_expr' must be in ascending order within each contig."),
                      lastCoord := getCoord(i))),
                  Code.whileLoop((idx < len) && cond, idx += 1),
                  sab.addInt(offset + idx),
                  sab.advance(),
                  i += 1),
                offset := offset + len)))
        }

        val srvb = new StagedRegionValueBuilder(r, rt)
        Code(Code(FastIndexedSeq(
          coords := _coords,
          radius := _radius,
          ncontigs := groupedT.loadLength(coords),
          totalLen := 0,
          forAllContigs(totalLen := totalLen + coordT.loadLength(coordsPerContig)),
          srvb.start(),
          srvb.addArray(PArray(PInt32()), addIdxWithCondition(startCond, _)),
          srvb.advance(),
          srvb.addArray(PArray(PInt32()), addIdxWithCondition(endCond, _)))),
          srvb.end())
    }

    registerCode("Locus", TString, tlocus("T"), null) {
      case (r, rt: PLocus, (strT, locusoff: Code[Long])) =>
        val slocus = asm4s.coerce[String](wrapArg(r, strT)(locusoff))
        val locus = Code
          .invokeScalaObject[String, ReferenceGenome, Locus](
          locusClass, "parse", slocus, rgCode(r.mb, rt.rg))
        emitLocus(r, locus, rt)
    }

    registerCode("Locus", TString, TInt32, tlocus("T"), null) {
      case (r, rt: PLocus, (contigT, contig: Code[Long]), (posT, pos: Code[Int])) =>
        Code.memoize(asm4s.coerce[Long](contig), "locus_contig", asm4s.coerce[Int](pos), "locus_pos") { (contig, pos) =>
          val srvb = new StagedRegionValueBuilder(r, rt)
          val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
          Code(
            rgCode(r.mb, rt.rg).invoke[String, Int, Unit]("checkLocus", scontig, pos),
            srvb.start(),
            srvb.addIRIntermediate(contigT)(contig),
            srvb.advance(),
            srvb.addInt(pos),
            srvb.offset)
        }
    }

    registerCode("LocusAlleles", TString, tvariant("T"), null) {
      case (r, rt: PStruct, (strT, variantoff: Code[Long])) =>
        val plocus = rt.types(0).asInstanceOf[PLocus]
        val svar = asm4s.coerce[String](wrapArg(r, strT)(variantoff))
        val variant = Code
          .invokeScalaObject[String, ReferenceGenome, (Locus, IndexedSeq[String])](
          VariantMethods.getClass, "parse", svar, rgCode(r.mb, plocus.rg))
        emitVariant(r, variant, rt)
    }

    registerCodeWithMissingness("LocusInterval", TString, TBoolean, tinterval("T"), null) {
      case (r: EmitRegion, rt: PInterval, (strT, ioff: EmitCode), (missingT, invalidMissing: EmitCode)) =>
        val plocus = rt.pointType.asInstanceOf[PLocus]
        val sinterval = asm4s.coerce[String](wrapArg(r, strT)(ioff.value[Long]))
        val intervalLocal = r.mb.newLocal[Interval](name="intervalObject")
        val interval = Code.invokeScalaObject[String, ReferenceGenome, Boolean, Interval](
          locusClass, "parseInterval", sinterval, rgCode(r.mb, plocus.rg), invalidMissing.value[Boolean])

        EmitCode(
          Code(ioff.setup, invalidMissing.setup),
          ioff.m || invalidMissing.m || Code(intervalLocal := interval, intervalLocal.isNull),
          PCode(rt, emitInterval(r, intervalLocal, rt))
        )
    }

    registerCodeWithMissingness("LocusInterval", TString, TInt32, TInt32, TBoolean, TBoolean, TBoolean, tinterval("T"), null) {
      case (r: EmitRegion, rt: PInterval,
      (locoffT, locoff: EmitCode),
      (pos1T, pos1: EmitCode),
      (pos2T, pos2: EmitCode),
      (include1T, include1: EmitCode),
      (include2T, include2: EmitCode),
      (invalidMissingT, invalidMissing: EmitCode)) =>
        val plocus = rt.pointType.asInstanceOf[PLocus]
        val sloc = asm4s.coerce[String](wrapArg(r, locoffT)(locoff.value[Long]))
        val intervalLocal = r.mb.newLocal[Interval]("intervalObject")
        val interval = Code.invokeScalaObject[String, Int, Int, Boolean, Boolean, ReferenceGenome, Boolean, Interval](
          locusClass, "makeInterval", sloc, pos1.value[Int], pos2.value[Int], include1.value[Boolean], include2.value[Boolean], rgCode(r.mb, plocus.rg), invalidMissing.value[Boolean])

        EmitCode(
          Code(locoff.setup, pos1.setup, pos2.setup, include1.setup, include2.setup, invalidMissing.setup),
          locoff.m || pos1.m || pos2.m || include1.m || include2.m || invalidMissing.m || Code(intervalLocal := interval, intervalLocal.isNull),
          PCode(rt, emitInterval(r, intervalLocal, rt))
        )
    }

    registerCode("globalPosToLocus", TInt64, tlocus("T"), null) {
      case (r, rt: PLocus, (globalPositionT, globalPosition: Code[Long])) =>
        val locus = rgCode(r.mb, rt.rg).invoke[Long, Locus]("globalPosToLocus", globalPosition)
        emitLocus(r, locus, rt)
    }

    registerCode("locusToGlobalPos", tlocus("T"), TInt64, null) {
      case (r, rt, (locusT: PLocus, locus: Code[Long])) =>
        val locusObject = Code.checkcast[Locus](wrapArg(r, locusT)(locus).asInstanceOf[Code[AnyRef]])
        unwrapReturn(r, rt)(rgCode(r.mb, locusT.rg).invoke[Locus, Long]("locusToGlobalPos", locusObject))
    }

    registerCodeWithMissingness("liftoverLocus", tlocus("T"), TFloat64, TStruct("result" -> tv("U", "locus"), "is_negative_strand" -> TBoolean), null) {
      case (r, rt: PStruct, (locT: PLocus, loc), (minMatchT, minMatch)) =>
        val srcRG = locT.rg
        val destRG = rt.types(0).asInstanceOf[PLocus].rg
        val locus = Code.checkcast[Locus](asm4s.coerce[AnyRef](wrapArg(r, locT)(loc.value[Long])))
        val tlocal = r.mb.newLocal[(Locus, Boolean)]()
        val lifted = rgCode(r.mb, srcRG).invoke[String, Locus, Double, (Locus, Boolean)]("liftoverLocus", destRG.name, locus, minMatch.value[Double])

        EmitCode(
          Code(loc.setup, minMatch.setup, tlocal := Code._null),
          loc.m || minMatch.m || Code(tlocal := lifted, tlocal.isNull),
          PCode(rt, emitLiftoverLocus(r, tlocal, rt))
        )
    }

    registerCodeWithMissingness("liftoverLocusInterval", tinterval("T"), TFloat64, TStruct("result" -> tinterval("U"), "is_negative_strand" -> TBoolean), null) {
      case (r, rt: PStruct, (iT: PInterval, i), (minMatchT, minMatch)) =>
        val srcRG = iT.pointType.asInstanceOf[PLocus].rg
        val destRG = rt.types(0).asInstanceOf[PInterval].pointType.asInstanceOf[PLocus].rg
        val interval = Code.checkcast[Interval](asm4s.coerce[AnyRef](wrapArg(r, iT)(i.value[Long])))
        val tlocal = r.mb.newLocal[(Interval, Boolean)]()
        val lifted = rgCode(r.mb, srcRG).invoke[String, Interval, Double, (Interval, Boolean)]("liftoverLocusInterval", destRG.name, interval, minMatch.value[Double])

        EmitCode(
          Code(i.setup, minMatch.setup, tlocal := Code._null),
          i.m || minMatch.m || Code(tlocal := lifted, tlocal.isNull),
          PCode(rt, emitLiftoverLocusInterval(r, tlocal, rt))
        )
    }
  }
}
