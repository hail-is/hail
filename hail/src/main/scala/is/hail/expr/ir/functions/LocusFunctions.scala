package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
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
    registerIR1(methodName, tlocus("T"), TBoolean)((_, a) => f(a))

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

    registerPCode1("contig", tlocus("T"), TString,
      (_: Type, x: PType) => x.asInstanceOf[PLocus].contigType) {
      case (r, rt, locus: PLocusCode) =>
        locus.contig()
    }

    registerCode1("position", tlocus("T"), TInt32, (_: Type, x: PType) => x.asInstanceOf[PLocus].positionType) {
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

    registerPCode2("min_rep", tlocus("T"), TArray(TString), TStruct("locus" -> tv("T"), "alleles" -> TArray(TString)), {
      (returnType: Type, _: PType, _: PType) => {
        val locusPT = PCanonicalLocus(returnType.asInstanceOf[TStruct].field("locus").typ.asInstanceOf[TLocus].rg, true)
        PCanonicalStruct("locus" -> locusPT, "alleles" -> PCanonicalArray(PCanonicalString(true), true))
      }
    }) {
      case (r, rt: PStruct, locus: PLocusCode, alleles: PIndexableCode) =>
        val tuple = Code.invokeScalaObject2[Locus, IndexedSeq[String], (Locus, IndexedSeq[String])](
          VariantMethods.getClass, "minRep",
          locus.getLocusObj(),
          Code.checkcast[IndexedSeq[String]](wrapArg(r, alleles.pt)(alleles.code).asInstanceOf[Code[AnyRef]]))

        val code = Code.memoize(tuple, "min_rep_tuple") { tuple =>
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
        PCode(rt, code)
    }

    registerCode2("locus_windows_per_contig", TArray(TArray(TFloat64)), TFloat64, TTuple(TArray(TInt32), TArray(TInt32)), {
      (_: Type, _: PType, _: PType) =>
        PCanonicalTuple(false, PCanonicalArray(PInt32(true), true), PCanonicalArray(PInt32(true), true))
    }) {
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
          srvb.addArray(PCanonicalArray(PInt32()), addIdxWithCondition(startCond, _)),
          srvb.advance(),
          srvb.addArray(PCanonicalArray(PInt32()), addIdxWithCondition(endCond, _)))),
          srvb.end())
    }

    registerCode1("Locus", TString, tlocus("T"), {
      (returnType: Type, _: PType) => PCanonicalLocus(returnType.asInstanceOf[TLocus].rg)
    }) {
      case (r, rt: PLocus, (strT, locusoff: Code[Long])) =>
        val slocus = asm4s.coerce[String](wrapArg(r, strT)(locusoff))
        val locus = Code
          .invokeScalaObject2[String, ReferenceGenome, Locus](
          locusClass, "parse", slocus, rgCode(r.mb, rt.rg))
        emitLocus(r, locus, rt)
    }

    registerCode2("Locus", TString, TInt32, tlocus("T"), {
      (returnType: Type, _: PType, _: PType) => PCanonicalLocus(returnType.asInstanceOf[TLocus].rg)
    }) {
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

    registerCode1("LocusAlleles", TString, tvariant("T"), {
      (returnType: Type, _: PType) => {
        val lTyp = returnType.asInstanceOf[TStruct].field("locus").typ.asInstanceOf[TLocus]
        PCanonicalStruct("locus" -> PCanonicalLocus(lTyp.rg, true), "alleles" -> PCanonicalArray(PCanonicalString(true), true))
      }
    }) {
      case (r, rt: PStruct, (strT, variantoff: Code[Long])) =>
        val plocus = rt.types(0).asInstanceOf[PLocus]
        val svar = asm4s.coerce[String](wrapArg(r, strT)(variantoff))
        val variant = Code
          .invokeScalaObject2[String, ReferenceGenome, (Locus, IndexedSeq[String])](
          VariantMethods.getClass, "parse", svar, rgCode(r.mb, plocus.rg))
        emitVariant(r, variant, rt)
    }

    registerEmitCode2("LocusInterval", TString, TBoolean, tinterval("T"), {
      (returnType: Type, _: PType, _: PType) => {
        val lPTyp = returnType.asInstanceOf[TInterval].pointType.asInstanceOf[TLocus]
        PCanonicalInterval(PCanonicalLocus(lPTyp.asInstanceOf[TLocus].rg))
      }
    }) {
      case (r: EmitRegion, rt: PInterval, ioff: EmitCode, invalidMissing: EmitCode) =>
        val plocus = rt.pointType.asInstanceOf[PLocus]
        val sinterval = asm4s.coerce[String](wrapArg(r, ioff.pt)(ioff.value[Long]))
        val intervalLocal = r.mb.newLocal[Interval](name="intervalObject")
        val interval = Code.invokeScalaObject3[String, ReferenceGenome, Boolean, Interval](
          locusClass, "parseInterval", sinterval, rgCode(r.mb, plocus.rg), invalidMissing.value[Boolean])

        EmitCode(
          Code(ioff.setup, invalidMissing.setup),
          ioff.m || invalidMissing.m || Code(intervalLocal := interval, intervalLocal.isNull),
          PCode(rt, emitInterval(r, intervalLocal, rt))
        )
    }

    registerEmitCode6("LocusInterval", TString, TInt32, TInt32, TBoolean, TBoolean, TBoolean, tinterval("T"), {
      (returnType: Type, _: PType, _: PType, _: PType, _: PType, _: PType, _: PType) => {
        val lPTyp = returnType.asInstanceOf[TInterval].pointType.asInstanceOf[TLocus]
        PCanonicalInterval(PCanonicalLocus(lPTyp.rg))
      }
    }) {
      case (r: EmitRegion, rt: PInterval,
      locoff: EmitCode,
      pos1: EmitCode,
      pos2: EmitCode,
      include1: EmitCode,
      include2: EmitCode,
      invalidMissing: EmitCode) =>
        val plocus = rt.pointType.asInstanceOf[PLocus]
        val sloc = asm4s.coerce[String](wrapArg(r, locoff.pt)(locoff.value[Long]))
        val intervalLocal = r.mb.newLocal[Interval]("intervalObject")
        val interval = Code.invokeScalaObject7[String, Int, Int, Boolean, Boolean, ReferenceGenome, Boolean, Interval](
          locusClass, "makeInterval", sloc, pos1.value[Int], pos2.value[Int], include1.value[Boolean], include2.value[Boolean], rgCode(r.mb, plocus.rg), invalidMissing.value[Boolean])

        EmitCode(
          Code(locoff.setup, pos1.setup, pos2.setup, include1.setup, include2.setup, invalidMissing.setup),
          locoff.m || pos1.m || pos2.m || include1.m || include2.m || invalidMissing.m || Code(intervalLocal := interval, intervalLocal.isNull),
          PCode(rt, emitInterval(r, intervalLocal, rt))
        )
    }

    registerCode1("globalPosToLocus", TInt64, tlocus("T"), {
      (returnType: Type, _: PType) =>
        PCanonicalLocus(returnType.asInstanceOf[TLocus].rg)
    }) {
      case (r, rt: PLocus, (globalPositionT, globalPosition: Code[Long])) =>
        val locus = rgCode(r.mb, rt.rg).invoke[Long, Locus]("globalPosToLocus", globalPosition)
        emitLocus(r, locus, rt)
    }

    registerCode1("locusToGlobalPos", tlocus("T"), TInt64, (_: Type, _: PType) => PInt64()) {
      case (r, rt, (locusT: PLocus, locus: Code[Long])) =>
        val locusObject = Code.checkcast[Locus](wrapArg(r, locusT)(locus).asInstanceOf[Code[AnyRef]])
        unwrapReturn(r, rt)(rgCode(r.mb, locusT.rg).invoke[Locus, Long]("locusToGlobalPos", locusObject))
    }

    registerEmitCode2("liftoverLocus", tlocus("T"), TFloat64, TStruct("result" -> tv("U", "locus"), "is_negative_strand" -> TBoolean), {
      (returnType: Type, _: PType, _: PType) => {
        val lTyp = returnType.asInstanceOf[TStruct].field("result").typ.asInstanceOf[TLocus]
        PCanonicalStruct("result" -> PCanonicalLocus(lTyp.rg, true), "is_negative_strand" -> PBoolean(true))
      }
    }) {
      case (r, rt: PStruct, loc, minMatch) =>
        val locT = loc.pt.asInstanceOf[PLocus]
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

    registerEmitCode2("liftoverLocusInterval", tinterval("T"), TFloat64, TStruct("result" -> tinterval("U"), "is_negative_strand" -> TBoolean), {
      (returnType: Type, _: PType, _: PType) => {
        val lTyp = returnType.asInstanceOf[TStruct].field("result").typ.asInstanceOf[TInterval].pointType.asInstanceOf[TLocus]
        PCanonicalStruct("result" -> PCanonicalInterval(PCanonicalLocus(lTyp.rg, true), true), "is_negative_strand" -> PBoolean(true))
      }
    }) {
      case (r, rt: PStruct, i, minMatch) =>
        val iT = i.pt.asInstanceOf[PInterval]
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
