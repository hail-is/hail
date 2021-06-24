package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir.{EmitMethodBuilder, _}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SType}
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SBaseStructPointerCode, SCanonicalLocusPointer, SCanonicalLocusPointerCode, SIntervalPointer, SIntervalPointerCode, SStringPointer}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SFloat64Code, SInt32, SInt64}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant._

object LocusFunctions extends RegistryFunctions {

  def rgCode(mb: EmitMethodBuilder[_], rg: ReferenceGenome): Code[ReferenceGenome] =
    mb.getReferenceGenome(rg)

  def tlocus(name: String): Type = tv(name, "locus")

  def tvariant(name: String): TStruct = TStruct("locus" -> tlocus(name), "alleles" -> TArray(TString))

  def tinterval(name: String): TInterval = TInterval(tlocus(name))

  def emitLocus(cb: EmitCodeBuilder, r: Value[Region], locus: Code[Locus], rt: PCanonicalLocus): SCanonicalLocusPointerCode = {
    val loc = cb.newLocal[Locus]("emit_locus_memo", locus)
    rt.constructFromPositionAndString(cb, r,
      loc.invoke[String]("contig"),
      loc.invoke[Int]("position"))
  }

  def emitVariant(cb: EmitCodeBuilder, r: Value[Region], variantCode: Code[(Locus, IndexedSeq[String])], rt: PCanonicalStruct): SBaseStructPointerCode = {

    val variant = cb.newLocal[(Locus, IndexedSeq[String])]("emit_variant_variant", variantCode)

    val locus = EmitCode.fromI(cb.emb) { cb => IEmitCode.present(cb, emitLocus(cb, r, variant.invoke[Locus]("_1"), rt.types(0).asInstanceOf[PCanonicalLocus])) }
    val alleles = EmitCode.fromI(cb.emb) { cb =>
      val pAlleles = rt.types(1).asInstanceOf[PCanonicalArray]
      val all = cb.newLocal[IndexedSeq[String]]("locus_alleles_parsed_alleles", variant.invoke[IndexedSeq[String]]("_2"))
      val len = cb.newLocal[Int]("locus_alleles_n_alleles", all.invoke[Int]("length"))
      val ps = pAlleles.elementType.setRequired(false).asInstanceOf[PCanonicalString]
      val ss = SStringPointer(ps)
      val (push, finish) = pAlleles.constructFromFunctions(cb, r, len, deepCopy = false)
      val i = cb.newLocal[Int]("locus_alleles_i", 0)
      cb.whileLoop(i < len, {
        push(cb, IEmitCode.present(cb, ss.constructFromString(cb, r, all.invoke[Int, String]("apply", i))))
        cb.assign(i, i + 1)
      })
      IEmitCode.present(cb, finish(cb))
    }
    rt.constructFromFields(cb, r, FastIndexedSeq(locus, alleles), deepCopy = false)
  }

  def emitLocusInterval(cb: EmitCodeBuilder, r: Value[Region], intervalCode: Code[Interval], pt: PCanonicalInterval): SIntervalPointerCode = {
    val interval = cb.newLocal[Interval]("emit_interval_interval", intervalCode)
    val pointType = pt.pointType.asInstanceOf[PCanonicalLocus]
    pt.constructFromCodes(cb,
      r,
      EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, emitLocus(cb, r, interval.invoke[Locus]("start"), pointType))),
      EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, emitLocus(cb, r, interval.invoke[Locus]("end"), pointType))),
      EmitCode.present(cb.emb, primitive(interval.invoke[Boolean]("includesStart"))),
      EmitCode.present(cb.emb, primitive(interval.invoke[Boolean]("includesEnd")))
    )
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
      (_: Type, x: SType) => x.asInstanceOf[SLocus].contigType) {
      case (r, cb, rt, locus: SLocusCode) =>
        locus.contig(cb)
    }

    registerPCode1("position", tlocus("T"), TInt32, (_: Type, x: SType) => SInt32) {
      case (r, cb, rt, pc: SLocusCode) =>
        val locus = pc.memoize(cb, "locus_position_locus")
        primitive(locus.position(cb))
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
      (returnType: Type, _: SType, _: SType) => {
        val locusPT = PCanonicalLocus(returnType.asInstanceOf[TStruct].field("locus").typ.asInstanceOf[TLocus].rg, true)
        PCanonicalStruct("locus" -> locusPT, "alleles" -> PCanonicalArray(PCanonicalString(true), true)).sType
      }
    }) {
      case (r, cb, SBaseStructPointer(rt: PCanonicalStruct), locus: SLocusCode, alleles: SIndexableCode) =>
        val variantTuple = Code.invokeScalaObject2[Locus, IndexedSeq[String], (Locus, IndexedSeq[String])](
          VariantMethods.getClass, "minRep",
          locus.getLocusObj(cb),
          Code.checkcast[IndexedSeq[String]](scodeToJavaValue(cb, r.region, alleles)))

        emitVariant(cb, r.region, variantTuple, rt)
    }

    registerPCode2("locus_windows_per_contig", TArray(TArray(TFloat64)), TFloat64, TTuple(TArray(TInt32), TArray(TInt32)), {
      (_: Type, _: SType, _: SType) =>
        PCanonicalTuple(false, PCanonicalArray(PInt32(true), true), PCanonicalArray(PInt32(true), true)).sType
    }) {
      case (r: EmitRegion, cb: EmitCodeBuilder, SBaseStructPointer(rt: PCanonicalTuple), groupedCode: SIndexableCode, radiusCode: SFloat64Code) =>

        val grouped = groupedCode.memoize(cb, "locuswindows_grouped")
        val radius = cb.newLocal("locuswindows_radius", radiusCode.doubleCode(cb))

        val ncontigs = grouped.loadLength()
        val totalLen = cb.newLocal[Int]("locuswindows_totallen", 0)

        def forAllContigs(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, Value[Int], SIndexableValue) => Unit): Unit = {
          val iContig = cb.newLocal[Int]("locuswindows_icontig", 0)
          cb.whileLoop(iContig < ncontigs, {
            val coordPerContig = grouped.loadElement(cb, iContig).get(cb, "locus_windows group cannot be missing")
              .asIndexable
              .memoize(cb, "locuswindows_coord_per_contig")
            f(cb, iContig, coordPerContig)
            cb.assign(iContig, iContig + 1)
          })
        }

        val arrayType = PCanonicalArray(PInt32(true), true)
        assert(rt.types(0) == arrayType)
        assert(rt.types(1) == arrayType)

        def addIdxWithCondition(cb: EmitCodeBuilder)(cond: (EmitCodeBuilder, Value[Int], Value[Int], SIndexableValue) => Code[Boolean]): IEmitCode = {

          val (pushElement, finish) = arrayType.constructFromFunctions(cb, r.region, totalLen, deepCopy = false)
          val offset = cb.newLocal[Int]("locuswindows_offset", 0)

          val lastCoord = cb.newLocal[Double]("locuswindows_coord")

          forAllContigs(cb) { case (cb, contigIdx, coords) =>
            val i = cb.newLocal[Int]("locuswindows_i", 0)
            val idx = cb.newLocal[Int]("locuswindows_idx", 0)
            val len = coords.loadLength()
            cb.ifx(len.ceq(0),
              cb.assign(lastCoord, 0.0),
              cb.assign(lastCoord, coords.loadElement(cb, 0).get(cb, "locus_windows: missing value for 'coord_expr'").asDouble.doubleCode(cb)))
            cb.whileLoop(i < len, {

              coords.loadElement(cb, i).consume(cb,
                cb += Code._fatal[Unit](
                  const("locus_windows: missing value for 'coord_expr' at row ")
                    .concat((offset + i).toS)),
                { sc =>
                  val currentCoord = cb.newLocal[Double]("locuswindows_coord_i", sc.asDouble.doubleCode(cb))
                  cb.ifx(lastCoord > currentCoord,
                    cb += Code._fatal[Unit]("locus_windows: 'coord_expr' must be in ascending order within each contig."),
                    cb.assign(lastCoord, currentCoord)
                  )
                })

              val Lstart = CodeLabel()
              val Lbreak = CodeLabel()

              cb.define(Lstart)
              cb.ifx(idx >= len,
                cb.goto(Lbreak)
              )
              cb.ifx(cond(cb, i, idx, coords),
                {
                  cb.assign(idx, idx + 1)
                  cb.goto(Lstart)
                },
                cb.goto(Lbreak)
              )
              cb.define(Lbreak)

              pushElement(cb, IEmitCode.present(cb, primitive(offset + idx)))

              cb.assign(i, i + 1)
            })
            cb.assign(offset, offset + len)
          }
          IEmitCode.present(cb, finish(cb))
        }


        forAllContigs(cb) { case (cb, _, coordsPerContig) =>
          cb.assign(totalLen, totalLen + coordsPerContig.loadLength())
        }

        rt.constructFromFields(cb, r.region,
          FastIndexedSeq[EmitCode](
            EmitCode.fromI(cb.emb)(cb => addIdxWithCondition(cb) { case (cb, i, idx, coords) => coords.loadElement(cb, i)
              .get(cb, "locus_windows: missing value for 'coord_expr'")
              .asDouble.doubleCode(cb) > (coords.loadElement(cb, idx)
              .get(cb, "locus_windows: missing value for 'coord_expr'").asDouble.doubleCode(cb) + radius)
            }),
            EmitCode.fromI(cb.emb)(cb => addIdxWithCondition(cb) { case (cb, i, idx, coords) => coords.loadElement(cb, i)
              .get(cb, "locus_windows: missing value for 'coord_expr'")
              .asDouble.doubleCode(cb) >= (coords.loadElement(cb, idx)
              .get(cb, "locus_windows: missing value for 'coord_expr'").asDouble.doubleCode(cb) - radius)
            })
          ), deepCopy = false)
    }

    registerPCode1("Locus", TString, tlocus("T"), {
      (returnType: Type, _: SType) => PCanonicalLocus(returnType.asInstanceOf[TLocus].rg).sType
    }) {
      case (r, cb, SCanonicalLocusPointer(rt: PCanonicalLocus), str: SStringCode) =>
        val slocus = str.loadString()
        emitLocus(cb,
          r.region,
          Code.invokeScalaObject2[String, ReferenceGenome, Locus](locusClass, "parse", slocus, rgCode(r.mb, rt.rg)),
          rt)
    }

    registerPCode2("Locus", TString, TInt32, tlocus("T"), {
      (returnType: Type, _: SType, _: SType) => PCanonicalLocus(returnType.asInstanceOf[TLocus].rg).sType
    }) {
      case (r, cb, SCanonicalLocusPointer(rt: PCanonicalLocus), contig, pos) =>
        val contigMemo = contig.memoize(cb, "locus_contig")
        val posMemo = pos.memoize(cb, "locus_pos")
        cb += rgCode(r.mb, rt.rg).invoke[String, Int, Unit]("checkLocus", contigMemo.asString.loadString(), posMemo.asInt.intCode(cb))
        rt.constructFromPositionAndString(cb, r.region, contigMemo.asString.loadString(), posMemo.asInt.intCode(cb))
    }

    registerPCode1("LocusAlleles", TString, tvariant("T"), {
      (returnType: Type, _: SType) => {
        val lTyp = returnType.asInstanceOf[TStruct].field("locus").typ.asInstanceOf[TLocus]
        PCanonicalStruct("locus" -> PCanonicalLocus(lTyp.rg, true), "alleles" -> PCanonicalArray(PCanonicalString(true), true)).sType
      }
    }) {
      case (r, cb, SBaseStructPointer(rt: PCanonicalStruct), variantStr) =>

        val svar = variantStr.asString.loadString()
        val plocus = rt.types(0).asInstanceOf[PCanonicalLocus]
        val variant = Code
          .invokeScalaObject2[String, ReferenceGenome, (Locus, IndexedSeq[String])](
            VariantMethods.getClass, "parse", svar, rgCode(r.mb, plocus.rg))
        emitVariant(cb, r.region, variant, rt)
    }

    registerIEmitCode2("LocusInterval", TString, TBoolean, tinterval("T"), {
      (returnType: Type, _: EmitType, _: EmitType) => {
        val lPTyp = returnType.asInstanceOf[TInterval].pointType.asInstanceOf[TLocus]
        EmitType(PCanonicalInterval(PCanonicalLocus(lPTyp.asInstanceOf[TLocus].rg)).sType, false)
      }
    }) { case (cb: EmitCodeBuilder, r: Value[Region], SIntervalPointer(rt: PCanonicalInterval), locusStrEC: EmitCode, invalidMissingEC: EmitCode) =>
      val plocus = rt.pointType.asInstanceOf[PLocus]


      locusStrEC.toI(cb).flatMap(cb) { locusStr =>
        invalidMissingEC.toI(cb).flatMap(cb) { invalidMissing =>

          val Lmissing = CodeLabel()
          val Ldefined = CodeLabel()

          val interval = cb.newLocal[Interval]("locus_interval_interval",
            Code.invokeScalaObject3[String, ReferenceGenome, Boolean, Interval](
              locusClass, "parseInterval",
              locusStr.asString.loadString(),
              rgCode(cb.emb, plocus.rg),
              invalidMissing.asBoolean.boolCode(cb)))

          cb.ifx(interval.isNull, cb.goto(Lmissing))

          val intervalCode = emitLocusInterval(cb, r, interval, rt)
          cb.goto(Ldefined)
          IEmitCode(Lmissing, Ldefined, intervalCode, false)
        }
      }
    }

    registerIEmitCode6("LocusInterval", TString, TInt32, TInt32, TBoolean, TBoolean, TBoolean, tinterval("T"), {
      (returnType: Type, _: EmitType, _: EmitType, _: EmitType, _: EmitType, _: EmitType, _: EmitType) => {
        val lPTyp = returnType.asInstanceOf[TInterval].pointType.asInstanceOf[TLocus]
        EmitType(PCanonicalInterval(PCanonicalLocus(lPTyp.rg)).sType, false)
      }
    }) {
      case (cb: EmitCodeBuilder, r: Value[Region],
      SIntervalPointer(rt: PCanonicalInterval),
      locusString: EmitCode,
      pos1: EmitCode,
      pos2: EmitCode,
      include1: EmitCode,
      include2: EmitCode,
      invalidMissing: EmitCode) =>
        val plocus = rt.pointType.asInstanceOf[PLocus]

        locusString.toI(cb).flatMap(cb) { locusString =>
          pos1.toI(cb).flatMap(cb) { pos1 =>
            pos2.toI(cb).flatMap(cb) { pos2 =>
              include1.toI(cb).flatMap(cb) { include1 =>
                include2.toI(cb).flatMap(cb) { include2 =>
                  invalidMissing.toI(cb).flatMap(cb) { invalidMissing =>

                    val Lmissing = CodeLabel()
                    val Ldefined = CodeLabel()

                    val interval = cb.newLocal[Interval]("locus_interval_interval",
                      Code.invokeScalaObject7[String, Int, Int, Boolean, Boolean, ReferenceGenome, Boolean, Interval](
                        locusClass, "makeInterval",
                        locusString.asString.loadString(),
                        pos1.asInt.intCode(cb),
                        pos2.asInt.intCode(cb),
                        include1.asBoolean.boolCode(cb),
                        include2.asBoolean.boolCode(cb),
                        rgCode(cb.emb, plocus.rg),
                        invalidMissing.asBoolean.boolCode(cb)))

                    cb.ifx(interval.isNull, cb.goto(Lmissing))

                    val intervalCode = emitLocusInterval(cb, r, interval, rt)
                    cb.goto(Ldefined)
                    IEmitCode(Lmissing, Ldefined, intervalCode, false)
                  }
                }
              }
            }
          }
        }
    }

    registerPCode1("globalPosToLocus", TInt64, tlocus("T"), {
      (returnType: Type, _: SType) =>
        PCanonicalLocus(returnType.asInstanceOf[TLocus].rg).sType
    }) {
      case (r, cb, SCanonicalLocusPointer(rt: PCanonicalLocus), globalPos) =>
        val locus = cb.newLocal[Locus]("global_pos_locus",
          rgCode(r.mb, rt.rg).invoke[Long, Locus]("globalPosToLocus", globalPos.asLong.longCode(cb)))
        rt.constructFromPositionAndString(cb, r.region, locus.invoke[String]("contig"), locus.invoke[Int]("position"))
    }

    registerPCode1("locusToGlobalPos", tlocus("T"), TInt64, (_: Type, _: SType) => SInt64) {
      case (r, cb, rt, locus: SLocusCode) =>
        val locusObject = locus.memoize(cb, "locus_to_global_pos")
          .getLocusObj(cb)
        val globalPos = rgCode(r.mb, locus.st.rg).invoke[Locus, Long]("locusToGlobalPos", locusObject)
        primitive(globalPos)
    }

    registerIEmitCode2("liftoverLocus", tlocus("T"), TFloat64, TStruct("result" -> tv("U", "locus"), "is_negative_strand" -> TBoolean), {
      (returnType: Type, _: EmitType, _: EmitType) => {
        val lTyp = returnType.asInstanceOf[TStruct].field("result").typ.asInstanceOf[TLocus]
        EmitType(PCanonicalStruct("result" -> PCanonicalLocus(lTyp.rg, true), "is_negative_strand" -> PBoolean(true)).sType, false)
      }
    }) {
      case (cb, r, SBaseStructPointer(rt: PCanonicalStruct), loc, minMatch) =>
        loc.toI(cb).flatMap(cb) { loc =>
          minMatch.toI(cb).flatMap(cb) { minMatch =>

            val Lmissing = CodeLabel()
            val Ldefined = CodeLabel()

            val locT = loc.asLocus.st
            val srcRG = locT.rg

            val destRG = rt.types(0).asInstanceOf[PLocus].rg
            val locusObj = loc.asLocus.memoize(cb, "liftover_locus").getLocusObj(cb)
            val lifted = cb.newLocal[(Locus, Boolean)]("lifterover_locus_ lifted",
              rgCode(cb.emb, srcRG).invoke[String, Locus, Double, (Locus, Boolean)]("liftoverLocus",
                destRG.name, locusObj, minMatch.asDouble.doubleCode(cb)))

            cb.ifx(lifted.isNull, cb.goto(Lmissing))

            val locType = rt.types(0).asInstanceOf[PCanonicalLocus]
            val locusCode = EmitCode.present(cb.emb, emitLocus(cb, r, Code.checkcast[Locus](lifted.getField[java.lang.Object]("_1")), locType))

            val negativeStrandCode = EmitCode.present(cb.emb,
              primitive(Code.checkcast[java.lang.Boolean](lifted.getField[java.lang.Object]("_2"))
                .invoke[Boolean]("booleanValue")))

            val structCode = rt.constructFromFields(cb, r, FastIndexedSeq(locusCode, negativeStrandCode), deepCopy = false)

            cb.goto(Ldefined)
            IEmitCode(Lmissing, Ldefined, structCode, false)
          }
        }
    }

    registerIEmitCode2("liftoverLocusInterval", tinterval("T"), TFloat64, TStruct("result" -> tinterval("U"), "is_negative_strand" -> TBoolean), {
      (returnType: Type, _: EmitType, _: EmitType) => {
        val lTyp = returnType.asInstanceOf[TStruct].field("result").typ.asInstanceOf[TInterval].pointType.asInstanceOf[TLocus]
        EmitType(PCanonicalStruct("result" -> PCanonicalInterval(PCanonicalLocus(lTyp.rg, true), true), "is_negative_strand" -> PBoolean(true)).sType, false)
      }
    }) {
      case (cb, r, SBaseStructPointer(rt: PCanonicalStruct), interval, minMatch) =>
        interval.toI(cb).flatMap(cb) { interval =>
          minMatch.toI(cb).flatMap(cb) { minMatch =>

            val Lmissing = CodeLabel()
            val Ldefined = CodeLabel()


            val iT = interval.st.asInstanceOf[SInterval]
            val srcRG = iT.pointType.asInstanceOf[SLocus].rg
            val destRG = rt.types(0).asInstanceOf[PInterval].pointType.asInstanceOf[PLocus].rg
            val er = EmitRegion(cb.emb, r)
            val intervalObj = Code.checkcast[Interval](scodeToJavaValue(cb, r, interval))
            val lifted = cb.newLocal[(Interval, Boolean)]("liftover_locus_interval_lifted",
              rgCode(cb.emb, srcRG).invoke[String, Interval, Double, (Interval, Boolean)]("liftoverLocusInterval",
                destRG.name, intervalObj, minMatch.asDouble.doubleCode(cb)))


            cb.ifx(lifted.isNull, cb.goto(Lmissing))

            val iType = rt.types(0).asInstanceOf[PCanonicalInterval]
            val intervalCode = EmitCode.present(cb.emb, emitLocusInterval(cb, r, Code.checkcast[Interval](lifted.getField[java.lang.Object]("_1")), iType))

            val negativeStrandCode = EmitCode.present(cb.emb,
              primitive(Code.checkcast[java.lang.Boolean](lifted.getField[java.lang.Object]("_2"))
                .invoke[Boolean]("booleanValue")))

            val structCode = rt.constructFromFields(cb, r, FastIndexedSeq(intervalCode, negativeStrandCode), deepCopy = false)


            cb.goto(Ldefined)
            IEmitCode(Lmissing, Ldefined, structCode, false)
          }
        }
    }
  }
}
