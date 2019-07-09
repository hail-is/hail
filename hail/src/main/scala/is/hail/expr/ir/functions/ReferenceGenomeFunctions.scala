package is.hail.expr.ir.functions

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types
import is.hail.expr.types.physical.{PBaseStruct, PString, PStruct, PType}
import is.hail.expr.types.virtual._
import is.hail.utils._
import is.hail.variant.{Locus, RGBase, ReferenceGenome, VariantMethods}

class ReferenceGenomeFunctions(rg: ReferenceGenome) extends RegistryFunctions {

  def rgCode(mb: EmitMethodBuilder): Code[ReferenceGenome] = mb.getReferenceGenome(rg)

  def emitLocus(r: EmitRegion, locus: Code[Locus]): Code[Long] = {
    val srvb = new StagedRegionValueBuilder(r, tlocus.physicalType)
    Code(emitLocus(srvb, locus), srvb.offset)
  }

  def emitLocus(srvb: StagedRegionValueBuilder, locus: Code[Locus]): Code[Unit] = {
    val llocal = srvb.mb.newLocal[Locus]
    Code(
      llocal := locus,
      srvb.start(),
      srvb.addString(locus.invoke[String]("contig")),
      srvb.advance(),
      srvb.addInt(locus.invoke[Int]("position")),
      srvb.advance()
    )
  }

  def emitVariant(r: EmitRegion, variant: Code[(Locus, IndexedSeq[String])]): Code[Long] = {
    val vlocal = r.mb.newLocal[(Locus, IndexedSeq[String])]
    val alocal = r.mb.newLocal[IndexedSeq[String]]
    val len = r.mb.newLocal[Int]
    val srvb = new StagedRegionValueBuilder(r, tvariant.physicalType)
    val addLocus = { srvb: StagedRegionValueBuilder =>
      emitLocus(srvb, Code.checkcast[Locus](vlocal.get[java.lang.Object]("_1")))
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
      alocal := Code.checkcast[IndexedSeq[String]](vlocal.get[java.lang.Object]("_2")),
      len := alocal.invoke[Int]("size"),
      srvb.start(),
      srvb.addBaseStruct(types.coerce[PStruct](tlocus.fundamentalType.physicalType), addLocus),
      srvb.advance(),
      srvb.addArray(talleles.physicalType, addAlleles),
      srvb.advance(),
      srvb.offset)
  }

  def emitInterval(r: EmitRegion, interval: Code[Interval]): Code[Long] = {
    val srvb = new StagedRegionValueBuilder(r, tinterval.physicalType)
    Code(emitInterval(srvb, interval), srvb.offset)
  }

  def emitInterval(srvb: StagedRegionValueBuilder, interval: Code[Interval]): Code[Unit] = {
    val ilocal = srvb.mb.newLocal[Interval]
    val addLocus = { (srvb: StagedRegionValueBuilder, point: String) =>
      emitLocus(srvb, Code.checkcast[Locus](ilocal.invoke[java.lang.Object](point)))
    }

    asm4s.coerce[Unit](Code(
      ilocal := interval,
      srvb.start(),
      srvb.addBaseStruct(types.coerce[PBaseStruct](tlocus.fundamentalType.physicalType), addLocus(_, "start")),
      srvb.advance(),
      srvb.addBaseStruct(types.coerce[PBaseStruct](tlocus.fundamentalType.physicalType), addLocus(_, "end")),
      srvb.advance(),
      srvb.addBoolean(ilocal.invoke[Boolean]("includesStart")),
      srvb.advance(),
      srvb.addBoolean(ilocal.invoke[Boolean]("includesEnd")),
      srvb.advance()))
  }

  def emitLiftoverLocus(r: EmitRegion, result: Code[(Locus, Boolean)]): Code[Long] = {
    val rlocal = r.mb.newLocal[(Locus, Boolean)]
    val blocal = r.mb.newLocal[Boolean]
    val srvb = new StagedRegionValueBuilder(r, TTuple(tlocus, TBoolean()).physicalType)
    val addLocus = { srvb: StagedRegionValueBuilder =>
      emitLocus(srvb, Code.checkcast[Locus](rlocal.get[java.lang.Object]("_1")))
    }

    Code(
      rlocal := result,
      blocal := Code.checkcast[java.lang.Boolean](rlocal.get[java.lang.Object]("_2")).invoke[Boolean]("booleanValue"),
      srvb.start(),
      srvb.addBaseStruct(types.coerce[PStruct](tlocus.fundamentalType.physicalType), addLocus),
      srvb.advance(),
      srvb.addBoolean(blocal),
      srvb.advance(),
      srvb.offset)
  }

  def emitLiftoverLocusInterval(r: EmitRegion, result: Code[(Interval, Boolean)]): Code[Long] = {
    val rlocal = r.mb.newLocal[(Interval, Boolean)]
    val ilocal = r.mb.newLocal[Interval]
    val blocal = r.mb.newLocal[Boolean]
    val srvb = new StagedRegionValueBuilder(r, TTuple(tinterval, TBoolean()).physicalType)
    val addInterval = { srvb: StagedRegionValueBuilder =>
      emitInterval(srvb, ilocal)
    }

    Code(
      rlocal := result,
      ilocal := Code.checkcast[Interval](rlocal.get[java.lang.Object]("_1")),
      blocal := Code.checkcast[java.lang.Boolean](rlocal.get[java.lang.Object]("_2")).invoke[Boolean]("booleanValue"),
      srvb.start(),
      srvb.addBaseStruct(types.coerce[PStruct](tinterval.fundamentalType.physicalType), addInterval),
      srvb.advance(),
      srvb.addBoolean(blocal),
      srvb.advance(),
      srvb.offset)
  }

  var registered: Set[String] = Set[String]()

  val tlocus = TLocus(rg)
  val talleles = TArray(TString())
  val tvariant = TStruct("locus" -> tlocus, "alleles" -> talleles)
  val tinterval = TInterval(tlocus)

  def removeRegisteredFunctions(): Unit =
    registered.foreach(IRFunctionRegistry.removeIRFunction)

  def registerRGCode(
    mname: String, args: Array[Type], rt: Type)(
    impl: (EmitRegion, Array[(PType, Code[_])]) => Code[_]
  ): Unit = {
    val newName = rg.wrapFunctionName(mname)
    registered += newName
    registerCode(newName, args, rt)(impl)
  }

  def registerRGCodeWithMissingness(
    mname: String, args: Array[Type], rt: Type)(
    impl: (EmitRegion, Array[(PType, EmitTriplet)]) => EmitTriplet
  ): Unit = {
    val newName = rg.wrapFunctionName(mname)
    registered += newName
    registerCodeWithMissingness(newName, args, rt)(impl)
  }

  def registerRGCode[A1](
    mname: String, arg1: Type, rt: Type)(
    impl: (EmitRegion, (PType, Code[A1])) => Code[_]
  ): Unit =
    registerRGCode(mname, Array[Type](arg1), rt) {
      case (r, Array(a1: (PType, Code[A1]) @unchecked)) => impl(r, a1)
    }

  def registerRGCode[A1, A2](
    mname: String, arg1: Type, arg2: Type, rt: Type)(
    impl: (EmitRegion, (PType, Code[A1]), (PType, Code[A2])) => Code[_]
  ): Unit =
    registerRGCode(mname, Array[Type](arg1, arg2), rt) {
      case (r, Array(a1: (PType, Code[A1]) @unchecked, a2: (PType, Code[A2]) @unchecked)) => impl(r, a1, a2)
    }

  def registerRGCodeWithMissingness(
    mname: String, arg1: Type, arg2: Type, rt: Type)(
    impl: (EmitRegion, (PType, EmitTriplet), (PType, EmitTriplet)) => EmitTriplet
  ): Unit =
    registerRGCodeWithMissingness(mname, Array[Type](arg1, arg2), rt) {
      case (mb, Array(a1, a2)) => impl(mb, a1, a2)
    }

  def registerRGCode[A1, A2, A3, A4](
    mname: String, arg1: Type, arg2: Type, arg3: Type, arg4: Type, rt: Type)(
    impl: (EmitRegion, (PType, Code[A1]), (PType, Code[A2]), (PType, Code[A3]), (PType, Code[A4])) => Code[_]
  ): Unit =
    registerRGCode(mname, Array[Type](arg1, arg2, arg3, arg4), rt) {
      case (r, Array(
      a1: (PType, Code[A1]) @unchecked,
      a2: (PType, Code[A2]) @unchecked,
      a3: (PType, Code[A3]) @unchecked,
      a4: (PType, Code[A4]) @unchecked)) => impl(r, a1, a2, a3, a4)
    }

  def registerRGCode[A1, A2, A3, A4, A5](
    mname: String, arg1: Type, arg2: Type, arg3: Type, arg4: Type, arg5: Type, rt: Type)(
    impl: (EmitRegion, (PType, Code[A1]), (PType, Code[A2]), (PType, Code[A3]), (PType, Code[A4]), (PType, Code[A5])) => Code[_]
  ): Unit =
    registerRGCode(mname, Array[Type](arg1, arg2, arg3, arg4, arg5), rt) {
      case (r, Array(
      a1: (PType, Code[A1]) @unchecked,
      a2: (PType, Code[A2]) @unchecked,
      a3: (PType, Code[A3]) @unchecked,
      a4: (PType, Code[A4]) @unchecked,
      a5: (PType, Code[A5]) @unchecked)) => impl(r, a1, a2, a3, a4, a5)
    }

  def registerRGCode[A1, A2, A3, A4, A5, A6](
    mname: String, arg1: Type, arg2: Type, arg3: Type, arg4: Type, arg5: Type, arg6: Type, rt: Type)(
    impl: (EmitRegion, (PType, Code[A1]), (PType, Code[A2]), (PType, Code[A3]), (PType, Code[A4]), (PType, Code[A5]), (PType, Code[A6])) => Code[_]
  ): Unit =
    registerRGCode(mname, Array[Type](arg1, arg2, arg3, arg4, arg5, arg6), rt) {
      case (r, Array(
      a1: (PType, Code[A1]) @unchecked,
      a2: (PType, Code[A2]) @unchecked,
      a3: (PType, Code[A3]) @unchecked,
      a4: (PType, Code[A4]) @unchecked,
      a5: (PType, Code[A5]) @unchecked,
      a6: (PType, Code[A6]) @unchecked)) => impl(r, a1, a2, a3, a4, a5, a6)
    }

  def registerRGCodeWithMissingness(
    mname: String, arg1: Type, arg2: Type, arg3: Type, arg4: Type, arg5: Type, arg6: Type, rt: Type)(
    impl: (EmitRegion, (PType, EmitTriplet), (PType, EmitTriplet),
      (PType, EmitTriplet), (PType, EmitTriplet), (PType, EmitTriplet), (PType, EmitTriplet)) => EmitTriplet
  ): Unit =
    registerRGCodeWithMissingness(mname, Array[Type](arg1, arg2, arg3, arg4, arg5, arg6), rt) {
      case (mb, Array(a1, a2, a3, a4, a5, a6)) => impl(mb, a1, a2, a3, a4, a5, a6)
    }

  def registerAll() {

    val locusClass = Locus.getClass

    registerRGCode("Locus", TString(), TLocus(rg)) {
      case (r, (strT, locusoff: Code[Long])) =>
        val slocus = asm4s.coerce[String](wrapArg(r, strT)(locusoff))
        val locus = Code
          .invokeScalaObject[String, RGBase, Locus](
          locusClass, "parse", slocus, rgCode(r.mb))
        emitLocus(r, locus)
    }

    registerRGCode("Locus", TString(), TInt32(), TLocus(rg)) {
      case (r, (contigT, contig: Code[Long]), (posT, pos: Code[Int])) =>
        val srvb = new StagedRegionValueBuilder(r, PType.canonical(tlocus))
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        Code(
          rgCode(r.mb).invoke[String, Int, Unit]("checkLocus", scontig, pos),
          srvb.start(),
          srvb.addIRIntermediate(PString())(contig),
          srvb.advance(),
          srvb.addInt(pos),
          srvb.offset)
    }

    registerRGCode("LocusAlleles", TString(), tvariant) {
      case (r, (strT, variantoff: Code[Long])) =>
        val svar = asm4s.coerce[String](wrapArg(r, strT)(variantoff))
        val variant = Code
          .invokeScalaObject[String, RGBase, (Locus, IndexedSeq[String])](
          VariantMethods.getClass, "parse", svar, rgCode(r.mb))
        emitVariant(r, variant)
    }

    registerRGCodeWithMissingness("LocusInterval", TString(), TBoolean(), tinterval) {
      case (r: EmitRegion, (strT, ioff: EmitTriplet), (missingT, invalidMissing: EmitTriplet)) =>
        val sinterval = asm4s.coerce[String](wrapArg(r, strT)(ioff.value[Long]))
        val intervalLocal = r.mb.newLocal[Interval](name="intervalObject")
        val interval = Code.invokeScalaObject[String, RGBase, Boolean, Interval](
          locusClass, "parseInterval", sinterval, rgCode(r.mb), invalidMissing.value[Boolean])

        EmitTriplet(
          Code(ioff.setup, invalidMissing.setup),
          ioff.m || invalidMissing.m || Code(intervalLocal := interval, intervalLocal.load().isNull),
          emitInterval(r, interval)
        )
    }

    registerRGCodeWithMissingness("LocusInterval", TString(), TInt32(), TInt32(), TBoolean(), TBoolean(), TBoolean(), tinterval) {
      case (r: EmitRegion,
      (locoffT, locoff: EmitTriplet),
      (pos1T, pos1: EmitTriplet),
      (pos2T, pos2: EmitTriplet),
      (include1T, include1: EmitTriplet),
      (include2T, include2: EmitTriplet),
      (invalidMissingT, invalidMissing: EmitTriplet)) =>
        val sloc = asm4s.coerce[String](wrapArg(r, locoffT)(locoff.value[Long]))
        val intervalLocal = r.mb.newLocal[Interval]("intervalObject")
        val interval = Code.invokeScalaObject[String, Int, Int, Boolean, Boolean, RGBase, Boolean, Interval](
          locusClass, "makeInterval", sloc, pos1.value[Int], pos2.value[Int], include1.value[Boolean], include2.value[Boolean], rgCode(r.mb), invalidMissing.value[Boolean])

        EmitTriplet(
          Code(locoff.setup, pos1.setup, pos2.setup, include1.setup, include2.setup, invalidMissing.setup),
          locoff.m || pos1.m || pos2.m || include1.m || include2.m || invalidMissing.m || Code(intervalLocal := interval, intervalLocal.load().isNull),
          emitInterval(r, interval)
      )
    }

    registerRGCode("isValidContig", TString(), TBoolean()) {
      case (r, (contigT, contig: Code[Long])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        rgCode(r.mb).invoke[String, Boolean]("isValidContig", scontig)
    }

    registerRGCode("isValidLocus", TString(), TInt32(), TBoolean()) {
      case (r, (contigT, contig: Code[Long]), (posT, pos: Code[Int])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        rgCode(r.mb).invoke[String, Int, Boolean]("isValidLocus", scontig, pos)
    }

    registerRGCode("getReferenceSequenceFromValidLocus", TString(), TInt32(), TInt32(), TInt32(), TString()) {
      case (r, (contigT, contig: Code[Long]), (posT, pos: Code[Int]), (beforeT, before: Code[Int]), (afterT, after: Code[Int])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        unwrapReturn(r, TString())(rgCode(r.mb).invoke[String, Int, Int, Int, String]("getSequence", scontig, pos, before, after))
    }

    registerIR(rg.wrapFunctionName("getReferenceSequence"), TString(), TInt32(), TInt32(), TInt32(), TString()) {
      (contig, pos, before, after) =>
        val getRef = IRFunctionRegistry.lookupConversion(
          rg.wrapFunctionName("getReferenceSequenceFromValidLocus"),
          Seq(TString(), TInt32(), TInt32(), TInt32())).get
        val isValid = IRFunctionRegistry.lookupConversion(
          rg.wrapFunctionName("isValidLocus"),
          Seq(TString(), TInt32())).get
        If(isValid(Array(contig, pos)), getRef(Array(contig, pos, before, after)), NA(TString()))
    }

    registerRGCode("globalPosToLocus", TInt64(), TLocus(rg)) {
      case (r, (globalPositionT, globalPosition: Code[Long])) =>
        val locus = rgCode(r.mb).invoke[Long, Locus]("globalPosToLocus", globalPosition)
        emitLocus(r, locus)
    }

    registerRGCode("locusToGlobalPos", TLocus(rg), TInt64()) {
      case (r, (locusT, locus: Code[Long])) =>
        val locusObject = Code.checkcast[Locus](wrapArg(r, locusT)(locus).asInstanceOf[Code[AnyRef]])
        unwrapReturn(r, TInt64())(rgCode(r.mb).invoke[Locus, Long]("locusToGlobalPos", locusObject))
    }
  }
}

class LiftoverFunctions(rg: ReferenceGenome, destRG: ReferenceGenome) extends ReferenceGenomeFunctions(rg) {

  def registerLiftoverCode(
    mname: String, args: Array[Type], rt: Type)(
    impl: (EmitRegion, Array[(PType, EmitTriplet)]) => EmitTriplet
  ): Unit = {
    val newName = destRG.wrapFunctionName(rg.wrapFunctionName(mname))
    registered += newName
    registerCodeWithMissingness(newName, args, rt)(impl)
  }

  def registerLiftoverCode(
    mname: String, arg1: Type, arg2: Type, rt: Type)(
    impl: (EmitRegion, (PType, EmitTriplet), (PType, EmitTriplet)) => EmitTriplet
  ): Unit =
    registerLiftoverCode(mname, Array[Type](arg1, arg2), rt) {
      case (r, Array(a1, a2)) => impl(r, a1, a2)
    }

  override def registerAll() {

    registerLiftoverCode("liftoverLocus", tlocus, TFloat64(), TStruct("result" -> TLocus(destRG), "is_negative_strand" -> TBoolean())) {
      case (r, (locT, loc), (minMatchT, minMatch)) =>
        val locus = Code.checkcast[Locus](asm4s.coerce[AnyRef](wrapArg(r, locT)(loc.value[Long])))
        val tlocal = r.mb.newLocal[(Locus, Boolean)]
        val lifted = rgCode(r.mb).invoke[String, Locus, Double, (Locus, Boolean)]("liftoverLocus", destRG.name, locus, minMatch.value[Double])

        EmitTriplet(
          Code(loc.setup, minMatch.setup, tlocal := Code._null),
          loc.m || minMatch.m || Code(tlocal := lifted, tlocal.isNull),
          emitLiftoverLocus(r, tlocal)
        )
    }

    registerLiftoverCode("liftoverLocusInterval", tinterval, TFloat64(), TStruct("result" -> TInterval(TLocus(destRG)), "is_negative_strand" -> TBoolean())) {
      case (r, (iT, i), (minMatchT, minMatch)) =>
        val interval = Code.checkcast[Interval](asm4s.coerce[AnyRef](wrapArg(r, iT)(i.value[Long])))
        val tlocal = r.mb.newLocal[(Interval, Boolean)]
        val lifted = rgCode(r.mb).invoke[String, Interval, Double, (Interval, Boolean)]("liftoverLocusInterval", destRG.name, interval, minMatch.value[Double])

        EmitTriplet(
          Code(i.setup, minMatch.setup, tlocal := Code._null),
          i.m || minMatch.m || Code(tlocal := lifted, tlocal.isNull),
          emitLiftoverLocusInterval(r, tlocal)
        )
    }
  }
}
