package is.hail.expr.ir.functions

import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.physical.{PBoolean, PCanonicalString, PInt32, PLocus, PType}
import is.hail.expr.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

class ReferenceGenomeFunctions(rg: ReferenceGenome) extends RegistryFunctions {
  def rgCode(mb: EmitMethodBuilder[_], rg: ReferenceGenome): Code[ReferenceGenome] = mb.getReferenceGenome(rg)
  var registered: Set[(String, Type, Seq[Type], Seq[Type])] = Set()

  def registerRGCode(mname: String, typeArgs: Array[Type], args: Array[Type], rt: Type, pt: (Type, Seq[PType]) => PType)(
                      impl: (EmitRegion, PType, Array[Type], Array[(PType, Code[_])]) => Code[_]
                    ): Unit = {
    registered += ((mname, rt, typeArgs, args))
    registerCode(mname, typeArgs, args, rt, pt)(impl)
  }

  def registerRGCode[A1](mname: String, typeArg1: Type, arg1: Type, rt: Type, pt: (Type, PType) => PType)(
    impl: (EmitRegion, PType, Type, (PType, Code[A1])) => Code[_]): Unit =
    registerRGCode(mname, Array(typeArg1), Array[Type](arg1), rt, unwrappedApply(pt)) {
      case (r, rt, Array(t1: TLocus), Array(a1: (PType, Code[A1]) @unchecked)) => impl(r, rt, t1, a1)
    }

  def registerRGCode[A1, A2](
                              mname: String, typeArg1: Type, arg1: Type, arg2: Type, rt: Type, pt: (Type, PType, PType) => PType)(
                              impl: (EmitRegion, PType, Type, (PType, Code[A1]), (PType, Code[A2])) => Code[_]
                            ): Unit =
    registerRGCode(mname, Array(typeArg1), Array[Type](arg1, arg2), rt, unwrappedApply(pt)) {
      case (r, rt, Array(t1: TLocus), Array(a1: (PType, Code[A1]) @unchecked, a2: (PType, Code[A2]) @unchecked)) => impl(r, rt, t1, a1, a2)
    }

  def registerRGCode[A1, A2, A3, A4](
                                      mname: String, typeArg1: Type, arg1: Type, arg2: Type, arg3: Type, arg4: Type, rt: Type, pt: (Type, PType, PType, PType, PType) => PType)(
                                      impl: (EmitRegion, PType, Type, (PType, Code[A1]), (PType, Code[A2]), (PType, Code[A3]), (PType, Code[A4])) => Code[_]
                                    ): Unit =
    registerRGCode(mname, Array(typeArg1), Array[Type](arg1, arg2, arg3, arg4), rt, unwrappedApply(pt)) {
      case (r, rt, Array(t1: TLocus), Array(
      a1: (PType, Code[A1]) @unchecked,
      a2: (PType, Code[A2]) @unchecked,
      a3: (PType, Code[A3]) @unchecked,
      a4: (PType, Code[A4]) @unchecked)) => impl(r, rt, t1, a1, a2, a3, a4)
    }

  def registerAll() {
    registerRGCode("isValidContig", TLocus(rg), TString, TBoolean, (_: Type, _: PType) => PBoolean()) {
      case (r, rt, typeArg: TLocus, (contigT, contig: Code[Long])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        rgCode(r.mb, typeArg.rg).invoke[String, Boolean]("isValidContig", scontig)
    }

    registerRGCode("isValidLocus", TLocus(rg), TString, TInt32, TBoolean, (_: Type, _: PType, _: PType) => PBoolean()) {
      case (r, rt, typeArg: TLocus, (contigT, contig: Code[Long]), (posT, pos: Code[Int])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        rgCode(r.mb, typeArg.rg).invoke[String, Int, Boolean]("isValidLocus", scontig, pos)
    }

    registerRGCode("getReferenceSequenceFromValidLocus", TLocus(rg), TString, TInt32, TInt32, TInt32, TString, (_: Type, _: PType, _: PType, _: PType, _: PType) => PCanonicalString()) {
      case (r, rt, typeArg: TLocus, (contigT, contig: Code[Long]), (posT, pos: Code[Int]), (beforeT, before: Code[Int]), (afterT, after: Code[Int])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        unwrapReturn(r, rt)(rgCode(r.mb, typeArg.rg).invoke[String, Int, Int, Int, String]("getSequence", scontig, pos, before, after))
    }

    registerIR("getReferenceSequence", TString, TInt32, TInt32, TInt32, TString) {
      (contig, pos, before, after) =>
        //lookupConversion(name: String, rt: Type, typeArgs: Seq[Type], args: Seq[Type]): Option[Seq[IR] => IR]
        val getRef = IRFunctionRegistry.lookupConversion(
          name = "getReferenceSequenceFromValidLocus",
          rt = TString,
          typeArgs = Array(TLocus(rg)),
          args = Seq(TString, TInt32, TInt32, TInt32)).get
        val isValid = IRFunctionRegistry.lookupConversion(
          "isValidLocus",
          TBoolean,
          Seq(TString, TInt32)).get
        If(isValid(Array(contig, pos)), getRef(Array(contig, pos, before, after)), NA(TString))
    }

    registerRGCode("contigLength", TLocus(rg), TString, TInt32, (_: Type, _: PType) => PInt32()) {
      case (r, rt, typeArg: TLocus, (contigT, contig: Code[Long])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        rgCode(r.mb, typeArg.rg).invoke[String, Int]("contigLength", scontig)
    }
  }
}
