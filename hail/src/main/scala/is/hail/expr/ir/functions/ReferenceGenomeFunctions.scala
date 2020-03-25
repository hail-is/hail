package is.hail.expr.ir.functions

import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

class ReferenceGenomeFunctions(rg: ReferenceGenome) extends RegistryFunctions {

  def rgCode(mb: EmitMethodBuilder[_]): Code[ReferenceGenome] = mb.getReferenceGenome(rg)

  var registered: Set[String] = Set[String]()

  def removeRegisteredFunctions(): Unit =
    registered.foreach(IRFunctionRegistry.removeIRFunction)

  def registerRGCode(
    mname: String, args: Array[Type], rt: Type, pt: Seq[PType] => PType)(
    impl: (EmitRegion, PType, Array[(PType, Code[_])]) => Code[_]
  ): Unit = {
    val newName = rg.wrapFunctionName(mname)
    registered += newName
    registerCode(newName, args, rt, pt)(impl)
  }

  def registerRGCode[A1](
    mname: String, arg1: Type, rt: Type, pt: PType => PType)(
    impl: (EmitRegion, PType, (PType, Code[A1])) => Code[_]
  ): Unit =
    registerRGCode(mname, Array[Type](arg1), rt, unwrappedApply(pt)) {
      case (r, rt, Array(a1: (PType, Code[A1]) @unchecked)) => impl(r, rt, a1)
    }

  def registerRGCode[A1, A2](
    mname: String, arg1: Type, arg2: Type, rt: Type, pt: (PType, PType) => PType)(
    impl: (EmitRegion, PType, (PType, Code[A1]), (PType, Code[A2])) => Code[_]
  ): Unit =
    registerRGCode(mname, Array[Type](arg1, arg2), rt, unwrappedApply(pt)) {
      case (r, rt, Array(a1: (PType, Code[A1]) @unchecked, a2: (PType, Code[A2]) @unchecked)) => impl(r, rt, a1, a2)
    }

  def registerRGCode[A1, A2, A3, A4](
    mname: String, arg1: Type, arg2: Type, arg3: Type, arg4: Type, rt: Type, pt: (PType, PType, PType) => PType)(
    impl: (EmitRegion, PType, (PType, Code[A1]), (PType, Code[A2]), (PType, Code[A3]), (PType, Code[A4])) => Code[_]
  ): Unit =
    registerRGCode(mname, Array[Type](arg1, arg2, arg3, arg4), rt, unwrappedApply(pt)) {
      case (r, rt, Array(
      a1: (PType, Code[A1]) @unchecked,
      a2: (PType, Code[A2]) @unchecked,
      a3: (PType, Code[A3]) @unchecked,
      a4: (PType, Code[A4]) @unchecked)) => impl(r, rt, a1, a2, a3, a4)
    }

  def registerAll() {
    registerRGCode("isValidContig", TString, TBoolean, null) {
      case (r, rt, (contigT, contig: Code[Long])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        rgCode(r.mb).invoke[String, Boolean]("isValidContig", scontig)
    }

    registerRGCode("isValidLocus", TString, TInt32, TBoolean, null) {
      case (r, rt, (contigT, contig: Code[Long]), (posT, pos: Code[Int])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        rgCode(r.mb).invoke[String, Int, Boolean]("isValidLocus", scontig, pos)
    }

    registerRGCode("getReferenceSequenceFromValidLocus", TString, TInt32, TInt32, TInt32, TString, null) {
      case (r, rt, (contigT, contig: Code[Long]), (posT, pos: Code[Int]), (beforeT, before: Code[Int]), (afterT, after: Code[Int])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        unwrapReturn(r, rt)(rgCode(r.mb).invoke[String, Int, Int, Int, String]("getSequence", scontig, pos, before, after))
    }

    registerIR(rg.wrapFunctionName("getReferenceSequence"), TString, TInt32, TInt32, TInt32, TString) {
      (contig, pos, before, after) =>
        val getRef = IRFunctionRegistry.lookupConversion(
          rg.wrapFunctionName("getReferenceSequenceFromValidLocus"),
          TString,
          Seq(TString, TInt32, TInt32, TInt32)).get
        val isValid = IRFunctionRegistry.lookupConversion(
          rg.wrapFunctionName("isValidLocus"),
          TBoolean,
          Seq(TString, TInt32)).get
        If(isValid(Array(contig, pos)), getRef(Array(contig, pos, before, after)), NA(TString))
    }

    registerRGCode("contigLength", TString, TInt32, null) {
      case (r, rt, (contigT, contig: Code[Long])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        rgCode(r.mb).invoke[String, Int]("contigLength", scontig)
    }
  }
}
