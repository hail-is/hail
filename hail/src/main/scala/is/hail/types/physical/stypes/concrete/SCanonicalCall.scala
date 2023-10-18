package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.{SCall, SCallValue, SIndexableValue}
import is.hail.types.physical.stypes.primitives.SInt64Value
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.physical.{PCall, PCanonicalCall, PType}
import is.hail.types.virtual.{TCall, Type}
import is.hail.utils._
import is.hail.variant._


case object SCanonicalCall extends SCall {
  def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    value.st match {
      case SCanonicalCall => value
    }
  }

  lazy val virtualType: Type = TCall

  override def castRename(t: Type): SType = this

  def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(IntInfo)

  def fromSettables(settables: IndexedSeq[Settable[_]]): SCanonicalCallSettable = {
    val IndexedSeq(call: Settable[Int@unchecked]) = settables
    assert(call.ti == IntInfo)
    new SCanonicalCallSettable(call)
  }

  def fromValues(values: IndexedSeq[Value[_]]): SCanonicalCallValue = {
    val IndexedSeq(call: Value[Int@unchecked]) = values
    assert(call.ti == IntInfo)
    new SCanonicalCallValue(call)
  }

  def storageType(): PType = PCanonicalCall(false)

  def copiedType: SType = this

  def containsPointers: Boolean = false

  def constructFromIntRepr(cb: EmitCodeBuilder, c: Code[Int]): SCanonicalCallValue =
    new SCanonicalCallValue(cb.memoize(c))
}

class SCanonicalCallValue(val call: Value[Int]) extends SCallValue {
  val pt: PCall = PCanonicalCall(false)

  override def unphase(cb: EmitCodeBuilder): SCanonicalCallValue = {
    val repr = cb.newLocal[Int]("unphase_call", call)
    cb.if_(isPhased(cb),
      cb.assign(repr, Code.invokeScalaObject1[Int, Int](Call.getClass, "unphase", call)),
      cb.assign(repr, call))
    new SCanonicalCallValue(repr)
  }

  def containsAllele(cb: EmitCodeBuilder, allele: Value[Int]): Value[Boolean] = {
    cb.memoize[Boolean](Code.invokeScalaObject2[Int, Int, Boolean](
      Call.getClass, "containsAllele", call, allele), "contains_allele")
  }

  override def canonicalCall(cb: EmitCodeBuilder): Value[Int] = call

  override val st: SCanonicalCall.type = SCanonicalCall

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(call)

  override def ploidy(cb: EmitCodeBuilder): Value[Int] =
    cb.memoize((call >>> 1) & 0x3)

  override def isPhased(cb: EmitCodeBuilder): Value[Boolean] =
    cb.memoize((call & 0x1).ceq(1))

  override def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit = {
    val call2 = cb.memoize(call >>> 3)
    val p = ploidy(cb)
    val j = cb.newLocal[Int]("fea_j")
    val k = cb.newLocal[Int]("fea_k")

    cb.if_(p.ceq(2), {
      cb.if_(call2 < Genotype.nCachedAllelePairs, {
        cb.assign(j, Code.invokeScalaObject1[Int, Int](Genotype.getClass, "cachedAlleleJ", call2))
        cb.assign(k, Code.invokeScalaObject1[Int, Int](Genotype.getClass, "cachedAlleleK", call2))
      }, {
        cb.assign(k, (Code.invokeStatic1[Math, Double, Double]("sqrt", const(8d) * call2.toD + 1d) / 2d - 0.5).toI)
        cb.assign(j, call2 - (k * (k + 1) / 2))
      })
      alleleCode(j)
      cb.if_(isPhased(cb), cb.assign(k, k - j))
      alleleCode(k)
    }, {
      cb.if_(p.ceq(1),
        alleleCode(call2),
        cb.if_(p.cne(0),
          cb.append(Code._fatal[Unit](const("invalid ploidy: ").concat(p.toS)))))
    })
  }

  override def lgtToGT(cb: EmitCodeBuilder, localAlleles: SIndexableValue, errorID: Value[Int]): SCallValue = {

    def checkAndTranslate(cb: EmitCodeBuilder, allele: Code[Int]): Code[Int] = {
      val av = cb.newLocal[Int](s"allele", allele)
      cb.if_(av >= localAlleles.loadLength(),
        cb._fatalWithError(errorID,
          s"lgt_to_gt: found allele ", av.toS, ", but there are only ", localAlleles.loadLength().toS, " local alleles"))
      localAlleles.loadElement(cb, av).get(cb, const("lgt_to_gt: found missing value in local alleles at index ").concat(av.toS), errorID = errorID)
              .asInt.value
    }

    val repr = cb.newLocal[Int]("lgt_to_gt_repr")
    cb.switch(ploidy(cb),
      cb._fatalWithError(errorID, s"ploidy above 2 is not currently supported"),
      FastSeq(
        { () => cb.assign(repr, call) }, // ploidy 0
        { () =>
          val allele = Code.invokeScalaObject1[Int, Int](Call.getClass, "alleleRepr", call)
          val newCall = Code.invokeScalaObject2[Int, Boolean, Int](Call1.getClass, "apply",
            checkAndTranslate(cb, allele), isPhased(cb)
          )
          cb.assign(repr, newCall)
        }, // ploidy 1
        { () =>
          val allelePair = cb.newLocal[Int]("allelePair", Code.invokeScalaObject1[Int, Int](Call.getClass, "allelePairUnchecked", call))
          val j = cb.newLocal[Int]("allele_j", Code.invokeScalaObject1[Int, Int](AllelePair.getClass, "j", allelePair))
          val k = cb.newLocal[Int]("allele_k", Code.invokeScalaObject1[Int, Int](AllelePair.getClass, "k", allelePair))

          cb.if_(j >= localAlleles.loadLength(), cb._fatalWithError(errorID, "invalid lgt_to_gt: allele "))

          cb.assign(repr, Code.invokeScalaObject4[Int, Int, Boolean, Int, Int](Call2.getClass, "withErrorID",
            checkAndTranslate(cb, j),
            checkAndTranslate(cb, k),
            isPhased(cb),
            errorID)
          )
        } // ploidy 2
      )
    )
    new SCanonicalCallValue(repr)
  }

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = new SInt64Value(this.pt.byteSize)
}

object SCanonicalCallSettable {
  def apply(sb: SettableBuilder, name: String): SCanonicalCallSettable =
    new SCanonicalCallSettable(sb.newSettable[Int](s"${ name }_call"))
}

final class SCanonicalCallSettable(override val call: Settable[Int]) extends SCanonicalCallValue(call) with SSettable {
  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(call, v.asInstanceOf[SCanonicalCallValue].call)

  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(call)
}
