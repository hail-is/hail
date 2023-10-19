package is.hail.types.encoded

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{ArraySorter, EmitCodeBuilder, EmitMethodBuilder, EmitRegion, StagedArrayBuilder}
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.virtual._
import is.hail.types.physical._
import is.hail.types.physical.stypes.SingleCodeType
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.utils._

final case class EDictAsUnsortedArrayOfPairs(val elementType: EType, override val required: Boolean = false) extends EContainer {
  assert(elementType.isInstanceOf[EBaseStruct])

  private[this] val arrayRepr = EArray(elementType, required)

  def _decodedSType(requestedType: Type): SType = {
    val elementPType = elementType.decodedPType(requestedType.asInstanceOf[TContainer].elementType)
    requestedType match {
      case _: TDict =>
        val et = elementPType.asInstanceOf[PStruct]
        SIndexablePointer(PCanonicalDict(et.fieldType("key"), et.fieldType("value"), false))
    }
  }

  def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {
    // Anything we have to encode from a region should already be sorted so we don't
    // have to do anything else
    arrayRepr._buildEncoder(cb, v, out)
  }

  def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SValue = {
    val tmpRegion = cb.memoize(Region.stagedCreate(Region.REGULAR, region.getPool()), "tmp_region")

    val decodedUnsortedArray = arrayRepr._buildDecoder(cb, t, tmpRegion, in).asInstanceOf[SIndexablePointerValue]
    val sct = SingleCodeType.fromSType(decodedUnsortedArray.st.elementType)

    val ab = new StagedArrayBuilder(sct, true, cb.emb, 0)
    cb.append(ab.ensureCapacity(decodedUnsortedArray.length))
    decodedUnsortedArray.forEachDefined(cb) { (cb, i, res) =>
      cb.append(ab.add(ab.elt.coerceSCode(cb, res, region, false).code))
    }

    val sorter = new ArraySorter(EmitRegion(cb.emb, region), ab)
    def lessThan(cb: EmitCodeBuilder, region: Value[Region], l: Value[_], r: Value[_]): Value[Boolean] = {
      val lk = cb.memoize(sct.loadToSValue(cb, l).asBaseStruct.loadField(cb, 0))
      val rk = cb.memoize(sct.loadToSValue(cb, r).asBaseStruct.loadField(cb, 0))

      cb.emb.ecb.getOrdering(lk.st, rk.st)
        .lt(cb, lk, rk, missingEqual = true)
    }

    sorter.sort(cb, region, lessThan)
    val ret = sorter.toRegion(cb, t)
    cb.append(tmpRegion.invalidate())
    ret
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    arrayRepr._buildSkip(cb, r, in)
  }

  def _asIdent = s"dict_of_${elementType.asIdent}"
  def _toPretty = s"EDictAsUnsortedArrayOfPairs[$elementType]"
  def setRequired(newRequired: Boolean): EType = EDictAsUnsortedArrayOfPairs(elementType, newRequired)
}
