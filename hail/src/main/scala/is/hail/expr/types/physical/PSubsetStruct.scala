package is.hail.expr.types.physical

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s.{Code, Settable, SettableBuilder, Value, const}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode}
import is.hail.expr.types.BaseStruct
import is.hail.expr.types.virtual.TStruct
import is.hail.utils._

object PSubsetStruct {
  def apply(ps: PStruct, fieldNames: String*): PSubsetStruct = {
    val f = fieldNames.map(f => ps.field(f)).toFastIndexedSeq
    PSubsetStruct(ps, f)
  }
}

// PSubsetStruct is a view of some other PStruct
// Operations it can take:
// 1) Modify the view by changing which fields from the backing struct it has access to (add/delete, but don't modify backing PStruct)
// 2) Modify the backing values (setting requiredeness)
// 3) Modify properties of the fields in the view (rename), which requires the generation of a new backing PStruct
// Creating entirely new fields seems out of scope
final case class PSubsetStruct(ps: PStruct, fields: IndexedSeq[PField]) extends PStruct {
  if (fields == ps.fields) {
    log.warn("PSubsetStruct used without subsetting input PStruct")
  }

  val idxMap: IndexedSeq[Int] = fields.zipWithIndex.map { case (f, i) => {
    // TODO: if we want to allow rename, this can't be true unless we modify the backing PCanonicalStruct
    // or are ok with index-only match
    val psField = ps.field(f.name)
    assert(f == psField)
    psField.index
  }
  }

  val required = ps.required

  override lazy val virtualType = TStruct(fields.map(f => (f.name -> f.typ.virtualType)):_*)
  override val types: Array[PType] = fields.map(_.typ).toArray

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (compact) {
      sb.append("PSubsetStruct{")
      fields.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
      sb += '}'
    } else {
      if (size == 0)
        sb.append("Struct { }")
      else {
        sb.append("Struct {")
        sb += '\n'
        fields.foreachBetween(_.pretty(sb, indent + 4, compact))(sb.append(",\n"))
        sb += '\n'
        sb.append(" " * indent)
        sb += '}'
      }
    }
  }

  override def truncate(newSize: Int): PSubsetStruct =
    PSubsetStruct(ps, fields.take(newSize))

  override def deleteField(key: String): PStruct = {
    assert(fieldIdx.contains(key))
    val index = fieldIdx(key)
    val newFields = Array.fill[PField](fields.length - 1)(null)
    for (i <- 0 until index)
      newFields(i) = fields(i)
    for (i <- index + 1 until fields.length)
      newFields(i - 1) = fields(i).copy(index = i - 1)
    PSubsetStruct(ps, newFields)
  }

  // TODO: This feels potentially outside the scope of a view of the backing struct.
  // If we choose to have this, need to decide whether we modify the backing PCanonicalStruct's field names,
  // or just the view's.
  // If just the view's, should have a nameMap in addition to idxMap
  override def rename(m: Map[String, String]): PStruct = ???

  // TODO: what are the semantics of this? I think most straightforward is concatenating a 2nd view of the same
  // PStruct, rather than modify the backing PStruct (else we're combining views of different PStructs
  override def ++(that: PStruct): PSubsetStruct = {
    assert(that.isInstanceOf[PSubsetStruct])
    val thatSubset = that.asInstanceOf[PSubsetStruct]
    assert(thatSubset.ps == ps)
    val overlapping = fields.map(_.name).toSet.intersect(
      that.fields.map(_.name).toSet)
    if (overlapping.nonEmpty)
      fatal(s"overlapping fields in PSubsetStruct concatenation: ${overlapping.mkString(", ")}")

    PSubsetStruct(ps, fields ++ that.fields)
  }

  def selectFields(names: Seq[String]): PSubsetStruct =
    PSubsetStruct(ps, names.map(f => field(f)).toIndexedSeq)

  def dropFields(names: Set[String]): PSubsetStruct =
    selectFields(fieldNames.filter(!names.contains(_)))

  def typeAfterSelect(keep: IndexedSeq[Int]): PSubsetStruct =
    PSubsetStruct(ps, keep.map(i => fields(i)))

  // TODO: The constructible type seems like it should be the canonical version of this type (same fields)
  // but how is this used? If used in copyFromType/constructAtAddress from something of the backing PStruct's fields
  // this won't work, so just keep as the entire ps
  lazy val structFundamentalType: PStruct = ps.structFundamentalType

  override def loadField(structAddress: Code[Long], fieldName: String): Code[Long] = ps.loadField(structAddress, fieldName)

  override def isFieldMissing(structAddress: Code[Long], fieldName: String): Code[Boolean] = ps.isFieldMissing(structAddress, fieldName)

  override def fieldOffset(structAddress: Code[Long], fieldName: String): Code[Long] = ps.fieldOffset(structAddress, fieldName)

  override def setFieldPresent(structAddress: Code[Long], fieldName: String): Code[Unit] = ps.setFieldPresent(structAddress, fieldName)

  override def setFieldMissing(structAddress: Code[Long], fieldName: String): Code[Unit] = ps.setFieldMissing(structAddress, fieldName)

  val missingIdx: Array[Int] = ps.missingIdx
  val nMissing: Int = ps.nMissing

  // TODO: do we want the fieldIdx here refer to the fields in backing PStruct, or this PSubsetStruct?
  // currently it's the PSubsetStruct indices
  override def isFieldDefined(structAddress: Long, fieldIdx: Int): Boolean =
    ps.isFieldDefined(structAddress, idxMap(fieldIdx))

  override def isFieldMissing(structAddress: Code[Long], fieldIdx: Int): Code[Boolean] =
    ps.isFieldMissing(structAddress, idxMap(fieldIdx))

  override def setFieldMissing(structAddress: Long, fieldIdx: Int): Unit =
    ps.setFieldMissing(structAddress, idxMap(fieldIdx))

  override def setFieldMissing(structAddress: Code[Long], fieldIdx: Int): Code[Unit] =
    ps.setFieldMissing(structAddress, idxMap(fieldIdx))

  override def setFieldPresent(structAddress: Long, fieldIdx: Int): Unit =
    ps.setFieldPresent(structAddress, idxMap(fieldIdx))

  override def setFieldPresent(structAddress: Code[Long], fieldIdx: Int): Code[Unit] =
    ps.setFieldPresent(structAddress, idxMap(fieldIdx))

  override def fieldOffset(structAddress: Long, fieldIdx: Int): Long =
    ps.fieldOffset(structAddress, idxMap(fieldIdx))

  override def fieldOffset(structAddress: Code[Long], fieldIdx: Int): Code[Long] =
    ps.fieldOffset(structAddress, idxMap(fieldIdx))

  override def loadField(structAddress: Long, fieldIdx: Int): Long =
    ps.loadField(structAddress, idxMap(fieldIdx))

  override def loadField(structAddress: Code[Long], fieldIdx: Int): Code[Long] =
    ps.loadField(structAddress, idxMap(fieldIdx))

  // FIXME: goal is to ensure isn't constructed
  // byteSize, alignment needed in InferPType...
  // FIXME: the correct answer here depends on our structFundamentalType decision
  override val byteSize = ps.byteSize
  override val alignment = ps.alignment

  override def appendKey(key: String, sig: PType): PStruct = ???

  def insertFields(fieldsToInsert: TraversableOnce[(String, PType)]): PSubsetStruct = ???

  override def initialize(structAddress: Long, setMissing: Boolean): Unit = ???

  override def stagedInitialize(structAddress: Code[Long], setMissing: Boolean): Code[Unit] = ???

  override def allocate(region: Region): Long = ???

  override def allocate(region: Code[Region]): Code[Long] = ???

  // This would mean changing the requiredeness of the backing PStruct, since requiredeness in the view
  // is derived from the backign PStruct
  override def setRequired(required: Boolean): PType = ???

  override def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = ???

  override def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] = ???

  override protected def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = ???

  override def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] = ???

  override def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = ???
}

object PSubsetStructSettable {
  def apply(cb: EmitCodeBuilder, pt: PSubsetStruct, name: String, sb: SettableBuilder): PSubsetStructSettable = {
    new PSubsetStructSettable(pt, sb.newSettable(name))
  }
}

class PSubsetStructSettable(val pt: PSubsetStruct, val a: Settable[Long]) extends PBaseStructValue with PSettable {
  def get: PSubsetStructCode = new PSubsetStructCode(pt, a)

  // Again, is fieldIdx here the backing PStruct?
  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    IEmitCode(cb,
      pt.isFieldMissing(a, fieldIdx),
      pt.fields(fieldIdx).typ.load(pt.fieldOffset(a, fieldIdx)))
  }

  def store(pv: PCode): Code[Unit] = {
    a := pv.asInstanceOf[PSubsetStructCode].a
  }
}

// TODO: We don't allow construction, so what do we do here for store?
class PSubsetStructCode(val pt: PSubsetStruct, val a: Code[Long]) extends PBaseStructCode {
  def code: Code[_] = a

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PBaseStructValue = {
    val s = PSubsetStructSettable(cb, pt, name, sb)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PBaseStructValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PBaseStructValue = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = ???
}
