package is.hail.types.physical

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s.{Code, Settable, SettableBuilder, Value, coerce, const}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode}
import is.hail.types.BaseStruct
import is.hail.types.virtual.TStruct
import is.hail.utils._

object PSubsetStruct {
  def apply(ps: PStruct, fieldNames: String*): PSubsetStruct = {
    val f = fieldNames.toArray
    PSubsetStruct(ps, f)
  }
}

// Semantics: PSubsetStruct is a non-constructible view of another PStruct, which is not allowed to mutate
// that underlying PStruct's region data
final case class PSubsetStruct(ps: PStruct, _fieldNames: Array[String]) extends PStruct {
  val fields: IndexedSeq[PField] = _fieldNames.zipWithIndex.map { case (name, i) => PField(name, ps.fieldType(name), i)}
  val required = ps.required

  if (fields == ps.fields) {
    log.warn("PSubsetStruct used without subsetting input PStruct")
  }

  private val idxMap: Array[Int] = _fieldNames.map(f => ps.fieldIdx(f))

  lazy val missingIdx: Array[Int] = idxMap.map(i => ps.missingIdx(i))
  lazy val nMissing: Int = missingIdx.length

  override lazy val virtualType = TStruct(fields.map(f => (f.name -> f.typ.virtualType)):_*)
  override val types: Array[PType] = fields.map(_.typ).toArray

  lazy val structFundamentalType: PStruct = PSubsetStruct(ps.structFundamentalType, _fieldNames)
  lazy val structEncodableType: PStruct = PSubsetStruct(ps.structEncodableType, _fieldNames)
  override val byteSize: Long = 8

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

  override def rename(m: Map[String, String]): PStruct = {
    val newNames = fieldNames.map(fieldName => m.getOrElse(fieldName, fieldName))
    val newPStruct = ps.rename(m)

    PSubsetStruct(newPStruct, newNames)
  }

  override def isFieldMissing(structAddress: Code[Long], fieldName: String): Code[Boolean] =
    ps.isFieldMissing(structAddress, fieldName)

  override def fieldOffset(structAddress: Code[Long], fieldName: String): Code[Long] =
    ps.fieldOffset(structAddress, fieldName)

  override def isFieldDefined(structAddress: Long, fieldIdx: Int): Boolean =
    ps.isFieldDefined(structAddress, idxMap(fieldIdx))

  override def isFieldMissing(structAddress: Code[Long], fieldIdx: Int): Code[Boolean] =
    ps.isFieldMissing(structAddress, idxMap(fieldIdx))

  override def fieldOffset(structAddress: Long, fieldIdx: Int): Long =
    ps.fieldOffset(structAddress, idxMap(fieldIdx))

  override def fieldOffset(structAddress: Code[Long], fieldIdx: Int): Code[Long] =
    ps.fieldOffset(structAddress, idxMap(fieldIdx))

  def loadField(structAddress: Code[Long], fieldName: String): Code[Long] =
    ps.loadField(structAddress, fieldName)

  override def loadField(structAddress: Long, fieldIdx: Int): Long =
    ps.loadField(structAddress, idxMap(fieldIdx))

  override def loadField(structAddress: Code[Long], fieldIdx: Int): Code[Long] =
    ps.loadField(structAddress, idxMap(fieldIdx))

  override def setFieldPresent(structAddress: Code[Long], fieldName: String): Code[Unit] = ???

  override def setFieldMissing(structAddress: Code[Long], fieldName: String): Code[Unit] = ???

  override def setFieldMissing(structAddress: Long, fieldIdx: Int): Unit = ???

  override def setFieldMissing(structAddress: Code[Long], fieldIdx: Int): Code[Unit] = ???

  override def setFieldPresent(structAddress: Long, fieldIdx: Int): Unit = ???

  override def setFieldPresent(structAddress: Code[Long], fieldIdx: Int): Code[Unit] = ???

  def insertFields(fieldsToInsert: TraversableOnce[(String, PType)]): PSubsetStruct = ???

  override def initialize(structAddress: Long, setMissing: Boolean): Unit =
    ps.initialize(structAddress, setMissing)

  override def stagedInitialize(structAddress: Code[Long], setMissing: Boolean): Code[Unit] =
    ps.stagedInitialize(structAddress, setMissing)

  def allocate(region: Region): Long =
    ps.allocate(region)

  def allocate(region: Code[Region]): Code[Long] =
    ps.allocate(region)

  override def setRequired(required: Boolean): PType =
    PSubsetStruct(ps.setRequired(required).asInstanceOf[PStruct], _fieldNames)

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    val srcPSubsetStruct = srcPType.asInstanceOf[PSubsetStruct]
    ps.copyFromType(mb, region, srcPSubsetStruct.ps, srcAddress, deepCopy)
  }

  override def copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    val srcPSubsetStruct = srcPType.asInstanceOf[PSubsetStruct]
    ps.copyFromAddress(region, srcPSubsetStruct.ps, srcAddress, deepCopy)
  }

  override def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    val srcPSubsetStruct = srcPType.asInstanceOf[PSubsetStruct]
    ps.copyFromAddress(region, srcPSubsetStruct.ps, srcAddress, deepCopy)
  }

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] =
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], deepCopy)

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] = {
    val srcPSubsetStruct = srcPType.asInstanceOf[PSubsetStruct]
    Region.storeAddress(addr, ps.copyFromType(mb, region, srcPSubsetStruct.ps, srcAddress, deepCopy))
  }

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    val srcPSubsetStruct = srcPType.asInstanceOf[PSubsetStruct]
    Region.storeAddress(addr, ps.copyFromAddress(region, srcPSubsetStruct.ps, srcAddress, deepCopy))
  }
}

object PSubsetStructSettable {
  def apply(cb: EmitCodeBuilder, pt: PSubsetStruct, name: String, sb: SettableBuilder): PSubsetStructSettable = {
    new PSubsetStructSettable(pt, sb.newSettable(name))
  }
}

class PSubsetStructSettable(val pt: PSubsetStruct, a: Settable[Long]) extends PBaseStructValue with PSettable {
  def get: PSubsetStructCode = new PSubsetStructCode(pt, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    IEmitCode(cb,
      pt.isFieldMissing(a, fieldIdx),
      pt.fields(fieldIdx).typ.load(pt.fieldOffset(a, fieldIdx)))
  }

  def isFieldMissing(fieldIdx: Int): Code[Boolean] =
    pt.isFieldMissing(a, fieldIdx)

  def store(pv: PCode): Code[Unit] = {
    a := pv.asInstanceOf[PSubsetStructCode].a
  }
}

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

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] =
    pt.ps.constructAtAddress(mb, dst, r, pt.ps, a, deepCopy = false)
}
