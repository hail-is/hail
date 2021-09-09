package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder}
import is.hail.types.physical.stypes.{SCode, SValue}
import is.hail.types.physical.stypes.concrete.{SIntervalPointer, SIntervalPointerCode, SIntervalPointerValue, SStackStruct, SUnreachableInterval}
import is.hail.types.physical.stypes.interfaces.{SIntervalValue, primitive}
import is.hail.types.physical.stypes.primitives.SBooleanCode
import is.hail.types.virtual.{TInterval, Type}
import is.hail.utils.{FastIndexedSeq, Interval}
import org.apache.spark.sql.Row

final case class PCanonicalInterval(pointType: PType, override val required: Boolean = false) extends PInterval {

  def byteSize: Long = representation.byteSize
  override def alignment: Long = representation.alignment

  def _asIdent = s"interval_of_${ pointType.asIdent }"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCInterval[")
    pointType.pretty(sb, indent, compact)
    sb.append("]")
  }

  val representation: PCanonicalStruct = PCanonicalStruct(
    required,
    "start" -> pointType,
    "end" -> pointType,
    "includesStart" -> PBooleanRequired,
    "includesEnd" -> PBooleanRequired)

  def setRequired(required: Boolean): PCanonicalInterval =
    if (required == this.required) this else PCanonicalInterval(this.pointType, required)

  def startOffset(off: Code[Long]): Code[Long] = representation.fieldOffset(off, 0)

  def endOffset(off: Code[Long]): Code[Long] = representation.fieldOffset(off, 1)

  def loadStart(off: Long): Long = representation.loadField(off, 0)

  def loadStart(off: Code[Long]): Code[Long] = representation.loadField(off, 0)

  def loadEnd(off: Long): Long = representation.loadField(off, 1)

  def loadEnd(off: Code[Long]): Code[Long] = representation.loadField(off, 1)

  def startDefined(off: Long): Boolean = representation.isFieldDefined(off, 0)

  def endDefined(off: Long): Boolean = representation.isFieldDefined(off, 1)

  def includesStart(off: Long): Boolean = Region.loadBoolean(representation.loadField(off, 2))

  def includesEnd(off: Long): Boolean = Region.loadBoolean(representation.loadField(off, 3))

  def startDefined(off: Code[Long]): Code[Boolean] = representation.isFieldDefined(off, 0)

  def endDefined(off: Code[Long]): Code[Boolean] = representation.isFieldDefined(off, 1)

  def includesStart(off: Code[Long]): Code[Boolean] =
    Region.loadBoolean(representation.loadField(off, 2))

  def includesEnd(off: Code[Long]): Code[Boolean] =
    Region.loadBoolean(representation.loadField(off, 3))

  override def deepRename(t: Type) = deepRenameInterval(t.asInstanceOf[TInterval])

  private def deepRenameInterval(t: TInterval) =
    PCanonicalInterval(this.pointType.deepRename(t.pointType), this.required)

  def containsPointers: Boolean = representation.containsPointers

  def sType: SIntervalPointer = SIntervalPointer(setRequired(false))

  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SIntervalPointerValue =
    new SIntervalPointerCode(sType, addr).memoize(cb, "loadCheapSCode")

  def loadCheapSCodeField(cb: EmitCodeBuilder, addr: Code[Long]): SIntervalPointerValue =
    new SIntervalPointerCode(sType, addr).memoizeField(cb, "loadCheapSCodeField")

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): Value[Long] = {
    value.st match {
      case SIntervalPointer(t: PCanonicalInterval) =>
        representation.store(cb, region, t.representation.loadCheapSCode(cb, value.asInstanceOf[SIntervalPointerValue].a), deepCopy)
      case _ =>
        val interval = value.asInterval
        val start = EmitCode.fromI(cb.emb)(cb => interval.loadStart(cb))
        val stop = EmitCode.fromI(cb.emb)(cb => interval.loadEnd(cb))
        val includesStart = EmitCode.present(cb.emb, new SBooleanCode(interval.includesStart()))
        val includesStop = EmitCode.present(cb.emb, new SBooleanCode(interval.includesEnd()))
        representation.store(cb, region,
          SStackStruct.constructFromArgs(cb, region, representation.virtualType,
            start, stop, includesStart, includesStop).memoize(cb, "store"), deepCopy)
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    value.st match {
      case SIntervalPointer(t: PCanonicalInterval) =>
        representation.storeAtAddress(cb, addr, region, t.representation.loadCheapSCode(cb, value.asInstanceOf[SIntervalPointerCode].a).get, deepCopy)
      case _ =>
        val interval = value.asInterval.memoize(cb, "pcinterval_store_at_addr")
        val start = EmitCode.fromI(cb.emb)(cb => interval.loadStart(cb))
        val stop = EmitCode.fromI(cb.emb)(cb => interval.loadEnd(cb))
        val includesStart = EmitCode.present(cb.emb, new SBooleanCode(interval.includesStart()))
        val includesStop = EmitCode.present(cb.emb, new SBooleanCode(interval.includesEnd()))
        representation.storeAtAddress(cb, addr, region,
          SStackStruct.constructFromArgs(cb, region, representation.virtualType,
            start, stop, includesStart, includesStop),
          deepCopy)
    }
  }
  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    srcPType match {
      case t: PCanonicalInterval =>
        representation.unstagedStoreAtAddress(addr, region, t.representation, srcAddress, deepCopy)
    }
  }

  override def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    srcPType match {
      case t: PCanonicalInterval =>
        representation._copyFromAddress(region, t.representation, srcAddress, deepCopy)
    }
  }

  def loadFromNested(addr: Code[Long]): Code[Long] = representation.loadFromNested(addr)

  override def unstagedLoadFromNested(addr: Long): Long = representation.unstagedLoadFromNested(addr)

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
    val jInterval = annotation.asInstanceOf[Interval]
    representation.unstagedStoreJavaObjectAtAddress(
      addr,
      Row(jInterval.start, jInterval.end, jInterval.includesStart, jInterval.includesEnd),
      region
    )
  }

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = {
    val addr = representation.allocate(region)
    unstagedStoreJavaObjectAtAddress(addr, annotation, region)
    addr
  }

  def constructFromCodes(cb: EmitCodeBuilder, region: Value[Region],
    start: EmitCode, end: EmitCode, includesStart: EmitCode, includesEnd: EmitCode): SIntervalPointerCode = {
    val sc = representation.constructFromFields(cb, region, FastIndexedSeq(start, end, includesStart, includesEnd), deepCopy = false)
    new SIntervalPointerCode(sType, sc.a)
  }

  override def copiedType: PType = {
    val copiedPoint = pointType.copiedType
    if (copiedPoint.eq(pointType))
      this
    else
      PCanonicalInterval(copiedPoint, required)
  }
}
