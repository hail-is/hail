package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, _}
import is.hail.backend.HailStateManager
import is.hail.check.Gen
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.SBaseStructValue
import is.hail.utils._

object PBaseStruct {
  def alignment(types: Array[PType]): Long = {
    if (types.isEmpty)
      1
    else
      types.map(_.alignment).max
  }
}

abstract class PBaseStruct extends PType {
  val types: Array[PType]

  val fields: IndexedSeq[PField]

  final lazy val fieldRequired: Array[Boolean] = types.map(_.required)
  final lazy val allFieldsRequired: Boolean = fieldRequired.forall(_ == true)

  final lazy val fieldIdx: Map[String, Int] =
    fields.map(f => (f.name, f.index)).toMap

  final lazy val fieldNames: Array[String] = fields.map(_.name).toArray

  def fieldByName(name: String): PField = fields(fieldIdx(name))

  def index(str: String): Option[Int] = fieldIdx.get(str)

  def selfField(name: String): Option[PField] = fieldIdx.get(name).map(i => fields(i))

  def hasField(name: String): Boolean = fieldIdx.contains(name)

  def field(name: String): PField = fields(fieldIdx(name))

  def fieldType(name: String): PType = types(fieldIdx(name))

  def size: Int = fields.length

  def isIsomorphicTo(other: PBaseStruct) = {
    this.fields.size == other.fields.size && this.isCompatibleWith(other)
  }

  def _toPretty: String = {
    val sb = new StringBuilder
    _pretty(sb, 0, compact = true)
    sb.result()
  }

  def identBase: String

  def _asIdent: String = {
    val sb = new StringBuilder
    sb.append(identBase)
    sb.append("_of_")
    types.foreachBetween { ty =>
      sb.append(ty.asIdent)
    } {
      sb.append("AND")
    }
    sb.append("END")
    sb.result()
  }

  def isPrefixOf(other: PBaseStruct): Boolean =
    size <= other.size && isCompatibleWith(other)

  def isCompatibleWith(other: PBaseStruct): Boolean =
    fields.zip(other.fields).forall{ case (l, r) => l.typ isOfType r.typ }

  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering =
    unsafeOrdering(sm, this)

  override def unsafeOrdering(sm: HailStateManager, rightType: PType): UnsafeOrdering = {
    require(this isOfType rightType)

    val right = rightType.asInstanceOf[PBaseStruct]
    val fieldOrderings: Array[UnsafeOrdering] =
      types.zip(right.types).map { case (l, r) => l.unsafeOrdering(sm, r)}

    new UnsafeOrdering {
      def compare(o1: Long, o2: Long): Int = {
        var i = 0
        while (i < types.length) {
          val leftDefined = isFieldDefined(o1, i)
          val rightDefined = right.isFieldDefined(o2, i)

          if (leftDefined && rightDefined) {
            val c = fieldOrderings(i).compare(loadField(o1, i), right.loadField(o2, i))
            if (c != 0)
              return c
          } else if (leftDefined != rightDefined) {
            val c = if (leftDefined) -1 else 1
            return c
          }
          i += 1
        }
        0
      }
    }
  }

  def nMissing: Int

  def missingIdx: Array[Int]

  def allocate(region: Region): Long

  def allocate(region: Code[Region]): Code[Long]

  def initialize(structAddress: Long, setMissing: Boolean = false): Unit

  def stagedInitialize(cb: EmitCodeBuilder, structAddress: Code[Long], setMissing: Boolean = false): Unit

  def isFieldDefined(offset: Long, fieldIdx: Int): Boolean

  def isFieldMissing(off: Long, fieldIdx: Int): Boolean = !isFieldDefined(off, fieldIdx)

  def isFieldMissing(cb: EmitCodeBuilder, offset: Code[Long], fieldIdx: Int): Value[Boolean]

  def isFieldDefined(cb: EmitCodeBuilder, offset: Code[Long], fieldIdx: Int): Value[Boolean] =
    cb.memoize(!isFieldMissing(cb, offset, fieldIdx))

  def setFieldMissing(offset: Long, fieldIdx: Int): Unit

  def setFieldMissing(cb: EmitCodeBuilder, offset: Code[Long], fieldIdx: Int): Unit

  def setFieldPresent(offset: Long, fieldIdx: Int): Unit

  def setFieldPresent(cb: EmitCodeBuilder, offset: Code[Long], fieldIdx: Int): Unit

  def fieldOffset(structAddress: Long, fieldIdx: Int): Long

  def fieldOffset(structAddress: Code[Long], fieldIdx: Int): Code[Long]

  def loadField(offset: Long, fieldIdx: Int): Long

  def loadField(offset: Code[Long], fieldIdx: Int): Code[Long]

  override def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SBaseStructValue

  override lazy val containsPointers: Boolean = types.exists(_.containsPointers)

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = {
    if (types.isEmpty) {
      Gen.const(Annotation.empty)
    } else
      Gen.uniformSequence(types.map(t => t.genValue(sm))).map(a => Annotation(a: _*))
  }
}
