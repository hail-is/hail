package is.hail.types

import is.hail.annotations.RegionValue
import is.hail.asm4s._
import scala.language.implicitConversions

package object physical {
  implicit def pvalueToPCode(pv: PValue): PCode = pv.get

  def typeToTypeInfo(t: PType): TypeInfo[_] = t.fundamentalType match {
    case _: PInt32 => typeInfo[Int]
    case _: PInt64 => typeInfo[Long]
    case _: PFloat32 => typeInfo[Float]
    case _: PFloat64 => typeInfo[Double]
    case _: PBoolean => typeInfo[Boolean]
    case PVoid => typeInfo[Unit]
    case _: PBinary => typeInfo[Long]
    case _: PStream => classInfo[Iterator[RegionValue]]
    case _: PBaseStruct => typeInfo[Long]
    case _: PNDArray => typeInfo[Long]
    case _: PContainer => typeInfo[Long]
    case _ => throw new RuntimeException(s"unsupported type found, $t")
  }

  def defaultValue(t: PType): Code[_] = defaultValue(typeToTypeInfo(t))

  def defaultValue(ti: TypeInfo[_]): Code[_] = ti match {
    case UnitInfo => Code._empty
    case BooleanInfo => false
    case IntInfo => 0
    case LongInfo => 0L
    case FloatInfo => 0.0f
    case DoubleInfo => 0.0
    case _: ClassInfo[_] => Code._null
    case ti => throw new RuntimeException(s"unsupported type found: $ti")
  }
}
