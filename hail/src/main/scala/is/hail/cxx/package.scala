package is.hail

import is.hail.expr.types.physical.{PNDArray, PType}
import is.hail.expr.types.virtual._

package object cxx {

  def typeToCXXType(pType: PType): Type = {
    pType.virtualType.fundamentalType match {
      case _: TBinary | _: TArray | _: TBaseStruct => "const char *"
      case _ => typeToNonConstCXXType(pType)
    }
  }

  def typeToNonConstCXXType(pType: PType): Type = {
    pType.virtualType.fundamentalType match {
      case _: TInt32 => "int"
      case _: TInt64 => "long"
      case _: TFloat32 => "float"
      case _: TFloat64 => "double"
      case _: TBoolean => "bool"
      case _: TBinary => "char *"
      case _: TArray => "char *"
      case _: TBaseStruct => "char *"
      case _: TNDArray => "NDArray"
      case TVoid => "void"
      case _ => throw new RuntimeException(s"unsupported type found, $pType")
    }
  }

  def typeDefaultValue(pType: PType): Type = {
    pType.virtualType.fundamentalType match {
      case _: TInt32 => "0"
      case _: TInt64 => "0l"
      case _: TFloat32 => "0.0f"
      case _: TFloat64 => "0.0"
      case _: TBoolean => "false"
      case _: TBinary => "(char *)nullptr"
      case _: TArray => "(char *)nullptr"
      case _: TBaseStruct => "(char *)nullptr"
      case TVoid => ""
      case _ => throw new RuntimeException(s"unsupported type found, $pType")
    }
  }

  def loadIRIntermediate(pType: PType, a: Code): Code = {
    pType.virtualType.fundamentalType match {
      case _: TInt32 => s"load_int($a)"
      case _: TInt64 => s"load_long($a)"
      case _: TFloat32 => s"load_float($a)"
      case _: TFloat64 => s"load_double($a)"
      case _: TBoolean => s"load_bool($a)"
      case _: TBinary => s"load_address($a)"
      case _: TArray => s"load_address($a)"
      case _: TBaseStruct => a
      case _ => throw new RuntimeException(s"unsupported type found, $pType")
    }
  }

  def storeIRIntermediate(pType: PType, a: Code, v: Code): Code = {
    pType.virtualType.fundamentalType match {
      case _: TInt32 => s"store_int($a, $v)"
      case _: TInt64 => s"store_long($a, $v)"
      case _: TFloat32 => s"store_float($a, $v)"
      case _: TFloat64 => s"store_double($a, $v)"
      case _: TBoolean => s"store_bool($a, $v)"
      case _: TBinary => s"store_address($a, $v)"
      case _: TArray => s"store_address($a, $v)"
      case _: TBaseStruct => s"memcpy($a, $v, ${ pType.byteSize })"
      case _ => throw new RuntimeException(s"unsupported type found, $pType")
    }
  }

  type Code = String
  type Type = String
}
