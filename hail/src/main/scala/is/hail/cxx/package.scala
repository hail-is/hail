package is.hail

import is.hail.expr.types
import is.hail.expr.types.physical.PType

package object cxx {

  def typeToCXXType(pType: PType): Type = {
    pType.virtualType match {
      case _: types.TInt32 => "int"
      case _: types.TInt64 => "long"
      case _: types.TFloat32 => "float"
      case _: types.TFloat64 => "double"
      case _: types.TBoolean => "bool"
      case _: types.TBinary => "char *"
      case _: types.TArray => "char *"
      case _: types.TBaseStruct => "char *"
      case types.TVoid => "void"
      case _ => throw new RuntimeException(s"unsupported type found, $pType")
    }
  }

  def typeDefaultValue(pType: PType): Type = {
    pType.virtualType match {
      case _: types.TInt32 => "0"
      case _: types.TInt64 => "0l"
      case _: types.TFloat32 => "0.0f"
      case _: types.TFloat64 => "0.0"
      case _: types.TBoolean => "false"
      case _: types.TBinary => "(char *)nullptr"
      case _: types.TArray => "char *)nullptr"
      case _: types.TBaseStruct => "(char *)nullptr"
      case types.TVoid => ""
      case _ => throw new RuntimeException(s"unsupported type found, $pType")
    }
  }

  def loadIRIntermediate(pType: PType, a: Code): Code = {
    pType.virtualType match {
      case _: types.TInt32 => s"load_int($a)"
      case _: types.TInt64 => s"load_long($a)"
      case _: types.TFloat32 => s"load_float($a)"
      case _: types.TFloat64 => s"load_double($a)"
      case _: types.TBoolean => s"load_bool($a)"
      case _: types.TBinary => s"load_address($a)"
      case _: types.TArray => s"load_address($a)"
      case _: types.TBaseStruct => a
      case _ => throw new RuntimeException(s"unsupported type found, $pType")
    }
  }

  def storeIRIntermediate(pType: PType, a: Code, v: Code): Code = {
    pType.virtualType match {
      case _: types.TInt32 => s"store_int($a, $v)"
      case _: types.TInt64 => s"store_long($a, $v)"
      case _: types.TFloat32 => s"store_float($a, $v)"
      case _: types.TFloat64 => s"store_double($a, $v)"
      case _: types.TBoolean => s"store_bool($a, $v)"
      case _: types.TBinary => s"store_address($a, $v)"
      case _: types.TArray => s"store_address($a, $v)"
      // FIXME case _: types.TBaseStruct => a
      case _ => throw new RuntimeException(s"unsupported type found, $pType")
    }
  }

  var symCounter: Long = 0

  def genSym(name: String): String = {
    symCounter += 1
    s"${ name }_$symCounter"
  }

  type Code = String
  type Type = String
}
