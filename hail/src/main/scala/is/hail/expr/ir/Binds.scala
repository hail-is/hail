package is.hail.expr.ir

import is.hail.expr.types.virtual.{TArray, TContainer}

object Binds {
  def apply(x: IR, v: String, i: Int): Boolean = {
    x match {
      case Let(n, _, _) =>
        v == n && i == 1
      case ArrayMap(_, n, _) =>
        v == n && i == 1
      case ArrayFlatMap(_, n, _) =>
        v == n && i == 1
      case ArrayFilter(_, n, _) =>
        v == n && i == 1
      case ArrayFold(_, _, accumName, valueName, _) =>
        (v == accumName || v == valueName) && i == 2
      case ArrayScan(_, _, accumName, valueName, _) =>
        (v == accumName || v == valueName) && i == 2
      case ArrayLeftJoinDistinct(_, _, l, r, _, _) =>
        (v == l || v == r) && i == 2
      case AggExplode(_, n, _) =>
        v == n && i == 1
      case _ =>
        false
    }
  }
}

object Bindings {
  private val empty = Array[String]()
  private val emptyRef = Array[Ref]()

  def apply(x: IR): Array[String] = x match {
    case Let(name, _, _) => Array(name)
    case ArrayMap(_, name, _) => Array(name)
    case ArrayFor(_, name, _) => Array(name)
    case ArrayFlatMap(_, name, _) => Array(name)
    case ArrayFilter(_, name, _) => Array(name)
    case ArrayFold(_, _, accumName, valueName, _) => Array(accumName, valueName)
    case ArrayScan(_, _, accumName, valueName, _) => Array(accumName, valueName)
    case ArrayLeftJoinDistinct(_, _, l, r, _, _) => Array(l, r)
    case AggExplode(_, n, _) => Array(n)
    case ArrayAgg(_, name, _) => Array(name)
    case _ => empty
  }

  def getRefs(x: IR): Array[Ref] = x match {
    case Let(name, value, _) => Array(Ref(name, value.typ))
    case ArrayMap(a, name, _) => Array(Ref(name, a.typ.asInstanceOf[TArray].elementType))
    case ArrayFor(a, name, _) => Array(Ref(name, a.typ.asInstanceOf[TArray].elementType))
    case ArrayFlatMap(a, name, _) => Array(Ref(name, a.typ.asInstanceOf[TArray].elementType))
    case ArrayFilter(a, name, _) => Array(Ref(name, a.typ.asInstanceOf[TArray].elementType))
    case ArrayFold(a, zero, accumName, valueName, _) => Array(Ref(accumName, zero.typ), Ref(valueName, a.typ.asInstanceOf[TArray].elementType))
    case ArrayScan(a, zero, accumName, valueName, _) => Array(Ref(accumName, zero.typ), Ref(valueName, a.typ.asInstanceOf[TArray].elementType))
    case ArrayLeftJoinDistinct(la, ra, l, r, _, _) => Array(Ref(l, la.typ.asInstanceOf[TArray].elementType), Ref(r, ra.typ.asInstanceOf[TArray].elementType))
    case AggExplode(agg, n, _) => Array(Ref(n, agg.typ.asInstanceOf[TContainer].elementType))
    case ArrayAgg(agg, name, _) => Array(Ref(name, agg.typ.asInstanceOf[TArray].elementType))
    case _ => emptyRef
  }

  def copyBindings(x: IR, newBindings: Array[String], newChildren: IndexedSeq[BaseIR]): IR = {
    x match {
      case _: Let =>
        val Array(newName) = newBindings
        val IndexedSeq(newValue: IR, newBody: IR) = newChildren
        Let(newName, newValue, newBody)
      case _: ArrayMap =>
        val Array(newName) = newBindings
        val IndexedSeq(newArray: IR, newBody: IR) = newChildren
        ArrayMap(newArray, newName, newBody)
      case _: ArrayFor =>
        val Array(newName) = newBindings
        val IndexedSeq(newArray: IR, newBody: IR) = newChildren
        ArrayFor(newArray, newName, newBody)
      case _: ArrayFlatMap =>
        val Array(newName) = newBindings
        val IndexedSeq(newArray: IR, newBody: IR) = newChildren
        ArrayFlatMap(newArray, newName, newBody)
      case _: ArrayFilter =>
        val Array(newName) = newBindings
        val IndexedSeq(newArray: IR, newBody: IR) = newChildren
        ArrayFilter(newArray, newName, newBody)
      case _: ArrayFold =>
        val Array(newAccumName, newValueName) = newBindings
        val IndexedSeq(newArray: IR, newZero: IR, newBody: IR) = newChildren
        ArrayFold(newArray, newZero, newAccumName, newValueName, newBody)
      case _: ArrayScan =>
        val Array(newAccumName, newValueName) = newBindings
        val IndexedSeq(newArray: IR, newZero: IR, newBody: IR) = newChildren
        ArrayScan(newArray, newZero, newAccumName, newValueName, newBody)
      case _: ArrayLeftJoinDistinct =>
        val Array(newL, newR) = newBindings
        val IndexedSeq(newLeft: IR, newRight: IR, newKeyF: IR, newJoinF: IR) = newChildren
        ArrayLeftJoinDistinct(newLeft, newRight, newL, newR, newKeyF, newJoinF)
      case _: AggExplode =>
        val Array(newName) = newBindings
        val IndexedSeq(newAgg: IR, newBody: IR) = newChildren
        AggExplode(newAgg, newName, newBody)
      case _: ArrayAgg =>
        val Array(newName) = newBindings
        val IndexedSeq(newAgg: IR, newBody: IR) = newChildren
        ArrayAgg(newAgg, newName, newBody)
    }
  }
}
