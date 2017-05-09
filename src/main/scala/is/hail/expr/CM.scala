package is.hail.expr

import is.hail.asm4s.{Code, _}
import is.hail.variant.GenomeReference

import scala.collection.mutable
import scala.reflect.ClassTag

object preCM {

  case class E(m: Map[String, (Type, Code[AnyRef])], fb: Function2Builder[Array[AnyRef], mutable.ArrayBuffer[AnyRef], AnyRef])

  case class S(fa: Array[AnyRef], ec: EvalContext)

  def emptyE = E(Map[String, (Type, Code[AnyRef])](), new Function2Builder[Array[AnyRef], mutable.ArrayBuffer[AnyRef], AnyRef]())

  def emptyS(ec: EvalContext) = S(Array[AnyRef](), ec)
}

import preCM._

case class CM[+T](mt: (E, S) => (T, S)) {
  def map[U](f: (T) => U): CM[U] = CM { (e, s1) =>
    val (t, s2) = mt(e, s1)
    (f(t), s2)
  }

  def flatMap[U](cmu: (T) => CM[U]): CM[U] = CM { (e, s1) =>
    val (t, s2) = mt(e, s1)
    val CM(mu) = cmu(t)
    mu(e, s2)
  }

  def withFilter(test: (T) => Boolean): CM[T] = CM { (e, s1) =>
    val (t, s2) = mt(e, s1)
    if (test(t))
      (t, s2)
    else
      ???
  }

  def filter(test: (T) => Boolean): CM[T] = withFilter(test)

  def run(ec: EvalContext)(implicit ev: T <:< Code[AnyRef]): () => AnyRef = {
    val e = emptyE
    val (code, s2) = mt(e, emptyS(ec))
    val primitiveFunctionArray = s2.fa.reverse
    e.fb.emit(Code._return(code))

    val localReference = GenomeReference.genomeReference
    val init = () => {
      GenomeReference.setReference(localReference)
    }
    val f = e.fb.result(init)

    { () =>
      try {
        f()(primitiveFunctionArray, mutable.ArrayBuffer[AnyRef]())
      } catch {
        case e: java.lang.reflect.InvocationTargetException =>
          throw e.getCause()
      }
    }
  }

  def run(bindings: Seq[(String, Type, AnyRef)], ec: EvalContext)(implicit ev: T <:< Code[AnyRef]): () => AnyRef = {
    val typedNames = bindings.map { case (name, typ, _) => (name, typ) }
    val values: mutable.ArrayBuffer[AnyRef] = bindings.map(_._3).to[mutable.ArrayBuffer]
    val f: mutable.ArrayBuffer[AnyRef] => AnyRef = runWithDelayedValues(typedNames, ec)

    () => f(values)
  }

  def runWithDelayedValues(typedNames: Seq[(String, Type)], ec: EvalContext)(implicit ev: T <:< Code[AnyRef]): mutable.ArrayBuffer[AnyRef] => AnyRef = {
    val e = emptyE
    val codeBindings = typedNames.zipWithIndex.map { case ((name, typ), i) =>
      (name, (typ, Code.checkcast(e.fb.arg2.invoke[Int, AnyRef]("apply", i))(typ.scalaClassTag)))
    }
    val e2 = e.copy(m = codeBindings.toMap)

    val (code, s2) = mt(e2, emptyS(ec))
    val primitiveFunctionArray = s2.fa.reverse
    e2.fb.emit(Code._return(code))

    val localReference = GenomeReference.genomeReference
    val init = () => {
      GenomeReference.setReference(localReference)
    }

    val f = e2.fb.result(init)

    { (x: mutable.ArrayBuffer[AnyRef]) =>
      try {
        f()(primitiveFunctionArray, x)
      } catch {
        case e: java.lang.reflect.InvocationTargetException =>
          throw e.getCause()
      }
    }
  }
}

object CM {
  def sequence[T](mts: TraversableOnce[CM[T]])(implicit tct: ClassTag[T]): CM[IndexedSeq[T]] = {
    var mresult = CM.ret(Array[T](): IndexedSeq[T])
    mts.foreach(mt => mresult = mresult.flatMap(r => mt.map(t => t +: r)))
    mresult.map(_.reverse)
  }

  def ret[T](t: T): CM[T] = CM((e, a) => (t, a))

  def fb(): CM[Function2Builder[Array[AnyRef], mutable.ArrayBuffer[AnyRef], AnyRef]] = CM { (e, s) => (e.fb, s) }

  def availableBindings(): CM[Map[String, (Type, Code[AnyRef])]] = CM { case (e, s) => (e.m, s) }

  def addForeignFun(f: AnyRef): CM[Int] = CM { (e, s) => (s.fa.length, s.copy(fa = f +: s.fa)) }

  def foreignFunArray(): CM[Code[Array[AnyRef]]] = fb().map(_.arg1)

  def foreignFun[T <: AnyRef](x: T): CM[Code[T]] = for (
    i <- addForeignFun(x);
    ffa <- foreignFunArray()
  ) yield ffa(i).asInstanceOf[Code[T]]

  def initialValueArray(): CM[Code[mutable.ArrayBuffer[AnyRef]]] = fb().map(_.arg2)

  def ec(): CM[EvalContext] = CM { (e, s) => (s.ec, s) }

  def currentSymbolTable(): CM[SymbolTable] = CM { (e, s) => (s.ec.st, s) }

  def ecNewPosition(): CM[(Int, mutable.ArrayBuffer[Any])] = CM { (e, s) =>
    val idx = s.ec.a.length
    val localA = s.ec.a
    localA += null
    ((idx, localA), s)
  }

  // returns a thunk that looks up the result of this aggregation
  def addAggregation(lhs: AST, agg: Aggregator): CM[() => Any] = CM { (e, s) =>
    val b = RefBox(null)
    s.ec.aggregations += ((b, lhs.runAggregator(s.ec), agg))
    (() => b.v, s)
  }

  def newLocal[T](implicit tti: TypeInfo[T]): CM[LocalRef[T]] = fb().map(_.newLocal[T])

  def memoize[T](mc: CM[Code[T]])(implicit tti: TypeInfo[T]): CM[(Code[Unit], Code[T])] = for (
    c <- mc;
    x <- newLocal[T]
  ) yield (x.store(c), x.load().asInstanceOf[Code[T]])

  def memoize[T](c: Code[T])(implicit tti: TypeInfo[T]): CM[(Code[Unit], Code[T])] = for (
    x <- newLocal[T]
  ) yield (x.store(c), x.load().asInstanceOf[Code[T]])

  // references to its argument will be duplicated
  def bindInRaw[T](name: String, typ: Type, c: Code[AnyRef])(body: CM[Code[T]]): CM[Code[T]] = CM { case (e, s) =>
    body.mt(e.copy(m = e.m + ((name, (typ, c)))), s)
  }

  def bindRepInRaw[T](bindings: Seq[(String, Type, CM[Code[AnyRef]])])(body: CM[Code[T]]): CM[Code[T]] = bindings match {
    case Seq() => body
    case Seq((name, typ, cmc), rest@_*) =>
      cmc.flatMap(c => bindInRaw(name, typ, c)(bindRepInRaw(rest)(body)))
  }

  // its argument is stored in a local variable, only refernece to the variable
  // are dupliacted
  def bindIn[T](name: String, typ: Type, c: Code[AnyRef])(body: CM[Code[T]]): CM[Code[T]] =
  memoize(c).flatMap { case (st, v) => bindInRaw(name, typ, v)(body.map(x => Code(st, x))) }

  def bindRepIn[T](bindings: Seq[(String, Type, CM[Code[AnyRef]])])(body: CM[Code[T]]): CM[Code[T]] = bindings match {
    case Seq() => body
    case Seq((name, typ, cmc), rest@_*) =>
      cmc.flatMap(c => bindIn(name, typ, c)(bindRepIn(rest)(body)))
  }

  def lookup(name: String): CM[Code[AnyRef]] = CM { case (e, s) =>
    (e.m(name)._2, s)
  }

  def invokePrimitive1Pure[A, R](f: (A) => R)
    (implicit act: ClassTag[A], rct: ClassTag[R]): CM[Code[A] => Code[R]] =
    foreignFun(f).map((ff: Code[(A) => R]) => (a: Code[A]) => ff.invoke[A, R]("apply", a))

  def invokePrimitive0[R](f: () => R)()(implicit rct: ClassTag[R]): CM[Code[R]] =
    foreignFun(f).map(_.invoke[R]("apply"))

  def invokePrimitive1[A, R](f: (A) => R)(a: Code[A])
    (implicit act: ClassTag[A], rct: ClassTag[R]): CM[Code[R]] =
    foreignFun(f).map(_.invoke[A, R]("apply", a))

  def invokePrimitive2[A, B, R](f: (A, B) => R)(a: Code[A], b: Code[B])
    (implicit act: ClassTag[A], bct: ClassTag[B], rct: ClassTag[R]): CM[Code[R]] =
    foreignFun(f).map(_.invoke[A, B, R]("apply", a, b))

  def invokePrimitive3[A, B, C, R](f: (A, B, C) => R)(a: Code[A], b: Code[B], c: Code[C])
    (implicit act: ClassTag[A], bct: ClassTag[B], cct: ClassTag[C], rct: ClassTag[R]): CM[Code[R]] =
    foreignFun(f).map(_.invoke[A, B, C, R]("apply", a, b, c))

  def invokePrimitive4[A, B, C, D, R](f: (A, B, C, D) => R)(a: Code[A], b: Code[B], c: Code[C], d: Code[D])
    (implicit act: ClassTag[A], bct: ClassTag[B], cct: ClassTag[C], dct: ClassTag[D], rct: ClassTag[R]): CM[Code[R]] =
    foreignFun(f).map(_.invoke[R]("apply", Array[Class[_]](act.runtimeClass, bct.runtimeClass, cct.runtimeClass, dct.runtimeClass), Array(a, b, c, d)))

  def invokePrimitive5[A, B, C, D, E, R](f: (A, B, C, D, E) => R)(a: Code[A], b: Code[B], c: Code[C], d: Code[D], e: Code[E])
    (implicit act: ClassTag[A], bct: ClassTag[B], cct: ClassTag[C], dct: ClassTag[D], ect: ClassTag[E], rct: ClassTag[R]): CM[Code[R]] =
    foreignFun(f).map(_.invoke[R]("apply", Array[Class[_]](act.runtimeClass, bct.runtimeClass, cct.runtimeClass, dct.runtimeClass, ect.runtimeClass), Array(a, b, c, d, e)))

  def invokePrimitive6[A, B, C, D, E, F, R](fn: (A, B, C, D, E, F) => R)(a: Code[A], b: Code[B], c: Code[C], d: Code[D], e: Code[E], f: Code[F])
    (implicit act: ClassTag[A], bct: ClassTag[B], cct: ClassTag[C], dct: ClassTag[D], ect: ClassTag[E], fct: ClassTag[F], rct: ClassTag[R]): CM[Code[R]] =
    foreignFun(fn).map(_.invoke[R]("apply", Array[Class[_]](act.runtimeClass, bct.runtimeClass, cct.runtimeClass, dct.runtimeClass, ect.runtimeClass, fct.runtimeClass), Array(a, b, c, d, e, f)))

  def createLambda[T <: AnyRef](name: String, typ: Type, body: CM[Code[T]])(implicit tti: TypeInfo[T]): CM[Code[(AnyRef) => T]] = {
    val cc = (for (
      v <- initialValueArray();
      result <- bindInRaw(name, typ, Code.checkcast(v.invoke[Int, AnyRef]("apply", 0))(typ.scalaClassTag))(body)
    ) yield result)
    for (
      bindings <- availableBindings();
      cvalues = CompilationHelp.arrayOf(bindings.map { case (_, (_, code)) => code }.toIndexedSeq);
      ec <- ec();
      compiledCode = cc.runWithDelayedValues((name, typ) +: bindings.map { case (name, (typ, _)) => (name, typ) }.toSeq, ec);
      f <- invokePrimitive1(((vs: Array[AnyRef]) => (x: AnyRef) => compiledCode((x +: vs).to[mutable.ArrayBuffer])).asInstanceOf[AnyRef => AnyRef])(cvalues)
    ) yield f.asInstanceOf[Code[AnyRef => T]]
  }

  def oOrNull[T >: Null](x: Code[Option[T]])(implicit tct: ClassTag[T]): CM[Code[T]] =
    CM.memoize(x).map { case (st, opt) => Code(st, opt.invoke[Boolean]("isEmpty").mux(Code._null[T], opt.invoke[T]("get"))) }

  def mapO[T, U >: Null](x: Code[Option[T]], f: Code[T] => Code[U])(implicit tct: ClassTag[T]): CM[Code[U]] =
    CM.memoize(x).map { case (st, opt) => Code(st, opt.invoke[Boolean]("isEmpty").mux(Code._null[U], f(opt.invoke[T]("get")))) }

  def mapOM[T, U >: Null](x: Code[Option[T]], f: Code[T] => CM[Code[U]])(implicit tct: ClassTag[T], uti: TypeInfo[U]): CM[Code[U]] = for (
    (st, opt) <- CM.memoize(x);
    out <- f(opt.invoke[T]("get"))
  ) yield
    Code(st, opt.invoke[Boolean]("isEmpty").mux(Code._null[U], out))

  def mapIS[T, U](ca: Code[IndexedSeq[T]], f: Code[T] => CM[Code[U]])(implicit tct: ClassTag[T], tti: TypeInfo[T], atct: ClassTag[Array[T]], uct: ClassTag[U], uti: TypeInfo[U], auti: TypeInfo[Array[U]]): CM[Code[IndexedSeq[U]]] = for (
    (sta, a) <- CM.memoize(ca);
    (stn, n) <- CM.memoize(a.invoke[Int]("size"));
    (stb, b) <- CM.memoize(Code.newArray[U](n));
    i <- CM.newLocal[Int];
    oldElement = a.invoke[Int, T]("apply", i);
    newElement <- f(oldElement)
  ) yield Code(
    i.store(0),
    sta, stn, stb,
    Code.whileLoop(i < n,
      Code(b.update(i, newElement), i.store(i + 1))
    ),
    CompilationHelp.arrayToWrappedArray(b)
  )
}
