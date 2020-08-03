package is.hail.asm4s

import java.io._

import is.hail.utils._
import is.hail.lir

import org.apache.spark.TaskContext
import org.objectweb.asm.tree._
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.util.{Textifier, TraceClassVisitor}
import org.objectweb.asm.ClassReader

import scala.collection.mutable
import scala.reflect.ClassTag

class Field[T: TypeInfo](classBuilder: ClassBuilder[_], val name: String) {
  val ti: TypeInfo[T] = implicitly

  val lf: lir.Field = classBuilder.lclass.newField(name, typeInfo[T])

  def get(obj: Code[_]): Code[T] = Code(obj, lir.getField(lf))

  def putAny(obj: Code[_], v: Code[_]): Code[Unit] = put(obj, coerce[T](v))

  def put(obj: Code[_], v: Code[T]): Code[Unit] = {
    obj.end.append(lir.goto(v.start))
    v.end.append(lir.putField(lf, obj.v, v.v))
    val newC = new VCode(obj.start, v.end, null)
    obj.clear()
    v.clear()
    newC
  }
}

class StaticField[T: TypeInfo](classBuilder: ClassBuilder[_], val name: String) {
  val ti: TypeInfo[T] = implicitly

  val lf: lir.StaticField = classBuilder.lclass.newStaticField(name, typeInfo[T])

  def get(): Code[T] = Code(lir.getStaticField(lf))

  def put(v: Code[T]): Code[Unit] = {
    v.end.append(lir.putStaticField(lf, v.v))
    val newC = new VCode(v.start, v.end, null)
    v.clear()
    newC
  }
}

class ClassesBytes(classesBytes: Array[(String, Array[Byte])]) extends Serializable {
  @transient @volatile var loaded: Boolean = false

  def load(): Unit = {
    if (!loaded) {
      synchronized {
        if (!loaded) {
          classesBytes.foreach { case (n, bytes) =>
            try {
              HailClassLoader.loadOrDefineClass(n, bytes)
            } catch {
              case e: Exception =>
                FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
                throw e
            }
          }
        }
        loaded = true
      }
    }
  }
}

class AsmTuple[C](val cb: ClassBuilder[C], val fields: IndexedSeq[Field[_]], val ctor: MethodBuilder[C]) {
  def newTuple(elems: IndexedSeq[Code[_]]): Code[C] = Code.newInstance(cb, ctor, elems)

  def loadElementsAny(t: Value[_]): IndexedSeq[Code[_]] = fields.map(_.get(coerce[C](t) ))

  def loadElements(t: Value[C]): IndexedSeq[Code[_]] = fields.map(_.get(t))
}

trait WrappedModuleBuilder {
  def modb: ModuleBuilder

  def newClass[C](name: String)(implicit cti: TypeInfo[C]): ClassBuilder[C] = modb.newClass[C](name)

  def genClass[C](baseName: String)(implicit cti: TypeInfo[C]): ClassBuilder[C] = modb.genClass[C](baseName)

  def classesBytes(print: Option[PrintWriter] = None): ClassesBytes = modb.classesBytes(print)
}

class ModuleBuilder() {
  val classes = new mutable.ArrayBuffer[ClassBuilder[_]]()

  def newClass[C](name: String)(implicit cti: TypeInfo[C]): ClassBuilder[C] = {
    val c = new ClassBuilder[C](this, name)
    if (cti != UnitInfo)
      c.addInterface(cti.iname)
    classes += c
    c
  }

  private val tuples = mutable.Map[IndexedSeq[TypeInfo[_]], AsmTuple[_]]()

  def tupleClass(fieldTypes: IndexedSeq[TypeInfo[_]]): AsmTuple[_] = {
    tuples.getOrElseUpdate(fieldTypes, {
      val cb = genClass[AnyRef]("Tuple")
      val fields = fieldTypes.zipWithIndex.map { case (ti, i) =>
        cb.newField(s"_$i")(ti)
      }
      val ctor = cb.newMethod("<init>", fieldTypes, UnitInfo)
      ctor.emitWithBuilder { cb =>
        fields.zipWithIndex.foreach { case (f, i) =>
            cb += f.putAny(ctor._this, ctor.getArg(i + 1)(f.ti).get)
        }
        Code._empty
      }
      new AsmTuple(cb, fields, ctor)
    })
  }

  def genClass[C](baseName: String)(implicit cti: TypeInfo[C]): ClassBuilder[C] = newClass[C](genName("C", baseName))

  var classesBytes: ClassesBytes = _

  def classesBytes(print: Option[PrintWriter] = None): ClassesBytes = {
    if (classesBytes == null) {
      classesBytes = new ClassesBytes(
        classes
          .iterator
          .flatMap(c => c.classBytes(print))
          .toArray)

    }
    classesBytes
  }
}

trait WrappedClassBuilder[C] extends WrappedModuleBuilder {
  def cb: ClassBuilder[C]

  def modb: ModuleBuilder = cb.modb

  def className: String = cb.className

  def ti: TypeInfo[_] = cb.ti

  def addInterface(name: String): Unit = cb.addInterface(name)

  def emitInit(c: Code[Unit]): Unit = cb.emitInit(c)

  def emitClinit(c: Code[Unit]): Unit = cb.emitClinit(c)

  def newField[T: TypeInfo](name: String): Field[T] = cb.newField[T](name)

  def newStaticField[T: TypeInfo](name: String): StaticField[T] = cb.newStaticField[T](name)

  def newStaticField[T: TypeInfo](name: String, init: Code[T]): StaticField[T] = cb.newStaticField[T](name, init)

  def genField[T: TypeInfo](baseName: String): Field[T] = cb.genField(baseName)

  def genFieldThisRef[T: TypeInfo](name: String = null): ThisFieldRef[T] = cb.genFieldThisRef[T](name)

  def genLazyFieldThisRef[T: TypeInfo](setup: Code[T], name: String = null): Value[T] = cb.genLazyFieldThisRef(setup, name)

  def getOrDefineLazyField[T: TypeInfo](setup: Code[T], id: Any): Value[T] = cb.getOrDefineLazyField(setup, id)

  def fieldBuilder: SettableBuilder = cb.fieldBuilder

  def newMethod(name: String, parameterTypeInfo: IndexedSeq[TypeInfo[_]], returnTypeInfo: TypeInfo[_]): MethodBuilder[C] =
    cb.newMethod(name, parameterTypeInfo, returnTypeInfo)

  def newMethod(name: String,
    maybeGenericParameterTypeInfo: IndexedSeq[MaybeGenericTypeInfo[_]],
    maybeGenericReturnTypeInfo: MaybeGenericTypeInfo[_]): MethodBuilder[C] =
    cb.newMethod(name, maybeGenericParameterTypeInfo, maybeGenericReturnTypeInfo)

  def newStaticMethod(name: String, parameterTypeInfo: IndexedSeq[TypeInfo[_]], returnTypeInfo: TypeInfo[_]): MethodBuilder[C] =
    cb.newStaticMethod(name, parameterTypeInfo, returnTypeInfo)

  def getOrGenMethod(
    baseName: String, key: Any, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]
  )(body: MethodBuilder[C] => Unit): MethodBuilder[C] =
    cb.getOrGenMethod(baseName, key, argsInfo, returnInfo)(body)

  def result(print: Option[PrintWriter] = None): () => C = cb.result(print)

  def _this: Value[C] = cb._this

  def genMethod(baseName: String, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilder[C] =
    cb.genMethod(baseName, argsInfo, returnInfo)

  def genMethod[R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[R](baseName)

  def genMethod[A: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[A, R](baseName)

  def genMethod[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[A1, A2, R](baseName)

  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[A1, A2, A3, R](baseName)

  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[A1, A2, A3, A4, R](baseName)

  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, A5: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[A1, A2, A3, A4, A5, R](baseName)

  def genStaticMethod(name: String, parameterTypeInfo: IndexedSeq[TypeInfo[_]], returnTypeInfo: TypeInfo[_]): MethodBuilder[C] =
    cb.genStaticMethod(name, parameterTypeInfo, returnTypeInfo)
}

class ClassBuilder[C](
  val modb: ModuleBuilder,
  val className: String) extends WrappedModuleBuilder {

  val ti: TypeInfo[C] = new ClassInfo[C](className)

  val lclass = new lir.Classx[C](className, "java/lang/Object")

  val methods: mutable.ArrayBuffer[MethodBuilder[C]] = new mutable.ArrayBuffer[MethodBuilder[C]](16)
  val fields: mutable.ArrayBuffer[FieldNode] = new mutable.ArrayBuffer[FieldNode](16)

  val lazyFieldMemo: mutable.Map[Any, Value[_]] = mutable.Map.empty

  val lInit = lclass.newMethod("<init>", FastIndexedSeq(), UnitInfo)

  var initBody: Code[Unit] = {
    val L = new lir.Block()
    L.append(
      lir.methodStmt(INVOKESPECIAL,
        "java/lang/Object",
        "<init>",
        "()V",
        false,
        UnitInfo,
        FastIndexedSeq(lir.load(lInit.getParam(0)))))
    L.append(lir.returnx())
    new VCode(L, L, null)
  }

  private var lClinit: lir.Method = _

  var clinitBody: Option[Code[Unit]] = None

  def emitInit(c: Code[Unit]): Unit = {
    initBody = Code(initBody, c)
  }

  def emitClinit(c: Code[Unit]): Unit = {
    clinitBody match {
      case None =>
        lClinit = lclass.newMethod("<clinit>", FastIndexedSeq(), UnitInfo, isStatic = true)
        clinitBody = Some(c)
      case Some(body) =>
        clinitBody = Some(Code(body, c))
    }
  }

  def addInterface(name: String): Unit = lclass.addInterface(name)

  def newMethod(name: String, parameterTypeInfo: IndexedSeq[TypeInfo[_]], returnTypeInfo: TypeInfo[_]): MethodBuilder[C] = {
    val mb = new MethodBuilder[C](this, name, parameterTypeInfo, returnTypeInfo)
    methods.append(mb)
    mb
  }

  def newMethod(name: String,
    maybeGenericParameterTypeInfo: IndexedSeq[MaybeGenericTypeInfo[_]],
    maybeGenericReturnTypeInfo: MaybeGenericTypeInfo[_]): MethodBuilder[C] = {

    val parameterTypeInfo: IndexedSeq[TypeInfo[_]] = maybeGenericParameterTypeInfo.map(_.base)
    val returnTypeInfo: TypeInfo[_] = maybeGenericReturnTypeInfo.base
    val m = newMethod(name, parameterTypeInfo, returnTypeInfo)
    if (maybeGenericParameterTypeInfo.exists(_.isGeneric) || maybeGenericReturnTypeInfo.isGeneric) {
      val generic = newMethod(name, maybeGenericParameterTypeInfo.map(_.generic), maybeGenericReturnTypeInfo.generic)
      generic.emit(
        maybeGenericReturnTypeInfo.castToGeneric(
          m.invoke(maybeGenericParameterTypeInfo.zipWithIndex.map { case (ti, i) =>
            ti.castFromGeneric(generic.getArg(i + 1)(ti.generic).get)
          }: _*)))
    }
    m
  }

  def newStaticMethod(name: String, parameterTypeInfo: IndexedSeq[TypeInfo[_]], returnTypeInfo: TypeInfo[_]): MethodBuilder[C] = {
    val mb = new MethodBuilder[C](this, name, parameterTypeInfo, returnTypeInfo, isStatic = true)
    methods.append(mb)
    mb
  }

  def genDependentFunction[A1 : TypeInfo, R : TypeInfo](baseName: String): DependentFunctionBuilder[AsmFunction1[A1, R]] = {
    val depCB = modb.genClass[AsmFunction1[A1, R]](baseName)
    val apply = depCB.newMethod("apply", Array(GenericTypeInfo[A1]), GenericTypeInfo[R])
    val dep_apply_method = new DependentMethodBuilder(apply)
    new DependentFunctionBuilder[AsmFunction1[A1, R]](dep_apply_method)
  }

  def newField[T: TypeInfo](name: String): Field[T] = new Field[T](this, name)

  def newStaticField[T: TypeInfo](name: String): StaticField[T] = new StaticField[T](this, name)

  def newStaticField[T: TypeInfo](name: String, init: Code[T]): StaticField[T] = {
    val f = new StaticField[T](this, name)
    emitClinit(f.put(init))
    f
  }

  def genField[T: TypeInfo](baseName: String): Field[T] = newField(genName("f", baseName))

  private[this] val methodMemo: mutable.Map[Any, MethodBuilder[C]] = mutable.HashMap.empty

  def getOrGenMethod(baseName: String, key: Any, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_])
    (f: MethodBuilder[C] => Unit): MethodBuilder[C] = {
    methodMemo.get(key) match {
      case Some(mb) => mb
      case None =>
        val mb = newMethod(genName("M", baseName), argsInfo, returnInfo)
        f(mb)
        methodMemo(key) = mb
        mb
    }
  }

  def classBytes(print: Option[PrintWriter] = None): Array[(String, Array[Byte])] = {
    assert(initBody.start != null)
    lInit.setEntry(initBody.start)

    clinitBody match {
      case None => // do nothing
      case Some(body) =>
        assert(body.start != null)
        body.end.append(lir.returnx())
        val nbody = new VCode(body.start, body.end, null)
        body.clear()
        lClinit.setEntry(nbody.start)
    }

    lclass.asBytes(print)
  }

  def result(print: Option[PrintWriter] = None): () => C = {
    val n = className.replace("/", ".")
    val classesBytes = modb.classesBytes()

    assert(TaskContext.get() == null,
      "FunctionBuilder emission should happen on master, but happened on worker")

    new (() => C) with java.io.Serializable {
      @transient @volatile private var theClass: Class[_] = null

      def apply(): C = {
        if (theClass == null) {
          this.synchronized {
            if (theClass == null) {
              classesBytes.load()
              theClass = loadClass(n)
            }
          }
        }

        theClass.newInstance().asInstanceOf[C]
      }
    }
  }

  def _this: Value[C] = new LocalRef[C](new lir.Parameter(null, 0, ti))

  val fieldBuilder: SettableBuilder = new SettableBuilder {
    def newSettable[T](name: String)(implicit tti: TypeInfo[T]): Settable[T] = genFieldThisRef[T](name)
  }

  def genFieldThisRef[T: TypeInfo](name: String = null): ThisFieldRef[T] =
    new ThisFieldRef[T](this, genField[T](name))

  def genLazyFieldThisRef[T: TypeInfo](setup: Code[T], name: String = null): Value[T] =
    new ThisLazyFieldRef[T](this, name, setup)

  def getOrDefineLazyField[T: TypeInfo](setup: Code[T], id: Any): Value[T] = {
    lazyFieldMemo.getOrElseUpdate(id, genLazyFieldThisRef[T](setup)).asInstanceOf[ThisLazyFieldRef[T]]
  }

  def genMethod(baseName: String, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilder[C] =
    newMethod(genName("m", baseName), argsInfo, returnInfo)

  def genMethod[R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](), typeInfo[R])

  def genMethod[A: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A]), typeInfo[R])

  def genMethod[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2]), typeInfo[R])

  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2], typeInfo[A3]), typeInfo[R])

  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2], typeInfo[A3], typeInfo[A4]), typeInfo[R])

  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, A5: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2], typeInfo[A3], typeInfo[A4], typeInfo[A5]), typeInfo[R])

  def genStaticMethod(baseName: String, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilder[C] =
    newStaticMethod(genName("sm", baseName), argsInfo, returnInfo)
}

object FunctionBuilder {
  val stderrAndLoggerErrorOS = getStderrAndLogOutputStream[FunctionBuilder[_]]

  def bytesToBytecodeString(bytes: Array[Byte], out: OutputStream) {
    val tcv = new TraceClassVisitor(null, new Textifier, new PrintWriter(out))
    new ClassReader(bytes).accept(tcv, 0)
  }

  def apply[F](
    baseName: String,
    argInfo: IndexedSeq[MaybeGenericTypeInfo[_]],
    returnInfo: MaybeGenericTypeInfo[_]
  )(implicit fti: TypeInfo[F]): FunctionBuilder[F] = {
    val modb: ModuleBuilder = new ModuleBuilder()
    val cb: ClassBuilder[F] = modb.genClass[F](baseName)
    val apply = cb.newMethod("apply", argInfo, returnInfo)
    new FunctionBuilder[F](apply)
  }

  def apply[R: TypeInfo](baseName: String): FunctionBuilder[AsmFunction0[R]] =
    apply[AsmFunction0[R]](baseName, FastIndexedSeq.empty[MaybeGenericTypeInfo[_]], GenericTypeInfo[R])

  def apply[A1: TypeInfo, R: TypeInfo](baseName: String): FunctionBuilder[AsmFunction1[A1, R]] =
    apply[AsmFunction1[A1, R]](baseName, Array(GenericTypeInfo[A1]), GenericTypeInfo[R])

  def apply[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](baseName: String): FunctionBuilder[AsmFunction2[A1, A2, R]] =
    apply[AsmFunction2[A1, A2, R]](baseName, Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R])

  def apply[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo](baseName: String): FunctionBuilder[AsmFunction3[A1, A2, A3, R]] =
    apply[AsmFunction3[A1, A2, A3, R]](baseName, Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3]), GenericTypeInfo[R])

  def apply[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, R: TypeInfo](baseName: String): FunctionBuilder[AsmFunction4[A1, A2, A3, A4, R]] =
    apply[AsmFunction4[A1, A2, A3, A4, R]](baseName, Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3], GenericTypeInfo[A4]), GenericTypeInfo[R])
}

trait WrappedMethodBuilder[C] extends WrappedClassBuilder[C] {
  def mb: MethodBuilder[C]

  def cb: ClassBuilder[C] = mb.cb

  def methodName: String = mb.methodName

  def parameterTypeInfo: IndexedSeq[TypeInfo[_]] = mb.parameterTypeInfo

  def returnTypeInfo: TypeInfo[_] = mb.returnTypeInfo

  def newLocal[T: TypeInfo](name: String = null): LocalRef[T] = mb.newLocal(name)

  def localBuilder: SettableBuilder = mb.localBuilder

  def getArg[T: TypeInfo](i: Int): LocalRef[T] = mb.getArg[T](i)

  def emitStartup(c: Code[Unit]): Unit = mb.emitStartup(c)

  def emit(body: Code[_]): Unit = mb.emit(body)

  def invoke[T](args: Code[_]*): Code[T] = mb.invoke(args: _*)
}

class MethodBuilder[C](
  val cb: ClassBuilder[C], _mname: String,
  val parameterTypeInfo: IndexedSeq[TypeInfo[_]],
  val returnTypeInfo: TypeInfo[_],
  val isStatic: Boolean = false
) extends WrappedClassBuilder[C] {
  val methodName: String = _mname.substring(0, scala.math.min(_mname.length, 65535))

  if (methodName != "<init>" && !isJavaIdentifier(methodName))
    throw new IllegalArgumentException(s"Illegal method name, not Java identifier: $methodName")

  val lmethod: lir.Method = cb.lclass.newMethod(methodName, parameterTypeInfo, returnTypeInfo, isStatic)

  val localBuilder: SettableBuilder = new SettableBuilder {
    def newSettable[T](name: String)(implicit tti: TypeInfo[T]): Settable[T] = newLocal[T](name)
  }

  def newLocal[T: TypeInfo](name: String = null): LocalRef[T] =
    new LocalRef[T](lmethod.newLocal(name, typeInfo[T]))

  def getArg[T: TypeInfo](i: Int): LocalRef[T] = {
    val ti = implicitly[TypeInfo[T]]

    if (i == 0 && !isStatic)
      assert(ti == cb.ti, s"$ti != ${ cb.ti }")
    else {
      val static = (!isStatic).toInt
      assert(ti == parameterTypeInfo(i - static), s"$ti != ${ parameterTypeInfo(i - static) }")
    }
    new LocalRef(lmethod.getParam(i))
  }

  private var emitted = false

  private var startup: Code[Unit] = Code._empty

  def emitStartup(c: Code[Unit]): Unit = {
    assert(!emitted)
    startup = Code(startup, c)
  }

  def emitWithBuilder[T](f: (CodeBuilder) => Code[T]): Unit = emit(CodeBuilder.scopedCode[T](this)(f))

  def emit(body: Code[_]): Unit = {
    assert(!emitted)
    emitted = true

    val start = startup.start
    startup.end.append(lir.goto(body.start))
    body.end.append(
      if (body.v != null)
        lir.returnx(body.v)
      else
        lir.returnx())
    assert(start != null)
    lmethod.setEntry(start)

    body.clear()
  }

  def invoke[T](args: Code[_]*): Code[T] = {
    val (start, end, argvs) = Code.sequenceValues(args.toFastIndexedSeq)
    if (returnTypeInfo eq UnitInfo) {
      if (isStatic) {
        end.append(lir.methodStmt(INVOKESTATIC, lmethod, argvs))
      } else {
        end.append(
          lir.methodStmt(INVOKEVIRTUAL, lmethod,
            lir.load(new lir.Parameter(null, 0, cb.ti)) +: argvs))
      }
      new VCode(start, end, null)
    } else {
      val value = if (isStatic) {
        lir.methodInsn(INVOKESTATIC, lmethod, argvs)
      } else {
        lir.methodInsn(INVOKEVIRTUAL, lmethod,
          lir.load(new lir.Parameter(null, 0, cb.ti)) +: argvs)
      }
      new VCode(start, end, value)
    }
  }
}

class DependentMethodBuilder[C](val mb: MethodBuilder[C]) extends WrappedMethodBuilder[C] {
  var setFields: mutable.ArrayBuffer[(lir.ValueX) => Code[Unit]] = new mutable.ArrayBuffer()

  def newDepField[T : TypeInfo](value: Code[T]): Value[T] = {
    val cfr = genFieldThisRef[T]()
    setFields += { (obj: lir.ValueX) =>
      value.end.append(lir.putField(cb.className, cfr.name, typeInfo[T], obj, value.v))
      val newC = new VCode(value.start, value.end, null)
      value.clear()
      newC
    }
    cfr
  }

  def newDepFieldAny[T: TypeInfo](value: Code[_]): Value[T] =
    newDepField(value.asInstanceOf[Code[T]])

  def newInstance(mb: MethodBuilder[_]): Code[C] = {
    val L = new lir.Block()

    val obj = new lir.Local(null, "new_dep_fun", cb.ti)
    L.append(lir.store(obj, lir.newInstance(cb.ti, cb.lInit, FastIndexedSeq.empty[lir.ValueX])))

    var end = L
    setFields.foreach { f =>
      val c = f(lir.load(obj))
      end.append(lir.goto(c.start))
      end = c.end
    }
    new VCode(L, end, lir.load(obj))
  }

  override def result(pw: Option[PrintWriter]): () => C =
    throw new UnsupportedOperationException("cannot call result() on a dependent function")
}

trait WrappedDependentMethodBuilder[C] extends WrappedMethodBuilder[C] {
  def dmb: DependentMethodBuilder[C]

  def mb: MethodBuilder[C] = dmb.mb

  def newDepField[T : TypeInfo](value: Code[T]): Value[T] = dmb.newDepField(value)

  def newDepFieldAny[T: TypeInfo](value: Code[_]): Value[T] = dmb.newDepFieldAny[T](value)

  def newInstance(mb: MethodBuilder[_]): Code[C] = dmb.newInstance(mb)
}

class DependentFunctionBuilder[F](apply_method: DependentMethodBuilder[F]) extends WrappedDependentMethodBuilder[F] {
  def dmb: DependentMethodBuilder[F] = apply_method
}

class FunctionBuilder[F](
  val apply_method: MethodBuilder[F]
) extends WrappedMethodBuilder[F] {
  val mb: MethodBuilder[F] = apply_method
}
