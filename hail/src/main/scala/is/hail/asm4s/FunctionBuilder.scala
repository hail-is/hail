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
  val lf: lir.Field = classBuilder.lclass.newField(name, typeInfo[T])

  def get(obj: Code[_]): Code[T] = Code(obj, lir.getField(lf))

  def put(obj: Code[_], v: Code[T]): Code[Unit] = {
    obj.end.append(lir.goto(v.start))
    v.end.append(lir.putField(lf, obj.v, v.v))
    new Code(obj.start, v.end, null)
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

class ModuleBuilder {
  val classes = new mutable.ArrayBuffer[ClassBuilder[_]]()

  def newClass[C](name: String = null): ClassBuilder[C] = {
    val c = new ClassBuilder[C](this, name)
    classes += c
    c
  }

  var classesBytes: ClassesBytes = _

  def classesBytes(print: Option[PrintWriter] = None): ClassesBytes = {
    if (classesBytes == null) {
      classesBytes = new ClassesBytes(
        classes
          .iterator
          .map(c => (c.name.replace("/", "."), c.classAsBytes(print)))
          .toArray)

    }
    classesBytes
  }
}

class ClassBuilder[C](
  module: ModuleBuilder,
  val name: String = null) {
  // FIXME C is wrong
  val ti: TypeInfo[_] = new ClassInfo[C](name)

  val lclass = new lir.Classx[C](name, "java/lang/Object")

  // FIXME use newClass
  module.classes += this

  var nameCounter: Int = 0

  val methods: mutable.ArrayBuffer[MethodBuilder] = new mutable.ArrayBuffer[MethodBuilder](16)
  val fields: mutable.ArrayBuffer[FieldNode] = new mutable.ArrayBuffer[FieldNode](16)

  val lazyFieldMemo: mutable.Map[Any, LazyFieldRef[_]] = mutable.Map.empty

  // methods
  def genName(tag: String): String = {
    nameCounter += 1
    s"__$tag$nameCounter"
  }

  def genName(tag: String, suffix: String): String = {
    if (suffix == null)
      return genName(tag)

    nameCounter += 1
    s"__$tag$nameCounter$suffix"
  }

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
    new Code(L, L, null)
  }

  def addInitInstructions(c: Code[Unit]): Unit = {
    initBody = Code(initBody, c)
  }

  def addInterface(name: String): Unit = lclass.addInterface(name)

  def addMethod(m: MethodBuilder): Unit = {
    methods.append(m)
  }

  def newDependentFunction[A1 : TypeInfo, R : TypeInfo]: DependentFunction[AsmFunction1[A1, R]] = {
    new DependentFunctionBuilder[AsmFunction1[A1, R]](
      Array(GenericTypeInfo[A1]),
      GenericTypeInfo[R],
      initModule = module)
  }

  def newField[T: TypeInfo](name: String): Field[T] = new Field[T](this, name)

  def genField[T: TypeInfo](): Field[T] = newField[T](genName("f"))

  def genField[T: TypeInfo](suffix: String): Field[T] = newField(genName("f", suffix))

  def classAsBytes(print: Option[PrintWriter] = None): Array[Byte] = {
    // FIXME build incrementally?
    lInit.setEntry(initBody.start)

    lclass.asBytes(print)
  }

  def result(print: Option[PrintWriter] = None): () => C = {
    val n = name.replace("/", ".")
    val classesBytes = module.classesBytes()

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
}

object FunctionBuilder {
  val stderrAndLoggerErrorOS = getStderrAndLogOutputStream[FunctionBuilder[_]]

  var count = 0

  def newUniqueID(): Int = {
    val id = count
    count += 1
    id
  }

  def bytesToBytecodeString(bytes: Array[Byte], out: OutputStream) {
    val tcv = new TraceClassVisitor(null, new Textifier, new PrintWriter(out))
    new ClassReader(bytes).accept(tcv, 0)
  }

  def functionBuilder[R: TypeInfo]: FunctionBuilder[AsmFunction0[R]] =
    new FunctionBuilder(Array[MaybeGenericTypeInfo[_]](), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction1[A, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction2[A, B, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction3[A, B, C, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction4[A, B, C, D, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction5[A, B, C, D, E, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction6[A, B, C, D, E, F, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, G: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction7[A, B, C, D, E, F, G, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F], GenericTypeInfo[G]), GenericTypeInfo[R])
}

object MethodBuilder {
  private val localInsnRefs = mutable.Map[AbstractInsnNode, LocalRef[_]]()

  def registerLocalInsn(x: AbstractInsnNode, lr: LocalRef[_]): Unit = {
    assert(!localInsnRefs.contains(x))
    localInsnRefs += (x -> lr)
  }

  def popInsnRef(x: AbstractInsnNode): Option[LocalRef[_]] = {
    val lr = localInsnRefs.get(x)
    lr match {
      case Some(_) =>
        localInsnRefs -= x
      case None =>
    }
    lr
  }
}

class MethodBuilder(val fb: FunctionBuilder[_], _mname: String, val parameterTypeInfo: IndexedSeq[TypeInfo[_]], val returnTypeInfo: TypeInfo[_]) {
  val mname = {
    val s = _mname.substring(0, scala.math.min(_mname.length, 65535))
    require(java.lang.Character.isJavaIdentifierStart(s.head), "invalid java identifier, " + s)
    require(s.forall(java.lang.Character.isJavaIdentifierPart), "invalid java identifer, " + s)
    s
  }

  val lmethod: lir.Method = fb.classBuilder.lclass.newMethod(mname, parameterTypeInfo, returnTypeInfo)

  def newLocal[T](implicit tti: TypeInfo[T]): LocalRef[T] =
    newLocal()

  def newLocal[T](name: String = null)(implicit tti: TypeInfo[T]): LocalRef[T] =
    new LocalRef[T](lmethod.newLocal(name, tti))

  def newField[T: TypeInfo]: ClassFieldRef[T] = newField[T]()

  def newField[T: TypeInfo](name: String = null): ClassFieldRef[T] = fb.newField[T](name)

  def newLazyField[T: TypeInfo](setup: Code[T], name: String = null): LazyFieldRef[T] = fb.newLazyField(setup, name)

  def getArg[T](i: Int)(implicit tti: TypeInfo[T]): LocalRef[T] =
    new LocalRef(lmethod.getParam(i))

  private var emitted = false

  private var startup: Code[Unit] = Code._empty

  def emitStartup(c: Code[Unit]): Unit = {
    assert(!emitted)
    startup = Code(startup, c)
  }

  def emit(body: Code[_]) {
    assert(!emitted)
    emitted = true

    val start = startup.start
    startup.end.append(lir.goto(body.start))
    body.end.append(
      if (body.v != null)
        lir.returnx(body.v)
      else
        lir.returnx())
    lmethod.setEntry(start)
  }

  def invoke[T](args: Code[_]*): Code[T] = {
    val (start, end, argvs) = Code.sequenceValues(args.toFastIndexedSeq)
    if (returnTypeInfo.desc == "V") {
      val L = new lir.Block()
      L.append(
        lir.methodStmt(INVOKEVIRTUAL, lmethod,
          lir.load(new lir.Parameter(null, 0, fb.classBuilder.ti)) +: argvs))
      new Code(L, L, null)
    } else {
      new Code(start, end,
        lir.methodInsn(INVOKEVIRTUAL, lmethod,
          lir.load(new lir.Parameter(null, 0, fb.classBuilder.ti)) +: argvs))
    }
  }
}

trait DependentFunction[F >: Null <: AnyRef] extends FunctionBuilder[F] {
  var setFields: mutable.ArrayBuffer[(lir.ValueX) => Code[Unit]] = new mutable.ArrayBuffer()

  def addField[T : TypeInfo](value: Code[T]): ClassFieldRef[T] = {
    val cfr = newField[T]
    setFields += { (obj: lir.ValueX) =>
      value.end.append(lir.putField(name, cfr.name, typeInfo[T], obj, value.v))
      new Code(value.start, value.end, null)
    }
    cfr
  }

  def addFieldAny[T](value: Code[_])(implicit ti: TypeInfo[T]): ClassFieldRef[T] =
    addField(value.asInstanceOf[Code[T]])


  def newInstance(mb: MethodBuilder)(implicit fct: ClassTag[F]): Code[F] = {
    val L = new lir.Block()

    val obj = mb.lmethod.genLocal("new_dep_fun", classBuilder.ti)
    L.append(lir.store(obj, lir.newInstance(classBuilder.ti)))
    L.append(lir.methodStmt(INVOKESPECIAL, classBuilder.lInit, Array(lir.load(obj))))

    var end = L
    setFields.foreach { f =>
      val c = f(lir.load(obj))
      end.append(lir.goto(c.start))
      end = c.end
    }
    new Code(L, end, lir.load(obj))
  }

  override def result(pw: Option[PrintWriter]): () => F =
    throw new UnsupportedOperationException("cannot call result() on a dependent function")

}

class DependentFunctionBuilder[F >: Null <: AnyRef : TypeInfo : ClassTag](
  parameterTypeInfo: Array[MaybeGenericTypeInfo[_]],
  returnTypeInfo: MaybeGenericTypeInfo[_],
  packageName: String = "is/hail/codegen/generated",
  initModule: ModuleBuilder = null
) extends FunctionBuilder[F](parameterTypeInfo, returnTypeInfo, packageName, initModule = initModule) with DependentFunction[F]

class FunctionBuilder[F >: Null](
  val parameterTypeInfo: Array[MaybeGenericTypeInfo[_]],
  val returnTypeInfo: MaybeGenericTypeInfo[_],
  val packageName: String = "is/hail/codegen/generated",
  namePrefix: String = null,
  initModule: ModuleBuilder = null)(implicit val interfaceTi: TypeInfo[F]) {
  import FunctionBuilder._

  val module: ModuleBuilder =
    if (initModule == null)
      new ModuleBuilder()
  else
      initModule

  val classBuilder: ClassBuilder[F] = new ClassBuilder[F](
    module,
    packageName + "/C" + Option(namePrefix).map(n => s"_${n}_").getOrElse("") + newUniqueID())

  val name: String = classBuilder.name

  private[this] val methodMemo: mutable.Map[Any, MethodBuilder] = mutable.HashMap.empty

  def getOrDefineMethod(suffix: String, key: Any, argsInfo: Array[TypeInfo[_]], returnInfo: TypeInfo[_])
    (f: MethodBuilder => Unit): MethodBuilder = {
    methodMemo.get(key) match {
      case Some(mb) => mb
      case None =>
        val mb = newMethod(suffix, argsInfo, returnInfo)
        f(mb)
        methodMemo(key) = mb
        mb
    }
  }

  private[this] lazy val _apply_method: MethodBuilder = {
    val m = new MethodBuilder(this, "apply", parameterTypeInfo.map(_.base), returnTypeInfo.base)
    if (parameterTypeInfo.exists(_.isGeneric) || returnTypeInfo.isGeneric) {
      val generic = new MethodBuilder(this, "apply", parameterTypeInfo.map(_.generic), returnTypeInfo.generic)
      classBuilder.addMethod(generic)
      generic.emit(
        returnTypeInfo.castToGeneric(
          m.invoke(parameterTypeInfo.zipWithIndex.map { case (ti, i) =>
            ti.castFromGeneric(generic.getArg(i + 1)(ti.generic))
          }: _*)))
    }
    m
  }

  def apply_method: MethodBuilder = _apply_method

  def newField[T: TypeInfo]: ClassFieldRef[T] = newField()

  def newField[T: TypeInfo](name: String = null): ClassFieldRef[T] =
    new ClassFieldRef[T](this, classBuilder.genField[T](name))

  def newLazyField[T: TypeInfo](setup: => Code[T], name: String = null): LazyFieldRef[T] =
    new LazyFieldRef[T](this, name, setup)

  val lazyFieldMemo: mutable.Map[Any, LazyFieldRef[_]] = mutable.Map.empty

  def getOrDefineLazyField[T: TypeInfo](setup: Code[T], id: Any): LazyFieldRef[T] = {
    lazyFieldMemo.getOrElseUpdate(id, newLazyField[T](setup)).asInstanceOf[LazyFieldRef[T]]
  }

  def newLocal[T](implicit tti: TypeInfo[T]): LocalRef[T] = newLocal()

  def newLocal[T](name: String = null)(implicit tti: TypeInfo[T]): LocalRef[T] = apply_method.newLocal[T](name)

  def getArg[T](i: Int)(implicit tti: TypeInfo[T]): Settable[T] = apply_method.getArg[T](i)

  def emit(c: Code[_]) = apply_method.emit(c)

  def newMethod(suffix: String, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilder = {
    val mb = new MethodBuilder(this, classBuilder.genName("m", suffix), argsInfo, returnInfo)
    classBuilder.addMethod(mb)
    mb
  }

  def newMethod(argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilder =
    newMethod("method", argsInfo, returnInfo)

  def newMethod[R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](), typeInfo[R])

  def newMethod[A: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C], typeInfo[D]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C], typeInfo[D], typeInfo[E]), typeInfo[R])

  def classAsBytes(print: Option[PrintWriter] = None): Array[Byte] = classBuilder.classAsBytes(print)

  classBuilder.addInterface(interfaceTi.iname)

  def result(print: Option[PrintWriter] = None): () => F = classBuilder.result(print)
}

class Function2Builder[A1 : TypeInfo, A2 : TypeInfo, R : TypeInfo](name: String)
  extends FunctionBuilder[AsmFunction2[A1, A2, R]](Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R], namePrefix = name) {

  def arg1 = getArg[A1](1)

  def arg2 = getArg[A2](2)
}

class Function3Builder[A1 : TypeInfo, A2 : TypeInfo, A3 : TypeInfo, R : TypeInfo](name: String)
  extends FunctionBuilder[AsmFunction3[A1, A2, A3, R]](Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3]), GenericTypeInfo[R], namePrefix = name) {

  def arg1 = getArg[A1](1)

  def arg2 = getArg[A2](2)

  def arg3 = getArg[A3](3)
}

class Function4Builder[A1 : TypeInfo, A2 : TypeInfo, A3 : TypeInfo, A4 : TypeInfo, R : TypeInfo](name: String)
  extends FunctionBuilder[AsmFunction4[A1, A2, A3, A4, R]](Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3], GenericTypeInfo[A4]), GenericTypeInfo[R], namePrefix = name) {

  def arg1 = getArg[A1](1)

  def arg2 = getArg[A2](2)

  def arg3 = getArg[A3](3)

  def arg4 = getArg[A4](4)
}

class Function5Builder[A1 : TypeInfo, A2 : TypeInfo, A3 : TypeInfo, A4 : TypeInfo, A5 : TypeInfo, R : TypeInfo](name: String)
  extends FunctionBuilder[AsmFunction5[A1, A2, A3, A4, A5, R]](Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3], GenericTypeInfo[A4], GenericTypeInfo[A5]), GenericTypeInfo[R], namePrefix = name) {

  def arg1 = getArg[A1](1)

  def arg2 = getArg[A2](2)

  def arg3 = getArg[A3](3)

  def arg4 = getArg[A4](4)

  def arg5 = getArg[A5](5)
}
