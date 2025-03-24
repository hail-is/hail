package is.hail.backend

import is.hail.{HailContext, HailFeatureFlags}
import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s.HailClassLoader
import is.hail.backend.local.LocalTaskContext
import is.hail.expr.ir.{BaseIR, CodeCacheKey, CompiledFunction}
import is.hail.expr.ir.LoweredTableReader.LoweredTableReaderCoercer
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.io.fs.FS
import is.hail.linalg.BlockMatrix
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.collection.mutable

import java.io._
import java.security.SecureRandom

import sourcecode.Enclosing

trait TempFileManager extends AutoCloseable {
  def newTmpPath(tmpdir: String, prefix: String, extension: String = null): String
}

class OwningTempFileManager(fs: FS) extends TempFileManager {
  private[this] val tmpPaths = mutable.ArrayBuffer[String]()

  override def newTmpPath(tmpdir: String, prefix: String, extension: String): String = {
    val tmp = ExecuteContext.createTmpPathNoCleanup(tmpdir, prefix, extension)
    tmpPaths += tmp
    tmp
  }

  override def close(): Unit = {
    for (p <- tmpPaths)
      fs.delete(p, recursive = true)
    tmpPaths.clear()
  }
}

class NonOwningTempFileManager private (owner: TempFileManager) extends TempFileManager {
  override def newTmpPath(tmpdir: String, prefix: String, extension: String): String =
    owner.newTmpPath(tmpdir, prefix, extension)

  override def close(): Unit = ()
}

object NonOwningTempFileManager {
  def apply(owner: TempFileManager): TempFileManager =
    owner match {
      case _: NonOwningTempFileManager => owner
      case _ => new NonOwningTempFileManager(owner)
    }
}

object ExecuteContext {
  def scoped[T](f: ExecuteContext => T)(implicit E: Enclosing): T = {
    val result = HailContext.sparkBackend("ExecuteContext.scoped").withExecuteContext(
      selfContainedExecution = false
    )(f)
    result
  }

  def scoped[T](
    tmpdir: String,
    localTmpdir: String,
    backend: Backend,
    references: Map[String, ReferenceGenome],
    fs: FS,
    timer: ExecutionTimer,
    tempFileManager: TempFileManager,
    theHailClassLoader: HailClassLoader,
    flags: HailFeatureFlags,
    backendContext: BackendContext,
    irMetadata: IrMetadata,
    blockMatrixCache: mutable.Map[String, BlockMatrix],
    codeCache: mutable.Map[CodeCacheKey, CompiledFunction[_]],
    irCache: mutable.Map[Int, BaseIR],
    coercerCache: mutable.Map[Any, LoweredTableReaderCoercer],
  )(
    f: ExecuteContext => T
  ): T = {
    RegionPool.scoped { pool =>
      pool.scopedRegion { region =>
        using(new ExecuteContext(
          tmpdir,
          localTmpdir,
          backend,
          references,
          fs,
          region,
          timer,
          tempFileManager,
          theHailClassLoader,
          flags,
          backendContext,
          irMetadata,
          blockMatrixCache,
          codeCache,
          irCache,
          coercerCache,
        ))(f(_))
      }
    }
  }

  def createTmpPathNoCleanup(tmpdir: String, prefix: String, extension: String = null): String = {
    val random = new SecureRandom()
    val alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    val token = (0 until 22).map(_ => alphabet(random.nextInt(alphabet.length))).mkString
    if (extension != null)
      s"$tmpdir/$prefix-$token.$extension"
    else
      s"$tmpdir/$prefix-$token"
  }
}

class ExecuteContext(
  val tmpdir: String,
  val localTmpdir: String,
  val backend: Backend,
  val references: Map[String, ReferenceGenome],
  val fs: FS,
  val r: Region,
  val timer: ExecutionTimer,
  _tempFileManager: TempFileManager,
  val theHailClassLoader: HailClassLoader,
  val flags: HailFeatureFlags,
  val backendContext: BackendContext,
  val irMetadata: IrMetadata,
  val BlockMatrixCache: mutable.Map[String, BlockMatrix],
  val CodeCache: mutable.Map[CodeCacheKey, CompiledFunction[_]],
  val PersistedIrCache: mutable.Map[Int, BaseIR],
  val PersistedCoercerCache: mutable.Map[Any, LoweredTableReaderCoercer],
) extends Closeable {

  val rngNonce: Long =
    try
      java.lang.Long.decode(getFlag("rng_nonce"))
    catch {
      case exc: NumberFormatException =>
        fatal(
          s"Could not parse flag rng_nonce as a 64-bit signed integer: ${getFlag("rng_nonce")}",
          exc,
        )
    }

  val stateManager = HailStateManager(references)

  val tempFileManager: TempFileManager =
    if (_tempFileManager != null) _tempFileManager else new OwningTempFileManager(fs)

  def fsBc: BroadcastValue[FS] = fs.broadcast

  val memo: mutable.Map[Any, Any] = new mutable.HashMap[Any, Any]()

  val taskContext: HailTaskContext = new LocalTaskContext(0, 0)

  def scopedExecution[T](
    f: (HailClassLoader, FS, HailTaskContext, Region) => T
  )(implicit E: Enclosing
  ): T =
    using(new LocalTaskContext(0, 0)) { tc =>
      time {
        f(theHailClassLoader, fs, tc, r)
      }
    }

  def createTmpPath(prefix: String, extension: String = null, local: Boolean = false): String =
    tempFileManager.newTmpPath(if (local) localTmpdir else tmpdir, prefix, extension)

  def getFlag(name: String): String = flags.get(name)

  def shouldWriteIRFiles(): Boolean = getFlag("write_ir_files") != null

  def shouldNotLogIR(): Boolean = flags.get("no_ir_logging") != null

  def shouldLogIR(): Boolean = !shouldNotLogIR()

  def close(): Unit = {
    tempFileManager.close()
    taskContext.close()
  }

  def time[A](block: => A)(implicit E: Enclosing): A =
    timer.time(E.value)(block)

  def local[A](
    tmpdir: String = this.tmpdir,
    localTmpdir: String = this.localTmpdir,
    backend: Backend = this.backend,
    references: Map[String, ReferenceGenome] = this.references,
    fs: FS = this.fs,
    r: Region = this.r,
    timer: ExecutionTimer = this.timer,
    tempFileManager: TempFileManager = NonOwningTempFileManager(this.tempFileManager),
    theHailClassLoader: HailClassLoader = this.theHailClassLoader,
    flags: HailFeatureFlags = this.flags,
    backendContext: BackendContext = this.backendContext,
    irMetadata: IrMetadata = this.irMetadata,
    blockMatrixCache: mutable.Map[String, BlockMatrix] = this.BlockMatrixCache,
    codeCache: mutable.Map[CodeCacheKey, CompiledFunction[_]] = this.CodeCache,
    persistedIrCache: mutable.Map[Int, BaseIR] = this.PersistedIrCache,
    persistedCoercerCache: mutable.Map[Any, LoweredTableReaderCoercer] = this.PersistedCoercerCache,
  )(
    f: ExecuteContext => A
  ): A =
    using(new ExecuteContext(
      tmpdir,
      localTmpdir,
      backend,
      references,
      fs,
      r,
      timer,
      tempFileManager,
      theHailClassLoader,
      flags,
      backendContext,
      irMetadata,
      blockMatrixCache,
      codeCache,
      persistedIrCache,
      persistedCoercerCache,
    ))(f)
}
